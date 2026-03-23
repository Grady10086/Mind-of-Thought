"""
使用Transformers Trainer微调Qwen3-VL on VSI-590K
"""
import os
import json
import torch
from dataclasses import dataclass
from typing import Dict, List
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
import decord
decord.bridge.set_bridge('torch')

# 环境配置
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'

# 路径配置
MODEL_PATH = "/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct"
DATA_PATH = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train.jsonl"
OUTPUT_DIR = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/qwen3vl_vsi590k_finetuned"

class VSI590KDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_frames=16):
        self.processor = processor
        self.max_frames = max_frames
        self.samples = []
        
        print(f"加载数据: {jsonl_path}")
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        print(f"加载完成: {len(self.samples)} 条样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 提取对话
        question = sample['conversations'][0]['value']
        answer = sample['conversations'][1]['value']
        
        # 加载视频
        video_path = sample.get('video')
        if video_path and os.path.exists(video_path):
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            indices = torch.linspace(0, total_frames-1, self.max_frames).long()
            frames = vr.get_batch(indices).permute(0, 3, 1, 2)  # (T, C, H, W)
        else:
            # 如果视频不存在，返回空
            return None
        
        # 使用processor处理
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        
        inputs = self.processor(
            text=[text],
            videos=[frames],
            padding=True,
            return_tensors="pt"
        )
        
        inputs['labels'] = inputs['input_ids'].clone()
        return {k: v.squeeze(0) for k, v in inputs.items()}

def main():
    print("="*60)
    print("Qwen3-VL 微调 on VSI-590K subset")
    print("="*60)
    
    # 加载模型和processor
    print(f"\n加载模型: {MODEL_PATH}")
    processor = Qwen2VLProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 准备数据
    print(f"\n加载数据集...")
    train_dataset = VSI590KDataset(DATA_PATH, processor)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        deepspeed="ds_config_zero2.json",  # 需要创建deepspeed配置
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("\n开始训练...")
    trainer.train()
    
    print(f"\n保存模型到: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n微调完成！")

if __name__ == "__main__":
    main()
