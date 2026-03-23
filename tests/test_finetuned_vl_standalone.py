#!/usr/bin/env python3
"""
测试微调后的Qwen3-VL模型 - 独立测试模式
不使用V7的Rule和Mind Map,仅测试VL模型本身的性能

Test 1: 微调模型单独推理 (VL Only)
"""

import os
import sys
import json
import torch
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

def load_finetuned_model(base_model_path, adapter_path, device='cuda'):
    """加载微调后的模型"""
    print(f"Loading base model from {base_model_path}...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        cache_dir='/home/tione/notebook/tianjungu/hf_cache',
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # 合并LoRA权重
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        cache_dir='/home/tione/notebook/tianjungu/hf_cache',
        trust_remote_code=True
    )
    
    return model, processor

def sample_frames(video_path, num_frames=8):
    """从视频中采样帧"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def vl_inference(model, processor, video_frames, question, options=None):
    """VL模型推理 - 简单prompt,不使用mind map"""
    if options:
        # 选择题
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        prompt = f"""Please answer the following question based on the video frames.

Question: {question}

Options:
{options_text}

Please respond with ONLY the option letter (A/B/C/D).
"""
    else:
        # 数值题
        prompt = f"""Please answer the following question based on the video frames.

Question: {question}

Please provide a numerical answer or direct answer.
"""
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_frames, "fps": 1.0},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to('cuda')
    
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
        )
    
    response = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()
    
    return response

def worker_process(gpu_id, samples, base_model_path, adapter_path, result_queue):
    """单GPU工作进程"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 加载模型
    model, processor = load_finetuned_model(base_model_path, adapter_path, device='cuda')
    
    results = []
    for sample in tqdm(samples, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            video_path = sample['video_path']
            question = sample['question']
            options = sample.get('options')
            ground_truth = sample['answer']
            
            # 采样视频帧
            frames = sample_frames(video_path, num_frames=8)
            
            # VL推理
            prediction = vl_inference(model, processor, frames, question, options)
            
            # 评估
            correct = (prediction == ground_truth)
            
            results.append({
                'question_id': sample['question_id'],
                'question_type': sample['question_type'],
                'prediction': prediction,
                'ground_truth': ground_truth,
                'correct': correct,
            })
            
        except Exception as e:
            print(f"GPU {gpu_id} - Error processing {sample['question_id']}: {e}")
            results.append({
                'question_id': sample['question_id'],
                'question_type': sample['question_type'],
                'prediction': 'ERROR',
                'ground_truth': ground_truth,
                'correct': False,
                'error': str(e)
            })
    
    result_queue.put((gpu_id, results))

def main():
    # 参数配置
    BASE_MODEL = '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
    ADAPTER_PATH = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/qwen3vl_10pct_fast/v0-20260211-225557/checkpoint-147'
    NUM_GPUS = 8
    
    print(f"\n{'='*80}")
    print("测试微调后的Qwen3-VL模型 - 独立测试 (无Rule, 无Mind Map)")
    print(f"{'='*80}\n")
    print(f"Base Model: {BASE_MODEL}")
    print(f"LoRA Adapter: {ADAPTER_PATH}")
    print(f"GPUs: {NUM_GPUS}")
    
    # 加载VSIBench数据集 (从JSON文件)
    print("\nLoading VSIBench dataset...")
    vsibench_json = "/home/tione/notebook/tianjungu/projects/Spatial-MLLM/evaluate/annotation/eval_vsibench.json"
    
    with open(vsibench_json, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples from JSON")
    
    # 准备样本
    samples = []
    VSIBENCH_VIDEO_BASE = "/home/tione/notebook/tianjungu/hf_cache/vsibench"
    
    for item in dataset:
        # Get video path from 'path' field (e.g., ./arkitscenes/41069025.mp4)
        rel_path = item.get('path', '')
        if rel_path.startswith('./'):
            rel_path = rel_path[2:]
        
        video_path = os.path.join(VSIBENCH_VIDEO_BASE, rel_path)
        if not os.path.exists(video_path):
            continue
        
        # Parse question and ground truth
        question = item['problem']  # Question text
        solution = item.get('solution', '')  # e.g., "<answer>4</answer>"
        
        # Extract answer from <answer> tags
        import re
        match = re.search(r'<answer>(.*?)</answer>', solution)
        ground_truth = match.group(1) if match else solution
        
        samples.append({
            'question_id': item.get('problem_id', ''),
            'question_type': item.get('original_question_type', 'unknown'),
            'video_path': video_path,
            'question': question,
            'options': item.get('options', None),
            'answer': ground_truth
        })
    
    print(f"Total samples: {len(samples)}")
    
    # 分配样本到各GPU
    samples_per_gpu = len(samples) // NUM_GPUS
    gpu_samples = [samples[i*samples_per_gpu:(i+1)*samples_per_gpu] for i in range(NUM_GPUS)]
    gpu_samples[-1].extend(samples[NUM_GPUS*samples_per_gpu:])
    
    # 多进程测试
    result_queue = mp.Queue()
    processes = []
    
    print(f"\nStarting {NUM_GPUS} GPU processes...")
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(target=worker_process, args=(
            gpu_id, gpu_samples[gpu_id], BASE_MODEL, ADAPTER_PATH, result_queue
        ))
        p.start()
        processes.append(p)
    
    # 收集结果
    all_results = []
    for _ in range(NUM_GPUS):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        print(f"GPU {gpu_id} completed: {len(results)} samples")
    
    for p in processes:
        p.join()
    
    # 统计结果
    by_task = {}
    for r in all_results:
        task = r['question_type']
        if task not in by_task:
            by_task[task] = {'correct': 0, 'total': 0}
        by_task[task]['total'] += 1
        if r['correct']:
            by_task[task]['correct'] += 1
    
    # 输出结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results_finetuned_vl_standalone_{timestamp}.json'
    
    summary = {
        'test_name': 'Finetuned VL Standalone (No Rule, No Mind Map)',
        'model': BASE_MODEL,
        'adapter': ADAPTER_PATH,
        'total_samples': len(all_results),
        'by_task': {}
    }
    
    print(f"\n{'='*80}")
    print("Results - Finetuned VL Standalone")
    print(f"{'='*80}\n")
    
    total_correct = 0
    total_samples = 0
    
    for task in sorted(by_task.keys()):
        correct = by_task[task]['correct']
        total = by_task[task]['total']
        acc = correct / total * 100 if total > 0 else 0
        print(f"{task:40s}: {correct:4d}/{total:4d} = {acc:5.1f}%")
        summary['by_task'][task] = {'accuracy': acc, 'correct': correct, 'total': total}
        total_correct += correct
        total_samples += total
    
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"\n{'Overall':40s}: {total_correct:4d}/{total_samples:4d} = {overall_acc:5.1f}%")
    summary['overall_accuracy'] = overall_acc
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump({'summary': summary, 'details': all_results}, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
