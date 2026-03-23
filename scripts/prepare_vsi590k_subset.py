"""
准备VSI-590K微调数据（仅使用已下载的adt和arkitscenes）
"""
import json
import os
from pathlib import Path

# 路径配置
JSONL_PATH = '/home/tione/notebook/tianjungu/hf_cache/datasets--nyu-visionx--VSI-590K/blobs/025355ea328c0f1f73ba997fd52070a1f487248acd1fb9d8ac727886b319d231'
OUTPUT_DIR = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset'
VSI_DATA_ROOT = '/home/tione/notebook/tianjungu/hf_cache/VSI-590K'

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 过滤：只保留adt和arkitscenes数据
available_datasets = {'adt', 'arkitscenes'}
output_jsonl = os.path.join(OUTPUT_DIR, 'train.jsonl')

print(f"读取: {JSONL_PATH}")
print(f"输出: {output_jsonl}")
print(f"仅保留数据集: {available_datasets}")

total = 0
kept = 0
skipped_datasets = set()

with open(JSONL_PATH, 'r') as fin, open(output_jsonl, 'w') as fout:
    for line in fin:
        total += 1
        sample = json.loads(line.strip())
        
        # 获取数据集名称
        if 'video' in sample:
            dataset = sample['video'].split('/')[0]
            media_path = sample['video']
        elif 'image' in sample:
            dataset = sample['image'].split('/')[0]
            media_path = sample['image']
        else:
            continue
        
        # 仅保留已下载的数据集
        if dataset in available_datasets:
            # 更新路径为绝对路径
            if 'video' in sample:
                sample['video'] = os.path.join(VSI_DATA_ROOT, sample['video'])
            elif 'image' in sample:
                sample['image'] = os.path.join(VSI_DATA_ROOT, sample['image'])
            
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
            kept += 1
        else:
            skipped_datasets.add(dataset)
        
        if total % 100000 == 0:
            print(f"处理: {total} 条，保留: {kept} 条")

print(f"\n完成！")
print(f"总样本: {total}")
print(f"保留样本: {kept} ({kept/total*100:.2f}%)")
print(f"跳过的数据集: {skipped_datasets}")
print(f"\n输出文件: {output_jsonl}")
