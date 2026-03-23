#!/usr/bin/env python3
"""分析VSI-590K中各dataset的数据分布"""
import json
from collections import Counter, defaultdict

vsi590k_path = "/home/tione/notebook/tianjungu/hf_cache/datasets--nyu-visionx--VSI-590K/snapshots/346fbd4e41dec974bf24894d0541a49327ee6669/vsi_590k.jsonl"

dataset_counts = Counter()
task_by_dataset = defaultdict(Counter)

print("分析VSI-590K数据集...")
with open(vsi590k_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        video = data.get('video', '')
        task = data.get('question_type', 'unknown')
        
        dataset = video.split('/')[0] if video else 'unknown'
        dataset_counts[dataset] += 1
        task_by_dataset[dataset][task] += 1

print("\n数据集分布:")
for dataset, count in sorted(dataset_counts.items(), key=lambda x: -x[1]):
    print(f"  {dataset:20s}: {count:6d} 样本")

print("\nscannet中每个任务的样本数:")
if 'scannet' in task_by_dataset:
    for task, count in sorted(task_by_dataset['scannet'].items(), key=lambda x: -x[1]):
        print(f"  {task:35s}: {count:5d}")
    print(f"\n  总计: {sum(task_by_dataset['scannet'].values())}")
