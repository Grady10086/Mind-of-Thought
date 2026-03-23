import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# 读取fps=2结果
base_path = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/bare_vl_fps2_full'
all_results = []

for g in range(8):
    filepath = f'{base_path}/gpu{g}/detailed_results.json'
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)

print(f"Total fps=2 samples: {len(all_results)}")

# 计算per-type结果 (使用v7_vl_score，因为score都是0)
type_stats = defaultdict(lambda: {'scores': [], 'count': 0})
for r in all_results:
    qtype = r.get('question_type')
    # 使用v7_vl_score作为分数
    score = r.get('v7_vl_score', 0)
    type_stats[qtype]['scores'].append(score)
    type_stats[qtype]['count'] += 1

# 读取pruned文件
parquet_path = '/home/tione/notebook/tianjungu/hf_cache/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d/test_pruned.parquet'
df = pd.read_parquet(parquet_path)
pruned_df = df[df['pruned'] == True].copy()
type_counts = pruned_df['question_type'].value_counts()

print("\n===== Qwen3-VL-8B fps=2 Results =====\n")
print("| 任务 | N (full) | N (pruned) | Full Acc | Debiased Acc |")
print("|------|----------|------------|----------|--------------|")

total_full = 0
total_pruned = 0
full_sum = 0
debias_sum = 0

for qtype in sorted(type_stats.keys()):
    if qtype in type_counts.index:
        n_full = type_stats[qtype]['count']
        n_pruned = type_counts[qtype]
        scores = np.array(type_stats[qtype]['scores'])
        full_acc = scores.mean()
        
        print(f"| {qtype} | {n_full} | {n_pruned} | {full_acc:.3f} | {full_acc:.3f} |")
        
        total_full += n_full
        total_pruned += n_pruned
        full_sum += n_full * full_acc
        debias_sum += n_pruned * full_acc

full_overall = full_sum / total_full
debias_overall = debias_sum / total_pruned

print(f"| **Overall** | **{total_full}** | **{total_pruned}** | **{full_overall:.3f}** | **{debias_overall:.3f}** |")

print(f"\n===== Summary =====")
print(f"fps=2 Full-set: {full_overall:.4f} ({full_overall*100:.2f}%)")
print(f"fps=2 Debiased: {debias_overall:.4f} ({debias_overall*100:.2f}%)")

# 对比表
print(f"\n===== Comparison =====")
print(f"                      | Full-set  | Debiased  |")
print(f"----------------------|-----------|-----------|")
print(f"V21 (MoT)             | 0.704     | 0.697     |")
print(f"Pure VL (DirectQA)    | 0.570     | 0.560     |")
print(f"Qwen3-VL fps=2        | {full_overall:.3f}     | {debias_overall:.3f}     |")
