import json
import pandas as pd
import numpy as np

# 读取Pure VL (DirectQA) 结果
with open('/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/directqa_v21_test_20260128_130722/results.json') as f:
    pure_vl_data = json.load(f)

pure_vl_summary = pure_vl_data['summary']
pure_vl_overall = pure_vl_data['overall'] / 100  # 转换为小数

print("===== Pure VL (Qwen3-VL-8B DirectQA) Results =====\n")
print(f"Overall Full-set Accuracy: {pure_vl_overall:.4f} ({pure_vl_overall*100:.2f}%)")

# 转换为统一格式
pure_vl_results = {}
for qtype, stats in pure_vl_summary.items():
    pure_vl_results[qtype] = {
        'N': stats['count'],
        'accuracy': stats['mean']  # DirectQA已经是binary accuracy
    }

# 读取pruned文件
parquet_path = '/home/tione/notebook/tianjungu/hf_cache/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d/test_pruned.parquet'
df = pd.read_parquet(parquet_path)
pruned_df = df[df['pruned'] == True].copy()

print("\n===== Pure VL Debiased Results =====\n")
print("| 任务 | N (full) | N (pruned) | Full Acc | Debiased Acc |")
print("|------|----------|------------|----------|--------------|")

type_counts = pruned_df['question_type'].value_counts()

total_full = 0
total_pruned = 0
full_sum = 0
debias_sum = 0

for qtype in sorted(pure_vl_results.keys()):
    if qtype in type_counts.index:
        n_full = pure_vl_results[qtype]['N']
        n_pruned = type_counts[qtype]
        full_acc = pure_vl_results[qtype]['accuracy']
        
        print(f"| {qtype} | {n_full} | {n_pruned} | {full_acc:.3f} | {full_acc:.3f} |")
        
        total_full += n_full
        total_pruned += n_pruned
        full_sum += n_full * full_acc
        debias_sum += n_pruned * full_acc

full_overall = full_sum / total_full
debias_overall = debias_sum / total_pruned

print(f"| **Overall** | **{total_full}** | **{total_pruned}** | **{full_overall:.3f}** | **{debias_overall:.3f}** |")

print(f"\n===== Summary =====")
print(f"Pure VL Full-set: {full_overall:.4f} ({full_overall*100:.2f}%)")
print(f"Pure VL Debiased: {debias_overall:.4f} ({debias_overall*100:.2f}%)")
print(f"Correct/Total (debiased): {int(debias_overall * total_pruned)}/{total_pruned}")

# 对比V21
print(f"\n===== V21 vs Pure VL Comparison =====")
print(f"                    | Full-set  | Debiased  |")
print(f"--------------------|-----------|-----------|")
print(f"V21 (MoT)           | {0.704:.3f}     | {0.697:.3f}     |")
print(f"Pure VL (Qwen3-VL)  | {full_overall:.3f}     | {debias_overall:.3f}     |")
print(f"--------------------|-----------|-----------|")
print(f"Gain (V21 - PureVL) | +{0.704-full_overall:.3f}    | +{0.697-debias_overall:.3f}    |")
