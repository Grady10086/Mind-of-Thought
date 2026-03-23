import pandas as pd
import numpy as np

# 你提供的V21正确结果 (基于Qwen3-VL-8B)
v21_results = {
    'object_rel_direction_easy': {'N': 217, 'weighted': 0.576, 'binary': 0.576},
    'object_rel_direction_hard': {'N': 373, 'weighted': 0.421, 'binary': 0.421},
    'obj_appearance_order': {'N': 618, 'weighted': 0.618, 'binary': 0.618},
    'object_abs_distance': {'N': 834, 'weighted': 0.803, 'binary': 0.803},
    'object_rel_direction_medium': {'N': 378, 'weighted': 0.500, 'binary': 0.500},
    'object_rel_distance': {'N': 710, 'weighted': 0.504, 'binary': 0.504},
    'route_planning': {'N': 194, 'weighted': 0.320, 'binary': 0.320},
    'object_counting': {'N': 565, 'weighted': 0.843, 'binary': 0.843},
    'object_size_estimation': {'N': 953, 'weighted': 0.955, 'binary': 0.955},
    'room_size_estimation': {'N': 288, 'weighted': 0.976, 'binary': 0.976},
}

# 读取pruned文件
parquet_path = '/home/tione/notebook/tianjungu/hf_cache/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d/test_pruned.parquet'
df = pd.read_parquet(parquet_path)
pruned_df = df[df['pruned'] == True].copy()

print("===== V21 (Qwen3-VL-8B) Debiased Results =====\n")

# 按类型统计pruned样本
type_counts = pruned_df['question_type'].value_counts()

print("| 任务 | N (pruned) | V21 Weighted | V21 Binary |")
print("|------|------------|--------------|------------|")

total_pruned = 0
total_weighted_sum = 0
total_binary_sum = 0

for qtype in sorted(v21_results.keys()):
    if qtype in type_counts.index:
        n_pruned = type_counts[qtype]
        weighted_acc = v21_results[qtype]['weighted']
        binary_acc = v21_results[qtype]['binary']
        
        print(f"| {qtype} | {n_pruned} | {weighted_acc:.3f} | {binary_acc:.3f} |")
        
        total_pruned += n_pruned
        total_weighted_sum += n_pruned * weighted_acc
        total_binary_sum += n_pruned * binary_acc

weighted_overall = total_weighted_sum / total_pruned
binary_overall = total_binary_sum / total_pruned

print(f"| **Overall** | **{total_pruned}** | **{weighted_overall:.3f}** | **{binary_overall:.3f}** |")

print(f"\n===== Summary =====")
print(f"Debiased Weighted Accuracy: {weighted_overall:.4f} ({weighted_overall*100:.2f}%)")
print(f"Debiased Binary Accuracy: {binary_overall:.4f} ({binary_overall*100:.2f}%)")
print(f"Correct/Total: {int(binary_overall * total_pruned)}/{total_pruned}")

# 与原始V21 Overall对比
original_overall = 0.704
print(f"\nOriginal V21 Overall: {original_overall:.4f}")
print(f"Debiased V21 Overall: {weighted_overall:.4f}")
print(f"Drop: {original_overall - weighted_overall:.4f} ({(original_overall - weighted_overall)/original_overall*100:.1f}%)")
