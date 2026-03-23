import json
import numpy as np
import pandas as pd
from collections import defaultdict

# 读取V20结果
with open('/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v20_ref/detailed_results_merged.json') as f:
    v20_results = json.load(f)

print(f"V20总样本数: {len(v20_results)}")

# 分析CODER使用情况和准确率
coder_used_samples = [r for r in v20_results if r.get('coder_used', False)]
coder_not_used_samples = [r for r in v20_results if not r.get('coder_used', False)]

print(f"\nCODER使用样本: {len(coder_used_samples)}")
print(f"CODER未使用样本: {len(coder_not_used_samples)}")

# 计算CODER使用vs不使用的准确率
if coder_used_samples:
    coder_scores = np.array([r['score'] for r in coder_used_samples])
    coder_acc = coder_scores.mean()
    coder_binary = (coder_scores >= 0.999).mean()
    print(f"\nCODER使用样本 - Weighted: {coder_acc:.4f}, Binary: {coder_binary:.4f}")

if coder_not_used_samples:
    no_coder_scores = np.array([r['score'] for r in coder_not_used_samples])
    no_coder_acc = no_coder_scores.mean()
    no_coder_binary = (no_coder_scores >= 0.999).mean()
    print(f"CODER未使用样本 - Weighted: {no_coder_acc:.4f}, Binary: {no_coder_binary:.4f}")

# 按类型分析CODER使用情况
print("\n===== Per-Type CODER Analysis =====")
type_stats = defaultdict(lambda: {'coder_used': [], 'coder_not_used': []})

for r in v20_results:
    qtype = r['question_type']
    score = r['score']
    if r.get('coder_used', False):
        type_stats[qtype]['coder_used'].append(score)
    else:
        type_stats[qtype]['coder_not_used'].append(score)

print("| 任务 | N(用CODER) | N(不用CODER) | Acc(用CODER) | Acc(不用CODER) |")
print("|------|-----------|-------------|--------------|----------------|")

for qtype in sorted(type_stats.keys()):
    coder_scores = type_stats[qtype]['coder_used']
    no_coder_scores = type_stats[qtype]['coder_not_used']
    
    n_coder = len(coder_scores)
    n_no_coder = len(no_coder_scores)
    
    acc_coder = np.mean(coder_scores) if coder_scores else 0
    acc_no_coder = np.mean(no_coder_scores) if no_coder_scores else 0
    
    print(f"| {qtype} | {n_coder} | {n_no_coder} | {acc_coder:.3f} | {acc_no_coder:.3f} |")

# 计算Debiased结果
parquet_path = '/home/tione/notebook/tianjungu/hf_cache/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d/test_pruned.parquet'
df = pd.read_parquet(parquet_path)
pruned_df = df[df['pruned'] == True].copy()

# 创建V20字典
v20_dict = {(r['scene_name'], r['question']): r for r in v20_results}

# 匹配pruned样本
matched_coder = []
matched_no_coder = []

for _, row in pruned_df.iterrows():
    key = (row['scene_name'], row['question'])
    if key in v20_dict:
        if v20_dict[key].get('coder_used', False):
            matched_coder.append(v20_dict[key]['score'])
        else:
            matched_no_coder.append(v20_dict[key]['score'])

print(f"\n===== Debiased CODER Analysis =====")
print(f"Debiased CODER使用样本: {len(matched_coder)}")
print(f"Debiased CODER未使用样本: {len(matched_no_coder)}")

if matched_coder:
    print(f"Debiased CODER使用 - Weighted: {np.mean(matched_coder):.4f}, Binary: {(np.array(matched_coder) >= 0.999).mean():.4f}")
if matched_no_coder:
    print(f"Debiased CODER未使用 - Weighted: {np.mean(matched_no_coder):.4f}, Binary: {(np.array(matched_no_coder) >= 0.999).mean():.4f}")

# V20整体结果
all_scores = np.array([r['score'] for r in v20_results])
print(f"\n===== V20 Overall =====")
print(f"V20 Full-set: {all_scores.mean():.4f} ({all_scores.mean()*100:.2f}%)")
print(f"V20 Binary: {(all_scores >= 0.999).mean():.4f}")

# Debiased整体
matched_scores = matched_coder + matched_no_coder
if matched_scores:
    print(f"V20 Debiased: {np.mean(matched_scores):.4f} ({np.mean(matched_scores)*100:.2f}%)")
