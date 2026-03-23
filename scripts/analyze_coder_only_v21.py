import json
import os
import numpy as np
import pandas as pd
import re
from collections import defaultdict

# 读取V21结果
all_results = []
base_path = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'
for g in range(8):
    filepath = f'{base_path}/gpu{g}/detailed_results.json'
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)

print(f"V21总样本数: {len(all_results)}")

# 从reasoning字段判断是否是CODER-only路径
coder_only_samples = []
vl_involved_samples = []

for r in all_results:
    reasoning = r.get('reasoning', '')
    # CODER-only: 包含[num_coder_path]或[coder]但没有VL轮次标记
    is_coder_only = '[num_coder_path]' in reasoning or ('[coder]' in reasoning and '[P1:vl_global]' not in reasoning and '[vl_full]' not in reasoning)
    
    # VL参与: 有[P1:vl_global]或[vl_full]或[R1]等标记
    has_vl = '[P1:vl_global]' in reasoning or '[vl_full]' in reasoning or '[R1' in reasoning
    
    if is_coder_only and not has_vl:
        coder_only_samples.append(r)
    else:
        vl_involved_samples.append(r)

print(f"\nCODER-only样本: {len(coder_only_samples)}")
print(f"VL参与样本: {len(vl_involved_samples)}")

# 计算准确率
if coder_only_samples:
    coder_scores = np.array([r['score'] for r in coder_only_samples])
    print(f"\nCODER-only - Weighted: {coder_scores.mean():.4f}, Binary: {(coder_scores >= 0.999).mean():.4f}")
    
    # 按类型分析
    type_scores = defaultdict(list)
    for r in coder_only_samples:
        type_scores[r['question_type']].append(r['score'])
    
    print("\n===== CODER-only Per Type =====")
    for qtype in sorted(type_scores.keys()):
        scores = np.array(type_scores[qtype])
        print(f"{qtype}: {scores.mean():.4f} ({(scores >= 0.999).sum()}/{len(scores)})")

if vl_involved_samples:
    vl_scores = np.array([r['score'] for r in vl_involved_samples])
    print(f"\nVL-involved - Weighted: {vl_scores.mean():.4f}, Binary: {(vl_scores >= 0.999).mean():.4f}")

# 计算Debiased结果
parquet_path = '/home/tione/notebook/tianjungu/hf_cache/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d/test_pruned.parquet'
df = pd.read_parquet(parquet_path)
pruned_df = df[df['pruned'] == True].copy()

v21_dict = {(r['scene_name'], r['question']): r for r in all_results}

matched_coder_only = []
matched_vl_involved = []

for _, row in pruned_df.iterrows():
    key = (row['scene_name'], row['question'])
    if key in v21_dict:
        r = v21_dict[key]
        reasoning = r.get('reasoning', '')
        is_coder_only = '[num_coder_path]' in reasoning or ('[coder]' in reasoning and '[P1:vl_global]' not in reasoning and '[vl_full]' not in reasoning)
        has_vl = '[P1:vl_global]' in reasoning or '[vl_full]' in reasoning or '[R1' in reasoning
        
        if is_coder_only and not has_vl:
            matched_coder_only.append(r['score'])
        else:
            matched_vl_involved.append(r['score'])

print(f"\n===== Debiased Results =====")
print(f"Debiased CODER-only样本: {len(matched_coder_only)}")
print(f"Debiased VL-involved样本: {len(matched_vl_involved)}")

if matched_coder_only:
    print(f"Debiased CODER-only - Weighted: {np.mean(matched_coder_only):.4f}, Binary: {(np.array(matched_coder_only) >= 0.999).mean():.4f}")
if matched_vl_involved:
    print(f"Debiased VL-involved - Weighted: {np.mean(matched_vl_involved):.4f}, Binary: {(np.array(matched_vl_involved) >= 0.999).mean():.4f}")
