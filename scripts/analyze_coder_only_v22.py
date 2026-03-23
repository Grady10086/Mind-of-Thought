import json
import os
import numpy as np
import pandas as pd
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

# 更精确地判断CODER-only路径
coder_only_samples = []
vl_with_coder_samples = []
vl_only_samples = []

for r in all_results:
    reasoning = r.get('reasoning', '')
    question_type = r.get('question_type', '')
    
    # NA任务 (numerical tasks)
    is_numerical = question_type in ['object_abs_distance', 'object_counting', 'object_size_estimation', 'room_size_estimation']
    
    # CODER-only的判断：
    # 1. 有[num_coder_path]标记
    # 2. 或者[numerical]开头，有[coder]，但没有[P1:vl_global]/[vl_full]/[R1]
    has_num_coder_path = '[num_coder_path]' in reasoning
    has_numerical_tag = '[numerical]' in reasoning
    has_coder = '[coder]' in reasoning
    has_vl = '[P1:vl_global]' in reasoning or '[vl_full]' in reasoning or '[R1' in reasoning
    
    if has_num_coder_path or (has_numerical_tag and has_coder and not has_vl):
        coder_only_samples.append(r)
    elif has_coder and has_vl:
        vl_with_coder_samples.append(r)
    elif has_vl:
        vl_only_samples.append(r)
    else:
        # 其他情况，可能是不涉及CODER的MCA任务
        vl_only_samples.append(r)

print(f"\nCODER-only样本: {len(coder_only_samples)}")
print(f"VL+CODER样本: {len(vl_with_coder_samples)}")
print(f"VL-only样本: {len(vl_only_samples)}")

# 计算各类别的准确率
def calc_stats(samples, name):
    if not samples:
        return
    scores = np.array([r['score'] for r in samples])
    print(f"\n{name}:")
    print(f"  Count: {len(samples)}")
    print(f"  Weighted: {scores.mean():.4f}")
    print(f"  Binary: {(scores >= 0.999).mean():.4f}")
    
    # Per type
    type_scores = defaultdict(list)
    for r in samples:
        type_scores[r['question_type']].append(r['score'])
    
    print(f"  Per-type:")
    for qtype in sorted(type_scores.keys()):
        ts = np.array(type_scores[qtype])
        print(f"    {qtype}: {ts.mean():.4f} ({(ts >= 0.999).sum()}/{len(ts)})")

calc_stats(coder_only_samples, "CODER-only")
calc_stats(vl_with_coder_samples, "VL+CODER")
calc_stats(vl_only_samples, "VL-only")

# Debiased分析
parquet_path = '/home/tione/notebook/tianjungu/hf_cache/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d/test_pruned.parquet'
df = pd.read_parquet(parquet_path)
pruned_df = df[df['pruned'] == True].copy()

v21_dict = {(r['scene_name'], r['question']): r for r in all_results}

def classify_sample(r):
    reasoning = r.get('reasoning', '')
    has_num_coder_path = '[num_coder_path]' in reasoning
    has_numerical_tag = '[numerical]' in reasoning
    has_coder = '[coder]' in reasoning
    has_vl = '[P1:vl_global]' in reasoning or '[vl_full]' in reasoning or '[R1' in reasoning
    
    if has_num_coder_path or (has_numerical_tag and has_coder and not has_vl):
        return 'coder_only'
    elif has_coder and has_vl:
        return 'vl_with_coder'
    else:
        return 'vl_only'

matched_coder_only = []
matched_vl_with_coder = []
matched_vl_only = []

for _, row in pruned_df.iterrows():
    key = (row['scene_name'], row['question'])
    if key in v21_dict:
        r = v21_dict[key]
        category = classify_sample(r)
        if category == 'coder_only':
            matched_coder_only.append(r['score'])
        elif category == 'vl_with_coder':
            matched_vl_with_coder.append(r['score'])
        else:
            matched_vl_only.append(r['score'])

print(f"\n===== Debiased Results =====")
if matched_coder_only:
    print(f"Debiased CODER-only: {len(matched_coder_only)} samples, Acc={np.mean(matched_coder_only):.4f}")
if matched_vl_with_coder:
    print(f"Debiased VL+CODER: {len(matched_vl_with_coder)} samples, Acc={np.mean(matched_vl_with_coder):.4f}")
if matched_vl_only:
    print(f"Debiased VL-only: {len(matched_vl_only)} samples, Acc={np.mean(matched_vl_only):.4f}")
