import pandas as pd
import json
import os
import numpy as np
from collections import Counter

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

print(f'V21总样本数: {len(all_results)}')

# 整体准确率
all_scores = [r['score'] for r in all_results]
print(f'V21整体Weighted Accuracy: {np.mean(all_scores):.4f}')
print(f'V21整体Binary Accuracy: {(np.array(all_scores) >= 0.999).mean():.4f}')

# 读取pruned文件
parquet_path = '/home/tione/notebook/tianjungu/hf_cache/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d/test_pruned.parquet'
df = pd.read_parquet(parquet_path)
pruned_df = df[df['pruned'] == True].copy()

print(f'\nPruned样本数: {len(pruned_df)}')

# 检查原始V21中的问题类型分布
v21_types = Counter([r['question_type'] for r in all_results])
pruned_types = Counter(pruned_df['question_type'].tolist())

print('\nV21问题类型分布:')
for t, c in sorted(v21_types.items(), key=lambda x: -x[1]):
    print(f'  {t}: {c}')

print('\nPruned问题类型分布:')
for t, c in sorted(pruned_types.items(), key=lambda x: -x[1]):
    print(f'  {t}: {c}')

# 匹配并计算
v21_dict = {(r.get('scene_name'), r.get('question')): r for r in all_results}

matched_scores = []
for _, row in pruned_df.iterrows():
    key = (row['scene_name'], row['question'])
    if key in v21_dict:
        matched_scores.append(v21_dict[key]['score'])

print(f'\n匹配成功: {len(matched_scores)}/{len(pruned_df)}')

if matched_scores:
    scores = np.array(matched_scores)
    print(f'\n===== V21 Debiased Results =====')
    print(f'Weighted Accuracy: {scores.mean():.4f}')
    print(f'Binary Accuracy:   {(scores >= 0.999).mean():.4f}')
    print(f'Correct/Total:     {(scores >= 0.999).sum()}/{len(scores)}')
    
    # 按类型分析
    print('\n===== Per Type Debiased Results =====')
    for qtype in sorted(pruned_types.keys()):
        type_scores = []
        for _, row in pruned_df.iterrows():
            if row['question_type'] == qtype:
                key = (row['scene_name'], row['question'])
                if key in v21_dict:
                    type_scores.append(v21_dict[key]['score'])
        if type_scores:
            ts = np.array(type_scores)
            print(f'{qtype}: {ts.mean():.4f} ({(ts >= 0.999).sum()}/{len(ts)})')
