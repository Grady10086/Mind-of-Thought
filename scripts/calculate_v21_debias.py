import json
import os
import pandas as pd
import numpy as np

# 1. 读取V21结果
all_results = []
base_path = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'
for g in range(8):
    filepath = f'{base_path}/gpu{g}/detailed_results.json'
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict) and 'results' in data:
                all_results.extend(data['results'])

print(f'V21总样本数: {len(all_results)}')

# 2. 读取pruned文件
parquet_path = '/home/tione/notebook/tianjungu/hf_cache/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d/test_pruned.parquet'
df = pd.read_parquet(parquet_path)

print(f'Pruned文件总样本数: {len(df)}')
print(f'Pruned=True: {df["pruned"].sum()}')
print(f'Pruned=False: {(~df["pruned"]).sum()}')

# 3. 筛选pruned=True的样本
pruned_df = df[df['pruned'] == True].copy()
print(f'\n用于debias评估的样本数: {len(pruned_df)}')

# 4. 创建V21结果的字典，用于匹配
# 使用(scene_name, question)作为key
v21_dict = {}
for r in all_results:
    key = (r.get('scene_name'), r.get('question'))
    v21_dict[key] = r

print(f'V21结果字典大小: {len(v21_dict)}')

# 5. 匹配并计算debias准确率
matched_scores = []
unmatched = []

for _, row in pruned_df.iterrows():
    key = (row['scene_name'], row['question'])
    if key in v21_dict:
        matched_scores.append(v21_dict[key]['score'])
    else:
        unmatched.append(key)

print(f'\n匹配成功: {len(matched_scores)}/{len(pruned_df)}')
print(f'未匹配: {len(unmatched)}')

if matched_scores:
    scores = np.array(matched_scores)
    debias_acc = scores.mean()
    binary_acc = (scores >= 0.999).mean()
    print(f'\n===== V21 Debiased Results =====')
    print(f'Weighted Accuracy: {debias_acc:.4f}')
    print(f'Binary Accuracy:   {binary_acc:.4f}')
    print(f'Correct/Total:     {(scores >= 0.999).sum()}/{len(scores)}')
