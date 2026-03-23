#!/usr/bin/env python3
"""分析缺失物体 - 从数据集中获取原始问题"""
import os
import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 加载数据集
from datasets import load_dataset
ds = load_dataset('nyu-visionx/VSI-Bench', split='test',
                  cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench')

# 构建 scene_name -> question 映射
direction_types = ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard']
scene_to_question = {}
for item in ds:
    if item['question_type'] in direction_types:
        scene_to_question[item['scene_name']] = item['question']

# 读取 DA3 1.1 结果
results_file = Path('outputs/baseline_comparison_20260207_082815/results_da3_11.json')
with open(results_file) as f:
    results = json.load(f)

def parse_question(q):
    q_lower = q.lower()
    pattern = r'standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+)'
    match = re.search(pattern, q_lower)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

missing = {'standing': [], 'facing': [], 'target': []}
found_correct = []
found_wrong = []

for r in results:
    scene = r['scene_name']
    objs = r.get('objects_found', {})
    q = scene_to_question.get(scene, '')
    
    if not q:
        continue
    
    standing, facing, target = parse_question(q)
    
    if not objs.get('standing') and standing:
        missing['standing'].append(standing)
    if not objs.get('facing') and facing:
        missing['facing'].append(facing)
    if not objs.get('target') and target:
        missing['target'].append(target)
    
    # 统计找到物体时的正确/错误
    if objs.get('standing') and objs.get('facing') and objs.get('target'):
        if r.get('correct'):
            found_correct.append({'standing': standing, 'facing': facing, 'target': target, 'debug': r.get('debug_info', {})})
        else:
            found_wrong.append({'standing': standing, 'facing': facing, 'target': target, 'debug': r.get('debug_info', {}), 'gt': r.get('ground_truth'), 'pred': r.get('predicted_direction')})

print('='*60)
print('DA3 1.1 (16帧) 缺失物体分析')
print('='*60)

print(f'\n总样本: {len(results)}')
print(f'找到所有物体: {len(found_correct) + len(found_wrong)}')
print(f'  - 正确: {len(found_correct)}')
print(f'  - 错误: {len(found_wrong)}')

print('\n=== 缺失物体统计 ===')
all_missing = []
for role, objs in missing.items():
    if objs:
        counts = Counter(objs)
        print(f'\n{role} 缺失:')
        for obj, cnt in counts.most_common(10):
            print(f'  {obj}: {cnt}次')
            all_missing.extend([obj] * cnt)

print('\n=== 总体缺失物体统计 ===')
total_counts = Counter(all_missing)
for obj, cnt in total_counts.most_common(15):
    print(f'  {obj}: {cnt}次')

print('\n=== 错误案例分析 ===')
for i, w in enumerate(found_wrong[:5]):
    print(f'\n案例 {i+1}:')
    print(f'  问题: standing_by={w["standing"]}, facing={w["facing"]}, target={w["target"]}')
    print(f'  GT: {w["gt"]}, Pred: {w["pred"]}')
    print(f'  Debug: {w["debug"]}')
