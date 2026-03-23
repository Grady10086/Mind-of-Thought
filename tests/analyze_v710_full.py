#!/usr/bin/env python3
import json

with open('outputs/v710_cross_product_20260206_142754/results_neg_y_axis.json') as f:
    data = json.load(f)

print('='*80)
print('V7.10 neg_y_axis 详细分析')
print('='*80)

# 分类统计
total = len(data)
found_all = [r for r in data if r.get('objects_found',{}).get('standing') and 
             r.get('objects_found',{}).get('facing') and r.get('objects_found',{}).get('target')]
not_found = [r for r in data if r not in found_all]

correct_found = [r for r in found_all if r.get('correct')]
wrong_found = [r for r in found_all if not r.get('correct')]

print(f'\n总样本: {total}')
print(f'找到所有3个物体: {len(found_all)} ({len(found_all)/total*100:.1f}%)')
print(f'  - 正确: {len(correct_found)} ({len(correct_found)/len(found_all)*100:.1f}%)')
print(f'  - 错误: {len(wrong_found)} ({len(wrong_found)/len(found_all)*100:.1f}%)')
print(f'未找到所有物体: {len(not_found)} ({len(not_found)/total*100:.1f}%)')

print('\n' + '='*80)
print('问题1: 未找到所有物体的案例分析')
print('='*80)

# 分析未找到的原因
missing_objects = {}
for r in not_found:
    reasoning = r.get('reasoning', '')
    if 'Missing:' in reasoning:
        missing_part = reasoning.split('Missing:')[1].strip()
        for item in missing_part.split(','):
            item = item.strip()
            if '(' in item:
                obj_type = item.split('(')[0].strip()
                obj_name = item.split('(')[1].replace(')', '').strip()
                if obj_name not in missing_objects:
                    missing_objects[obj_name] = 0
                missing_objects[obj_name] += 1

print('\n缺失物体频次统计:')
for obj, count in sorted(missing_objects.items(), key=lambda x: -x[1])[:15]:
    print(f'  {obj}: {count}次')

print('\n' + '='*80)
print('问题2: 找到物体但方向错误的案例 (9个)')
print('='*80)

for i, r in enumerate(wrong_found):
    # 获取GT方向
    opts = r.get('options', [])
    gt_dir = None
    for opt in opts:
        if opt.startswith(r['ground_truth'] + '.'):
            gt_dir = opt.split('.')[1].strip()
            break
    
    print(f'\n[错误{i+1}] Scene: {r["scene_name"]}')
    print(f'  Question: {r["question"][:70]}...')
    print(f'  GT: {r["ground_truth"]} ({gt_dir})')
    print(f'  Pred: {r["prediction"]} ({r.get("predicted_direction")})')
    di = r.get('debug_info', {})
    if 'proj_forward' in di:
        print(f'  proj_forward={di["proj_forward"]:.3f}, proj_right={di["proj_right"]:.3f}')
    print(f'  Reasoning: {r.get("reasoning", "")[:80]}')

print('\n' + '='*80)
print('问题3: 正确案例分析 (9个)')
print('='*80)

for i, r in enumerate(correct_found[:5]):
    opts = r.get('options', [])
    gt_dir = None
    for opt in opts:
        if opt.startswith(r['ground_truth'] + '.'):
            gt_dir = opt.split('.')[1].strip()
            break
    
    print(f'\n[正确{i+1}] Scene: {r["scene_name"]}')
    print(f'  GT: {r["ground_truth"]} ({gt_dir}), Pred: {r["prediction"]} ({r.get("predicted_direction")})')
    di = r.get('debug_info', {})
    if 'proj_forward' in di:
        print(f'  proj_forward={di["proj_forward"]:.3f}, proj_right={di["proj_right"]:.3f}')
