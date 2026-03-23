#!/usr/bin/env python3
import json
import glob

# V7.10 neg_y_axis
with open('outputs/v710_cross_product_20260206_142754/results_neg_y_axis.json') as f:
    v710 = json.load(f)

# V7.16 original  
dirs = sorted(glob.glob('outputs/v716_flip_test_*'))
with open(f'{dirs[-1]}/results_original.json') as f:
    v716 = json.load(f)

print('对比 V7.10 neg_y_axis vs V7.16 original')
print('='*60)

# 统计
v710_found = [r for r in v710 if r.get('objects_found',{}).get('standing') and 
              r.get('objects_found',{}).get('facing') and r.get('objects_found',{}).get('target')]
v716_found = [r for r in v716 if r.get('objects_found',{}).get('standing') and 
              r.get('objects_found',{}).get('facing') and r.get('objects_found',{}).get('target')]

v710_correct = sum(1 for r in v710_found if r.get('correct'))
v716_correct = sum(1 for r in v716_found if r.get('correct'))

print(f'V7.10: {v710_correct}/{len(v710_found)} = {v710_correct/len(v710_found)*100:.1f}%')
print(f'V7.16: {v716_correct}/{len(v716_found)} = {v716_correct/len(v716_found)*100:.1f}%')

# 检查具体差异
print('\n具体差异:')
v710_dict = {(r['scene_name'], r.get('ground_truth', '')): r for r in v710}
v716_dict = {(r['scene_name'], r.get('ground_truth', '')): r for r in v716}

for key in v710_dict:
    r710 = v710_dict[key]
    r716 = v716_dict.get(key)
    if not r716:
        continue
    
    # 检查 3D 坐标是否相同
    di710 = r710.get('debug_info', {})
    di716 = r716.get('debug_info', {})
    
    pf710 = di710.get('proj_forward', 0)
    pf716 = di716.get('proj_forward', 0)
    
    if abs(pf710 - pf716) > 1:
        print(f'Scene {key[0]}: proj_forward 差异大')
        print(f'  V7.10: {pf710:.2f}, V7.16: {pf716:.2f}')
