#!/usr/bin/env python3
"""分析缺失物体"""
import json
import re
from pathlib import Path
from collections import Counter

# 读取最新的测试结果
results_dir = sorted(Path('outputs').glob('baseline_comparison_*'))[-1]
print(f'分析目录: {results_dir}')

# 分析 DA3 1.1 16帧结果
results_file = results_dir / 'results_da3_11.json'
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    
    missing_objects = {
        'standing': [],
        'facing': [],
        'target': [],
    }
    
    for r in results:
        objs = r.get('objects_found', {})
        q = r.get('question', '') if 'question' not in r else ''
        
        # 从 scene_name 找原始问题
        if not objs.get('standing'):
            m = re.search(r'standing by (?:the )?(\w+)', q.lower())
            if m:
                missing_objects['standing'].append(m.group(1))
        if not objs.get('facing'):
            m = re.search(r'facing (?:the )?(\w+)', q.lower())
            if m:
                missing_objects['facing'].append(m.group(1))
        if not objs.get('target'):
            m = re.search(r'is (?:the )?(\w+)', q.lower())
            if m:
                missing_objects['target'].append(m.group(1))
    
    # 统计找到和未找到
    found_all = sum(1 for r in results if r.get('objects_found', {}).get('standing') 
                    and r.get('objects_found', {}).get('facing') 
                    and r.get('objects_found', {}).get('target'))
    
    print(f'\n总样本: {len(results)}')
    print(f'找到所有物体: {found_all}')
    print(f'未找到完整: {len(results) - found_all}')
    
    print('\n=== 缺失物体统计 ===')
    for role, objs in missing_objects.items():
        if objs:
            counts = Counter(objs)
            print(f'\n{role} 缺失:')
            for obj, cnt in counts.most_common(10):
                print(f'  {obj}: {cnt}次')
else:
    print(f'结果文件不存在: {results_file}')

# 也看看 32帧的情况
results_file_32 = results_dir / 'results_da3_11_32frames.json'
if results_file_32.exists():
    with open(results_file_32) as f:
        results_32 = json.load(f)
    
    found_all_32 = sum(1 for r in results_32 if r.get('objects_found', {}).get('standing') 
                       and r.get('objects_found', {}).get('facing') 
                       and r.get('objects_found', {}).get('target'))
    
    print(f'\n\n=== 32帧结果 ===')
    print(f'总样本: {len(results_32)}')
    print(f'找到所有物体: {found_all_32}')
