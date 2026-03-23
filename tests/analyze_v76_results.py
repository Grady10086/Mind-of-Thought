#!/usr/bin/env python3
"""分析V7.6测试结果"""
import json

with open('outputs/v76_da3_full_20260206_113228/results_gpu0.json') as f:
    results = json.load(f)

print('=== V7.6 详细结果分析 ===')
print()

# 统计
correct = sum(1 for r in results if r.get('correct', False))
matched = sum(1 for r in results if r.get('predicted_direction'))
total = len(results)

print(f'总样本: {total}')
print(f'正确: {correct} ({correct/total*100:.1f}%)')
print(f'方向匹配: {matched} ({matched/total*100:.1f}%)')
print()

# 查看具体案例
print('=== 前5个样本详情 ===')
for i, r in enumerate(results[:5]):
    print(f'\n样本 {i+1}:')
    print(f'  问题: {r["question"][:100]}...')
    print(f'  GT: {r["ground_truth"]}')
    print(f'  预测: {r["prediction"]}')
    print(f'  预测方向: {r.get("predicted_direction", "None")}')
    print(f'  正确: {r.get("correct", False)}')
    print(f'  推理: {r.get("reasoning", "None")[:150]}...')
    
    # 心智地图
    mm = r.get('mind_map_summary', {})
    if mm:
        print(f'  心智地图物体: {list(mm.keys())[:5]}')
        # 打印物体位置
        for label, info in list(mm.items())[:2]:
            pos = info.get('position_3d')
            if pos:
                print(f'    {label}: pos={[round(p,2) for p in pos]}')

# 分析问题模式
print('\n=== 问题模式分析 ===')
direction_patterns = {}
for r in results:
    if r.get('predicted_direction'):
        d = r['predicted_direction']
        direction_patterns[d] = direction_patterns.get(d, 0) + 1

print(f'预测方向分布: {direction_patterns}')

# 分析目标物体匹配
print('\n=== 物体匹配分析 ===')
target_found = sum(1 for r in results if r.get('target_object') and 
                   any(r.get('target_object').lower() in label.lower() 
                       for label in r.get('mind_map_summary', {}).keys()))
print(f'目标物体找到: {target_found}/{total} ({target_found/total*100:.1f}%)')
