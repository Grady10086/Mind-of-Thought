#!/usr/bin/env python3
"""分析V7.6b测试结果"""
import json
import glob

# 找到最新的结果文件
result_files = glob.glob('outputs/v76b_da3_full_*/results_gpu0.json')
if not result_files:
    print("No results found")
    exit()

result_file = sorted(result_files)[-1]
print(f"分析: {result_file}")

with open(result_file) as f:
    results = json.load(f)

print('=== V7.6b 详细分析 ===')

# 分析物体查找情况
found_all = [r for r in results if r.get('objects_found', {}).get('standing') and 
             r.get('objects_found', {}).get('facing') and r.get('objects_found', {}).get('target')]

print(f'\n三个物体都找到的样本: {len(found_all)}')
if found_all:
    correct_when_found = sum(1 for r in found_all if r.get('correct', False))
    print(f'  其中正确: {correct_when_found} ({correct_when_found/len(found_all)*100:.1f}%)')

# 查看具体案例
print('\n=== 三个物体都找到的案例 ===')
for i, r in enumerate(found_all[:5]):
    print(f'\n案例 {i+1}:')
    print(f'  问题: {r["question"][:80]}...')
    print(f'  解析: standing={r["parsed"]["standing_by"]}, facing={r["parsed"]["facing"]}, target={r["parsed"]["target"]}')
    print(f'  预测方向: {r.get("predicted_direction")}')
    print(f'  GT: {r["ground_truth"]}, 预测: {r["prediction"]}, 正确: {r.get("correct")}')
    print(f'  推理: {r.get("reasoning", "")[:100]}')
    
    # 检查选项
    for j, opt in enumerate(r.get('options', [])):
        marker = '✓' if chr(65+j) == r["ground_truth"].strip().upper()[0] else ' '
        print(f'    {marker} {opt}')

# 分析方向分布
print('\n=== 预测方向分布 ===')
dir_stats = {}
for r in results:
    d = r.get('predicted_direction', 'None')
    dir_stats[d] = dir_stats.get(d, 0) + 1

for d, count in sorted(dir_stats.items(), key=lambda x: -x[1]):
    print(f'  {d}: {count}')

# 分析物体缺失情况
print('\n=== 物体缺失分析 ===')
missing_stats = {'standing': 0, 'facing': 0, 'target': 0}
for r in results:
    objs = r.get('objects_found', {})
    for key in missing_stats:
        if not objs.get(key, True):  # 默认True避免旧数据问题
            missing_stats[key] += 1

for key, count in missing_stats.items():
    print(f'  {key} 物体缺失: {count}/{len(results)}')

# 查看心智地图中有哪些物体
print('\n=== 心智地图物体分布 ===')
all_labels = {}
for r in results:
    for label in r.get('mind_map_labels', []):
        all_labels[label] = all_labels.get(label, 0) + 1

for label, count in sorted(all_labels.items(), key=lambda x: -x[1])[:20]:
    print(f'  {label}: {count}')
