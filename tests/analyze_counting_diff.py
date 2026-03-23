#!/usr/bin/env python3
"""分析 V7 和 V7.2 counting 任务差异"""
import json

# 读取两个版本的详细结果
with open('/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/evolving_agent_v7_20260203_134612/detailed_results.json') as f:
    v7_results = json.load(f)

with open('/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/tests/outputs/evolving_agent_v72_da3_11_20260208_201751/detailed_results.json') as f:
    v72_results = json.load(f)

# 过滤 counting 任务
v7_counting = [r for r in v7_results if r.get('question_type') == 'object_counting']
v72_counting = [r for r in v72_results if r.get('question_type') == 'object_counting']

print(f"V7 counting samples: {len(v7_counting)}")
print(f"V7.2 counting samples: {len(v72_counting)}")

# 创建索引
v7_by_q = {r['question']: r for r in v7_counting}
v72_by_q = {r['question']: r for r in v72_counting}

# V7 rule 正确但 V7.2 rule 错误
v7_correct_v72_wrong = []
for q, v7_r in v7_by_q.items():
    if q in v72_by_q:
        v72_r = v72_by_q[q]
        if v7_r.get('rule_score', 0) == 1.0 and v72_r.get('rule_score', 0) == 0.0:
            v7_correct_v72_wrong.append((q, v7_r, v72_r))

print(f"\nV7 rule 正确但 V7.2 rule 错误的样本: {len(v7_correct_v72_wrong)}")

# 显示前 5 个例子
for i, (q, v7_r, v72_r) in enumerate(v7_correct_v72_wrong[:5]):
    print(f"\n{'='*60}")
    print(f"Example {i+1}")
    print(f"Question: {q[:100]}...")
    print(f"Ground truth: {v7_r.get('ground_truth')}")
    print(f"V7 rule pred: {v7_r.get('rule_prediction')}")
    print(f"V7.2 rule pred: {v72_r.get('rule_prediction')}")
    
    # 分析 mind_map 差异
    v7_mm = v7_r.get('mind_map_summary', {})
    v72_mm = v72_r.get('mind_map_summary', {})
    
    # 找问题中的目标物体
    import re
    match = re.search(r'How many (\w+)', q, re.IGNORECASE)
    if match:
        target = match.group(1).lower()
        print(f"Target object: {target}")
        
        # 在 mind_map 中查找
        v7_count = None
        v72_count = None
        for label, info in v7_mm.items():
            if target in label.lower() or label.lower() in target:
                v7_count = info.get('count')
                print(f"  V7 found '{label}': count={v7_count}")
        for label, info in v72_mm.items():
            if target in label.lower() or label.lower() in target:
                v72_count = info.get('count')
                print(f"  V7.2 found '{label}': count={v72_count}")
        
        if v7_count is None:
            print(f"  V7 没有找到 '{target}'")
        if v72_count is None:
            print(f"  V7.2 没有找到 '{target}'")

# 统计检测到的物体数量差异
print(f"\n{'='*60}")
print("检测物体数量统计")
v7_obj_counts = []
v72_obj_counts = []
for q in v7_by_q:
    if q in v72_by_q:
        v7_mm = v7_by_q[q].get('mind_map_summary', {})
        v72_mm = v72_by_q[q].get('mind_map_summary', {})
        v7_obj_counts.append(len(v7_mm))
        v72_obj_counts.append(len(v72_mm))

import numpy as np
print(f"V7 平均检测物体数: {np.mean(v7_obj_counts):.2f}")
print(f"V7.2 平均检测物体数: {np.mean(v72_obj_counts):.2f}")
print(f"V7 检测物体数范围: {min(v7_obj_counts)} - {max(v7_obj_counts)}")
print(f"V7.2 检测物体数范围: {min(v72_obj_counts)} - {max(v72_obj_counts)}")
