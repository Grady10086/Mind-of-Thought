#!/usr/bin/env python3
"""
转换平衡采样数据为VSIBench格式,以便V7处理
"""

import json
from pathlib import Path

input_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_per_task_full.jsonl")
output_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_per_task_full_vsibench.json")

print("=" * 80)
print("转换为VSIBench格式")
print("=" * 80)
print()

vsibench_data = []

print(f"📖 读取: {input_path}")
with open(input_path, 'r') as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line)
        
        # 提取信息
        question = data.get('conversations', [{}])[0].get('value', '')
        answer = data.get('conversations', [{}])[1].get('value', '') if len(data.get('conversations', [])) > 1 else ''
        
        # 简单提取问题(移除<video>标签后的内容)
        if '<video>' in question:
            question = question.split('<video>')[0].strip()
        
        # 构造VSIBench格式
        vsibench_item = {
            "scene_name": f"vsi590k_sample_{line_num:05d}",
            "video_path": data.get('video', ''),  # 如果有video字段
            "question": question,
            "question_type": data.get('question_type', 'unknown'),
            "ground_truth": answer,
            "options": data.get('options', [])
        }
        
        vsibench_data.append(vsibench_item)

print(f"✅ 转换完成: {len(vsibench_data)} 样本")
print()

print(f"💾 保存到: {output_path}")
with open(output_path, 'w') as f:
    json.dump(vsibench_data, f, ensure_ascii=False, indent=2)

print("✅ 保存成功!")
print()

# 统计
from collections import Counter
task_counts = Counter(item['question_type'] for item in vsibench_data)

print("📊 任务分布:")
for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
    pct = count / len(vsibench_data) * 100
    print(f"  {task:35s}: {count:5d} ({pct:5.1f}%)")
print()
print(f"  {'总计':35s}: {len(vsibench_data):5d} (100.0%)")
