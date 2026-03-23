#!/usr/bin/env python3
import json
from collections import defaultdict

# 检查Evolution覆盖率
with open('outputs/mindmap_full_evolution_9908.jsonl') as f:
    lines = f.readlines()

total = len(lines)
evo_count = 0
task_counts = defaultdict(int)
evo_by_task = defaultdict(int)

for line in lines:
    data = json.loads(line)
    conv = data.get('conversations', [])
    if len(conv) < 2:
        continue
    
    human_msg = conv[0].get('value', '')
    
    # 检查是否有Evolution内容
    has_evo = 'EVOLUTION' in human_msg or 'Evolution' in human_msg
    
    # 提取任务类型
    q_type = data.get('question_type', '').lower()
    for task in ['counting', 'size', 'distance', 'direction', 'appearance', 'route']:
        if task in q_type:
            task_counts[task] += 1
            if has_evo:
                evo_by_task[task] += 1
            break
    
    if has_evo:
        evo_count += 1

print(f'总样本数: {total}')
print(f'Evolution覆盖率: {evo_count}/{total} = {evo_count/total*100:.1f}%')
print()
print('各任务类型分布:')
for task in sorted(task_counts.keys()):
    cnt = task_counts[task]
    evo = evo_by_task[task]
    pct = evo/cnt*100 if cnt > 0 else 0
    print(f'  {task}: {cnt} 样本, Evolution {evo} ({pct:.1f}%)')

# 检查样本格式
print('\n样本格式示例:')
first_sample = json.loads(lines[0])
print(f"Keys: {first_sample.keys()}")
print(f"Question type: {first_sample.get('question_type')}")
