#!/usr/bin/env python3
"""测试任务类型推断"""
import json
import os
from collections import defaultdict

def infer_question_type(question):
    question_lower = question.lower()
    
    if 'how many' in question_lower:
        return 'object_counting'
    
    if any(word in question_lower for word in ['size', 'dimensions', 'length', 'width', 'height', 'how tall', 'how wide', 'how long', 'how big']):
        if 'room' in question_lower:
            return 'room_size_estimation'
        return 'object_size_estimation'
    
    if 'distance' in question_lower:
        if 'closest' in question_lower or 'nearest' in question_lower or 'farthest' in question_lower:
            return 'object_rel_distance'
        return 'object_abs_distance'
    
    # 比较距离任务（which of these objects is closest/nearest/farthest）
    if any(phrase in question_lower for phrase in ['which of these objects', 'which object is closest', 'which object is nearest', 'which object is farthest']):
        return 'object_rel_distance'
    
    if any(word in question_lower for word in ['left', 'right', 'front', 'behind', 'above', 'below', 'direction', 'relative position']):
        return 'object_rel_direction'
    
    if any(phrase in question_lower for phrase in ['first appear', 'appears first', 'order of appearance', 'which appears', 'seen first']):
        return 'obj_appearance_order'
    
    if any(phrase in question_lower for phrase in ['route', 'path', 'navigate', 'go from', 'get to', 'reach']):
        return 'route_planning'
    
    return 'other'

# 加载数据
with open('/home/tione/notebook/tianjungu/datasets/VLM_3R_Videos/VLM-3R-DATA/vsibench_train/merged_qa_scannet_train.json', 'r') as f:
    data = json.load(f)

# 统计
task_samples = defaultdict(list)
for sample in data:
    question = sample['conversations'][0]['value'] if sample['conversations'] else ''
    task_type = infer_question_type(question)
    task_samples[task_type].append(sample)

print('任务类型分布:')
total = 0
for task, samples in sorted(task_samples.items()):
    print(f'  {task}: {len(samples)}')
    total += len(samples)
print(f'  总计: {total}')

# 检查视频路径
print('\n检查视频路径:')
video_base = '/home/tione/notebook/tianjungu/datasets/VLM_3R_Videos'
for task, samples in list(task_samples.items())[:3]:
    sample = samples[0]
    video_path = os.path.join(video_base, sample['video'])
    exists = os.path.exists(video_path)
    print(f'  {sample["video"]}: {"存在" if exists else "不存在"}')

# 打印一些other类型的问题
print('\n"other"类型样本问题示例:')
for sample in task_samples['other'][:5]:
    q = sample['conversations'][0]['value'][:100]
    print(f'  - {q}...')
