#!/usr/bin/env python3
"""找到可用的测试样本"""
import json, os

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset'
with open(os.path.join(base, 'train_balanced_1k_per_task_full_vsibench_v2.json')) as f:
    dataset = json.load(f)

for qt in ['object_abs_distance', 'object_size_estimation', 'room_size_estimation']:
    for s in dataset:
        if s['question_type'] == qt and os.path.exists(s.get('video_path', '')):
            print(f'\n=== {qt} ===')
            print(f'scene: {s["scene_name"]}')
            print(f'video: {s["video_path"]}')
            print(f'question: {s["question"]}')
            print(f'gt: {s["ground_truth"]}')
            break
