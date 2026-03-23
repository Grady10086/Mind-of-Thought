#!/usr/bin/env python3
"""单样本深度测试"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['http_proxy'] = 'http://10.11.16.24:8118'
os.environ['https_proxy'] = 'http://10.11.16.24:8118'

import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')

import numpy as np
from datasets import load_dataset

# 官方 MRA
def mean_relative_accuracy(pred, target, start=0.05, end=0.5, interval=0.05):
    if pred is None or target is None:
        return 0.0
    epsilon = 1e-8
    rel_error = abs(pred - target) / (abs(target) + epsilon)
    thresholds = np.arange(start, end + interval / 2, interval)
    conditions = rel_error < (1 - thresholds)
    return conditions.astype(float).mean()

import re
def normalize_number(text):
    if text is None:
        return None
    text = re.sub(r'(meters?|m|cm|feet|ft|inches?|in|square)\b', '', str(text).lower())
    match = re.search(r'[-+]?\d*\.?\d+', text)
    if match:
        try:
            return float(match.group())
        except:
            return None
    return None

# 加载数据
ds = load_dataset('nyu-visionx/VSI-Bench', split='test', 
                  cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench')

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

def find_video(scene_name):
    for d in VIDEO_DIRS:
        p = os.path.join(d, f"{scene_name}.mp4")
        if os.path.exists(p):
            return p
    return None

# 测试所有 object_counting 样本
counting_samples = [d for d in ds if d['question_type'] == 'object_counting']

print(f"测试所有 {len(counting_samples)} 个 object_counting 样本")

# V3 builder
from tests.test_vsibench_mindmap_with_depth import MindMapBuilderWithDepth as V3Builder, MindMapQA3D

v3_builder = V3Builder(device='cuda', num_frames=16, box_threshold=0.25)

v3_scores = []
for i, sample in enumerate(counting_samples):
    scene = sample['scene_name']
    question = sample['question']
    gt = sample['ground_truth']
    
    video_path = find_video(scene)
    if not video_path:
        continue
    
    mind_map = v3_builder.build_from_video(video_path)
    pred = MindMapQA3D.answer_counting(mind_map, question)
    
    pred_num = normalize_number(pred)
    gt_num = normalize_number(gt)
    
    mra = mean_relative_accuracy(pred_num, gt_num) if pred_num is not None and gt_num is not None else 0.0
    
    v3_scores.append(mra)
    
    if (i + 1) % 50 == 0:
        print(f"V3: {i+1}/{len(counting_samples)}, 当前 MRA: {np.mean(v3_scores)*100:.2f}%")

print(f"\nV3 最终 MRA: {np.mean(v3_scores)*100:.2f}%")
