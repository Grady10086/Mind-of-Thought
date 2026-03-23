#!/usr/bin/env python3
"""对比 V3 和 V4 的 MindMapBuilder 结果"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['http_proxy'] = 'http://10.11.16.24:8118'
os.environ['https_proxy'] = 'http://10.11.16.24:8118'

import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')

import numpy as np
from datasets import load_dataset

# 加载数据
ds = load_dataset('nyu-visionx/VSI-Bench', split='test', 
                  cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench')

# 找视频路径
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

# 获取 object_counting 样本
counting_samples = [d for d in ds if d['question_type'] == 'object_counting'][:5]

print(f"测试 {len(counting_samples)} 个 object_counting 样本")

# 测试 V3 builder
print("\n" + "="*60)
print("V3 Builder (test_vsibench_mindmap_with_depth.py)")
print("="*60)

from tests.test_vsibench_mindmap_with_depth import MindMapBuilderWithDepth as V3Builder, MindMapQA3D

v3_builder = V3Builder(device='cuda', num_frames=16, box_threshold=0.25)

v3_results = []
for sample in counting_samples:
    scene = sample['scene_name']
    question = sample['question']
    gt = sample['ground_truth']
    
    video_path = find_video(scene)
    if not video_path:
        continue
    
    mind_map = v3_builder.build_from_video(video_path)
    pred = MindMapQA3D.answer_counting(mind_map, question)
    
    # 计算 MRA
    pred_num = float(pred) if pred.isdigit() else 0
    gt_num = float(gt) if str(gt).isdigit() else 0
    if gt_num > 0:
        mra = max(0, 1 - abs(pred_num - gt_num) / gt_num)
    else:
        mra = 1.0 if pred_num == 0 else 0.0
    
    print(f"Scene: {scene}, Q: {question[:40]}..., GT: {gt}, Pred: {pred}, MRA: {mra:.3f}")
    v3_results.append(mra)

v3_builder.unload()
import torch
torch.cuda.empty_cache()

print(f"\nV3 平均 MRA: {np.mean(v3_results)*100:.2f}%")

# 测试 V4 builder
print("\n" + "="*60)
print("V4 Builder (test_vsibench_mindmap_qwen_reasoning.py)")
print("="*60)

from tests.test_vsibench_mindmap_qwen_reasoning import MindMapBuilderWithDepth as V4Builder, DirectQA

v4_builder = V4Builder(device='cuda', num_frames=16, box_threshold=0.25)

v4_results = []
for sample in counting_samples:
    scene = sample['scene_name']
    question = sample['question']
    gt = sample['ground_truth']
    
    video_path = find_video(scene)
    if not video_path:
        continue
    
    mind_map = v4_builder.build_from_video(video_path)
    pred = DirectQA.answer_counting(mind_map, question)
    
    # 计算 MRA
    pred_num = float(pred) if pred.isdigit() else 0
    gt_num = float(gt) if str(gt).isdigit() else 0
    if gt_num > 0:
        mra = max(0, 1 - abs(pred_num - gt_num) / gt_num)
    else:
        mra = 1.0 if pred_num == 0 else 0.0
    
    print(f"Scene: {scene}, Q: {question[:40]}..., GT: {gt}, Pred: {pred}, MRA: {mra:.3f}")
    v4_results.append(mra)

print(f"\nV4 平均 MRA: {np.mean(v4_results)*100:.2f}%")

# 对比
print("\n" + "="*60)
print("对比")
print("="*60)
print(f"V3: {np.mean(v3_results)*100:.2f}%")
print(f"V4: {np.mean(v4_results)*100:.2f}%")
print(f"差异: {(np.mean(v4_results) - np.mean(v3_results))*100:.2f}%")
