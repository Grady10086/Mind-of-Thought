#!/usr/bin/env python3
"""对比 V3 和 V4 的 QA 函数在多个样本上的结果"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'

import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')

import json
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
counting_samples = [d for d in ds if d['question_type'] == 'object_counting'][:10]

print(f"测试 {len(counting_samples)} 个 object_counting 样本")

# 导入 V3 和 V4
from tests.test_vsibench_mindmap_with_depth import MindMapBuilderWithDepth as V3Builder, MindMapQA3D
from tests.test_vsibench_mindmap_qwen_reasoning import MindMapBuilderWithDepth as V4Builder, DirectQA

# 创建 builder (共用，节省加载时间)
builder = V3Builder(device='cuda', num_frames=16, box_threshold=0.25)

print("\n测试结果:")
for sample in counting_samples:
    scene = sample['scene_name']
    question = sample['question']
    gt = sample['ground_truth']
    
    video_path = find_video(scene)
    if not video_path:
        continue
    
    # 构建心智地图
    mind_map = builder.build_from_video(video_path)
    
    # V3 和 V4 的 QA 函数
    v3_pred = MindMapQA3D.answer_counting(mind_map, question)
    v4_pred = DirectQA.answer_counting(mind_map, question)
    
    print(f"\nScene: {scene}")
    print(f"Q: {question}")
    print(f"GT: {gt}")
    print(f"V3 pred: {v3_pred}")
    print(f"V4 pred: {v4_pred}")
    print(f"V3==V4: {v3_pred == v4_pred}")
