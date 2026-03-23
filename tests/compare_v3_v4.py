#!/usr/bin/env python3
"""对比 V3 和 V4 的心智地图构建结果"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'

import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')

import json

# 测试视频和问题
test_video = '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes/41069025.mp4'

print("="*60)
print("测试 V3 (test_vsibench_mindmap_with_depth.py)")
print("="*60)

# 导入 V3
from tests.test_vsibench_mindmap_with_depth import MindMapBuilderWithDepth as V3Builder, MindMapQA3D

v3_builder = V3Builder(device='cuda', num_frames=16, box_threshold=0.25)
v3_mindmap = v3_builder.build_from_video(test_video)

print("\nV3 心智地图:")
for label, entity in v3_mindmap.items():
    print(f"  {label}: count={entity.count}, size_3d={entity.size_3d}, depth={entity.depth_median:.2f}")

# 释放 V3 模型
v3_builder.unload()
import torch
torch.cuda.empty_cache()

print("\n" + "="*60)
print("测试 V4 (test_vsibench_mindmap_qwen_reasoning.py)")
print("="*60)

# 导入 V4
from tests.test_vsibench_mindmap_qwen_reasoning import MindMapBuilderWithDepth as V4Builder

v4_builder = V4Builder(device='cuda', num_frames=16, box_threshold=0.25)
v4_mindmap = v4_builder.build_from_video(test_video)

print("\nV4 心智地图:")
for label, entity in v4_mindmap.items():
    print(f"  {label}: count={entity.count}, size_3d={entity.size_3d}, depth={entity.depth_median:.2f}")

# 对比
print("\n" + "="*60)
print("差异分析")
print("="*60)

all_labels = set(v3_mindmap.keys()) | set(v4_mindmap.keys())
for label in sorted(all_labels):
    v3_e = v3_mindmap.get(label)
    v4_e = v4_mindmap.get(label)
    
    if v3_e and v4_e:
        count_diff = v4_e.count - v3_e.count
        size_diff = v4_e.size_3d - v3_e.size_3d if v4_e.size_3d is not None and v3_e.size_3d is not None else None
        depth_diff = v4_e.depth_median - v3_e.depth_median
        print(f"{label}: count差={count_diff}, depth差={depth_diff:.2f}")
        if size_diff is not None:
            print(f"        size差={size_diff}")
    elif v3_e:
        print(f"{label}: V4 缺失")
    else:
        print(f"{label}: V3 缺失")
