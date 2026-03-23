#!/usr/bin/env python3
"""测试两种深度估计方式的区别"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'

import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')

from core.perception import DepthEstimator
import cv2
import numpy as np

# 读取一个测试视频
test_video = '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes/41069025.mp4'
cap = cv2.VideoCapture(test_video)
frames = []
for _ in range(8):
    ret, frame = cap.read()
    if ret:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f'Loaded {len(frames)} frames, shape: {frames[0].shape}')

estimator = DepthEstimator(device='cuda')

# 方法 1: infer_single
print('\n=== infer_single (V3 方式) ===')
depths_single = []
for i, frame in enumerate(frames[:3]):
    depth_tensor, _ = estimator.infer_single(frame, normalize=False)
    d = depth_tensor.squeeze().cpu().numpy()
    median = np.median(d)
    print(f'Frame {i}: median depth = {median:.4f}, shape = {d.shape}')
    depths_single.append(d)

# 方法 2: infer_video
print('\n=== infer_video (V4 方式) ===')
depth_pred = estimator.infer_video(frames[:3])
if hasattr(depth_pred, 'depth_maps'):
    depth_tensor = depth_pred.depth_maps
    for i in range(depth_tensor.shape[0]):
        d = depth_tensor[i, 0].cpu().numpy()
        median = np.median(d)
        print(f'Frame {i}: median depth = {median:.4f}, shape = {d.shape}')
else:
    print('No depth_maps attribute')
    print(f'Type: {type(depth_pred)}')
