#!/usr/bin/env python3
"""测试两种帧采样方式的区别"""

import cv2
import numpy as np

test_video = '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes/41069025.mp4'
num_frames = 16

# 方式 1: V3 - 先读所有帧再采样
cap = cv2.VideoCapture(test_video)
all_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    all_frames.append(frame)
cap.release()

total = len(all_frames)
indices = np.linspace(0, total - 1, num_frames).astype(int)
v3_frames = [all_frames[i] for i in indices]

print(f"V3: 总帧数 {total}, 采样索引: {indices}")

# 方式 2: V4 - 直接跳帧读取
cap = cv2.VideoCapture(test_video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
v4_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
v4_frames = []
for idx in v4_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        v4_frames.append(frame)
cap.release()

print(f"V4: 总帧数 {total_frames}, 采样索引: {v4_indices}")

# 对比帧是否相同
print(f"\n帧对比:")
for i in range(min(len(v3_frames), len(v4_frames))):
    diff = np.abs(v3_frames[i].astype(float) - v4_frames[i].astype(float)).mean()
    print(f"Frame {i}: 差异 = {diff:.4f}")
