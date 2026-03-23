#!/usr/bin/env python3
"""
测试 DA3 多帧输出的坐标一致性
检查不同帧的相机位置是否在同一个世界坐标系中
"""
import sys
import numpy as np
sys.path.insert(0, '../Depth-Anything-3/src')

from depth_anything_3.api import DepthAnything3
import torch
from PIL import Image
import cv2

# 加载一个真实视频
video_path = '/home/tione/notebook/tianjungu/hf_cache/vsibench/arkitscenes/41069025.mp4'

# 采样帧
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int)

frames = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
cap.release()

print(f"采样 {len(frames)} 帧")

# 加载模型
print("加载 DA3 模型...")
model = DepthAnything3(model_name='da3nested-giant-large')
model = model.to('cuda')
model.eval()

# 推理
print("推理中 (use_ray_pose=True)...")
with torch.no_grad():
    prediction = model.inference(
        image=frames,
        use_ray_pose=True,
        ref_view_strategy='saddle_balanced',
        process_res=504,
    )

print(f"\n=== DA3 输出分析 ===")
print(f"depth shape: {prediction.depth.shape}")
print(f"extrinsics shape: {prediction.extrinsics.shape}")
print(f"intrinsics shape: {prediction.intrinsics.shape}")

# 分析相机位置
print(f"\n=== 各帧相机位置 ===")
camera_centers = []
for i in range(len(frames)):
    E = prediction.extrinsics[i]  # (3, 4) w2c
    R = E[:3, :3]
    t = E[:3, 3]
    camera_center = -R.T @ t
    camera_centers.append(camera_center)
    print(f"帧 {i}: 相机位置 = {camera_center.round(3)}")

# 计算相机之间的距离
camera_centers = np.array(camera_centers)
print(f"\n=== 相机轨迹分析 ===")
for i in range(1, len(camera_centers)):
    dist = np.linalg.norm(camera_centers[i] - camera_centers[i-1])
    print(f"帧 {i-1} -> 帧 {i}: 移动距离 = {dist:.3f}")

# 总体移动范围
print(f"\n=== 总体范围 ===")
print(f"X 范围: {camera_centers[:, 0].min():.3f} ~ {camera_centers[:, 0].max():.3f}")
print(f"Y 范围: {camera_centers[:, 1].min():.3f} ~ {camera_centers[:, 1].max():.3f}")
print(f"Z 范围: {camera_centers[:, 2].min():.3f} ~ {camera_centers[:, 2].max():.3f}")

total_range = camera_centers.max(axis=0) - camera_centers.min(axis=0)
print(f"总体移动范围: {total_range.round(3)}")

# 分析深度分布
print(f"\n=== 深度分布 ===")
for i in range(len(frames)):
    d = prediction.depth[i]
    print(f"帧 {i}: depth min={d.min():.3f}, max={d.max():.3f}, mean={d.mean():.3f}")

# 分析相机朝向
print(f"\n=== 相机朝向分析 ===")
for i in range(len(frames)):
    E = prediction.extrinsics[i]
    R = E[:3, :3]
    # 相机的前方向（Z轴负方向在相机坐标系，转到世界坐标系）
    forward = R.T @ np.array([0, 0, 1])  # 相机看向的方向
    up = R.T @ np.array([0, -1, 0])  # 相机的上方向
    right = R.T @ np.array([1, 0, 0])  # 相机的右方向
    print(f"帧 {i}:")
    print(f"  Forward: {forward.round(3)}")
    print(f"  Up: {up.round(3)}")
    print(f"  Right: {right.round(3)}")

# 将中心点的像素投影到世界坐标
print(f"\n=== 中心像素世界坐标 ===")
def pixel_to_world(u, v, depth, extrinsics, intrinsics):
    """将像素坐标转换为世界坐标"""
    K = intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # 像素 -> 相机坐标
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    cam_point = np.array([x, y, z])
    
    # 相机 -> 世界 (w2c: cam = R @ world + t, so world = R^T @ (cam - t))
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    world_point = R.T @ cam_point - R.T @ t
    
    return world_point

h, w = prediction.depth[0].shape
center_u, center_v = w // 2, h // 2

for i in range(len(frames)):
    depth = prediction.depth[i]
    center_depth = depth[center_v, center_u]
    world_pos = pixel_to_world(center_u, center_v, center_depth, prediction.extrinsics[i], prediction.intrinsics[i])
    print(f"帧 {i}: 中心像素深度={center_depth:.3f}, 世界坐标={world_pos.round(2)}")
