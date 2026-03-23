#!/usr/bin/env python3
"""
直接分析 DA3 坐标系 - 通过比较多帧的相机位置和朝向
"""
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DA3_PATH = PROJECT_ROOT.parent / "Depth-Anything-3"
if str(DA3_PATH / "src") not in sys.path:
    sys.path.insert(0, str(DA3_PATH / "src"))

from depth_anything_3.api import DepthAnything3

# 加载视频
video_path = '/home/tione/notebook/tianjungu/hf_cache/vsibench/arkitscenes/42446167.mp4'

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
print("推理...")
import torch
with torch.no_grad():
    prediction = model.inference(
        image=frames,
        use_ray_pose=True,
        ref_view_strategy='saddle_balanced',
        process_res=504,
    )

print(f"\n=== 分析 DA3 坐标系 ===")

# 提取相机参数
for i in range(len(frames)):
    E = prediction.extrinsics[i]  # (3, 4) w2c
    R = E[:3, :3]
    t = E[:3, 3]
    
    # 相机中心 (世界坐标)
    camera_center = -R.T @ t
    
    # 相机在世界坐标系中的朝向
    # 相机坐标系: Z 前, X 右, Y 下
    # forward = R^T @ [0, 0, 1] (相机 Z 轴在世界坐标系中)
    forward = R.T @ np.array([0, 0, 1])
    up = R.T @ np.array([0, -1, 0])  # 相机 Y 轴取反 (因为 Y 向下)
    right = R.T @ np.array([1, 0, 0])
    
    if i == 0:
        print(f"\n帧 {i} (第一帧):")
        print(f"  相机位置: {camera_center.round(3)}")
        print(f"  Forward: {forward.round(3)}")
        print(f"  Up: {up.round(3)}")
        print(f"  Right: {right.round(3)}")
        
        # 分析 forward 的主方向
        abs_fwd = np.abs(forward)
        main_axis = np.argmax(abs_fwd)
        print(f"\n  主要朝向: {'XYZ'[main_axis]} {'正' if forward[main_axis] > 0 else '负'}方向")
        
        # 分析 up 的主方向
        abs_up = np.abs(up)
        main_up = np.argmax(abs_up)
        print(f"  向上方向: {'XYZ'[main_up]} {'正' if up[main_up] > 0 else '负'}方向")

# 计算相机移动轨迹
print(f"\n=== 相机移动轨迹 ===")
camera_centers = []
for i in range(len(frames)):
    E = prediction.extrinsics[i]
    R = E[:3, :3]
    t = E[:3, 3]
    camera_center = -R.T @ t
    camera_centers.append(camera_center)

camera_centers = np.array(camera_centers)

# 移动范围
for axis, name in enumerate(['X', 'Y', 'Z']):
    range_val = camera_centers[:, axis].max() - camera_centers[:, axis].min()
    print(f"{name}: {camera_centers[:, axis].min():.3f} ~ {camera_centers[:, axis].max():.3f} (范围: {range_val:.3f})")

# 分析移动方向
print(f"\n=== 移动方向分析 ===")
total_move = camera_centers[-1] - camera_centers[0]
print(f"从第一帧到最后一帧的总移动: {total_move.round(3)}")
main_move_axis = np.argmax(np.abs(total_move))
print(f"主要移动轴: {'XYZ'[main_move_axis]}")

# 测试像素到世界坐标
print(f"\n=== 测试像素到世界坐标 ===")

def pixel_to_world(u, v, depth, extrinsics, intrinsics):
    K = intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # 像素 -> 相机
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth
    cam_point = np.array([x_cam, y_cam, z_cam])
    
    # 相机 -> 世界
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    world_point = R.T @ cam_point - R.T @ t
    
    return world_point

# 获取图像中心和四角的世界坐标
H, W = prediction.depth.shape[1:3]
test_points = [
    (W//2, H//2, "center"),
    (0, 0, "top-left"),
    (W-1, 0, "top-right"),
    (0, H-1, "bottom-left"),
    (W-1, H-1, "bottom-right"),
]

print(f"图像尺寸: {W} x {H}")
print(f"\n第一帧测试点:")
for u, v, name in test_points:
    depth = prediction.depth[0, int(v), int(u)]
    world = pixel_to_world(u, v, depth, prediction.extrinsics[0], prediction.intrinsics[0])
    print(f"  {name} ({u}, {v}): depth={depth:.2f}, world={world.round(2)}")
