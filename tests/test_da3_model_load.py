#!/usr/bin/env python3
"""测试 DA3 完整模型加载和推理"""
import sys
import numpy as np
sys.path.insert(0, '../Depth-Anything-3/src')

from depth_anything_3.api import DepthAnything3
import torch
from PIL import Image

print('加载 da3nested-giant-large 模型...')
model = DepthAnything3(model_name='da3nested-giant-large')
model = model.to('cuda')
model.eval()

print('✓ 模型加载成功')
print(f'模型名称: {model.model_name}')

# 创建测试图像
dummy_images = [
    Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    for _ in range(4)
]

print('\n测试多视图推理 (use_ray_pose=True)...')
with torch.no_grad():
    prediction = model.inference(
        image=dummy_images,
        extrinsics=None,
        intrinsics=None,
        use_ray_pose=True,
        ref_view_strategy='saddle_balanced',
        process_res=504,
    )

print(f'✓ 推理成功')
print(f'  depth shape: {prediction.depth.shape}')
print(f'  extrinsics: {prediction.extrinsics.shape if prediction.extrinsics is not None else "None"}')
print(f'  intrinsics: {prediction.intrinsics.shape if prediction.intrinsics is not None else "None"}')

# 测试坐标转换
if prediction.extrinsics is not None and prediction.intrinsics is not None:
    print('\n测试 3D 坐标计算...')
    
    # 获取中心像素的深度
    H, W = prediction.depth.shape[1:3]
    cx, cy = W // 2, H // 2
    depth = prediction.depth[0, cy, cx]
    
    # 获取内参
    K = prediction.intrinsics[0]
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    
    # 计算相机坐标
    x_cam = (cx - px) / fx * depth
    y_cam = (cy - py) / fy * depth
    z_cam = depth
    
    print(f'  中心像素深度: {depth:.3f}')
    print(f'  相机坐标: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})')
    print(f'  内参: fx={fx:.1f}, fy={fy:.1f}, cx={px:.1f}, cy={py:.1f}')
    
    # 获取外参
    E = prediction.extrinsics[0]
    print(f'  外参矩阵:\n{E}')
