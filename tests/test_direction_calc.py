#!/usr/bin/env python3
"""测试 DA3 坐标系和方向计算"""
import sys
import numpy as np
sys.path.insert(0, '../Depth-Anything-3/src')

# 模拟测试
# 案例: standing=stove=[35.59, 46.97], facing=dishwasher=[43.64, 1.07], target=refrigerator=[34.3, 48.0]
# GT: front-right

standing = np.array([35.59, 46.97])
facing = np.array([43.64, 1.07])
target = np.array([34.3, 48.0])

print("=== 方向计算测试 ===")
print(f"Standing: {standing}")
print(f"Facing: {facing}")
print(f"Target: {target}")

# 计算前方向量
forward = facing - standing
forward_norm = np.linalg.norm(forward)
print(f"\nForward vector (raw): {forward}")
print(f"Forward norm: {forward_norm}")

if forward_norm > 0.01:
    forward = forward / forward_norm
    print(f"Forward vector (normalized): {forward}")
    
    # 右方向量 (逆时针旋转90度)
    right = np.array([forward[1], -forward[0]])
    print(f"Right vector: {right}")
    
    # 目标相对位置
    target_rel = target - standing
    print(f"\nTarget relative: {target_rel}")
    
    # 投影
    proj_forward = np.dot(target_rel, forward)
    proj_right = np.dot(target_rel, right)
    print(f"Projection forward: {proj_forward}")
    print(f"Projection right: {proj_right}")
    
    # 判断方向
    threshold = 0.3
    directions = []
    if proj_forward > threshold:
        directions.append("front")
    elif proj_forward < -threshold:
        directions.append("back")
    if proj_right > threshold:
        directions.append("right")
    elif proj_right < -threshold:
        directions.append("left")
    
    result = "-".join(directions) if directions else "same-position"
    print(f"\nPredicted direction: {result}")
    print(f"Ground truth: front-right")

print("\n" + "="*50)
print("分析: 坐标系可能存在问题")
print("facing=[43.64, 1.07] 比 standing=[35.59, 46.97] 的 Y 小很多")
print("这意味着 forward 向量主要指向 Y 负方向")
print("但这可能不符合实际的 '面向' 方向")
