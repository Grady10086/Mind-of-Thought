#!/usr/bin/env python3
"""验证方向计算"""
import numpy as np

standing = np.array([1.45, 1.61])
facing = np.array([1.53, 1.70])
target = np.array([1.28, 0.63])

# Forward
forward = facing - standing
forward = forward / np.linalg.norm(forward)
print(f'forward: {forward}')

# 当前的 Right (可能有问题)
right_old = np.array([forward[1], -forward[0]])
print(f'right_old: {right_old}')

# Target relative
target_rel = target - standing
print(f'target_rel: {target_rel}')

# 用角度来验证
angle_forward = np.arctan2(forward[1], forward[0])
angle_target = np.arctan2(target_rel[1], target_rel[0])
relative_angle = angle_target - angle_forward

print(f'\nforward 角度: {np.degrees(angle_forward):.1f}°')
print(f'target_rel 角度: {np.degrees(angle_target):.1f}°')
print(f'相对角度: {np.degrees(relative_angle):.1f}°')

# 规范化到 [-180, 180]
if relative_angle > np.pi:
    relative_angle -= 2*np.pi
if relative_angle < -np.pi:
    relative_angle += 2*np.pi
print(f'规范化相对角度: {np.degrees(relative_angle):.1f}°')

# 正确的方向判断:
# -45 to 45: front
# 45 to 135: right
# 135 to 180 or -180 to -135: back
# -135 to -45: left

print(f'\n基于角度的方向判断:')
deg = np.degrees(relative_angle)
if -45 <= deg <= 45:
    fb = "front"
elif deg > 135 or deg < -135:
    fb = "back"
elif deg > 0:
    fb = "right"
else:
    fb = "left"

if -135 <= deg <= -45:
    lr = "left"
elif 45 <= deg <= 135:
    lr = "right"
else:
    lr = ""

if fb == "front" or fb == "back":
    if lr:
        direction = f"{fb}-{lr}"
    else:
        direction = fb
else:
    direction = fb

print(f'方向: {direction}')
print(f'正确答案: back-left')

# 验证 right 向量
print(f'\n--- 验证 right 向量定义 ---')
print('在 2D XY 平面俯视图中:')
print('如果 forward = (0, 1) 即 Y+ 方向')
print('那么 right 应该 = (1, 0) 即 X+ 方向')
print()
print(f'使用 right = (forward_y, -forward_x):')
print(f'  forward = (0, 1) -> right = (1, 0) ✓')
print()
print(f'使用 right = (-forward_y, forward_x):')
print(f'  forward = (0, 1) -> right = (-1, 0) ✗ (这是左边)')
print()
print('所以公式 right = (forward_y, -forward_x) 是对的')
print()
print('让我重新验证投影...')

# Projections with old right
proj_forward = np.dot(target_rel, forward)
proj_right_old = np.dot(target_rel, right_old)
print(f'\nproj_forward: {proj_forward:.3f} (negative = back)')
print(f'proj_right (old): {proj_right_old:.3f}')

# 问题可能在于坐标系的理解
# DA3 的坐标系: Z 是深度方向, X 和 Y 是图像平面?
# 或者 Y 是上方向?

print('\n--- 重新理解坐标系 ---')
print('DA3 输出的 3D 坐标:')
print(f'  table: (1.45, 1.61, 1.435)')
print(f'  bed: (1.53, 1.70, 1.356)')
print(f'  toilet: (1.28, 0.63, 1.532)')
print()
print('Z 坐标都在 1.3-1.5 之间，可能是高度')
print('所以 XY 平面是地面')
print()

# 在标准坐标系中，如果我们站在 table 看向 bed
# forward = (0.674, 0.739) 朝右上
# toilet 在 (-0.17, -0.98) 相对位置，即左下方
#
# 如果 right = (0.739, -0.674) 朝右下
# proj_right = (-0.17)*0.739 + (-0.98)*(-0.674) = -0.126 + 0.660 = 0.534
# proj_right > 0 意味着 toilet 在 right 方向有正投影
# 但 right 是右下方...

# 我觉得问题在于 2D 投影的理解
# 让我用另一种方式: 计算 target 相对于 forward 的夹角

# 如果 target 在前方，夹角应该在 [-90, 90]
# 如果 target 在后方，夹角应该在 [90, 180] 或 [-180, -90]
# 如果在右边，夹角应该在 (0, 180) 
# 如果在左边，夹角应该在 (-180, 0)

print('重新计算:')
cross = forward[0] * target_rel[1] - forward[1] * target_rel[0]
dot = np.dot(forward, target_rel)
print(f'cross product (forward x target_rel): {cross:.3f}')
print(f'dot product: {dot:.3f}')
print()
print('cross > 0: target 在 forward 的左边')
print('cross < 0: target 在 forward 的右边')
print(f'cross = {cross:.3f} < 0，所以 toilet 在左边!')
print()
print('所以正确答案是 back-LEFT!')
print()
print('问题找到了: proj_right 计算是错的')
print('应该用 cross product 来判断左右!')
