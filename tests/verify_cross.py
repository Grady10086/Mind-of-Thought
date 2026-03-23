#!/usr/bin/env python3
import numpy as np

# 案例：dishwasher -> chair，target stove
standing = np.array([10.62, -3.24])
facing = np.array([10.82, -3.53])
target = np.array([10.19, -2.74])

forward = facing - standing
forward = forward / np.linalg.norm(forward)
target_rel = target - standing

cross = forward[0] * target_rel[1] - forward[1] * target_rel[0]
dot = np.dot(forward, target_rel)

print(f'forward: {forward}')
print(f'target_rel: {target_rel}')
print(f'cross: {cross:.3f}, dot: {dot:.3f}')
print(f'代码结果: back-right, 正确答案: back-left')
print()

# 用角度验证
angle_forward = np.arctan2(forward[1], forward[0])
angle_target = np.arctan2(target_rel[1], target_rel[0])
rel = angle_target - angle_forward
if rel > np.pi: rel -= 2*np.pi
if rel < -np.pi: rel += 2*np.pi

print(f'forward 角度: {np.degrees(angle_forward):.1f}°')
print(f'target_rel 角度: {np.degrees(angle_target):.1f}°')
print(f'相对角度: {np.degrees(rel):.1f}°')
print()
print(f'相对角度 {np.degrees(rel):.1f}° 意味着:')
if rel > 0:
    print('  target 在 forward 的逆时针方向 -> 应该是左边')
else:
    print('  target 在 forward 的顺时针方向 -> 应该是右边')

print()
print('=== 验证 2D 叉积符号约定 ===')
# 简单例子：面向 Y+，target 在 X- (应该是左边)
f = np.array([0, 1])  # 面向 Y+
t = np.array([-1, 0])  # target 在 X- (左边)
c = f[0]*t[1] - f[1]*t[0]
print(f'面向 Y+, target 在 X-: cross = {c}')
print(f'cross < 0，但 X- 是左边!')

# 反过来
t2 = np.array([1, 0])  # target 在 X+ (右边)
c2 = f[0]*t2[1] - f[1]*t2[0]
print(f'面向 Y+, target 在 X+: cross = {c2}')
print(f'cross < 0，但 X+ 是右边!')

print()
print('结论: cross 计算没有区分 X+ 和 X-??')
print('让我重新算...')
print(f'f = [0, 1], t = [-1, 0]')
print(f'cross = f_x * t_y - f_y * t_x = 0*0 - 1*(-1) = 1')
