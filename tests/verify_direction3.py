#!/usr/bin/env python3
import numpy as np

print('DA3 世界坐标系分析:')
print('Forward 主要是 Y+, Right 主要是 X+, Up 主要是 Z+')
print('这是标准的 Y-forward, X-right, Z-up 右手坐标系')
print()

# 验证叉积约定
forward = np.array([0, 1])  # 面向 Y+
target_right = np.array([1, 0])  # X+ 方向 (右边)
target_left = np.array([-1, 0])  # X- 方向 (左边)

cross_right = forward[0] * target_right[1] - forward[1] * target_right[0]
cross_left = forward[0] * target_left[1] - forward[1] * target_left[0]

print(f'面向Y+, target在X+(右边): cross = {cross_right}')
print(f'面向Y+, target在X-(左边): cross = {cross_left}')
print()
print('代码逻辑: cross < 0 -> right, cross > 0 -> left')
print(f'所以: {cross_right} < 0 -> right ✓')
print(f'所以: {cross_left} > 0 -> left ✓')
print()
print('叉积约定是正确的!')
print()

# 但实际测试中为什么还是错?
# 让我用实际案例验证
print('=== 验证实际错误案例 ===')
standing = np.array([10.62, -3.24])  # dishwasher
facing = np.array([10.82, -3.53])    # chair  
target = np.array([10.19, -2.74])    # stove

forward = facing - standing
forward = forward / np.linalg.norm(forward)
target_rel = target - standing

cross = forward[0] * target_rel[1] - forward[1] * target_rel[0]
dot = np.dot(forward, target_rel)

print(f'standing (dishwasher): {standing}')
print(f'facing (chair): {facing}')
print(f'target (stove): {target}')
print()
print(f'forward: {forward}')
print(f'target_rel: {target_rel}')
print(f'cross: {cross:.3f}, dot: {dot:.3f}')
print()

# 分析方向
if cross > 0:
    lr = 'left'
elif cross < 0:
    lr = 'right'
else:
    lr = ''

if dot > 0:
    fb = 'front'
elif dot < 0:
    fb = 'back'
else:
    fb = ''

print(f'代码预测: {fb}-{lr} if fb and lr else fb or lr')
print(f'正确答案: back-left')
print()

# 画图理解
print('坐标分析:')
print(f'standing = (10.62, -3.24)')
print(f'facing = (10.82, -3.53) -> 相对 standing 是 X+ 且 Y-')
print(f'  即: 向右下看')
print(f'target = (10.19, -2.74) -> 相对 standing 是 X- 且 Y+')
print(f'  即: target 在 standing 的左上方')
print()
print('如果站在 standing 看向 facing (右下):')
print('  - 前方是右下')
print('  - 后方是左上')
print('  - 右边是...')
print()

# 计算右边方向
# forward 指向 (X+, Y-)
# 右边 = forward 顺时针90度 = (forward_y, -forward_x)
right = np.array([forward[1], -forward[0]])
print(f'right 方向: {right}')
print(f'right 主要指向 X- 且 Y- (左下)')
print()
print('所以:')
print('- forward = 右下')
print('- back = 左上')
print('- right = 左下')
print('- left = 右上')
print()
print('target 在左上 = back 方向')
print('target 在左上... 是 back 还是 left?')
print()

# 用投影判断
proj_right = np.dot(target_rel, right)
print(f'target_rel · right = {proj_right:.3f}')
print(f'proj_right < 0 意味着 target 在 left 方向')
print()
print(f'所以 target 应该在 back-left!')
print()
print('但是叉积判断是:')
print(f'cross = {cross:.3f}')
if cross < 0:
    print('cross < 0 -> right (错误!)')
else:
    print('cross > 0 -> left')
