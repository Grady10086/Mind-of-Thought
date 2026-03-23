#!/usr/bin/env python3
"""
彻底验证 2D 叉积和左右方向的关系
"""
import numpy as np

print("=== 2D 叉积与左右方向 ===\n")

# 场景：站在原点，面向不同方向
# 测试 target 在不同位置时，叉积的符号

scenarios = [
    # (forward, target, expected_lr)
    ((0, 1), (1, 0), "right"),   # 面向Y+, target在X+ (东)
    ((0, 1), (-1, 0), "left"),   # 面向Y+, target在X- (西)
    ((0, 1), (1, 1), "right"),   # 面向Y+, target在右前
    ((0, 1), (-1, 1), "left"),   # 面向Y+, target在左前
    ((1, 0), (0, 1), "left"),    # 面向X+, target在Y+ (左)
    ((1, 0), (0, -1), "right"),  # 面向X+, target在Y- (右)
    ((-1, 0), (0, 1), "right"),  # 面向X-, target在Y+ (右)
    ((-1, 0), (0, -1), "left"),  # 面向X-, target在Y- (左)
]

for fwd, tgt, expected in scenarios:
    fwd = np.array(fwd)
    tgt = np.array(tgt)
    
    # 叉积: fwd_x * tgt_y - fwd_y * tgt_x
    cross = fwd[0] * tgt[1] - fwd[1] * tgt[0]
    
    # 当前代码逻辑: cross > 0 -> left, cross < 0 -> right
    predicted = "left" if cross > 0 else "right"
    
    match = "✓" if predicted == expected else "✗"
    
    print(f"Forward={fwd}, Target={tgt}")
    print(f"  Cross={cross:+.1f}, Predicted={predicted}, Expected={expected} {match}")
    print()

# 验证结果
print("\n=== 验证标准数学定义 ===")
print("2D 叉积: a × b = a_x * b_y - a_y * b_x")
print("如果 a × b > 0, b 在 a 的逆时针方向 (左边)")
print("如果 a × b < 0, b 在 a 的顺时针方向 (右边)")
print()

# 但是人类的左右定义取决于坐标系约定！
print("=== 坐标系约定分析 ===")
print("如果使用标准数学坐标系 (X右, Y上):")
print("  面向 Y+ (上), 右边是 X+ (东)")
print("  从 Y+ 到 X+ 是顺时针方向")
print("  所以 cross < 0 表示右边 ✓")
print()
print("但如果使用图像坐标系 (X右, Y下):")
print("  面向 Y+ (下), 右边是 X- (西)")
print("  从 Y+ 到 X- 是...?")
print()

# 关键测试
print("=== 关键测试 ===")
print("假设面向 (0, 1) [Y+方向]")
print("右边应该是 X+ 还是 X-?")
print()
print("在标准数学坐标系中:")
print("  站在原点，看向北方 (Y+)")
print("  右手边是东方 (X+)")
print("  forward=(0,1), right=(1,0)")
print("  cross = 0*0 - 1*1 = -1 < 0")
print("  所以 cross < 0 确实表示右边 ✓")
print()
print("结论: 我们的叉积逻辑是正确的!")
print("cross > 0 -> left")
print("cross < 0 -> right")
