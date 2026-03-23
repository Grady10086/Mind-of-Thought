#!/usr/bin/env python3
"""
简化版: 直接运行V7两次,分别生成无Evolution和带Evolution的训练数据
基于现有的test_v7_with_finetuned_vl.py,去掉VL推理部分,只导出Mind Map
"""

import sys
import os
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')

# 直接执行实验1和实验2
print("=" * 80)
print("开始生成Mind Map训练数据")
print("=" * 80)
print()

# 由于生成Mind Map需要完整的V7代码,而代码已经很长
# 我采用pragmatic的方式: 直接调用export_v7_mindmap_for_training.py两次
# 第一次禁用Evolution, 第二次启用Evolution

import subprocess

data_file = "data/vsi590k_subset/train_balanced_1k_per_task_vsibench.json"

# 实验1: 无Evolution
print("┌" + "─" * 78 + "┐")
print("│ 实验1: 生成Mind Map (无Evolution)                                          │")
print("└" + "─" * 78 + "┘")
print()

cmd1 = [
    "python3", "scripts/v7_perception_for_training.py",
    "--input-file", data_file,
    "--output-file", "data/vsi590k_subset/train_balanced_mindmap_no_evo.jsonl",
    "--num-gpus", "8",
    "--disable-evolution"  # 关键: 禁用Evolution
]

# 检查脚本是否支持--disable-evolution参数
# 如果不支持,我们需要修改脚本

print(f"执行命令: {' '.join(cmd1)}")
print()

result = subprocess.run(cmd1, capture_output=False)

if result.returncode != 0:
    print()
    print("❌ 实验1失败")
    print("原因: scripts/v7_perception_for_training.py可能不支持--disable-evolution参数")
    print()
    print("建议:")
    print("1. 检查scripts/v7_perception_for_training.py")
    print("2. 或手动复制export_v7_mindmap_for_training.py并修改")
    sys.exit(1)

print()
print("✅ 实验1完成!")
print()

# 实验2: 带Evolution
print("┌" + "─" * 78 + "┐")
print("│ 实验2: 生成Mind Map (带Evolution)                                          │")
print("└" + "─" * 78 + "┘")
print()

cmd2 = [
    "python3", "scripts/v7_perception_for_training.py",
    "--input-file", data_file,
    "--output-file", "data/vsi590k_subset/train_balanced_mindmap_with_evo.jsonl",
    "--num-gpus", "8",
    # 不加--disable-evolution,默认启用
]

print(f"执行命令: {' '.join(cmd2)}")
print()

result = subprocess.run(cmd2, capture_output=False)

if result.returncode != 0:
    print()
    print("❌ 实验2失败")
    sys.exit(1)

print()
print("✅ 实验2完成!")
print()

print("=" * 80)
print("✅ 两个版本的Mind Map数据都已生成!")
print("=" * 80)
print()
print("输出文件:")
print("  1. data/vsi590k_subset/train_balanced_mindmap_no_evo.jsonl")
print("  2. data/vsi590k_subset/train_balanced_mindmap_with_evo.jsonl")
