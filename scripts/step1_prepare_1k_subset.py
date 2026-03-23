#!/usr/bin/env python3
"""
使用V7完整流程生成训练数据
直接复用test_v7_with_finetuned_vl.py的所有代码,只修改输出格式
"""

# 首先提取前1000个样本
import json
from pathlib import Path

input_file = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_filtered_10pct.jsonl"
output_subset = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_1k_for_mindmap.jsonl"

print(f"提取前1000个样本...")

with open(input_file, 'r') as f:
    samples = [json.loads(line) for line in f]

# 取前1000个
samples_1k = samples[:1000]

# 保存
with open(output_subset, 'w') as f:
    for sample in samples_1k:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"✅ 已保存 {len(samples_1k)} 个样本到: {output_subset}")
print("\n现在需要手动执行以下步骤:")
print("1. 修改 test_v7_with_finetuned_vl.py,在worker_process中添加Mind Map导出")
print("2. 运行修改后的脚本处理这1000个样本")
print("3. 将输出的Mind Map数据转换为训练格式")
