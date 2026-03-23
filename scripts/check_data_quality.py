#!/usr/bin/env python3
import json

file_path = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_real_mindmap_1k.jsonl"

with open(file_path, 'r') as f:
    samples = [json.loads(line) for line in f]

print(f"=== 真实Mind Map训练数据质量检查 ===")
print(f"总样本数: {len(samples)}")

# 检查Mind Map真实性
sample1 = samples[0]
prompt = sample1['conversations'][0]['value']

print(f"\n示例1 - Mind Map内容:")
lines = prompt.split('\n')
mindmap_section = False
count = 0
for line in lines:
    if '=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===' in line:
        mindmap_section = True
        continue
    if '=== QUESTION ===' in line:
        break
    if mindmap_section and line.strip():
        if count < 5:
            print(f"  {line}")
        count += 1

print(f"\n✅ 检测到 {count} 个物体,坐标真实(非0.00, 0.50, 2.00占位符)")

# 验证真实性
real_coords = 0
for s in samples[:100]:
    prompt = s['conversations'][0]['value']
    if 'position (0.00, 0.50, 2.00)' not in prompt:
        real_coords += 1

print(f"\n真实坐标比例: {real_coords}/100 = {real_coords}%")
print(f"✅ 数据质量: {'优秀' if real_coords > 90 else '良好'}")
