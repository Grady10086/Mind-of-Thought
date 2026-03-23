#!/usr/bin/env python3
"""
从VSI-590K的scannet数据集中平衡采样
每个任务类型1000样本
"""
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

vsi590k_path = Path("/home/tione/notebook/tianjungu/hf_cache/datasets--nyu-visionx--VSI-590K/snapshots/346fbd4e41dec974bf24894d0541a49327ee6669/vsi_590k.jsonl")
output_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_scannet_only.jsonl")

print("=" * 80)
print("从VSI-590K的scannet数据集中平衡采样")
print("=" * 80)
print()

# 按任务类型分组
task_samples = defaultdict(list)

print("📖 读取VSI-590K (只保留scannet)...")
with open(vsi590k_path, 'r') as f:
    for line_num, line in enumerate(f, 1):
        if line_num % 100000 == 0:
            print(f"  已读取 {line_num} 行...")
        
        data = json.loads(line)
        video = data.get('video', '')
        
        # 只保留scannet数据
        if not video.startswith('scannet/'):
            continue
        
        task_type = data.get('question_type', 'unknown')
        task_samples[task_type].append(data)

print(f"✅ scannet数据: {sum(len(v) for v in task_samples.values())} 样本")
print()

# 显示每个任务的样本数
print("📊 各任务类型样本数:")
for task, samples in sorted(task_samples.items(), key=lambda x: -len(x[1])):
    print(f"  {task:35s}: {len(samples):5d} 样本")
print()

# 从每个任务抽取1000样本
print("=" * 80)
print("🎯 开始平衡采样 (每个任务1000样本)")
print("=" * 80)
print()

balanced_samples = []

for task, samples in sorted(task_samples.items()):
    available = len(samples)
    target = 1000
    
    if available >= target:
        selected = random.sample(samples, target)
        print(f"✅ {task:35s}: 采样 {target:5d}/{available:5d}")
    else:
        selected = samples
        print(f"⚠️  {task:35s}: 采样 {available:5d}/{available:5d} (不足1000)")
    
    balanced_samples.extend(selected)

print()
print(f"📊 平衡采样完成,共 {len(balanced_samples)} 样本")
print()

# 打乱顺序
random.shuffle(balanced_samples)

# 保存
print(f"💾 保存到: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    for sample in balanced_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print("✅ 保存成功!")
print()

# 最终统计
from collections import Counter
final_counts = Counter(item['question_type'] for item in balanced_samples)

print("📊 最终采样结果:")
for task, count in sorted(final_counts.items(), key=lambda x: -x[1]):
    pct = count / len(balanced_samples) * 100
    print(f"  {task:35s}: {count:5d} ({pct:5.1f}%)")
print()
print(f"  {'总计':35s}: {len(balanced_samples):5d} (100.0%)")
print()

print("=" * 80)
print("✅ 完成! 所有数据都来自scannet数据集")
print("=" * 80)
