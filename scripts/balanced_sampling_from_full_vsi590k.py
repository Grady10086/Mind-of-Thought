#!/usr/bin/env python3
"""
从完整VSI-590K中按任务类型平衡采样
每个任务类型抽取1000样本,构建平衡训练集
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# 设置随机种子以保证可重复性
random.seed(42)

# 完整VSI-590K路径
vsi590k_full_path = Path("/home/tione/notebook/tianjungu/hf_cache/datasets--nyu-visionx--VSI-590K/snapshots/346fbd4e41dec974bf24894d0541a49327ee6669/vsi_590k.jsonl")
output_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_per_task_full.jsonl")

def main():
    print("=" * 80)
    print("从完整VSI-590K按任务类型平衡采样")
    print("=" * 80)
    print()
    
    # 读取所有数据并按任务类型分组
    task_samples = defaultdict(list)
    
    print("📖 读取完整VSI-590K数据...")
    with open(vsi590k_full_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"  已读取 {line_num} 行...")
            
            try:
                data = json.loads(line)
                task_type = data.get('question_type', 'unknown')
                task_samples[task_type].append(data)
                
            except json.JSONDecodeError:
                print(f"⚠️  第{line_num}行JSON解析失败")
                continue
    
    print(f"✅ 共读取 {sum(len(v) for v in task_samples.values())} 样本")
    print()
    
    # 显示每个任务的样本数
    print("📊 各任务类型样本数统计:")
    print()
    for task, samples in sorted(task_samples.items(), key=lambda x: -len(x[1])):
        print(f"  {task:35s}: {len(samples):6d} 样本")
    print()
    
    # 从每个任务抽取1000样本
    print("=" * 80)
    print("🎯 开始平衡采样 (每个任务1000样本)")
    print("=" * 80)
    print()
    
    balanced_samples = []
    sampling_stats = {}
    
    for task, samples in sorted(task_samples.items()):
        available = len(samples)
        target = 1000
        
        if available >= target:
            # 随机采样1000个
            selected = random.sample(samples, target)
            sampling_stats[task] = f"{target}/{available}"
            print(f"✅ {task:35s}: 采样 {target:6d}/{available:6d}")
        else:
            # 不足1000个,全部使用
            selected = samples
            sampling_stats[task] = f"{available}/{available} (不足1000)"
            print(f"⚠️  {task:35s}: 采样 {available:6d}/{available:6d} (不足1000)")
        
        balanced_samples.extend(selected)
    
    print()
    print(f"📊 平衡采样完成,共 {len(balanced_samples)} 样本")
    print()
    
    # 打乱顺序
    random.shuffle(balanced_samples)
    
    # 写入输出文件
    print(f"💾 保存到: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in balanced_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print("✅ 保存成功!")
    print()
    
    # 最终统计
    print("=" * 80)
    print("📊 平衡采样结果统计")
    print("=" * 80)
    print()
    
    final_task_counts = defaultdict(int)
    with open(output_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_type = data.get('question_type', 'unknown')
            final_task_counts[task_type] += 1
    
    total = sum(final_task_counts.values())
    for task, count in sorted(final_task_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {task:35s}: {count:5d} ({pct:5.1f}%)")
    print()
    print(f"  {'总计':35s}: {total:5d} (100.0%)")
    print()
    
    print("=" * 80)
    print("✅ 平衡采样完成!")
    print("=" * 80)

if __name__ == '__main__':
    main()
