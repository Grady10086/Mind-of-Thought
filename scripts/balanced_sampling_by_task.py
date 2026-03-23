#!/usr/bin/env python3
"""
从VSI-590K中按任务类型平衡采样
每个任务类型抽取1000样本,构建平衡训练集
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# 设置随机种子以保证可重复性
random.seed(42)

# VSI-590K路径
vsi590k_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_filtered_10pct.jsonl")
output_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_per_task.jsonl")

# VSI-590K到VSIBench的任务类型映射
TASK_TYPE_MAPPING = {
    # VSI-590K任务类型 -> VSIBench任务类型
    'relative_direction_object': 'relative_direction',  # 需要拆分难度
    'relative_size_object': 'object_size',
    'absolute_size_object': 'object_size', 
    'absolute_distance_object': 'absolute_distance',
    'relative_distance_object': 'relative_distance',
    'absolute_direction_object': 'absolute_direction',
    'absolute_size_room': 'room_size',
    'absolute_count': 'object_counting',
    'relative_count': 'object_counting',
    # 注意: appearance_order 和 route_planning 可能不在VSI-590K中
}

# VSIBench主要任务类型
VSIBENCH_TASKS = [
    'appearance_order',
    'absolute_distance',
    'object_counting', 
    'relative_direction',
    'relative_distance',
    'object_size',
    'room_size',
    'route_planning',
]

def main():
    print("=" * 80)
    print("从VSI-590K按任务类型平衡采样")
    print("=" * 80)
    print()
    
    # 读取所有数据并按任务类型分组
    task_samples = defaultdict(list)
    
    print("📖 读取VSI-590K数据...")
    with open(vsi590k_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                task_type = data.get('question_type', 'unknown')
                
                # 转换为VSIBench格式的任务类型
                mapped_task = TASK_TYPE_MAPPING.get(task_type, task_type)
                
                # 保存原始数据
                data['_original_task_type'] = task_type
                data['_mapped_task_type'] = mapped_task
                
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
        print(f"  {task:35s}: {len(samples):5d} 样本")
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
            print(f"✅ {task:35s}: 采样 {target}/{available}")
        else:
            # 不足1000个,全部使用
            selected = samples
            sampling_stats[task] = f"{available}/{available} (不足1000)"
            print(f"⚠️  {task:35s}: 采样 {available}/{available} (不足1000)")
        
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
            # 移除临时字段
            sample.pop('_original_task_type', None)
            sample.pop('_mapped_task_type', None)
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
    print()
    print("下一步:")
    print("  1. 转换为VSIBench格式")
    print("  2. 用V7生成Mind Map (无Evolution)")
    print("  3. 用V7生成Mind Map (带Evolution)")
    print("  4. 分别训练两个模型")
    print("  5. 测试对比")

if __name__ == '__main__':
    main()
