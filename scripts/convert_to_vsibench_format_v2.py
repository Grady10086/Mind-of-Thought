#!/usr/bin/env python3
"""
转换平衡采样数据为VSIBench格式,以便V7处理
V2: 修复视频路径问题
"""

import json
import os
from pathlib import Path

# 输入输出路径
input_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_per_task_full.jsonl")
output_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_per_task_full_vsibench_v2.json")

# 视频根目录
VIDEO_ROOT = "/home/tione/notebook/tianjungu/datasets/VLM_3R_Videos"

def convert_video_path(relative_path: str) -> str:
    """
    转换VSI-590K的相对视频路径为绝对路径
    
    Args:
        relative_path: 例如 "scannet/scene0085_00.mp4" 或 "arkitscenes/41069025.mp4"
    
    Returns:
        绝对路径: "/home/tione/notebook/tianjungu/datasets/VLM_3R_Videos/scannet/videos/scene0085_00.mp4"
    """
    if not relative_path:
        return ""
    
    # 分离dataset和文件名
    parts = relative_path.split('/')
    if len(parts) < 2:
        print(f"⚠️  无效路径格式: {relative_path}")
        return ""
    
    dataset = parts[0]  # scannet, arkitscenes等
    filename = parts[-1]  # scene0085_00.mp4
    
    # 构造完整路径: VIDEO_ROOT/{dataset}/videos/{filename}
    full_path = os.path.join(VIDEO_ROOT, dataset, "videos", filename)
    
    return full_path

print("=" * 80)
print("转换为VSIBench格式 (V2 - 修复视频路径)")
print("=" * 80)
print()

# 统计信息
total_samples = 0
valid_samples = 0
invalid_paths = 0

vsibench_data = []

print(f"📖 读取: {input_path}")
with open(input_path, 'r') as f:
    for line_num, line in enumerate(f, 1):
        total_samples += 1
        data = json.loads(line)
        
        # 提取信息
        question = data.get('conversations', [{}])[0].get('value', '')
        answer = data.get('conversations', [{}])[1].get('value', '') if len(data.get('conversations', [])) > 1 else ''
        
        # 简单提取问题(移除<image>标签后的内容)
        if '<image>' in question:
            question = question.split('<image>')[0].strip()
        
        # 转换视频路径
        relative_path = data.get('video', '')
        absolute_path = convert_video_path(relative_path)
        
        # 验证路径是否存在
        if absolute_path and os.path.exists(absolute_path):
            valid_samples += 1
        else:
            invalid_paths += 1
            if invalid_paths <= 5:  # 只打印前5个错误
                print(f"⚠️  视频不存在: {relative_path} -> {absolute_path}")
        
        # 构造VSIBench格式
        vsibench_item = {
            "scene_name": f"vsi590k_sample_{line_num:05d}",
            "video_path": absolute_path,  # 使用绝对路径
            "question": question,
            "question_type": data.get('question_type', 'unknown'),
            "ground_truth": answer,
            "options": data.get('options', [])
        }
        
        vsibench_data.append(vsibench_item)

print(f"✅ 转换完成: {len(vsibench_data)} 样本")
print(f"   - 有效视频路径: {valid_samples}/{total_samples}")
print(f"   - 无效视频路径: {invalid_paths}/{total_samples}")
print()

print(f"💾 保存到: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(vsibench_data, f, ensure_ascii=False, indent=2)

print("✅ 保存成功!")
print()

# 统计
from collections import Counter
task_counts = Counter(item['question_type'] for item in vsibench_data)

print("📊 任务分布:")
for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
    pct = count / len(vsibench_data) * 100
    print(f"  {task:35s}: {count:5d} ({pct:5.1f}%)")
print()
print(f"  {'总计':35s}: {len(vsibench_data):5d} (100.0%)")
print()

print("=" * 80)
print("✅ 转换完成! 可以开始生成Mind Map")
print("=" * 80)
