#!/usr/bin/env python3
"""转换scannet采样数据为VSIBench格式"""
import json
import os
from pathlib import Path

input_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_scannet_only.jsonl")
output_path = Path("/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_scannet_vsibench.json")

VIDEO_ROOT = "/home/tione/notebook/tianjungu/datasets/VLM_3R_Videos"

def convert_video_path(relative_path: str) -> str:
    """scannet/scene0085_00.mp4 -> /path/to/scannet/videos/scene0085_00.mp4"""
    if not relative_path or '/' not in relative_path:
        return ""
    parts = relative_path.split('/')
    dataset = parts[0]
    filename = parts[-1]
    return os.path.join(VIDEO_ROOT, dataset, "videos", filename)

print("=" * 80)
print("转换scannet数据为VSIBench格式")
print("=" * 80)
print()

vsibench_data = []
valid = 0
invalid = 0

print(f"📖 读取: {input_path}")
with open(input_path, 'r') as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line)
        
        question = data.get('conversations', [{}])[0].get('value', '')
        answer = data.get('conversations', [{}])[1].get('value', '') if len(data.get('conversations', [])) > 1 else ''
        
        if '<image>' in question:
            question = question.split('<image>')[0].strip()
        
        relative_path = data.get('video', '')
        absolute_path = convert_video_path(relative_path)
        
        if absolute_path and os.path.exists(absolute_path):
            valid += 1
        else:
            invalid += 1
            if invalid <= 3:
                print(f"⚠️  视频不存在: {relative_path}")
        
        vsibench_data.append({
            "scene_name": f"scannet_sample_{line_num:05d}",
            "video_path": absolute_path,
            "question": question,
            "question_type": data.get('question_type', 'unknown'),
            "ground_truth": answer,
            "options": data.get('options', [])
        })

print(f"✅ 转换完成: {len(vsibench_data)} 样本")
print(f"   有效视频: {valid}/{len(vsibench_data)} ({valid/len(vsibench_data)*100:.1f}%)")
print(f"   无效视频: {invalid}/{len(vsibench_data)}")
print()

print(f"💾 保存到: {output_path}")
with open(output_path, 'w') as f:
    json.dump(vsibench_data, f, ensure_ascii=False, indent=2)

print("✅ 保存成功!")

from collections import Counter
task_counts = Counter(item['question_type'] for item in vsibench_data)
print("\n📊 任务分布:")
for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
    print(f"  {task:35s}: {count:5d}")
print(f"\n  总计: {len(vsibench_data)}")
