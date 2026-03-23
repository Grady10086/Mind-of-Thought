#!/usr/bin/env python3
"""过滤VSI-590K数据集，只保留能在8卡上训练的样本"""

import json
import os
from pathlib import Path

# 配置
INPUT_JSONL = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train.jsonl"
OUTPUT_JSONL = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_filtered.jsonl"
MAX_TEXT_LENGTH = 500  # 文本字符数上限
MAX_VIDEO_FRAMES = 32  # 估算：假设视频帧数与文件大小成正比

def estimate_video_frames(video_path):
    """估算视频帧数（粗略方法：通过文件大小）"""
    if not os.path.exists(video_path):
        return 999  # 文件不存在，跳过
    
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    # 假设每帧平均0.5MB (压缩视频)
    estimated_frames = file_size_mb / 0.5
    return int(estimated_frames)

def get_text_length(conversations):
    """计算对话总文本长度"""
    total_text = ' '.join([turn['value'] for turn in conversations])
    return len(total_text)

def main():
    print(f"开始过滤数据集...")
    print(f"输入: {INPUT_JSONL}")
    print(f"输出: {OUTPUT_JSONL}")
    print(f"过滤条件: 文本长度<{MAX_TEXT_LENGTH}字符, 视频<{MAX_VIDEO_FRAMES}帧(估算)")
    print()
    
    total_samples = 0
    kept_samples = 0
    filtered_by_text = 0
    filtered_by_video = 0
    filtered_by_missing = 0
    
    with open(INPUT_JSONL, 'r') as fin, open(OUTPUT_JSONL, 'w') as fout:
        for line in fin:
            total_samples += 1
            
            try:
                data = json.loads(line)
                
                # 检查文本长度
                text_length = get_text_length(data['conversations'])
                if text_length > MAX_TEXT_LENGTH:
                    filtered_by_text += 1
                    continue
                
                # 检查视频路径
                if 'video' not in data:
                    filtered_by_missing += 1
                    continue
                
                video_path = data['video']
                
                # 估算视频帧数
                estimated_frames = estimate_video_frames(video_path)
                if estimated_frames > MAX_VIDEO_FRAMES:
                    filtered_by_video += 1
                    continue
                
                # 保留样本
                fout.write(line)
                kept_samples += 1
                
                if kept_samples % 1000 == 0:
                    print(f"已处理: {total_samples}, 保留: {kept_samples} ({kept_samples/total_samples*100:.1f}%)")
                    
            except Exception as e:
                print(f"处理样本 {total_samples} 时出错: {e}")
                continue
    
    print()
    print("=" * 60)
    print(f"过滤完成！")
    print(f"总样本数: {total_samples}")
    print(f"保留样本: {kept_samples} ({kept_samples/total_samples*100:.1f}%)")
    print(f"文本过长: {filtered_by_text} ({filtered_by_text/total_samples*100:.1f}%)")
    print(f"视频过长: {filtered_by_video} ({filtered_by_video/total_samples*100:.1f}%)")
    print(f"缺失文件: {filtered_by_missing} ({filtered_by_missing/total_samples*100:.1f}%)")
    print(f"输出文件: {OUTPUT_JSONL}")
    print("=" * 60)

if __name__ == "__main__":
    main()
