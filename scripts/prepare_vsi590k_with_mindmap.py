#!/usr/bin/env python3
"""
准备 VSI-590K 训练数据 - 添加心智地图信息

将 VSI-590K 数据转换为带心智地图的训练格式，让模型学会利用结构化空间信息。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# 数据路径
VSI_590K_DIR = "/home/tione/notebook/tianjungu/hf_cache/VSI-590K"
OUTPUT_DIR = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_mindmap"

# 问题类型映射到 VSIBench 类型
QUESTION_TYPE_MAP = {
    "object_count": "object_counting",
    "relative_direction_object": "object_rel_direction",
    "relative_direction_room": "object_rel_direction", 
    "absolute_distance": "object_abs_distance",
    "relative_distance": "object_rel_distance",
    "object_size": "object_size_estimation",
    "room_size": "room_size_estimation",
    "appearance_order": "obj_appearance_order",
    "route_planning": "route_planning",
}


def convert_to_swift_format(sample: Dict, add_mindmap_hint: bool = True) -> Optional[Dict]:
    """
    将 VSI-590K 样本转换为 ms-swift 训练格式
    
    Args:
        sample: VSI-590K 原始样本
        add_mindmap_hint: 是否添加心智地图使用提示
    
    Returns:
        ms-swift 格式的样本
    """
    conversations = sample.get("conversations", [])
    if len(conversations) < 2:
        return None
    
    human_msg = conversations[0].get("value", "")
    gpt_msg = conversations[1].get("value", "")
    
    # 获取媒体路径
    image_path = sample.get("image")
    video_path = sample.get("video")
    
    if not image_path and not video_path:
        return None
    
    # 构建媒体完整路径
    if image_path:
        # 解析 tar.gz 内的路径
        # 格式: "arkitscenes/xxx.jpg" -> 需要解压后的路径
        media_type = "image"
        media_path = image_path
    else:
        media_type = "video"
        media_path = video_path
    
    question_type = sample.get("question_type", "unknown")
    
    # 添加心智地图使用提示
    if add_mindmap_hint:
        mindmap_hint = """
When a mind map with object positions is provided, use it as follows:
- position=(X, Y, Z): X>0 means right, X<0 means left; Y is height; Z is depth (larger=farther)
- Compare X coordinates to determine left/right relationships
- Compare Z coordinates to determine closer/farther relationships
- Use size estimates for size-related questions
- Cross-reference with visual information for best accuracy
"""
        # 在问题前添加提示
        if "<image>" in human_msg or "<video>" in human_msg:
            human_msg = human_msg.replace("<image>", f"<image>\n{mindmap_hint}\n")
            human_msg = human_msg.replace("<video>", f"<video>\n{mindmap_hint}\n")
    
    # ms-swift 格式
    swift_sample = {
        "messages": [
            {"role": "user", "content": human_msg},
            {"role": "assistant", "content": gpt_msg}
        ],
        "question_type": question_type,
    }
    
    if media_type == "image":
        swift_sample["images"] = [media_path]
    else:
        swift_sample["videos"] = [media_path]
    
    return swift_sample


def process_vsi590k(
    input_jsonl: str,
    output_jsonl: str,
    add_mindmap_hint: bool = True,
    max_samples: int = None,
):
    """
    处理 VSI-590K 数据
    """
    print(f"Processing {input_jsonl}")
    
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(input_jsonl, 'r') as f_in, open(output_jsonl, 'w') as f_out:
        for line in tqdm(f_in, desc="Converting"):
            if max_samples and count >= max_samples:
                break
            
            try:
                sample = json.loads(line.strip())
                swift_sample = convert_to_swift_format(sample, add_mindmap_hint)
                
                if swift_sample:
                    f_out.write(json.dumps(swift_sample, ensure_ascii=False) + "\n")
                    count += 1
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
    
    print(f"Converted {count} samples to {output_jsonl}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare VSI-590K with mind map hints")
    parser.add_argument("--input", type=str, default=f"{VSI_590K_DIR}/vsi_590k.jsonl")
    parser.add_argument("--output", type=str, default=f"{OUTPUT_DIR}/vsi590k_swift.jsonl")
    parser.add_argument("--no-mindmap-hint", action="store_true", help="Don't add mind map hints")
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    
    process_vsi590k(
        args.input,
        args.output,
        add_mindmap_hint=not args.no_mindmap_hint,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
