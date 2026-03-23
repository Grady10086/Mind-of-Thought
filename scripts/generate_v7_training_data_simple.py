#!/usr/bin/env python3
"""
使用V7完整感知流程生成真实Mind Map训练数据
直接复用V7的所有组件,保证100%一致性
"""

import os
import sys
from pathlib import Path

# 导入V7完整代码
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 直接导入V7的perception模块
from tests.test_v7_with_finetuned_vl import (
    PerceptionPipeline,
    ScaleCalibrator,
    MindMapEvolver,
    process_single_video
)

import json
import argparse
from tqdm import tqdm
import torch

def format_mind_map_to_text(mind_map_entities: dict) -> str:
    """格式化Mind Map为文本"""
    if not mind_map_entities:
        return "No objects detected."
    
    lines = []
    for label, entity in mind_map_entities.items():
        if entity.position_3d is None:
            continue
        
        pos = entity.position_3d
        size = entity.size_3d if entity.size_3d is not None else 0.0
        count = len(entity.detections)
        
        line = f"{label}: position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        
        if isinstance(size, (list, tuple)):
            line += f", size ({size[0]:.2f}m × {size[1]:.2f}m × {size[2]:.2f}m)"
        elif size > 0:
            line += f", size ({size:.2f}m)"
        
        if count > 1:
            line += f", count: {count}"
        
        lines.append(line)
    
    return "\n".join(lines)

def get_task_instructions(question_type: str) -> str:
    """任务指导文本"""
    task_map = {
        'relative_direction_object': """=== TASK: RELATIVE DIRECTION ===
You need to determine the spatial relationship (direction) between objects.
- Consider the viewpoint/camera position when determining directions.
- Use the position data in detected objects as a reference.""",
        
        'relative_size_object': """=== TASK: RELATIVE SIZE ===
You need to compare the sizes of objects.
- Use the size information from detected objects.
- Consider both width, height, and depth dimensions.""",
        
        'absolute_distance_object': """=== TASK: ABSOLUTE DISTANCE ===
You need to estimate the distance between two objects in meters.
- Use the 3D positions from detected objects.
- Calculate Euclidean distance between object centers.""",
        
        'object_counting': """=== TASK: OBJECT COUNTING ===
You need to count how many instances of an object appear in the scene.
- The 'count' field in detected objects provides estimates.
- Verify with visual inspection of the video frames.""",
        
        'room_size_estimation': """=== TASK: ROOM SIZE ESTIMATION ===
You need to estimate the dimensions of the room.
- Use object sizes and positions as reference points.
- Consider standard furniture sizes (door ~2m, bed ~2m).""",
        
        'size_estimation': """=== TASK: OBJECT SIZE ESTIMATION ===
You need to estimate the size/dimensions of an object.
- Use the size information from detected objects.
- Cross-reference with standard object sizes."""
    }
    
    return task_map.get(question_type, """=== TASK: SPATIAL REASONING ===
Analyze the spatial relationships in the scene to answer the question.""")

def generate_v7_prompt(sample: dict, mind_map_entities: dict) -> str:
    """生成V7格式的prompt"""
    question = sample['conversations'][0]['value']
    
    # 清理原始prompt
    if '<image>' in question:
        question = question.replace('<image>\n', '')
    if 'These are frames of a video.\n' in question:
        question = question.replace('These are frames of a video.\n', '')
    
    question_type = sample.get('question_type', 'unknown')
    
    # 构建完整prompt
    prompt = """You are a spatial intelligence assistant analyzing a video of an indoor scene.

"""
    
    # 添加任务指导
    prompt += get_task_instructions(question_type) + "\n\n"
    
    # 添加真实Mind Map
    mind_map_text = format_mind_map_to_text(mind_map_entities)
    prompt += f"""=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

"""
    
    # 添加问题
    prompt += f"""=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Please analyze the video frames and the detected objects to answer the question.
"""
    
    if 'Options:' in question or 'A.' in question:
        prompt += "Answer with the option's letter from the given choices directly.\n"
    
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-samples', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    print(f"🚀 使用V7完整感知流程生成训练数据")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"最大样本数: {args.max_samples}\n")
    
    # 加载数据
    with open(args.input, 'r') as f:
        samples = [json.loads(line) for line in f][:args.max_samples]
    
    print(f"加载 {len(samples)} 个样本\n")
    
    # 初始化V7感知pipeline (会自动加载DA3, DINO等)
    print("初始化V7感知系统...")
    # 这部分由V7的process_single_video内部处理
    
    # 处理数据
    processed = []
    failed = 0
    
    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        try:
            video_path = sample['video']
            
            if not Path(video_path).exists():
                failed += 1
                continue
            
            # 调用V7的完整perception流程
            result = process_single_video(
                video_path=video_path,
                question=sample['conversations'][0]['value'],
                answer=sample['conversations'][1]['value'],
                question_type=sample.get('question_type', 'unknown'),
                device=args.device
            )
            
            if result is None or 'mind_map' not in result:
                failed += 1
                continue
            
            # 生成V7格式的prompt
            mind_map = result['mind_map']
            new_prompt = generate_v7_prompt(sample, mind_map)
            
            # 构建新样本
            new_sample = {
                'conversations': [
                    {'from': 'human', 'value': new_prompt},
                    {'from': 'gpt', 'value': sample['conversations'][1]['value']}
                ],
                'video': video_path,
                'question_type': sample.get('question_type', 'unknown')
            }
            
            processed.append(new_sample)
            
        except Exception as e:
            print(f"\n❌ 样本 {i} 失败: {str(e)}")
            failed += 1
        
        # 清理显存
        if (i + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n✅ 处理完成")
    print(f"成功: {len(processed)}")
    print(f"失败: {failed}")
    print(f"成功率: {len(processed) / len(samples) * 100:.1f}%\n")
    
    # 保存
    print(f"💾 保存到: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w') as f:
        for sample in processed:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(processed)} 个样本")
    
    # 显示示例
    if processed:
        print("\n" + "="*80)
        print("示例 (真实V7 Mind Map):")
        print("="*80)
        print(processed[0]['conversations'][0]['value'][:800])
        print("...")

if __name__ == '__main__':
    main()
