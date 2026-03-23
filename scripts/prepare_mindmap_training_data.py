#!/usr/bin/env python3
"""
为VSI-590K训练数据生成Mind Map增强的prompt
使其与V7测试时的prompt格式完全一致
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def format_mind_map_text(mind_map: Dict) -> str:
    """格式化Mind Map为文本 (与V7格式一致)"""
    if not mind_map or 'entities' not in mind_map:
        return "No objects detected."
    
    lines = []
    for obj_name, data in mind_map['entities'].items():
        pos = data.get('position_3d', [0, 0, 0])
        size = data.get('size', {})
        count = data.get('count', 1)
        
        line = f"{obj_name}:"
        line += f" position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        
        if size:
            h = size.get('height', 0)
            w = size.get('width', 0)
            d = size.get('depth', 0)
            line += f", size ({w:.2f}m × {d:.2f}m × {h:.2f}m)"
        
        if count > 1:
            line += f", count: {count}"
        
        lines.append(line)
    
    return "\n".join(lines)

def get_task_instructions(question_type: str) -> str:
    """根据任务类型返回指导文本 (与V7一致)"""
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

def generate_mindmap_prompt(sample: Dict, mind_map: Dict = None) -> str:
    """生成V7风格的Mind Map prompt"""
    question = sample['conversations'][0]['value']
    # 移除原始prompt中的<image>和简洁描述
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
    
    # 添加Mind Map信息
    if mind_map:
        mind_map_text = format_mind_map_text(mind_map)
        prompt += f"""=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

"""
    
    # 添加问题
    prompt += f"""=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Please analyze the video frames and the detected objects to answer the question.
"""
    
    # 如果是选择题,添加选项格式提示
    if 'Options:' in question or 'A.' in question:
        prompt += "Answer with the option's letter from the given choices directly.\n"
    
    return prompt

def process_dataset(
    input_file: str,
    output_file: str,
    use_dummy_mindmap: bool = True
):
    """
    处理训练数据集
    
    Args:
        input_file: 原始训练数据 (VSI-590K格式)
        output_file: 输出文件 (Mind Map增强格式)
        use_dummy_mindmap: 是否使用虚拟Mind Map (实际应该运行DA3+DINO)
    """
    print(f"Loading dataset from: {input_file}")
    
    with open(input_file, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    print(f"Total samples: {len(samples)}")
    
    processed = []
    
    for sample in tqdm(samples, desc="Processing"):
        # TODO: 这里应该实际运行DA3深度估计和GroundingDINO检测
        # 现在用虚拟Mind Map作为占位
        if use_dummy_mindmap:
            # 从问题中提取物体名称
            question = sample['conversations'][0]['value']
            # 简单的物体提取 (实际应该用NER或正则)
            common_objects = ['chair', 'table', 'sofa', 'bed', 'door', 'window']
            detected = [obj for obj in common_objects if obj in question.lower()]
            
            # 构建虚拟Mind Map
            mind_map = {'entities': {}}
            for i, obj in enumerate(detected):
                mind_map['entities'][obj] = {
                    'position_3d': [i*1.5, 0.5, 2.0 + i*0.5],
                    'size': {'height': 0.8, 'width': 0.5, 'depth': 0.5},
                    'count': 1
                }
        else:
            # TODO: 实际运行perception pipeline
            mind_map = None
        
        # 生成新prompt
        new_prompt = generate_mindmap_prompt(sample, mind_map)
        
        # 构建新样本
        new_sample = {
            'conversations': [
                {'from': 'human', 'value': new_prompt},
                {'from': 'gpt', 'value': sample['conversations'][1]['value']}
            ],
            'video': sample['video'],
            'question_type': sample.get('question_type', 'unknown')
        }
        
        processed.append(new_sample)
    
    # 保存
    print(f"Saving to: {output_file}")
    with open(output_file, 'w') as f:
        for sample in processed:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Processed {len(processed)} samples")
    
    # 显示示例
    print("\n" + "="*80)
    print("EXAMPLE PROMPT:")
    print("="*80)
    print(processed[0]['conversations'][0]['value'])
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Generate Mind Map training data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input training data (JSONL)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output training data (JSONL)')
    parser.add_argument('--real-mindmap', action='store_true',
                       help='Use real perception pipeline (DA3+DINO)')
    
    args = parser.parse_args()
    
    process_dataset(
        args.input,
        args.output,
        use_dummy_mindmap=not args.real_mindmap
    )

if __name__ == '__main__':
    main()
