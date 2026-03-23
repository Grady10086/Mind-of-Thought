#!/usr/bin/env python3
"""
生成真实Mind Map训练数据 - 完整感知流程
使用DA3深度估计 + GroundingDINO检测 + V7完整pipeline

严格要求:
1. 必须运行真实的感知系统 (DA3 + DINO)
2. 必须生成真实的3D坐标和尺寸
3. 不允许使用任何虚假/占位数据
4. 保证训练-测试分布完全一致
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import torch
import gc

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.perception import DepthEstimator
from core.calibrator import Calibrator
from core.evolver import Evolver
from core.mind_map import MindMap3D
from utils.video_utils import VideoReader


class RealMindMapGenerator:
    """真实Mind Map生成器 - 完整感知流程"""
    
    def __init__(self, device='cuda', half_precision=True):
        self.device = device
        print(f"🚀 初始化真实感知系统...")
        
        # 1. 深度估计器 (DA3)
        print("  [1/3] 加载 Depth Anything V3...")
        self.depth_estimator = DepthEstimator(
            device=device,
            half_precision=half_precision,
            model_size='large'
        )
        
        # 2. 校准器 (标定物检测)
        print("  [2/3] 加载 Calibrator (GroundingDINO)...")
        self.calibrator = Calibrator(device=device)
        
        # 3. 演化器 (Mind Map构建)
        print("  [3/3] 加载 Evolver...")
        self.evolver = Evolver()
        
        print("✅ 感知系统初始化完成!\n")
    
    def process_video(self, video_path: str) -> Dict:
        """
        处理单个视频,生成真实Mind Map
        
        Returns:
            mind_map: {
                'entities': {
                    'chair': {
                        'position_3d': [x, y, z],
                        'size': {'width': w, 'height': h, 'depth': d},
                        'count': n
                    },
                    ...
                }
            }
        """
        try:
            # 读取视频
            reader = VideoReader(video_path)
            frames = reader.read_frames(max_frames=32)  # 采样32帧
            
            if len(frames) == 0:
                return None
            
            # Step 1: 深度估计 (DA3)
            depth_pred = self.depth_estimator.infer_video(frames)
            
            # Step 2: 标定物检测和校准
            calibration_results = self.calibrator.detect_and_calibrate(
                frames=frames,
                depth_maps=depth_pred.depth_maps
            )
            
            # Step 3: 构建初始Mind Map
            mind_map = MindMap3D()
            mind_map.build_from_perception(
                frames=frames,
                depth_maps=depth_pred.depth_maps,
                calibration=calibration_results
            )
            
            # Step 4: 演化修正 (使用标定物)
            if calibration_results['detected_calibrators']:
                self.evolver.evolve_mind_map(
                    mind_map=mind_map,
                    calibrators=calibration_results['detected_calibrators']
                )
            
            # 返回Mind Map数据
            return mind_map.to_dict()
            
        except Exception as e:
            print(f"  ⚠️  处理视频失败: {video_path}")
            print(f"     错误: {str(e)}")
            return None
        finally:
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def format_mind_map_text(self, mind_map: Dict) -> str:
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
    
    def get_task_instructions(self, question_type: str) -> str:
        """根据任务类型返回指导文本"""
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
- Cross-reference with standard object sizes.""",
            
            'route_planning': """=== TASK: ROUTE PLANNING ===
You need to plan a path between locations.
- Use object positions to understand spatial layout.
- Consider obstacles and distances.""",
            
            'appearance_order': """=== TASK: APPEARANCE ORDER ===
You need to determine the order objects appear in the video.
- Analyze the temporal sequence of frames.
- Track when objects first become visible."""
        }
        
        return task_map.get(question_type, """=== TASK: SPATIAL REASONING ===
Analyze the spatial relationships in the scene to answer the question.""")
    
    def generate_prompt(self, sample: Dict, mind_map: Dict) -> str:
        """生成V7风格的完整prompt"""
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
        prompt += self.get_task_instructions(question_type) + "\n\n"
        
        # 添加真实Mind Map
        mind_map_text = self.format_mind_map_text(mind_map)
        prompt += f"""=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

"""
        
        # 添加问题
        prompt += f"""=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Please analyze the video frames and the detected objects to answer the question.
"""
        
        # 选择题格式提示
        if 'Options:' in question or 'A.' in question:
            prompt += "Answer with the option's letter from the given choices directly.\n"
        
        return prompt


def main():
    parser = argparse.ArgumentParser(
        description='生成真实Mind Map训练数据 - 完整感知流程'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入文件 (原始VSI-590K训练数据)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件 (真实Mind Map增强数据)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='最大处理样本数 (默认1000)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda/cpu)'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='使用的GPU数量'
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not Path(args.input).exists():
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    # 加载原始数据
    print(f"📂 加载原始数据: {args.input}")
    with open(args.input, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    # 限制样本数
    if len(samples) > args.max_samples:
        print(f"⚠️  数据集有 {len(samples)} 样本,限制为 {args.max_samples}")
        samples = samples[:args.max_samples]
    
    print(f"✅ 加载 {len(samples)} 个样本\n")
    
    # 初始化生成器
    generator = RealMindMapGenerator(device=args.device)
    
    # 处理数据
    processed_samples = []
    failed_count = 0
    
    print(f"🔄 开始处理视频并生成真实Mind Map...")
    print(f"=" * 80)
    
    for i, sample in enumerate(tqdm(samples, desc="Processing"), 1):
        video_path = sample.get('video', '')
        
        # 检查视频文件
        if not Path(video_path).exists():
            print(f"\n  ⚠️  [{i}/{len(samples)}] 视频不存在: {video_path}")
            failed_count += 1
            continue
        
        # 运行真实感知流程
        mind_map = generator.process_video(video_path)
        
        if mind_map is None:
            failed_count += 1
            continue
        
        # 生成新prompt
        new_prompt = generator.generate_prompt(sample, mind_map)
        
        # 构建新样本
        new_sample = {
            'conversations': [
                {'from': 'human', 'value': new_prompt},
                {'from': 'gpt', 'value': sample['conversations'][1]['value']}
            ],
            'video': video_path,
            'question_type': sample.get('question_type', 'unknown'),
            'mind_map': mind_map  # 保存原始Mind Map用于调试
        }
        
        processed_samples.append(new_sample)
        
        # 每100个样本显示进度
        if i % 100 == 0:
            success_rate = (i - failed_count) / i * 100
            print(f"\n  📊 进度: {i}/{len(samples)} | 成功率: {success_rate:.1f}%")
    
    print("\n" + "=" * 80)
    print(f"✅ 处理完成!")
    print(f"   成功: {len(processed_samples)} 样本")
    print(f"   失败: {failed_count} 样本")
    print(f"   成功率: {len(processed_samples) / len(samples) * 100:.1f}%")
    
    # 保存结果
    print(f"\n💾 保存到: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in processed_samples:
            # 移除mind_map字段 (仅用于调试)
            sample_to_save = sample.copy()
            sample_to_save.pop('mind_map', None)
            f.write(json.dumps(sample_to_save, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(processed_samples)} 个高质量样本")
    
    # 显示示例
    if processed_samples:
        print("\n" + "=" * 80)
        print("示例 PROMPT (真实Mind Map):")
        print("=" * 80)
        print(processed_samples[0]['conversations'][0]['value'][:1000])
        print("..." if len(processed_samples[0]['conversations'][0]['value']) > 1000 else "")
        print("=" * 80)


if __name__ == '__main__':
    main()
