#!/usr/bin/env python3
"""
V7.6 - DA3 完整能力测试 (Ray Map + 统一3D坐标系)

核心改进:
1. 使用 da3nested-giant-large 模型 (多视图融合)
2. 启用 use_ray_pose=True 获取相机位姿
3. 物体位置在统一的世界坐标系中表示
4. 使用 DA3 预测的相机内参进行精确 3D 投影

测试目标:
- 验证心智地图的 3D 坐标是否在统一世界坐标系中
- 测试方向推理的准确性
- 对比原有方法的改进
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import numpy as np
import cv2
import torch
from tqdm import tqdm

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 添加 DA3 路径
DA3_PATH = PROJECT_ROOT.parent / "Depth-Anything-3"
if str(DA3_PATH / "src") not in sys.path:
    sys.path.insert(0, str(DA3_PATH / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 视频目录
VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

# 扩展词汇表 (包含方向任务常见物体)
EXTENDED_VOCABULARY = [
    "chair", "table", "sofa", "couch", "stove", "tv", "television",
    "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator", "fridge",
    "sink", "toilet", "bathtub", "door", "window", "picture", "painting",
    "pillow", "cushion", "monitor", "backpack", "bag", "heater",
    "trash can", "trash bin", "mirror", "towel", "plant", "cup",
    "nightstand", "closet", "microwave", "printer", "washer", "dryer",
    "oven", "counter", "drawer", "curtain", "rug", "carpet", "clock",
    "fan", "air conditioner", "bookshelf", "armchair", "stool",
    # 补充词汇
    "dishwasher", "telephone", "keyboard", "laptop", "whiteboard",
    "radiator", "fireplace", "vase", "bottle", "box", "basket",
    "bench", "ottoman", "dresser", "wardrobe", "piano",
]

# 方向映射
DIRECTION_MAP = {
    'front': (0, -1),   # 相机前方 (Z 负方向)
    'back': (0, 1),     # 相机后方
    'left': (-1, 0),    # 相机左方 (X 负方向)
    'right': (1, 0),    # 相机右方
    'front-left': (-1, -1),
    'front-right': (1, -1),
    'back-left': (-1, 1),
    'back-right': (1, 1),
}


def find_video_path(scene_name: str) -> Optional[str]:
    """查找视频路径"""
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def get_synonyms(obj: str) -> List[str]:
    """获取物体的同义词"""
    SYNONYM_MAP = {
        'sofa': ['couch', 'settee'],
        'tv': ['television', 'tv screen'],
        'refrigerator': ['fridge'],
        'trash bin': ['trash can', 'garbage can'],
        'couch': ['sofa'],
    }
    obj_lower = obj.lower()
    synonyms = [obj_lower]
    if obj_lower in SYNONYM_MAP:
        synonyms.extend(SYNONYM_MAP[obj_lower])
    for key, values in SYNONYM_MAP.items():
        if obj_lower in values:
            synonyms.append(key)
    return list(set(synonyms))


def match_object_name(query: str, label: str) -> bool:
    """匹配物体名称"""
    query_lower = query.lower().strip()
    label_lower = label.lower().strip()
    
    if query_lower == label_lower or query_lower in label_lower or label_lower in query_lower:
        return True
    
    query_syns = get_synonyms(query_lower)
    label_syns = get_synonyms(label_lower)
    
    return bool(set(query_syns) & set(label_syns))


def parse_direction_question(question: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    解析方向问题，提取目标物体、参考物体和观察者朝向
    
    Returns:
        (target_object, reference_object, facing_direction)
    """
    patterns = [
        # "standing by the X, facing Y, is Z to my left/right/..."
        r'standing by the (\w+).*?facing the (\w+).*?is the (\w+) to my (\w+(?:-\w+)?)',
        r'standing by (\w+).*?facing (\w+).*?is (\w+) to my (\w+(?:-\w+)?)',
        # "from X perspective looking at Y, where is Z"
        r'from (?:the )?(\w+).*?looking at (?:the )?(\w+).*?where is (?:the )?(\w+)',
        # simpler patterns
        r'where is (?:the )?(\w+) relative to (?:the )?(\w+)',
        r'(\w+) is located.*?(\w+(?:-\w+)?) of the room',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question.lower())
        if match:
            groups = match.groups()
            if len(groups) >= 4:
                return groups[2], groups[0], groups[1]  # target, standing_by, facing
            elif len(groups) >= 2:
                return groups[0], groups[1], None
    
    return None, None, None


def compute_direction_from_position(
    target_pos: np.ndarray,
    reference_pos: np.ndarray,
    facing_direction: Optional[str] = None,
) -> str:
    """
    根据 3D 位置计算方向关系
    
    Args:
        target_pos: 目标物体的世界坐标 (x, y, z)
        reference_pos: 参考点的世界坐标 (x, y, z)
        facing_direction: 观察者朝向
        
    Returns:
        direction: 方向字符串 (如 "front-left", "back-right")
    """
    # 计算相对位置
    diff = target_pos - reference_pos
    dx, dy, dz = diff[0], diff[1], diff[2]
    
    # 默认坐标系: X 右, Y 下, Z 前
    # 但需要根据 facing_direction 旋转
    
    directions = []
    
    # 前后判断 (基于 Z 或深度)
    if abs(dz) > 0.3:
        directions.append("front" if dz > 0 else "back")
    
    # 左右判断 (基于 X)
    if abs(dx) > 0.3:
        directions.append("right" if dx > 0 else "left")
    
    if len(directions) == 0:
        return "same-position"
    elif len(directions) == 1:
        return directions[0]
    else:
        # 组合方向
        return "-".join(directions)


def worker_process_v76(gpu_id: int, samples: List[Dict], output_file: str):
    """V7.6 GPU Worker - 使用 DA3 完整能力"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 导入 DA3 完整估计器
    from core.perception_da3_full import MindMapBuilder3D
    
    # 创建构建器
    builder = MindMapBuilder3D(
        device='cuda',
        num_frames=16,
        box_threshold=0.25,
        model_name="da3nested-giant-large",
        use_ray_pose=True,
    )
    
    results = []
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 构建 3D 心智地图
            mind_map_3d, da3_prediction, sampled_frames = builder.build_from_video(
                video_path,
                target_objects=None,
                extended_vocabulary=EXTENDED_VOCABULARY,
            )
            
            # 解析方向问题
            target_obj, ref_obj, facing = parse_direction_question(question)
            
            # 查找物体
            target_entity = None
            ref_entity = None
            
            for label, entity in mind_map_3d.items():
                if target_obj and match_object_name(target_obj, label):
                    target_entity = entity
                if ref_obj and match_object_name(ref_obj, label):
                    ref_entity = entity
            
            # 计算方向
            predicted_direction = None
            reasoning = ""
            
            if target_entity and target_entity.get('position_3d') is not None:
                target_pos = target_entity['position_3d']
                
                if ref_entity and ref_entity.get('position_3d') is not None:
                    ref_pos = ref_entity['position_3d']
                    predicted_direction = compute_direction_from_position(target_pos, ref_pos, facing)
                    reasoning = f"Target {target_obj} at {target_pos}, Ref {ref_obj} at {ref_pos}, direction: {predicted_direction}"
                elif da3_prediction is not None:
                    # 使用相机位置作为参考
                    camera_center = da3_prediction.get_camera_center(0)
                    predicted_direction = compute_direction_from_position(target_pos, camera_center, facing)
                    reasoning = f"Target {target_obj} at {target_pos}, Camera at {camera_center}, direction: {predicted_direction}"
                else:
                    reasoning = f"Found target {target_obj} but no reference point"
            else:
                reasoning = f"Target object '{target_obj}' not found in mind map"
            
            # 匹配选项
            pred_answer = options[0][0] if options else "A"
            if predicted_direction and options:
                for i, opt in enumerate(options):
                    opt_lower = opt.lower()
                    # 检查方向是否匹配
                    if predicted_direction in opt_lower:
                        pred_answer = chr(65 + i)
                        break
                    # 检查方向的各部分
                    for part in predicted_direction.split('-'):
                        if part in opt_lower:
                            pred_answer = chr(65 + i)
                            break
            
            # 评估
            gt_norm = gt.strip().upper()
            if len(gt_norm) > 1 and gt_norm[1] in '.、':
                gt_norm = gt_norm[0]
            
            pred_norm = pred_answer.strip().upper()
            if len(pred_norm) > 1 and pred_norm[1] in '.、':
                pred_norm = pred_norm[0]
            
            correct = pred_norm == gt_norm
            
            results.append({
                'scene_name': sample['scene_name'],
                'question': question,
                'question_type': question_type,
                'ground_truth': gt,
                'options': options,
                'prediction': pred_answer,
                'predicted_direction': predicted_direction,
                'correct': correct,
                'reasoning': reasoning,
                'target_object': target_obj,
                'reference_object': ref_obj,
                'mind_map_summary': {
                    label: {
                        'count': e.get('count', 0),
                        'position_3d': e.get('position_3d', []).tolist() if e.get('position_3d') is not None else None,
                        'confidence': e.get('avg_confidence', 0),
                    }
                    for label, e in list(mind_map_3d.items())[:10]
                },
                'da3_info': {
                    'has_extrinsics': da3_prediction.extrinsics is not None if da3_prediction else False,
                    'has_intrinsics': da3_prediction.intrinsics is not None if da3_prediction else False,
                    'num_frames': da3_prediction.num_frames if da3_prediction else 0,
                },
            })
            
        except Exception as e:
            logger.error(f"Error processing {sample['scene_name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'scene_name': sample['scene_name'],
                'question': question,
                'question_type': question_type,
                'ground_truth': gt,
                'options': options,
                'prediction': '',
                'correct': False,
                'error': str(e),
            })
    
    # 清理
    builder.unload()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def load_vsibench_direction_samples() -> List[Dict]:
    """加载 VSI-Bench 方向任务数据"""
    from datasets import load_dataset
    
    logger.info("加载 VSI-Bench 数据集...")
    ds = load_dataset(
        'nyu-visionx/VSI-Bench',
        split='test',
        cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench'
    )
    
    # 只保留方向任务
    direction_types = [
        'object_rel_direction_easy',
        'object_rel_direction_medium',
        'object_rel_direction_hard',
    ]
    
    samples = []
    for item in ds:
        if item['question_type'] not in direction_types:
            continue
        
        scene_name = item['scene_name']
        video_path = find_video_path(scene_name)
        
        if not video_path:
            continue
        
        samples.append({
            'scene_name': scene_name,
            'video_path': video_path,
            'question': item['question'],
            'question_type': item['question_type'],
            'options': item.get('options', []),
            'ground_truth': item['ground_truth'],
        })
    
    logger.info(f"加载了 {len(samples)} 个方向任务样本")
    return samples


def main():
    parser = argparse.ArgumentParser(description='V7.6 - DA3 Full Capability Test')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--max-samples', type=int, default=50, help='Max samples')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # 加载数据
    samples = load_vsibench_direction_samples()
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    logger.info(f"测试 {len(samples)} 个样本，GPU: {args.gpu}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"v76_da3_full_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = str(output_dir / f"results_gpu{args.gpu}.json")
    
    # 运行测试
    results = worker_process_v76(args.gpu, samples, output_file)
    
    # 统计结果
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'matched': 0})
    
    for r in results:
        qtype = r['question_type']
        type_stats[qtype]['total'] += 1
        if r.get('correct', False):
            type_stats[qtype]['correct'] += 1
        if r.get('predicted_direction'):
            type_stats[qtype]['matched'] += 1
    
    # 打印结果
    print("\n" + "=" * 80)
    print("V7.6 DA3 完整能力测试结果")
    print("=" * 80)
    print(f"{'任务类型':<35} {'准确率':>12} {'匹配率':>12} {'样本数':>10}")
    print("-" * 80)
    
    total_correct = 0
    total_matched = 0
    total_samples = 0
    
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        n = stats['total']
        acc = stats['correct'] / n * 100 if n > 0 else 0
        match_rate = stats['matched'] / n * 100 if n > 0 else 0
        print(f"{qtype:<35} {acc:>11.2f}% {match_rate:>11.2f}% {n:>10}")
        
        total_correct += stats['correct']
        total_matched += stats['matched']
        total_samples += n
    
    print("-" * 80)
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    overall_match = total_matched / total_samples * 100 if total_samples > 0 else 0
    print(f"{'Overall':<35} {overall_acc:>11.2f}% {overall_match:>11.2f}% {total_samples:>10}")
    print("=" * 80)
    
    # 保存汇总
    summary = {
        'timestamp': timestamp,
        'config': {
            'model': 'da3nested-giant-large',
            'use_ray_pose': True,
            'num_frames': 16,
        },
        'results_by_type': {
            qtype: {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'match_rate': stats['matched'] / stats['total'] if stats['total'] > 0 else 0,
                'samples': stats['total'],
            }
            for qtype, stats in type_stats.items()
        },
        'overall': {
            'accuracy': overall_acc / 100,
            'match_rate': overall_match / 100,
            'total_samples': total_samples,
        },
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果保存到: {output_dir}")


if __name__ == '__main__':
    main()
