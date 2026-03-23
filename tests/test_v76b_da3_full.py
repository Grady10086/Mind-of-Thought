#!/usr/bin/env python3
"""
V7.6b - DA3 完整能力测试 (改进版方向计算)

改进:
1. 正确解析 "standing by X, facing Y, is Z to my..." 格式
2. 考虑观察者视角旋转 (facing direction)
3. 在观察者坐标系中计算相对方向
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
from pathlib import Path
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

DA3_PATH = PROJECT_ROOT.parent / "Depth-Anything-3"
if str(DA3_PATH / "src") not in sys.path:
    sys.path.insert(0, str(DA3_PATH / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

EXTENDED_VOCABULARY = [
    "chair", "table", "sofa", "couch", "stove", "tv", "television",
    "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator", "fridge",
    "sink", "toilet", "bathtub", "door", "window", "picture", "painting",
    "pillow", "cushion", "monitor", "backpack", "bag", "heater",
    "trash can", "trash bin", "mirror", "towel", "plant", "cup",
    "nightstand", "closet", "microwave", "printer", "washer", "dryer",
    "oven", "counter", "drawer", "curtain", "rug", "carpet", "clock",
    "fan", "air conditioner", "bookshelf", "armchair", "stool",
    "dishwasher", "telephone", "keyboard", "laptop", "whiteboard",
    "radiator", "fireplace", "vase", "bottle", "box", "basket",
    "bench", "ottoman", "dresser", "wardrobe", "piano",
]

SYNONYM_MAP = {
    'sofa': ['couch', 'settee'],
    'tv': ['television', 'tv screen', 'monitor'],
    'refrigerator': ['fridge'],
    'trash bin': ['trash can', 'garbage can'],
    'couch': ['sofa'],
    'nightstand': ['night stand', 'bedside table'],
    'armchair': ['arm chair'],
}


def find_video_path(scene_name: str) -> Optional[str]:
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def get_synonyms(obj: str) -> List[str]:
    obj_lower = obj.lower().strip()
    synonyms = [obj_lower]
    if obj_lower in SYNONYM_MAP:
        synonyms.extend(SYNONYM_MAP[obj_lower])
    for key, values in SYNONYM_MAP.items():
        if obj_lower in values:
            synonyms.append(key)
            synonyms.extend(v for v in values if v != obj_lower)
    return list(set(synonyms))


def match_object_name(query: str, label: str) -> bool:
    query_lower = query.lower().strip()
    label_lower = label.lower().strip()
    
    if query_lower == label_lower:
        return True
    if query_lower in label_lower or label_lower in query_lower:
        return True
    
    query_syns = get_synonyms(query_lower)
    label_syns = get_synonyms(label_lower)
    return bool(set(query_syns) & set(label_syns))


def parse_direction_question_v2(question: str) -> Dict[str, Optional[str]]:
    """
    解析方向问题 - 改进版
    
    格式: "If I am standing by the X and facing the Y, is the Z to my [direction]?"
    
    Returns:
        {
            'standing_by': X,
            'facing': Y,
            'target': Z,
            'options_direction': 提取的方向选项
        }
    """
    result = {
        'standing_by': None,
        'facing': None,
        'target': None,
    }
    
    q_lower = question.lower()
    
    # Pattern 1: "standing by the X and facing the Y, is the Z"
    pattern1 = r'standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+)'
    match = re.search(pattern1, q_lower)
    if match:
        result['standing_by'] = match.group(1)
        result['facing'] = match.group(2)
        result['target'] = match.group(3)
        return result
    
    # Pattern 2: "from the X, looking at Y, where is Z"
    pattern2 = r'from (?:the )?(\w+).*?looking at (?:the )?(\w+).*?(?:where is|is) (?:the )?(\w+)'
    match = re.search(pattern2, q_lower)
    if match:
        result['standing_by'] = match.group(1)
        result['facing'] = match.group(2)
        result['target'] = match.group(3)
        return result
    
    return result


def compute_direction_in_viewer_frame(
    target_pos: np.ndarray,
    standing_pos: np.ndarray,
    facing_pos: np.ndarray,
) -> str:
    """
    在观察者坐标系中计算目标方向
    
    使用投影方法:
    1. 计算前方向量 (从 standing 指向 facing)
    2. 计算右方向量 (前方向量顺时针旋转90度)
    3. 将 target 投影到前方和右方
    
    使用 X-Y 平面 (俯视图)，忽略 Z (高度)
    
    Args:
        target_pos: 目标物体位置 (世界坐标)
        standing_pos: 观察者位置 (世界坐标)
        facing_pos: 观察者面向的物体位置 (世界坐标)
        
    Returns:
        direction: "front-left", "back-right", etc.
    """
    # 使用 X-Y 平面 (俯视图)
    forward = facing_pos[:2] - standing_pos[:2]
    forward_norm = np.linalg.norm(forward)
    
    if forward_norm < 0.01:
        return "same-position"
    
    forward = forward / forward_norm
    
    # 右方向: 顺时针旋转90度
    # 在标准数学坐标系中 (X 右, Y 上):
    # 顺时针90度: (x, y) -> (y, -x)
    right = np.array([forward[1], -forward[0]])
    
    target_rel = target_pos[:2] - standing_pos[:2]
    target_norm = np.linalg.norm(target_rel)
    
    if target_norm < 0.01:
        return "same-position"
    
    # 投影到前方和右方
    proj_forward = np.dot(target_rel, forward)
    proj_right = np.dot(target_rel, right)
    
    # 使用相对阈值
    threshold = target_norm * 0.1
    
    directions = []
    
    # 前后
    if proj_forward > threshold:
        directions.append("front")
    elif proj_forward < -threshold:
        directions.append("back")
    
    # 左右
    if proj_right > threshold:
        directions.append("right")
    elif proj_right < -threshold:
        directions.append("left")
    
    if len(directions) == 0:
        return "same-position"
    elif len(directions) == 1:
        return directions[0]
    else:
        return "-".join(directions)


def direction_to_option(predicted_dir: str, options: List[str]) -> str:
    """将预测的方向映射到选项"""
    if not options:
        return "A"
    
    # 方向规范化
    dir_map = {
        'front-left': ['front-left', 'front left', 'left-front'],
        'front-right': ['front-right', 'front right', 'right-front'],
        'back-left': ['back-left', 'back left', 'left-back', 'rear-left'],
        'back-right': ['back-right', 'back right', 'right-back', 'rear-right'],
        'front': ['front', 'ahead', 'forward', 'in front'],
        'back': ['back', 'behind', 'rear', 'backward'],
        'left': ['left', 'to the left', 'on the left'],
        'right': ['right', 'to the right', 'on the right'],
    }
    
    # 获取预测方向的所有变体
    pred_variants = dir_map.get(predicted_dir, [predicted_dir])
    
    for i, opt in enumerate(options):
        opt_lower = opt.lower()
        for variant in pred_variants:
            if variant in opt_lower:
                return chr(65 + i)
    
    # 如果完全匹配失败，尝试部分匹配
    pred_parts = predicted_dir.split('-')
    for i, opt in enumerate(options):
        opt_lower = opt.lower()
        match_count = sum(1 for part in pred_parts if part in opt_lower)
        if match_count == len(pred_parts):
            return chr(65 + i)
    
    return "A"


def worker_process_v76b(gpu_id: int, samples: List[Dict], output_file: str):
    """V7.6b Worker"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from core.perception_da3_full import MindMapBuilder3D
    
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
            # 构建心智地图
            mind_map_3d, da3_prediction, sampled_frames = builder.build_from_video(
                video_path,
                target_objects=None,
                extended_vocabulary=EXTENDED_VOCABULARY,
            )
            
            # 解析问题
            parsed = parse_direction_question_v2(question)
            standing_by = parsed['standing_by']
            facing = parsed['facing']
            target = parsed['target']
            
            # 查找物体
            def find_entity(obj_name):
                if not obj_name:
                    return None
                for label, entity in mind_map_3d.items():
                    if match_object_name(obj_name, label):
                        return entity
                return None
            
            standing_entity = find_entity(standing_by)
            facing_entity = find_entity(facing)
            target_entity = find_entity(target)
            
            # 计算方向
            predicted_direction = None
            reasoning = ""
            
            if target_entity and target_entity.get('position_3d') is not None:
                target_pos = target_entity['position_3d']
                
                if standing_entity and standing_entity.get('position_3d') is not None:
                    standing_pos = standing_entity['position_3d']
                    
                    if facing_entity and facing_entity.get('position_3d') is not None:
                        facing_pos = facing_entity['position_3d']
                        
                        # 在观察者坐标系中计算方向
                        predicted_direction = compute_direction_in_viewer_frame(
                            target_pos, standing_pos, facing_pos
                        )
                        
                        reasoning = (
                            f"Standing at {standing_by}={standing_pos[:2].round(2)}, "
                            f"facing {facing}={facing_pos[:2].round(2)}, "
                            f"target {target}={target_pos[:2].round(2)} -> {predicted_direction}"
                        )
                    else:
                        reasoning = f"Facing object '{facing}' not found in mind map"
                else:
                    reasoning = f"Standing object '{standing_by}' not found in mind map"
            else:
                reasoning = f"Target object '{target}' not found in mind map"
            
            # 匹配选项
            pred_answer = direction_to_option(predicted_direction, options) if predicted_direction else "A"
            
            # 评估
            gt_norm = gt.strip().upper()[0] if gt else "A"
            pred_norm = pred_answer.strip().upper()[0] if pred_answer else "A"
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
                'parsed': parsed,
                'objects_found': {
                    'standing': standing_entity is not None,
                    'facing': facing_entity is not None,
                    'target': target_entity is not None,
                },
                'mind_map_labels': list(mind_map_3d.keys()),
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
    
    builder.unload()
    gc.collect()
    torch.cuda.empty_cache()
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def load_vsibench_direction_samples() -> List[Dict]:
    from datasets import load_dataset
    
    logger.info("加载 VSI-Bench 数据集...")
    ds = load_dataset(
        'nyu-visionx/VSI-Bench',
        split='test',
        cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench'
    )
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    samples = load_vsibench_direction_samples()
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    logger.info(f"测试 {len(samples)} 个样本，GPU: {args.gpu}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"v76b_da3_full_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = str(output_dir / f"results_gpu{args.gpu}.json")
    results = worker_process_v76b(args.gpu, samples, output_file)
    
    # 统计
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    objects_found_stats = {'all_3': 0, 'at_least_1': 0, 'none': 0}
    
    for r in results:
        qtype = r['question_type']
        type_stats[qtype]['total'] += 1
        if r.get('correct', False):
            type_stats[qtype]['correct'] += 1
        
        objs = r.get('objects_found', {})
        if objs.get('standing') and objs.get('facing') and objs.get('target'):
            objects_found_stats['all_3'] += 1
        elif any(objs.values()):
            objects_found_stats['at_least_1'] += 1
        else:
            objects_found_stats['none'] += 1
    
    print("\n" + "=" * 80)
    print("V7.6b DA3 完整能力测试结果 (改进版方向计算)")
    print("=" * 80)
    print(f"{'任务类型':<35} {'准确率':>12} {'样本数':>10}")
    print("-" * 80)
    
    total_correct = 0
    total_samples = 0
    
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        n = stats['total']
        acc = stats['correct'] / n * 100 if n > 0 else 0
        print(f"{qtype:<35} {acc:>11.2f}% {n:>10}")
        total_correct += stats['correct']
        total_samples += n
    
    print("-" * 80)
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"{'Overall':<35} {overall_acc:>11.2f}% {total_samples:>10}")
    print("=" * 80)
    
    print(f"\n物体查找统计:")
    print(f"  三个物体都找到: {objects_found_stats['all_3']}/{total_samples}")
    print(f"  至少找到一个: {objects_found_stats['at_least_1']}/{total_samples}")
    print(f"  一个都没找到: {objects_found_stats['none']}/{total_samples}")
    
    # 保存汇总
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'overall_accuracy': overall_acc / 100,
            'objects_found_stats': objects_found_stats,
            'total_samples': total_samples,
        }, f, indent=2)
    
    print(f"\n结果保存到: {output_dir}")


if __name__ == '__main__':
    main()
