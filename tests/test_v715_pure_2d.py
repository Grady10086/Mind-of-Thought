#!/usr/bin/env python3
"""
V7.15 - 纯 2D 图像坐标方法（简化版）

核心思想:
完全放弃 3D 重建，只使用第一帧图像的 2D 检测结果。

方向定义 (从图像角度):
- 如果相机从上方看场景（俯视），则图像的 X 轴对应左右，Y 轴对应前后
- 站在 A 面向 B：
  - 前方 = A 到 B 的方向
  - 右方 = 前方顺时针旋转 90 度

这个方法的假设:
- 视频是从大致水平的角度拍摄的
- 图像的上下大致对应场景的远近

2D 叉积判断左右:
forward_2d = B - A (在图像坐标系)
target_rel_2d = Target - A
cross = forward_2d.x * target_rel_2d.y - forward_2d.y * target_rel_2d.x
如果 cross > 0，target 在 A 的右边（图像坐标系 Y 向下）
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
    '/home/tione/notebook/tianjungu/hf_cache/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/hf_cache/vsibench/scannet',
    '/home/tione/notebook/tianjungu/hf_cache/vsibench/scannetpp',
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
]

SYNONYM_MAP = {
    'sofa': ['couch', 'settee'],
    'tv': ['television', 'tv screen', 'monitor'],
    'refrigerator': ['fridge'],
    'trash bin': ['trash can', 'garbage can'],
    'couch': ['sofa'],
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


def parse_direction_question(question: str) -> Dict[str, Optional[str]]:
    result = {'standing_by': None, 'facing': None, 'target': None}
    
    q_lower = question.lower()
    pattern = r'standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+)'
    match = re.search(pattern, q_lower)
    if match:
        result['standing_by'] = match.group(1)
        result['facing'] = match.group(2)
        result['target'] = match.group(3)
    
    return result


def compute_direction_2d_simple(
    target_2d: np.ndarray,
    standing_2d: np.ndarray,
    facing_2d: np.ndarray,
) -> Tuple[str, Dict]:
    """
    纯 2D 方向计算
    
    图像坐标系:
    - X: 向右
    - Y: 向下
    
    站在 A 面向 B:
    - forward_2d = B - A
    - 顺时针旋转 90 度得到 right_2d: (fx, fy) -> (fy, -fx)
    - 注意：图像 Y 向下，所以顺时针旋转是 (fx, fy) -> (fy, -fx)
    """
    forward_2d = facing_2d - standing_2d
    forward_norm = np.linalg.norm(forward_2d)
    
    if forward_norm < 1:
        return "same-position", {'error': 'standing == facing (2D)'}
    
    forward_2d = forward_2d / forward_norm
    
    target_rel_2d = target_2d - standing_2d
    target_rel_norm = np.linalg.norm(target_rel_2d)
    
    if target_rel_norm < 1:
        return "same-position", {'error': 'standing == target (2D)'}
    
    # 前后: 点积
    proj_forward = np.dot(target_rel_2d, forward_2d)
    
    # 左右: 2D 叉积 (图像坐标系 Y 向下)
    # cross = fx * ty - fy * tx
    # 如果 cross > 0，target 在 forward 的右边
    cross_2d = forward_2d[0] * target_rel_2d[1] - forward_2d[1] * target_rel_2d[0]
    
    debug_info = {
        'standing_2d': standing_2d.tolist(),
        'facing_2d': facing_2d.tolist(),
        'target_2d': target_2d.tolist(),
        'forward_2d': forward_2d.tolist(),
        'target_rel_2d': target_rel_2d.tolist(),
        'proj_forward': float(proj_forward),
        'cross_2d': float(cross_2d),
    }
    
    # 判断方向
    directions = []
    threshold = 1  # 像素阈值
    
    if proj_forward > threshold:
        directions.append("front")
    elif proj_forward < -threshold:
        directions.append("back")
    
    # 叉积判断左右
    if cross_2d > threshold:
        directions.append("right")
    elif cross_2d < -threshold:
        directions.append("left")
    
    if len(directions) == 0:
        return "same-position", debug_info
    elif len(directions) == 1:
        return directions[0], debug_info
    else:
        return "-".join(directions), debug_info


def direction_to_option(predicted_dir: str, options: List[str]) -> str:
    if not options:
        return "A"
    
    dir_map = {
        'front-left': ['front-left', 'front left'],
        'front-right': ['front-right', 'front right'],
        'back-left': ['back-left', 'back left'],
        'back-right': ['back-right', 'back right'],
        'front': ['front', 'ahead'],
        'back': ['back', 'behind'],
        'left': ['left'],
        'right': ['right'],
    }
    
    pred_variants = dir_map.get(predicted_dir, [predicted_dir])
    
    for i, opt in enumerate(options):
        opt_lower = opt.lower()
        for variant in pred_variants:
            if variant in opt_lower:
                return chr(65 + i)
    
    return "A"


def worker_process(gpu_id: int, samples: List[Dict], output_file: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    import cv2
    from core.semantic_labeler import GroundingDINOLabeler
    
    logger.info(f"GPU {gpu_id}: 加载 GroundingDINO...")
    labeler = GroundingDINOLabeler(
        model_id="IDEA-Research/grounding-dino-base",
        device='cuda',
        box_threshold=0.25,
        text_threshold=0.25,
    )
    labeler.load_model()
    
    vocab = EXTENDED_VOCABULARY
    prompt = " . ".join(vocab) + " ."
    
    results = []
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 读取第一帧
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                results.append({
                    'scene_name': sample['scene_name'],
                    'question': question,
                    'question_type': question_type,
                    'ground_truth': gt,
                    'prediction': 'A',
                    'correct': gt.strip().upper() == 'A',
                    'error': 'Failed to read frame',
                })
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = labeler.detect(frame_rgb, prompt)
            
            # 构建物体字典 (只有 2D 位置)
            object_info = {}
            for det in detections:
                label = det.label.strip().lower()
                if label.startswith('##'):
                    continue
                
                bbox = det.bbox_pixels
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                
                if label not in object_info:
                    object_info[label] = np.array([cx, cy])
            
            # 解析问题
            parsed = parse_direction_question(question)
            standing_by = parsed['standing_by']
            facing = parsed['facing']
            target = parsed['target']
            
            def find_object(obj_name):
                if not obj_name:
                    return None
                for label, pos in object_info.items():
                    if match_object_name(obj_name, label):
                        return pos
                return None
            
            standing_pos = find_object(standing_by)
            facing_pos = find_object(facing)
            target_pos = find_object(target)
            
            predicted_direction = None
            reasoning = ""
            debug_info = {}
            
            if standing_pos is not None and facing_pos is not None and target_pos is not None:
                predicted_direction, debug_info = compute_direction_2d_simple(
                    target_pos, standing_pos, facing_pos,
                )
                
                reasoning = (
                    f"Standing={standing_by}@{standing_pos.round(0)}, "
                    f"Facing={facing}@{facing_pos.round(0)}, "
                    f"Target={target}@{target_pos.round(0)} -> {predicted_direction}"
                )
            else:
                missing = []
                if standing_pos is None:
                    missing.append(f"standing({standing_by})")
                if facing_pos is None:
                    missing.append(f"facing({facing})")
                if target_pos is None:
                    missing.append(f"target({target})")
                reasoning = f"Missing: {', '.join(missing)}"
            
            pred_answer = direction_to_option(predicted_direction, options) if predicted_direction else "A"
            
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
                'debug_info': debug_info,
                'objects_found': {
                    'standing': standing_pos is not None,
                    'facing': facing_pos is not None,
                    'target': target_pos is not None,
                },
            })
            
        except Exception as e:
            logger.error(f"Error: {sample['scene_name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'scene_name': sample['scene_name'],
                'question': question,
                'question_type': question_type,
                'ground_truth': gt,
                'prediction': 'A',
                'correct': gt.strip().upper() == 'A' if gt else False,
                'error': str(e),
            })
    
    del labeler
    gc.collect()
    torch.cuda.empty_cache()
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def load_vsibench_direction_samples() -> List[Dict]:
    from datasets import load_dataset
    
    logger.info("加载 VSI-Bench...")
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
        
        video_path = find_video_path(item['scene_name'])
        if not video_path:
            continue
        
        samples.append({
            'scene_name': item['scene_name'],
            'video_path': video_path,
            'question': item['question'],
            'question_type': item['question_type'],
            'options': item.get('options', []),
            'ground_truth': item['ground_truth'],
        })
    
    logger.info(f"加载 {len(samples)} 个样本")
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
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"v715_pure_2d_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = str(output_dir / f"results_gpu{args.gpu}.json")
    results = worker_process(args.gpu, samples, output_file)
    
    # 统计
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'all_found': 0, 'all_found_correct': 0})
    
    for r in results:
        qtype = r['question_type']
        type_stats[qtype]['total'] += 1
        if r.get('correct', False):
            type_stats[qtype]['correct'] += 1
        
        objs = r.get('objects_found', {})
        if objs.get('standing') and objs.get('facing') and objs.get('target'):
            type_stats[qtype]['all_found'] += 1
            if r.get('correct', False):
                type_stats[qtype]['all_found_correct'] += 1
    
    print(f"\n{'='*80}")
    print("V7.15 纯 2D 图像坐标方法结果")
    print("=" * 80)
    
    total_correct = 0
    total_samples = 0
    total_found = 0
    total_found_correct = 0
    
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        n = stats['total']
        acc = stats['correct'] / n * 100 if n > 0 else 0
        found = stats['all_found']
        found_rate = found / n * 100 if n > 0 else 0
        found_acc = stats['all_found_correct'] / found * 100 if found > 0 else 0
        
        print(f"{qtype:<35} {acc:>6.1f}% ({stats['correct']}/{n})")
        print(f"  找到全部物体: {found_rate:.1f}% ({found}/{n}), 其中正确: {found_acc:.1f}%")
        
        total_correct += stats['correct']
        total_samples += n
        total_found += found
        total_found_correct += stats['all_found_correct']
    
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    overall_found_rate = total_found / total_samples * 100 if total_samples > 0 else 0
    overall_found_acc = total_found_correct / total_found * 100 if total_found > 0 else 0
    
    print("-" * 80)
    print(f"{'Overall':<35} {overall_acc:>6.1f}% ({total_correct}/{total_samples})")
    print(f"  找到全部物体: {overall_found_rate:.1f}% ({total_found}/{total_samples}), 其中正确: {overall_found_acc:.1f}%")
    print("=" * 80)
    
    print(f"\n结果保存到: {output_dir}")


if __name__ == '__main__':
    main()
