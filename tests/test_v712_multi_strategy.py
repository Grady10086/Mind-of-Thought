#!/usr/bin/env python3
"""
V7.12 - 多策略融合 + 置信度投票

核心思想:
DA3 的世界坐标系方向是任意的，但对于同一个场景，坐标系是一致的。
不同的 "up" 假设可能对不同场景有效。

策略:
1. 尝试多种 up 向量假设
2. 同时尝试两种叉积顺序 (正向和反向)
3. 基于一致性选择最佳结果
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
from collections import defaultdict, Counter
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


def extract_camera_basis_from_extrinsics(extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = extrinsics[:3, :3]
    right = R[0, :]
    down = R[1, :]
    forward = R[2, :]
    up = -down
    
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    forward = forward / np.linalg.norm(forward)
    
    return right, up, forward


def compute_direction_single(
    target_world: np.ndarray,
    standing_world: np.ndarray,
    facing_world: np.ndarray,
    up_vector: np.ndarray,
    flip_lr: bool = False,
) -> Tuple[str, Dict]:
    """
    使用单一策略计算方向
    """
    viewer_forward = facing_world - standing_world
    viewer_forward_norm = np.linalg.norm(viewer_forward)
    
    if viewer_forward_norm < 1e-6:
        return "same-position", {'error': 'standing == facing'}
    
    viewer_forward = viewer_forward / viewer_forward_norm
    
    target_rel = target_world - standing_world
    target_rel_norm = np.linalg.norm(target_rel)
    
    if target_rel_norm < 1e-6:
        return "same-position", {'error': 'standing == target'}
    
    # 叉积计算 viewer_right
    viewer_right = np.cross(viewer_forward, up_vector)
    viewer_right_norm = np.linalg.norm(viewer_right)
    
    if viewer_right_norm < 1e-6:
        return "unknown", {'error': 'parallel vectors'}
    
    viewer_right = viewer_right / viewer_right_norm
    
    # 可选：翻转左右
    if flip_lr:
        viewer_right = -viewer_right
    
    # 计算投影
    proj_forward = np.dot(target_rel, viewer_forward)
    proj_right = np.dot(target_rel, viewer_right)
    
    # 判断方向
    directions = []
    threshold = 0.01
    
    if proj_forward > threshold:
        directions.append("front")
    elif proj_forward < -threshold:
        directions.append("back")
    
    if proj_right > threshold:
        directions.append("right")
    elif proj_right < -threshold:
        directions.append("left")
    
    if len(directions) == 0:
        return "same-position", {}
    elif len(directions) == 1:
        return directions[0], {}
    else:
        return "-".join(directions), {}


def compute_direction_multi_strategy(
    target_world: np.ndarray,
    standing_world: np.ndarray,
    facing_world: np.ndarray,
    camera_up: np.ndarray,
) -> Tuple[str, Dict]:
    """
    使用多种策略计算方向，返回投票结果
    """
    # 定义多种 up 向量候选
    up_candidates = [
        ('camera_up', camera_up),
        ('neg_camera_up', -camera_up),
        ('y_axis', np.array([0, 1, 0])),
        ('neg_y_axis', np.array([0, -1, 0])),
        ('z_axis', np.array([0, 0, 1])),
        ('neg_z_axis', np.array([0, 0, -1])),
    ]
    
    results = []
    for name, up_vec in up_candidates:
        direction, _ = compute_direction_single(
            target_world, standing_world, facing_world, up_vec, flip_lr=False
        )
        if direction not in ('same-position', 'unknown'):
            results.append((name, direction))
    
    if not results:
        return "same-position", {'error': 'no valid direction from any strategy'}
    
    # 投票
    direction_counts = Counter([r[1] for r in results])
    most_common = direction_counts.most_common(1)[0]
    
    debug_info = {
        'all_results': results,
        'vote_counts': dict(direction_counts),
        'confidence': most_common[1] / len(results),
    }
    
    return most_common[0], debug_info


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
    
    from core.perception_da3_full import MindMapBuilder3D
    
    builder = MindMapBuilder3D(
        device='cuda',
        num_frames=16,
        box_threshold=0.25,
        model_name="da3nested-giant-large",
        use_ray_pose=True,
        use_camera_coords=False,
    )
    
    results = []
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            mind_map_3d, da3_prediction, sampled_frames = builder.build_from_video(
                video_path,
                target_objects=None,
                extended_vocabulary=EXTENDED_VOCABULARY,
            )
            
            # 提取相机基向量
            if da3_prediction is not None and da3_prediction.extrinsics is not None:
                _, camera_up, _ = extract_camera_basis_from_extrinsics(
                    da3_prediction.extrinsics[0]
                )
            else:
                camera_up = np.array([0, 1, 0])
            
            parsed = parse_direction_question(question)
            standing_by = parsed['standing_by']
            facing = parsed['facing']
            target = parsed['target']
            
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
            
            predicted_direction = None
            reasoning = ""
            debug_info = {}
            
            if (target_entity and target_entity.get('position_3d') is not None and
                standing_entity and standing_entity.get('position_3d') is not None and
                facing_entity and facing_entity.get('position_3d') is not None):
                
                target_pos = target_entity['position_3d']
                standing_pos = standing_entity['position_3d']
                facing_pos = facing_entity['position_3d']
                
                if np.allclose(standing_pos, facing_pos, atol=1e-3):
                    predicted_direction = "same-position"
                    debug_info = {'error': 'standing == facing'}
                else:
                    predicted_direction, debug_info = compute_direction_multi_strategy(
                        target_pos, standing_pos, facing_pos, camera_up,
                    )
                
                reasoning = (
                    f"Standing={standing_by}@{standing_pos.round(2)}, "
                    f"Facing={facing}@{facing_pos.round(2)}, "
                    f"Target={target}@{target_pos.round(2)} -> {predicted_direction}"
                )
            else:
                missing = []
                if standing_entity is None or standing_entity.get('position_3d') is None:
                    missing.append(f"standing({standing_by})")
                if facing_entity is None or facing_entity.get('position_3d') is None:
                    missing.append(f"facing({facing})")
                if target_entity is None or target_entity.get('position_3d') is None:
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
                    'standing': standing_entity is not None,
                    'facing': facing_entity is not None,
                    'target': target_entity is not None,
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
    output_dir = Path(args.output_dir) / f"v712_multi_strategy_{timestamp}"
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
    print("V7.12 多策略融合投票结果")
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
