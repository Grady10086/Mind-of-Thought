#!/usr/bin/env python3
"""
V7.11 - 使用纯 2D 投影方法（XZ 平面）

核心思想:
DA3 的世界坐标系虽然是任意的，但物体在场景中应该主要分布在某个"地面"平面上。
假设地面大致平行于 XZ 平面（Y 是高度方向），我们可以:
1. 将所有 3D 坐标投影到 XZ 平面
2. 使用 2D 向量运算判断方向

2D 叉积判断左右:
如果 viewer_forward_2d × target_rel_2d > 0，则 target 在右边（假设 Y 向上）
如果 viewer_forward_2d × target_rel_2d < 0，则 target 在左边

2D 点积判断前后:
如果 viewer_forward_2d · target_rel_2d > 0，则 target 在前面
如果 viewer_forward_2d · target_rel_2d < 0，则 target 在后面
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


def compute_direction_2d_xz(
    target_world: np.ndarray,
    standing_world: np.ndarray,
    facing_world: np.ndarray,
    plane: str = 'xz',
) -> Tuple[str, Dict]:
    """
    使用 2D 投影判断方向
    
    plane 参数:
    - 'xz': 投影到 XZ 平面 (Y 为高度)
    - 'xy': 投影到 XY 平面 (Z 为高度)
    - 'yz': 投影到 YZ 平面 (X 为高度)
    
    对于 XZ 平面:
    - 前后: 沿 viewer_forward 方向的点积
    - 左右: 2D 叉积 (假设 Y 向上时, 叉积 > 0 表示右)
    """
    # 选择投影平面
    if plane == 'xz':
        # X 和 Z 坐标
        idx1, idx2 = 0, 2
    elif plane == 'xy':
        idx1, idx2 = 0, 1
    elif plane == 'yz':
        idx1, idx2 = 1, 2
    else:
        idx1, idx2 = 0, 2
    
    # 投影到 2D
    target_2d = np.array([target_world[idx1], target_world[idx2]])
    standing_2d = np.array([standing_world[idx1], standing_world[idx2]])
    facing_2d = np.array([facing_world[idx1], facing_world[idx2]])
    
    # 观察者前方 (2D)
    viewer_forward_2d = facing_2d - standing_2d
    forward_norm = np.linalg.norm(viewer_forward_2d)
    
    if forward_norm < 1e-6:
        return "same-position", {'error': 'standing == facing (2D)'}
    
    viewer_forward_2d = viewer_forward_2d / forward_norm
    
    # 目标相对位置 (2D)
    target_rel_2d = target_2d - standing_2d
    target_rel_norm = np.linalg.norm(target_rel_2d)
    
    if target_rel_norm < 1e-6:
        return "same-position", {'error': 'standing == target (2D)'}
    
    # 前后判断: 点积
    proj_forward = np.dot(target_rel_2d, viewer_forward_2d)
    
    # 左右判断: 2D 叉积
    # cross_2d = viewer_forward_2d[0] * target_rel_2d[1] - viewer_forward_2d[1] * target_rel_2d[0]
    # 这是 forward × target_rel，正值表示 target_rel 在 forward 的右侧
    cross_2d = viewer_forward_2d[0] * target_rel_2d[1] - viewer_forward_2d[1] * target_rel_2d[0]
    
    debug_info = {
        'plane': plane,
        'target_2d': target_2d.tolist(),
        'standing_2d': standing_2d.tolist(),
        'facing_2d': facing_2d.tolist(),
        'viewer_forward_2d': viewer_forward_2d.tolist(),
        'target_rel_2d': target_rel_2d.tolist(),
        'proj_forward': float(proj_forward),
        'cross_2d': float(cross_2d),
    }
    
    # 判断方向
    directions = []
    threshold = 0.01
    
    if proj_forward > threshold:
        directions.append("front")
    elif proj_forward < -threshold:
        directions.append("back")
    
    # 叉积判断左右 (正值 = 右, 负值 = 左)
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


def worker_process(gpu_id: int, samples: List[Dict], output_file: str, plane: str = 'xz'):
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
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id} [{plane}]"):
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
                
                predicted_direction, debug_info = compute_direction_2d_xz(
                    target_pos, standing_pos, facing_pos, plane,
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


def print_stats(results: List[Dict], plane: str):
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
    print(f"V7.11 2D 投影方法 [{plane}]")
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
    
    return overall_acc, overall_found_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--plane', type=str, default='all', 
                        choices=['xz', 'xy', 'yz', 'all'])
    args = parser.parse_args()
    
    samples = load_vsibench_direction_samples()
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"v711_2d_projection_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    planes = ['xz', 'xy', 'yz'] if args.plane == 'all' else [args.plane]
    
    best_plane = None
    best_acc = 0
    
    for plane in planes:
        output_file = str(output_dir / f"results_{plane}.json")
        results = worker_process(args.gpu, samples, output_file, plane)
        acc, found_acc = print_stats(results, plane)
        
        if found_acc > best_acc:
            best_acc = found_acc
            best_plane = plane
    
    print(f"\n最佳平面: {best_plane} (找到物体时准确率: {best_acc:.1f}%)")
    print(f"\n结果保存到: {output_dir}")


if __name__ == '__main__':
    main()
