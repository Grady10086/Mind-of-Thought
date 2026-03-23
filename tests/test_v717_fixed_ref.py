#!/usr/bin/env python3
"""
V7.17 - 使用固定参考帧 (first) 来消除随机性

关键改动：
1. 使用 ref_view_strategy="first" 而不是 "saddle_balanced"
2. 这应该让 DA3 的输出更稳定
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
import cv2

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
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


def compute_direction(
    target_world: np.ndarray,
    standing_world: np.ndarray,
    facing_world: np.ndarray,
) -> Tuple[str, Dict]:
    """使用 neg_y_axis 作为 up 向量"""
    viewer_forward = facing_world - standing_world
    viewer_forward_norm = np.linalg.norm(viewer_forward)
    
    if viewer_forward_norm < 1e-6:
        return "same-position", {'error': 'standing == facing'}
    
    viewer_forward = viewer_forward / viewer_forward_norm
    
    target_rel = target_world - standing_world
    target_rel_norm = np.linalg.norm(target_rel)
    
    if target_rel_norm < 1e-6:
        return "same-position", {'error': 'standing == target'}
    
    # 重力方向: -Y (基于相机水平假设)
    up = np.array([0, -1, 0])
    
    # viewer_right = viewer_forward × up
    viewer_right = np.cross(viewer_forward, up)
    viewer_right_norm = np.linalg.norm(viewer_right)
    
    if viewer_right_norm < 1e-6:
        return "unknown", {'error': 'parallel vectors'}
    
    viewer_right = viewer_right / viewer_right_norm
    
    proj_forward = np.dot(target_rel, viewer_forward)
    proj_right = np.dot(target_rel, viewer_right)
    
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
    
    debug_info = {
        'proj_forward': float(proj_forward),
        'proj_right': float(proj_right),
    }
    
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


def worker_process(gpu_id: int, samples: List[Dict], output_file: str, ref_view_strategy: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from core.perception_da3_full import DA3FullEstimator
    from core.semantic_labeler import GroundingDINOLabeler
    
    # 加载模型
    da3 = DA3FullEstimator(
        model_name="da3nested-giant-large",
        device='cuda',
        use_ray_pose=True,
    )
    
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
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id} [{ref_view_strategy}]"):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 读取视频
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_frames = 16
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            sampled_frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            if len(sampled_frames) < 2:
                results.append({
                    'scene_name': sample['scene_name'],
                    'question_type': question_type,
                    'ground_truth': gt,
                    'prediction': 'A',
                    'correct': gt.strip().upper() == 'A',
                    'error': 'Insufficient frames',
                })
                continue
            
            # DA3 推理 - 使用指定的参考帧策略
            prediction = da3.estimate_multiview(sampled_frames, ref_view_strategy=ref_view_strategy)
            
            # 在每帧检测物体并聚合
            from collections import defaultdict
            object_positions = defaultdict(list)
            
            proc_H, proc_W = prediction.depth_maps.shape[1:3]
            
            for i, frame_rgb in enumerate(sampled_frames):
                orig_H, orig_W = frame_rgb.shape[:2]
                detections = labeler.detect(frame_rgb, prompt)
                
                for det in detections:
                    label = det.label.strip().lower()
                    if label.startswith('##'):
                        continue
                    
                    bbox = det.bbox_pixels
                    scale_x = proc_W / orig_W
                    scale_y = proc_H / orig_H
                    
                    bbox_scaled = (
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y,
                    )
                    
                    pos_3d = da3.compute_object_center_3d(prediction, i, bbox_scaled)
                    if pos_3d is not None:
                        object_positions[label].append(pos_3d)
            
            # 聚合为单个位置 (取中位数)
            mind_map = {}
            for label, positions in object_positions.items():
                if positions:
                    pos_array = np.array(positions)
                    median_pos = np.median(pos_array, axis=0)
                    mind_map[label] = {'position_3d': median_pos}
            
            # 解析问题
            parsed = parse_direction_question(question)
            standing_by, facing, target = parsed['standing_by'], parsed['facing'], parsed['target']
            
            def find_entity(obj_name):
                if not obj_name:
                    return None
                for label, entity in mind_map.items():
                    if match_object_name(obj_name, label):
                        return entity
                return None
            
            standing_entity = find_entity(standing_by)
            facing_entity = find_entity(facing)
            target_entity = find_entity(target)
            
            predicted_direction = None
            debug_info = {}
            
            if (target_entity and target_entity.get('position_3d') is not None and
                standing_entity and standing_entity.get('position_3d') is not None and
                facing_entity and facing_entity.get('position_3d') is not None):
                
                target_pos = target_entity['position_3d']
                standing_pos = standing_entity['position_3d']
                facing_pos = facing_entity['position_3d']
                
                if np.allclose(standing_pos, facing_pos, atol=1e-3):
                    predicted_direction = "same-position"
                else:
                    predicted_direction, debug_info = compute_direction(
                        target_pos, standing_pos, facing_pos
                    )
            
            pred_answer = direction_to_option(predicted_direction, options) if predicted_direction else "A"
            gt_norm = gt.strip().upper()[0] if gt else "A"
            pred_norm = pred_answer.strip().upper()[0] if pred_answer else "A"
            
            results.append({
                'scene_name': sample['scene_name'],
                'question_type': question_type,
                'ground_truth': gt,
                'prediction': pred_answer,
                'predicted_direction': predicted_direction,
                'correct': pred_norm == gt_norm,
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
    ds = load_dataset('nyu-visionx/VSI-Bench', split='test',
                      cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench')
    
    direction_types = ['object_rel_direction_easy', 'object_rel_direction_medium', 
                       'object_rel_direction_hard']
    
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
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--ref-strategy', type=str, default='first', 
                        choices=['first', 'middle', 'saddle_balanced'])
    args = parser.parse_args()
    
    samples = load_vsibench_direction_samples()
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs') / f"v717_fixed_ref_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = str(output_dir / f"results_{args.ref_strategy}.json")
    results = worker_process(args.gpu, samples, output_file, args.ref_strategy)
    
    # 统计
    found_all = [r for r in results if r.get('objects_found',{}).get('standing') and 
                 r.get('objects_found',{}).get('facing') and r.get('objects_found',{}).get('target')]
    correct = sum(1 for r in found_all if r.get('correct'))
    
    print(f"\n{'='*60}")
    print(f"V7.17 固定参考帧 [{args.ref_strategy}]")
    print('='*60)
    print(f"总样本: {len(results)}")
    print(f"找到所有物体: {len(found_all)} ({len(found_all)/len(results)*100:.1f}%)")
    print(f"找到物体时准确率: {correct}/{len(found_all)} = {correct/len(found_all)*100:.1f}%")
    print('='*60)


if __name__ == '__main__':
    main()
