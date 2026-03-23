#!/usr/bin/env python3
"""
基线对比测试 - 按顺序测试不同配置

测试项目：
1. 原始 V7 配置 (DA3-Large, 32帧)
2. DA3 1.1 新模型 
3. DA3 1.1 + 32帧

专注于 direction 任务的性能对比
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================

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


def compute_direction_v7_original(
    target_world: np.ndarray,
    standing_world: np.ndarray,
    facing_world: np.ndarray,
) -> Tuple[str, Dict]:
    """原始 V7 方向计算 - 基于 XZ 平面投影"""
    # V7 原始实现：使用 XZ 平面 (Y 是高度)
    standing_xz = np.array([standing_world[0], standing_world[2]])
    facing_xz = np.array([facing_world[0], facing_world[2]])
    target_xz = np.array([target_world[0], target_world[2]])
    
    viewer_forward = facing_xz - standing_xz
    forward_norm = np.linalg.norm(viewer_forward)
    
    if forward_norm < 1e-6:
        return "same-position", {'error': 'forward too small'}
    
    viewer_forward = viewer_forward / forward_norm
    
    # 2D 右方向 (在 XZ 平面: 顺时针旋转90度)
    viewer_right = np.array([viewer_forward[1], -viewer_forward[0]])
    
    target_rel = target_xz - standing_xz
    
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
        'method': 'v7_original_xz',
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


# ============================================================================
# 测试配置 1: 原始 V7 (DA3-Large, 单帧深度)
# ============================================================================

def test_v7_original(samples: List[Dict], num_frames: int = 32, model_name: str = "depth-anything/DA3-Large"):
    """测试原始 V7 配置"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试配置: V7 Original")
    logger.info(f"  - 深度模型: {model_name}")
    logger.info(f"  - 帧数: {num_frames}")
    logger.info(f"{'='*60}")
    
    from core.perception import DepthEstimator
    from core.semantic_labeler import GroundingDINOLabeler
    
    # 加载模型
    depth_estimator = DepthEstimator(
        model_name=model_name,
        device='cuda',
        half_precision=True,
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
    focal_length = 500
    
    for sample in tqdm(samples, desc="V7 Original"):
        video_path = sample['video_path']
        question = sample['question']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 读取视频帧
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            all_detections = defaultdict(list)
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                
                # 深度估计
                depth_result = depth_estimator.infer_single(frame_rgb)
                depth_map = depth_result[1].cpu().numpy() if isinstance(depth_result, tuple) else depth_result.cpu().numpy()
                if depth_map.shape[0] != h or depth_map.shape[1] != w:
                    depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # 物体检测
                detections = labeler.detect(frame_rgb, prompt)
                
                for det in detections:
                    label = det.label.strip().lower()
                    if label.startswith('##'):
                        continue
                    
                    box = det.bbox_pixels
                    conf = det.confidence
                    
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    cx = min(max(cx, 0), w - 1)
                    cy = min(max(cy, 0), h - 1)
                    depth = float(depth_map[cy, cx])
                    
                    pos_3d = np.array([
                        (cx - w / 2) * depth / focal_length,
                        (cy - h / 2) * depth / focal_length,
                        depth
                    ])
                    
                    all_detections[label].append({
                        'position_3d': pos_3d,
                        'confidence': conf,
                    })
            
            cap.release()
            
            # 聚合位置
            mind_map = {}
            for label, dets in all_detections.items():
                if dets:
                    positions = np.array([d['position_3d'] for d in dets])
                    median_pos = np.median(positions, axis=0)
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
            
            if (target_entity and standing_entity and facing_entity):
                target_pos = target_entity['position_3d']
                standing_pos = standing_entity['position_3d']
                facing_pos = facing_entity['position_3d']
                
                predicted_direction, debug_info = compute_direction_v7_original(
                    target_pos, standing_pos, facing_pos
                )
            
            pred_answer = direction_to_option(predicted_direction, options) if predicted_direction else "A"
            gt_norm = gt.strip().upper()[0] if gt else "A"
            pred_norm = pred_answer.strip().upper()[0] if pred_answer else "A"
            
            results.append({
                'scene_name': sample['scene_name'],
                'question_type': sample['question_type'],
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
            results.append({
                'scene_name': sample['scene_name'],
                'correct': False,
                'error': str(e),
            })
    
    # 清理
    del depth_estimator
    del labeler
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# ============================================================================
# 测试配置 2: DA3 Full (多帧 3D 重建)
# ============================================================================

def test_da3_full(samples: List[Dict], num_frames: int = 16, model_name: str = "da3nested-giant-large"):
    """测试 DA3 Full 配置"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试配置: DA3 Full")
    logger.info(f"  - 模型: {model_name}")
    logger.info(f"  - 帧数: {num_frames}")
    logger.info(f"{'='*60}")
    
    from core.perception_da3_full import DA3FullEstimator
    from core.semantic_labeler import GroundingDINOLabeler
    
    # 加载模型
    da3 = DA3FullEstimator(
        model_name=model_name,
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
    
    for sample in tqdm(samples, desc=f"DA3 Full ({model_name})"):
        video_path = sample['video_path']
        question = sample['question']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 读取视频帧
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                    'correct': False,
                    'error': 'Insufficient frames',
                })
                continue
            
            # DA3 推理 - 使用固定参考帧
            prediction = da3.estimate_multiview(sampled_frames, ref_view_strategy="first")
            
            # 在每帧检测物体并聚合
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
            
            # 聚合为单个位置
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
            
            if (target_entity and standing_entity and facing_entity):
                target_pos = target_entity['position_3d']
                standing_pos = standing_entity['position_3d']
                facing_pos = facing_entity['position_3d']
                
                # DA3 Full 使用 neg_y_axis 作为 up
                up = np.array([0, -1, 0])
                viewer_forward = facing_pos - standing_pos
                if np.linalg.norm(viewer_forward) > 1e-6:
                    viewer_forward = viewer_forward / np.linalg.norm(viewer_forward)
                    viewer_right = np.cross(viewer_forward, up)
                    if np.linalg.norm(viewer_right) > 1e-6:
                        viewer_right = viewer_right / np.linalg.norm(viewer_right)
                        
                        target_rel = target_pos - standing_pos
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
                        
                        if len(directions) == 0:
                            predicted_direction = "same-position"
                        elif len(directions) == 1:
                            predicted_direction = directions[0]
                        else:
                            predicted_direction = "-".join(directions)
                        
                        debug_info = {
                            'proj_forward': float(proj_forward),
                            'proj_right': float(proj_right),
                            'method': 'da3_full_neg_y',
                        }
            
            pred_answer = direction_to_option(predicted_direction, options) if predicted_direction else "A"
            gt_norm = gt.strip().upper()[0] if gt else "A"
            pred_norm = pred_answer.strip().upper()[0] if pred_answer else "A"
            
            results.append({
                'scene_name': sample['scene_name'],
                'question_type': sample['question_type'],
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
                'correct': False,
                'error': str(e),
            })
    
    # 清理
    del da3
    del labeler
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def compute_stats(results: List[Dict], name: str) -> Dict:
    """计算统计数据"""
    total = len(results)
    
    # 找到所有物体的样本
    found_all = [r for r in results if r.get('objects_found', {}).get('standing') and 
                 r.get('objects_found', {}).get('facing') and r.get('objects_found', {}).get('target')]
    
    # 按类型统计
    by_type = defaultdict(lambda: {'total': 0, 'found': 0, 'correct': 0})
    for r in results:
        qtype = r.get('question_type', 'unknown')
        by_type[qtype]['total'] += 1
        if r.get('objects_found', {}).get('standing') and r.get('objects_found', {}).get('facing') and r.get('objects_found', {}).get('target'):
            by_type[qtype]['found'] += 1
            if r.get('correct'):
                by_type[qtype]['correct'] += 1
    
    correct_all = sum(1 for r in results if r.get('correct'))
    correct_found = sum(1 for r in found_all if r.get('correct'))
    
    stats = {
        'name': name,
        'total_samples': total,
        'found_all_objects': len(found_all),
        'found_rate': len(found_all) / total * 100 if total > 0 else 0,
        'overall_accuracy': correct_all / total * 100 if total > 0 else 0,
        'accuracy_when_found': correct_found / len(found_all) * 100 if found_all else 0,
        'by_type': dict(by_type),
    }
    
    return stats


def print_stats(stats: Dict):
    """打印统计结果"""
    print(f"\n{'='*60}")
    print(f"测试结果: {stats['name']}")
    print('='*60)
    print(f"总样本: {stats['total_samples']}")
    print(f"找到所有物体: {stats['found_all_objects']} ({stats['found_rate']:.1f}%)")
    print(f"Overall 准确率: {stats['overall_accuracy']:.1f}%")
    print(f"找到物体时准确率: {stats['accuracy_when_found']:.1f}%")
    print()
    print("按类型:")
    for qtype, data in stats['by_type'].items():
        if data['found'] > 0:
            acc = data['correct'] / data['found'] * 100
            print(f"  {qtype}: {data['correct']}/{data['found']} = {acc:.1f}% (共{data['total']}样本)")
    print('='*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--test', type=str, default='all', 
                        choices=['all', 'v7', 'da3', 'da3_11', 'da3_32'])
    args = parser.parse_args()
    
    # 加载数据
    samples = load_vsibench_direction_samples()
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    logger.info(f"加载 {len(samples)} 个 direction 样本")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs') / f"baseline_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    
    # 测试 1: 原始 V7 (DA3-Large, 32帧)
    if args.test in ['all', 'v7']:
        results_v7 = test_v7_original(samples, num_frames=32, model_name="depth-anything/DA3-Large")
        stats_v7 = compute_stats(results_v7, "V7 Original (DA3-Large, 32帧)")
        print_stats(stats_v7)
        all_stats.append(stats_v7)
        
        with open(output_dir / "results_v7_original.json", 'w') as f:
            json.dump(results_v7, f, indent=2, default=str)
    
    # 测试 2: DA3 Full 1.0
    if args.test in ['all', 'da3']:
        results_da3 = test_da3_full(samples, num_frames=16, model_name="da3nested-giant-large")
        stats_da3 = compute_stats(results_da3, "DA3 Full 1.0 (16帧)")
        print_stats(stats_da3)
        all_stats.append(stats_da3)
        
        with open(output_dir / "results_da3_full.json", 'w') as f:
            json.dump(results_da3, f, indent=2, default=str)
    
    # 测试 3: DA3 Full 1.1
    if args.test in ['all', 'da3_11']:
        # 先检查模型路径
        da3_11_path = "/home/tione/notebook/tianjungu/hf_cache/DA3NESTED-GIANT-LARGE-1.1"
        if os.path.exists(da3_11_path):
            results_da3_11 = test_da3_full(samples, num_frames=16, model_name=da3_11_path)
            stats_da3_11 = compute_stats(results_da3_11, "DA3 Full 1.1 (16帧)")
            print_stats(stats_da3_11)
            all_stats.append(stats_da3_11)
            
            with open(output_dir / "results_da3_11.json", 'w') as f:
                json.dump(results_da3_11, f, indent=2, default=str)
        else:
            logger.warning(f"DA3 1.1 模型路径不存在: {da3_11_path}")
    
    # 测试 4: DA3 Full 1.1 + 32帧
    if args.test in ['all', 'da3_32']:
        da3_11_path = "/home/tione/notebook/tianjungu/hf_cache/DA3NESTED-GIANT-LARGE-1.1"
        if os.path.exists(da3_11_path):
            results_da3_32 = test_da3_full(samples, num_frames=32, model_name=da3_11_path)
            stats_da3_32 = compute_stats(results_da3_32, "DA3 Full 1.1 (32帧)")
            print_stats(stats_da3_32)
            all_stats.append(stats_da3_32)
            
            with open(output_dir / "results_da3_11_32frames.json", 'w') as f:
                json.dump(results_da3_32, f, indent=2, default=str)
        else:
            logger.warning(f"DA3 1.1 模型路径不存在: {da3_11_path}")
    
    # 保存汇总
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("基线对比总结")
    print('='*80)
    print(f"{'配置':<35} {'找到率':>10} {'Overall':>10} {'找到时准确率':>15}")
    print('-'*80)
    for s in all_stats:
        print(f"{s['name']:<35} {s['found_rate']:>9.1f}% {s['overall_accuracy']:>9.1f}% {s['accuracy_when_found']:>14.1f}%")
    print('='*80)


if __name__ == '__main__':
    main()
