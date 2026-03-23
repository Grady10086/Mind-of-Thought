#!/usr/bin/env python3
"""
VSIBench 完整测试 - 使用 DA3 1.1

对比原始 V7 (DA3-Large) 和 DA3 1.1 在完整 VSIBench 上的表现
支持多 GPU 并行
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
import multiprocessing as mp
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

NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 'room_size_estimation', 'object_abs_distance']
CHOICE_TASKS = ['object_rel_distance', 'object_rel_direction_easy', 'object_rel_direction_medium', 
                'object_rel_direction_hard', 'obj_appearance_order', 'route_planning']


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


def normalize_number(text: str) -> Optional[float]:
    if text is None:
        return None
    text = re.sub(r'(meters?|m|cm|feet|ft|inches?|in|square)\b', '', str(text).lower())
    match = re.search(r'[-+]?\d*\.?\d+', text)
    if match:
        try:
            return float(match.group())
        except:
            return None
    return None


def mean_relative_accuracy(pred: float, target: float, start=0.05, end=0.5, interval=0.05) -> float:
    if pred is None or target is None:
        return 0.0
    epsilon = 1e-8
    rel_error = abs(pred - target) / (abs(target) + epsilon)
    thresholds = np.arange(start, end + interval / 2, interval)
    conditions = rel_error < (1 - thresholds)
    return conditions.astype(float).mean()


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


def compute_direction(target_pos, standing_pos, facing_pos):
    """使用 neg_y_axis 作为 up 向量"""
    viewer_forward = facing_pos - standing_pos
    if np.linalg.norm(viewer_forward) < 1e-6:
        return None, {}
    viewer_forward = viewer_forward / np.linalg.norm(viewer_forward)
    
    up = np.array([0, -1, 0])
    viewer_right = np.cross(viewer_forward, up)
    if np.linalg.norm(viewer_right) < 1e-6:
        return None, {}
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
        return None, {}
    elif len(directions) == 1:
        return directions[0], {'proj_forward': proj_forward, 'proj_right': proj_right}
    else:
        return "-".join(directions), {'proj_forward': proj_forward, 'proj_right': proj_right}


def direction_to_option(predicted_dir: str, options: List[str]) -> str:
    if not options or not predicted_dir:
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


def worker_process(gpu_id: int, samples: List[Dict], model_path: str, num_frames: int, result_queue):
    """单 GPU 工作进程"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from core.perception_da3_full import DA3FullEstimator
    from core.semantic_labeler import GroundingDINOLabeler
    
    # 加载模型
    logger.info(f"GPU {gpu_id}: 加载模型")
    da3 = DA3FullEstimator(
        model_name=model_path,
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
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}", position=gpu_id):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 读取视频
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
                    'question_type': question_type,
                    'ground_truth': gt,
                    'rule_prediction': None,
                    'rule_score': 0,
                    'error': 'Insufficient frames',
                })
                continue
            
            # DA3 推理
            prediction = da3.estimate_multiview(sampled_frames, ref_view_strategy="first")
            
            # 检测物体
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
            
            # 构建心智地图
            mind_map = {}
            for label, positions in object_positions.items():
                if positions:
                    pos_array = np.array(positions)
                    median_pos = np.median(pos_array, axis=0)
                    mind_map[label] = {
                        'position_3d': median_pos,
                        'count': len(positions),
                    }
            
            # 根据任务类型计算分数
            score = 0
            rule_prediction = None
            
            if question_type in ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard']:
                # 方向任务
                parsed = parse_direction_question(question)
                
                def find_entity(obj_name):
                    if not obj_name:
                        return None
                    for label, entity in mind_map.items():
                        if match_object_name(obj_name, label):
                            return entity
                    return None
                
                standing = find_entity(parsed['standing_by'])
                facing = find_entity(parsed['facing'])
                target = find_entity(parsed['target'])
                
                if standing and facing and target:
                    direction, _ = compute_direction(
                        target['position_3d'],
                        standing['position_3d'],
                        facing['position_3d']
                    )
                    if direction:
                        rule_prediction = direction_to_option(direction, options)
                
                if rule_prediction is None:
                    rule_prediction = 'A'
                
                gt_norm = gt.strip().upper()[0] if gt else 'A'
                pred_norm = rule_prediction.strip().upper()[0] if rule_prediction else 'A'
                score = 1.0 if gt_norm == pred_norm else 0.0
                
            elif question_type == 'object_counting':
                # 计数任务
                match = re.search(r'how many (\w+)', question.lower())
                if match:
                    obj_name = match.group(1)
                    count = 0
                    for label, entity in mind_map.items():
                        if match_object_name(obj_name, label):
                            count = max(count, entity.get('count', 1))
                    rule_prediction = str(count)
                    
                    pred_num = normalize_number(rule_prediction)
                    gt_num = normalize_number(gt)
                    score = mean_relative_accuracy(pred_num, gt_num) if pred_num and gt_num else 0
                else:
                    rule_prediction = '0'
                    score = 0
                    
            elif question_type in ['object_size_estimation', 'room_size_estimation', 'object_abs_distance']:
                # 数值估计任务 - 使用简单估计
                rule_prediction = '1.5'  # 默认值
                pred_num = normalize_number(rule_prediction)
                gt_num = normalize_number(gt)
                score = mean_relative_accuracy(pred_num, gt_num) if pred_num and gt_num else 0
                
            elif question_type in ['object_rel_distance', 'obj_appearance_order', 'route_planning']:
                # 其他选择题 - 随机选A
                rule_prediction = 'A'
                gt_norm = gt.strip().upper()[0] if gt else 'A'
                score = 1.0 if gt_norm == 'A' else 0.0
            
            results.append({
                'scene_name': sample['scene_name'],
                'question_type': question_type,
                'ground_truth': gt,
                'rule_prediction': rule_prediction,
                'rule_score': score,
            })
            
        except Exception as e:
            logger.error(f"Error: {sample['scene_name']}: {e}")
            results.append({
                'scene_name': sample['scene_name'],
                'question_type': question_type,
                'ground_truth': gt,
                'rule_prediction': None,
                'rule_score': 0,
                'error': str(e),
            })
    
    # 清理
    del da3
    del labeler
    gc.collect()
    torch.cuda.empty_cache()
    
    result_queue.put(results)


def load_vsibench_samples(question_types=None) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset('nyu-visionx/VSI-Bench', split='test',
                      cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench')
    
    samples = []
    for item in ds:
        if question_types and item['question_type'] not in question_types:
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
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--model-path', type=str, 
                        default='/home/tione/notebook/tianjungu/hf_cache/DA3NESTED-GIANT-LARGE-1.1')
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--question-types', type=str, default=None,
                        help='Comma-separated question types, e.g. "object_rel_direction_easy,object_rel_direction_hard"')
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    
    # 解析问题类型
    question_types = None
    if args.question_types:
        question_types = [t.strip() for t in args.question_types.split(',')]
    
    # 加载数据
    samples = load_vsibench_samples(question_types)
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    logger.info(f"加载 {len(samples)} 个样本")
    logger.info(f"模型: {args.model_path}")
    logger.info(f"帧数: {args.num_frames}")
    logger.info(f"GPU 数: {args.num_gpus}")
    
    # 分配样本到 GPU
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    samples_per_gpu = len(samples) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])
    
    # 启动多进程
    result_queue = mp.Queue()
    processes = []
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_samples[gpu_id], args.model_path, args.num_frames, result_queue)
        )
        p.start()
        processes.append(p)
    
    # 收集结果
    all_results = []
    for _ in range(num_gpus):
        results = result_queue.get()
        all_results.extend(results)
    
    for p in processes:
        p.join()
    
    # 统计结果
    type_stats = defaultdict(lambda: {'total': 0, 'score_sum': 0})
    
    for r in all_results:
        qtype = r['question_type']
        type_stats[qtype]['total'] += 1
        type_stats[qtype]['score_sum'] += r.get('rule_score', 0)
    
    # 打印结果
    print("\n" + "=" * 80)
    print(f"VSIBench 测试结果 - DA3 1.1 ({args.num_frames}帧)")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print("-" * 80)
    print(f"{'任务类型':<40} {'准确率':>15} {'样本数':>10}")
    print("-" * 80)
    
    overall_score = 0
    overall_total = 0
    
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        n = stats['total']
        avg = stats['score_sum'] / n if n > 0 else 0
        print(f"{qtype:<40} {avg*100:>14.2f}% {n:>10}")
        overall_score += stats['score_sum']
        overall_total += n
    
    print("-" * 80)
    print(f"{'Overall':<40} {overall_score/overall_total*100:>14.2f}% {overall_total:>10}")
    print("=" * 80)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs') / f"vsibench_da3_11_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': timestamp,
        'config': {
            'model_path': args.model_path,
            'num_frames': args.num_frames,
            'num_gpus': num_gpus,
        },
        'results_by_type': {
            qtype: {
                'accuracy': stats['score_sum'] / stats['total'] if stats['total'] > 0 else 0,
                'samples': stats['total'],
            }
            for qtype, stats in type_stats.items()
        },
        'overall': {
            'accuracy': overall_score / overall_total if overall_total > 0 else 0,
        },
        'total_samples': overall_total,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
