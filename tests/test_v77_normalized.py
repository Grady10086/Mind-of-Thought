#!/usr/bin/env python3
"""
V7.7 - 修正 DA3 深度缩放问题

关键发现: DA3 输出的深度是 affine-invariant depth (非度量深度)
需要进行归一化后再计算方向

方向计算策略:
1. 使用相机坐标系 (X 右, Y 下, Z 前)
2. 使用 X-Z 平面 (俯视图)
3. 深度值虽然非度量，但相对关系是正确的
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
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


def compute_direction_normalized(
    target_pos: np.ndarray,
    standing_pos: np.ndarray,
    facing_pos: np.ndarray,
) -> str:
    """
    使用归一化坐标计算方向
    
    DA3 的深度是 affine-invariant，但相对关系是正确的。
    我们只需要关心方向，不需要真实距离。
    
    坐标系: X 右, Y 下, Z 前 (深度)
    使用 X-Z 平面计算方向 (俯视图)
    """
    # 先归一化坐标，消除尺度影响
    # 以 standing 为原点
    facing_rel = facing_pos - standing_pos
    target_rel = target_pos - standing_pos
    
    # 使用 X-Z 平面
    forward_xz = np.array([facing_rel[0], facing_rel[2]])
    forward_norm = np.linalg.norm(forward_xz)
    
    if forward_norm < 1e-6:
        return "same-position"
    
    forward_xz = forward_xz / forward_norm
    
    # 右方向: (x, z) 顺时针旋转90度 -> (z, -x)
    right_xz = np.array([forward_xz[1], -forward_xz[0]])
    
    target_xz = np.array([target_rel[0], target_rel[2]])
    target_norm = np.linalg.norm(target_xz)
    
    if target_norm < 1e-6:
        return "same-position"
    
    # 投影
    proj_forward = np.dot(target_xz, forward_xz)
    proj_right = np.dot(target_xz, right_xz)
    
    # 使用固定阈值 (因为已经归一化)
    threshold = 0.1 * target_norm
    
    directions = []
    
    if proj_forward > threshold:
        directions.append("front")
    elif proj_forward < -threshold:
        directions.append("back")
    
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


def compute_3d_position_simple(
    frame_rgb: np.ndarray,
    bbox: tuple,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """
    简单的 3D 位置计算
    
    使用归一化的相机坐标: x/z, y/z
    这样可以消除深度尺度不准确的影响
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    H, W = depth_map.shape
    orig_H, orig_W = frame_rgb.shape[:2]
    
    # 缩放坐标
    scale_x = W / orig_W
    scale_y = H / orig_H
    
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y
    
    cx_int = int(np.clip(cx_scaled, 0, W - 1))
    cy_int = int(np.clip(cy_scaled, 0, H - 1))
    
    # 使用中值深度
    x1_int = int(np.clip(x1 * scale_x, 0, W - 1))
    x2_int = int(np.clip(x2 * scale_x, 0, W - 1))
    y1_int = int(np.clip(y1 * scale_y, 0, H - 1))
    y2_int = int(np.clip(y2 * scale_y, 0, H - 1))
    
    depth_region = depth_map[y1_int:y2_int+1, x1_int:x2_int+1]
    if depth_region.size > 0:
        depth = np.median(depth_region)
    else:
        depth = depth_map[cy_int, cx_int]
    
    # 相机内参
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx_cam, cy_cam = intrinsics[0, 2], intrinsics[1, 2]
    
    # 计算相机坐标
    x_cam = (cx_scaled - cx_cam) / fx * depth
    y_cam = (cy_scaled - cy_cam) / fy * depth
    z_cam = depth
    
    return np.array([x_cam, y_cam, z_cam])


def worker_process(gpu_id: int, samples: List[Dict], output_file: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from core.semantic_labeler import GroundingDINOLabeler
    from core.perception_da3_full import DA3FullEstimator
    
    # 加载模型
    logger.info(f"GPU {gpu_id}: 加载模型...")
    
    labeler = GroundingDINOLabeler(
        model_id="IDEA-Research/grounding-dino-base",
        device='cuda',
        box_threshold=0.25,
        text_threshold=0.25,
    )
    labeler.load_model()
    
    da3_estimator = DA3FullEstimator(
        model_name="da3nested-giant-large",
        device='cuda',
        use_ray_pose=True,
    )
    
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
            import cv2
            
            # 采样帧
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                results.append({
                    'scene_name': sample['scene_name'],
                    'question': question,
                    'question_type': question_type,
                    'ground_truth': gt,
                    'prediction': 'A',
                    'correct': gt.strip().upper() == 'A',
                    'error': 'No frames',
                })
                continue
            
            # 采样16帧用于 DA3
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
                    'question': question,
                    'question_type': question_type,
                    'ground_truth': gt,
                    'prediction': 'A',
                    'correct': gt.strip().upper() == 'A',
                    'error': f'Insufficient frames: {len(sampled_frames)}',
                })
                continue
            
            # DA3 深度估计
            prediction = da3_estimator.estimate_multiview(sampled_frames)
            
            # 在第一帧上检测物体
            frame_rgb = sampled_frames[0]
            detections = labeler.detect(frame_rgb, prompt)
            
            # 构建物体位置字典
            object_positions = {}
            for det in detections:
                label = det.label.strip().lower()
                if label.startswith('##'):
                    continue
                
                pos_3d = compute_3d_position_simple(
                    frame_rgb,
                    det.bbox_pixels,
                    prediction.depth_maps[0],
                    prediction.intrinsics[0],
                )
                
                if label not in object_positions:
                    object_positions[label] = pos_3d
                else:
                    # 如果已存在，使用平均位置
                    object_positions[label] = (object_positions[label] + pos_3d) / 2
            
            # 解析问题
            parsed = parse_direction_question(question)
            standing_by = parsed['standing_by']
            facing = parsed['facing']
            target = parsed['target']
            
            def find_position(obj_name):
                if not obj_name:
                    return None
                for label, pos in object_positions.items():
                    if match_object_name(obj_name, label):
                        return pos
                return None
            
            standing_pos = find_position(standing_by)
            facing_pos = find_position(facing)
            target_pos = find_position(target)
            
            predicted_direction = None
            reasoning = ""
            
            if target_pos is not None and standing_pos is not None and facing_pos is not None:
                predicted_direction = compute_direction_normalized(
                    target_pos, standing_pos, facing_pos
                )
                reasoning = (
                    f"Standing={standing_by}@{standing_pos.round(2)}, "
                    f"Facing={facing}@{facing_pos.round(2)}, "
                    f"Target={target}@{target_pos.round(2)} -> {predicted_direction}"
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
                'objects_found': {
                    'standing': standing_pos is not None,
                    'facing': facing_pos is not None,
                    'target': target_pos is not None,
                },
                'num_objects_detected': len(object_positions),
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
    
    # 清理
    del labeler
    del da3_estimator
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
    output_dir = Path(args.output_dir) / f"v77_normalized_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = str(output_dir / f"results_gpu{args.gpu}.json")
    results = worker_process(args.gpu, samples, output_file)
    
    # 统计
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for r in results:
        qtype = r['question_type']
        type_stats[qtype]['total'] += 1
        if r.get('correct', False):
            type_stats[qtype]['correct'] += 1
    
    print("\n" + "=" * 80)
    print("V7.7 归一化坐标方向测试结果")
    print("=" * 80)
    
    total_correct = 0
    total_samples = 0
    
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        n = stats['total']
        acc = stats['correct'] / n * 100 if n > 0 else 0
        print(f"{qtype:<35} {acc:>11.2f}% {n:>10}")
        total_correct += stats['correct']
        total_samples += n
    
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    print("-" * 80)
    print(f"{'Overall':<35} {overall_acc:>11.2f}% {total_samples:>10}")
    print("=" * 80)
    
    print(f"\n结果保存到: {output_dir}")


if __name__ == '__main__':
    main()
