#!/usr/bin/env python3
"""
V7.9 - 在图像平面上进行方向判断

核心思路:
1. 物体的 2D 位置（像素坐标）直接反映了在图像中的"左右"
2. 深度值反映了"前后"（近/远）
3. 使用第一帧图像的 2D 坐标 + 深度进行方向计算

方向定义 (从观察者视角):
- 站在 A 面向 B
- 如果 Target 在图像中比 A 更靠左 -> Target 在 A 的左边
- 如果 Target 比 A 更靠近相机 (depth 更小) -> Target 在 A 的前面

关键改进:
- 使用 2D 像素坐标进行左右判断（因为这是直接可观察的）
- 使用深度进行前后判断
- 建立以 A->B 方向为"前"的局部坐标系
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
    'dishwasher': ['washer'],
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


def compute_direction_2d_plus_depth(
    target_2d: np.ndarray,  # (x, y) 像素坐标
    target_depth: float,
    standing_2d: np.ndarray,
    standing_depth: float,
    facing_2d: np.ndarray,
    facing_depth: float,
) -> Tuple[str, Dict]:
    """
    使用 2D 像素坐标 + 深度进行方向判断
    
    坐标系:
    - 图像 X 轴: 向右
    - 图像 Y 轴: 向下
    - 深度: 向前 (值越小越近)
    
    算法:
    1. 计算 facing 相对于 standing 的 2D 方向向量 (这是"前方")
    2. 右方向 = 前方顺时针旋转 90 度
    3. 将 target 相对于 standing 的向量投影到前方和右方
    """
    # 前方向量 (2D)
    forward_2d = facing_2d - standing_2d
    forward_norm = np.linalg.norm(forward_2d)
    
    if forward_norm < 1e-6:
        return "same-position", {'error': 'standing == facing (2D)'}
    
    forward_2d = forward_2d / forward_norm
    
    # 右方向量 (2D) - 顺时针旋转 90 度: (x, y) -> (y, -x)
    right_2d = np.array([forward_2d[1], -forward_2d[0]])
    
    # 目标相对向量 (2D)
    target_rel_2d = target_2d - standing_2d
    
    # 2D 投影
    proj_forward_2d = np.dot(target_rel_2d, forward_2d)
    proj_right_2d = np.dot(target_rel_2d, right_2d)
    
    # 深度差异 (负值表示更近，即"前方")
    depth_diff = target_depth - standing_depth
    facing_depth_diff = facing_depth - standing_depth
    
    # 综合前后判断：
    # - 如果 target 比 standing 更靠近 facing (深度更接近 facing) -> 前
    # - 或者 target 在 standing->facing 方向的 2D 投影为正 -> 前
    
    # 简化：使用 2D 投影作为主要判断依据
    # 因为 "站在 A 面向 B" 的 "前方" 就是 A->B 的方向
    
    debug_info = {
        'standing_2d': standing_2d.tolist(),
        'facing_2d': facing_2d.tolist(),
        'target_2d': target_2d.tolist(),
        'forward_2d': forward_2d.tolist(),
        'right_2d': right_2d.tolist(),
        'proj_forward_2d': float(proj_forward_2d),
        'proj_right_2d': float(proj_right_2d),
        'standing_depth': float(standing_depth),
        'facing_depth': float(facing_depth),
        'target_depth': float(target_depth),
    }
    
    # 阈值 - 使用相对值
    target_rel_norm = np.linalg.norm(target_rel_2d)
    if target_rel_norm < 1e-6:
        return "same-position", debug_info
    
    # 使用较小的阈值比例
    threshold = 0.05 * target_rel_norm
    
    directions = []
    
    # 前后判断 (基于 2D 投影)
    if proj_forward_2d > threshold:
        directions.append("front")
    elif proj_forward_2d < -threshold:
        directions.append("back")
    
    # 左右判断 (基于 2D 投影)
    if proj_right_2d > threshold:
        directions.append("right")
    elif proj_right_2d < -threshold:
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
    from core.perception_da3_full import DA3FullEstimator
    
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
            # 读取视频
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
            
            # 采样帧
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
            H, W = frame_rgb.shape[:2]
            detections = labeler.detect(frame_rgb, prompt)
            
            # 处理后的图像尺寸
            proc_H, proc_W = prediction.depth_maps.shape[1:3]
            scale_x = proc_W / W
            scale_y = proc_H / H
            
            # 构建物体信息字典 (2D 位置 + 深度)
            object_info = {}
            for det in detections:
                label = det.label.strip().lower()
                if label.startswith('##'):
                    continue
                
                # 2D 中心点 (原始图像坐标)
                bbox = det.bbox_pixels
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                
                # 获取深度 (处理后图像坐标)
                cx_scaled = int(np.clip(cx * scale_x, 0, proc_W - 1))
                cy_scaled = int(np.clip(cy * scale_y, 0, proc_H - 1))
                depth = prediction.depth_maps[0, cy_scaled, cx_scaled]
                
                if label not in object_info:
                    object_info[label] = {
                        'pos_2d': np.array([cx, cy]),
                        'depth': float(depth),
                    }
            
            # 解析问题
            parsed = parse_direction_question(question)
            standing_by = parsed['standing_by']
            facing = parsed['facing']
            target = parsed['target']
            
            def find_object(obj_name):
                if not obj_name:
                    return None
                for label, info in object_info.items():
                    if match_object_name(obj_name, label):
                        return info
                return None
            
            standing_info = find_object(standing_by)
            facing_info = find_object(facing)
            target_info = find_object(target)
            
            predicted_direction = None
            reasoning = ""
            debug_info = {}
            
            if standing_info and facing_info and target_info:
                predicted_direction, debug_info = compute_direction_2d_plus_depth(
                    target_info['pos_2d'], target_info['depth'],
                    standing_info['pos_2d'], standing_info['depth'],
                    facing_info['pos_2d'], facing_info['depth'],
                )
                
                reasoning = (
                    f"Standing={standing_by}@({standing_info['pos_2d'][0]:.0f},{standing_info['pos_2d'][1]:.0f},d={standing_info['depth']:.1f}), "
                    f"Facing={facing}@({facing_info['pos_2d'][0]:.0f},{facing_info['pos_2d'][1]:.0f},d={facing_info['depth']:.1f}), "
                    f"Target={target}@({target_info['pos_2d'][0]:.0f},{target_info['pos_2d'][1]:.0f},d={target_info['depth']:.1f}) -> {predicted_direction}"
                )
            else:
                missing = []
                if standing_info is None:
                    missing.append(f"standing({standing_by})")
                if facing_info is None:
                    missing.append(f"facing({facing})")
                if target_info is None:
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
                    'standing': standing_info is not None,
                    'facing': facing_info is not None,
                    'target': target_info is not None,
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
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    samples = load_vsibench_direction_samples()
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"v79_2d_depth_{timestamp}"
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
    
    print("\n" + "=" * 80)
    print("V7.9 2D+深度方向测试结果")
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
