#!/usr/bin/env python3
"""
V7.8 - 使用相机局部坐标系进行方向判断

核心改进:
1. 废弃对 DA3 世界坐标系轴向的任何假设
2. 从 extrinsics 提取当前帧的相机局部基向量 (Right, Up, Forward)
3. 使用点积投影到相机局部空间进行左右/前后判断

DA3 extrinsics 是 world-to-camera (w2c) 变换: [R|t]
- R 的列向量在相机坐标系中表示世界坐标轴
- R 的行向量在世界坐标系中表示相机坐标轴

相机坐标系约定 (OpenCV):
- X: 右
- Y: 下  
- Z: 前 (深度方向)

从 w2c 的 R 矩阵提取相机基向量 (在世界坐标系中的表示):
- Right = R[0, :] (R 的第一行)
- Down = R[1, :]  (R 的第二行)
- Forward = R[2, :] (R 的第三行)
- Up = -Down
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
    'dishwasher': ['washer'],  # 添加这个匹配
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
    """解析方向问题，提取三个物体"""
    result = {'standing_by': None, 'facing': None, 'target': None}
    
    q_lower = question.lower()
    # 匹配: "standing by X and facing Y, is Z to..."
    pattern = r'standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+)'
    match = re.search(pattern, q_lower)
    if match:
        result['standing_by'] = match.group(1)
        result['facing'] = match.group(2)
        result['target'] = match.group(3)
    
    return result


def extract_camera_basis_from_extrinsics(extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 extrinsics (w2c) 提取相机基向量 (在世界坐标系中的表示)
    
    DA3 extrinsics 是 (3, 4) 的 [R|t] 矩阵，表示 world-to-camera 变换
    cam_point = R @ world_point + t
    
    R 的行向量表示相机坐标轴在世界坐标系中的方向:
    - R[0, :] = Right (相机 X 轴在世界坐标系中)
    - R[1, :] = Down (相机 Y 轴在世界坐标系中)
    - R[2, :] = Forward (相机 Z 轴在世界坐标系中)
    
    Args:
        extrinsics: (3, 4) w2c 变换矩阵
        
    Returns:
        (right, up, forward): 三个单位向量，在世界坐标系中表示
    """
    R = extrinsics[:3, :3]  # (3, 3)
    
    # 从 w2c 的 R 矩阵提取相机基向量
    right = R[0, :]    # 相机 X 轴 (右)
    down = R[1, :]     # 相机 Y 轴 (下)
    forward = R[2, :]  # 相机 Z 轴 (前/深度方向)
    
    up = -down  # 上 = -下
    
    # 归一化
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    forward = forward / np.linalg.norm(forward)
    
    return right, up, forward


def compute_direction_in_camera_space(
    target_world: np.ndarray,
    standing_world: np.ndarray,
    facing_world: np.ndarray,
    camera_right: np.ndarray,
    camera_forward: np.ndarray,
) -> Tuple[str, Dict]:
    """
    在相机局部空间中计算方向
    
    问题: "站在 standing 处，面向 facing，target 在哪个方向？"
    
    算法:
    1. 计算观察者的前方向量: facing_dir = facing_world - standing_world
    2. 计算目标的相对向量: target_rel = target_world - standing_world
    3. 将这些向量投影到相机的 Right 和 Forward 基向量上
    4. 根据投影判断左右/前后
    
    关键改进: 
    - 不再假设世界坐标系有固定语义
    - 使用相机的局部基向量作为参考
    
    但这里有个问题: 问题说的是"站在 standing 处面向 facing"，
    所以我们需要用 standing->facing 的方向作为观察者的"前方"，
    而不是相机的前方。
    
    修正算法:
    1. 观察者的前方: viewer_forward = normalize(facing - standing)
    2. 观察者的右方: viewer_right = viewer_forward × camera_up (或用 camera_right 投影)
    3. 目标相对位置: target_rel = target - standing
    4. 判断: target_rel · viewer_forward (前后), target_rel · viewer_right (左右)
    
    更简单的方法 (纯世界坐标，使用叉积):
    - viewer_forward = facing - standing (在水平面投影)
    - 使用 2D 叉积判断左右
    """
    # 计算观察者前方向量
    viewer_forward = facing_world - standing_world
    viewer_forward_norm = np.linalg.norm(viewer_forward)
    
    if viewer_forward_norm < 1e-6:
        return "same-position", {'error': 'standing == facing'}
    
    viewer_forward = viewer_forward / viewer_forward_norm
    
    # 目标相对向量
    target_rel = target_world - standing_world
    target_rel_norm = np.linalg.norm(target_rel)
    
    if target_rel_norm < 1e-6:
        return "same-position", {'error': 'standing == target'}
    
    # 方法1: 使用相机的 Up 向量构建观察者的右方向
    # viewer_right = viewer_forward × camera_up
    # 这要求 camera_up 近似垂直于 viewer_forward
    
    # 方法2: 使用相机的 Right 和 Forward 向量
    # 将 viewer_forward 投影到相机的水平面 (由 camera_right 和 camera_forward 张成)
    # 然后旋转 90 度得到 viewer_right
    
    # 方法3 (最稳健): 使用 Gram-Schmidt 过程
    # 1. 以 viewer_forward 为基准
    # 2. 从 camera_right 中减去沿 viewer_forward 的分量，得到 viewer_right
    
    # 这里我们使用方法3
    proj = np.dot(camera_right, viewer_forward) * viewer_forward
    viewer_right = camera_right - proj
    viewer_right_norm = np.linalg.norm(viewer_right)
    
    if viewer_right_norm < 1e-6:
        # camera_right 与 viewer_forward 平行，使用 camera_forward
        proj = np.dot(camera_forward, viewer_forward) * viewer_forward
        viewer_right_candidate = camera_forward - proj
        # 旋转 90 度 (叉积)
        viewer_up_approx = np.cross(viewer_forward, viewer_right_candidate)
        viewer_right = np.cross(viewer_up_approx, viewer_forward)
        viewer_right_norm = np.linalg.norm(viewer_right)
        if viewer_right_norm < 1e-6:
            return "unknown", {'error': 'cannot determine right direction'}
    
    viewer_right = viewer_right / viewer_right_norm
    
    # 计算投影
    proj_forward = np.dot(target_rel, viewer_forward)
    proj_right = np.dot(target_rel, viewer_right)
    
    # 使用更小的阈值 - 只要有明显的侧向分量就判断左右
    # 关键改进：使用 proj_forward 作为参考，而不是 target_rel_norm
    # 如果 |proj_right| > |proj_forward| * 0.2，就认为有显著的侧向分量
    
    # 判断方向
    directions = []
    
    # 前后判断
    if proj_forward > 0.01:  # 很小的阈值，几乎任何正值都是前
        directions.append("front")
    elif proj_forward < -0.01:
        directions.append("back")
    
    # 左右判断 - 使用相对比例
    # 如果侧向投影的绝对值大于前向投影绝对值的 20%，就判断左右
    forward_abs = abs(proj_forward) + 0.01  # 避免除零
    right_threshold = max(0.05, forward_abs * 0.2)
    
    if proj_right > right_threshold:
        directions.append("right")
    elif proj_right < -right_threshold:
        directions.append("left")
    
    debug_info = {
        'viewer_forward': viewer_forward.tolist(),
        'viewer_right': viewer_right.tolist(),
        'target_rel': target_rel.tolist(),
        'proj_forward': float(proj_forward),
        'proj_right': float(proj_right),
        'right_threshold': float(right_threshold),
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


def worker_process(gpu_id: int, samples: List[Dict], output_file: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    from core.perception_da3_full import MindMapBuilder3D, DA3FullPrediction
    
    # 使用世界坐标 (不是相机坐标)
    builder = MindMapBuilder3D(
        device='cuda',
        num_frames=16,
        box_threshold=0.25,
        model_name="da3nested-giant-large",
        use_ray_pose=True,
        use_camera_coords=False,  # 使用世界坐标
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
            
            # 提取第一帧的相机基向量
            if da3_prediction is not None and da3_prediction.extrinsics is not None:
                camera_right, camera_up, camera_forward = extract_camera_basis_from_extrinsics(
                    da3_prediction.extrinsics[0]
                )
            else:
                camera_right = np.array([1, 0, 0])
                camera_up = np.array([0, 1, 0])
                camera_forward = np.array([0, 0, 1])
            
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
                
                predicted_direction, debug_info = compute_direction_in_camera_space(
                    target_pos, standing_pos, facing_pos,
                    camera_right, camera_forward,
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
                'camera_basis': {
                    'right': camera_right.tolist(),
                    'up': camera_up.tolist(),
                    'forward': camera_forward.tolist(),
                },
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
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    samples = load_vsibench_direction_samples()
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"v78_camera_basis_{timestamp}"
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
    print("V7.8 相机局部基向量方向测试结果")
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
