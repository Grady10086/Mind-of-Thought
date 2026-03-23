#!/usr/bin/env python3
"""
VSIBench 最优配置测试 - 基于消融实验结果

核心策略：
- 对大多数任务使用尺度校准（提升 direction, distance, size）
- 对 room_size_estimation 保持原始中值深度校准（避免回归）

作者: tianjungu
日期: 2026-01-28
"""

import os
import sys
import json
import gc
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import re
import logging

import numpy as np
import cv2

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from tests.test_vsibench_directqa import (
    normalize_number, mean_relative_accuracy,
    NUMERICAL_TASKS, CHOICE_TASKS,
    EXTENDED_VOCABULARY, SYNONYM_MAP,
    get_synonyms, match_object_name,
    DirectQA,
    find_video_path, get_scene_source,
    VIDEO_DIRS, MindMapEntity3D,
)


# ============================================================================
# 改进版心智地图构建器
# ============================================================================

class MindMapBuilderOptimal:
    """最优配置的心智地图构建器"""
    
    # 已知物体尺寸（用于尺度校准）
    KNOWN_SIZES = {
        'door': {'height': 2.0, 'width': 0.9},
        'bed': {'height': 0.5, 'width': 1.5, 'length': 2.0},
        'sofa': {'height': 0.85, 'width': 0.9},
        'couch': {'height': 0.85, 'width': 0.9},
        'refrigerator': {'height': 1.7, 'width': 0.7},
        'fridge': {'height': 1.7, 'width': 0.7},
        'chair': {'height': 0.9, 'width': 0.5},
        'table': {'height': 0.75, 'width': 0.8},
        'desk': {'height': 0.75, 'width': 0.6},
        'toilet': {'height': 0.4, 'width': 0.4},
        'tv': {'height': 0.6, 'width': 1.0},
        'television': {'height': 0.6, 'width': 1.0},
        'monitor': {'height': 0.35, 'width': 0.55},
    }
    
    def __init__(self, device: str = 'cuda', num_frames: int = 32,
                 box_threshold: float = 0.25):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        
        self._labeler = None
        self._depth_estimator = None
        
        self.focal_length = 500
        self.principal_point = None
        
        # 尺度校准历史
        self.scale_history: List[float] = []
        
        # 存储中间结果
        self.all_detections: Dict[str, List[Dict]] = {}
        self.frame_indices: List[int] = []
        
        # 双重深度图存储：原始和校准后的
        self.depth_maps_original: List[np.ndarray] = []
        self.depth_maps_calibrated: List[np.ndarray] = []
    
    def _load_models(self):
        """加载模型"""
        import torch
        
        if self._labeler is None:
            from core.semantic_labeler import GroundingDINOLabeler
            self._labeler = GroundingDINOLabeler(
                model_id="IDEA-Research/grounding-dino-base",
                device=self.device,
                box_threshold=self.box_threshold,
                text_threshold=0.25,
            )
            self._labeler.load_model()
        
        if self._depth_estimator is None:
            from core.perception import DepthEstimator
            self._depth_estimator = DepthEstimator(
                model_name="depth-anything/DA3-Large",
                device=self.device,
                half_precision=True,
            )
    
    def unload(self):
        """释放模型"""
        import torch
        
        if self._labeler is not None:
            try:
                del self._labeler.model
                del self._labeler.processor
            except:
                pass
            self._labeler = None
        
        if self._depth_estimator is not None:
            try:
                del self._depth_estimator.model
            except:
                pass
            self._depth_estimator = None
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    
    def _sample_frames(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:
        n = min(self.num_frames, len(frames))
        indices = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in indices], indices.tolist()
    
    def _calibrate_scale(self, detections: List[Dict], depth_map: np.ndarray) -> float:
        """动态尺度校准"""
        scale_estimates = []
        
        for det in detections:
            label = det['label'].lower()
            
            known_size = None
            for key in self.KNOWN_SIZES:
                if key in label or label in key:
                    known_size = self.KNOWN_SIZES[key]
                    break
            
            if known_size is None:
                continue
            
            ref_dim = known_size.get('height', known_size.get('width'))
            if ref_dim is None:
                continue
            
            x1, y1, x2, y2 = det['bbox']
            h_pixel = y2 - y1
            
            depth_roi = depth_map[y1:y2, x1:x2]
            if depth_roi.size == 0:
                continue
            depth_median = np.median(depth_roi)
            if depth_median <= 0:
                continue
            
            h_estimated = h_pixel / self.focal_length * depth_median
            
            if h_estimated > 0.01:
                scale = ref_dim / h_estimated
                if 0.1 < scale < 10:
                    scale_estimates.append(scale)
        
        if scale_estimates:
            scale = float(np.median(scale_estimates))
            self.scale_history.append(scale)
            if len(self.scale_history) > 10:
                self.scale_history = self.scale_history[-10:]
            return scale
        
        if self.scale_history:
            return float(np.median(self.scale_history))
        
        # 默认中值校准
        median_depth = np.median(depth_map)
        return 2.5 / median_depth if median_depth > 0 else 1.0
    
    def build_from_video(self, video_path: str, 
                         target_objects: List[str] = None,
                         question_type: str = None) -> Dict[str, MindMapEntity3D]:
        """
        构建心智地图
        
        Args:
            question_type: 问题类型，用于决定是否使用尺度校准
        """
        self._load_models()
        
        # 重置
        self.scale_history.clear()
        self.depth_maps_original.clear()
        self.depth_maps_calibrated.clear()
        
        # 提取帧
        all_frames = self._extract_frames(video_path)
        if not all_frames:
            return {}
        
        frames, self.frame_indices = self._sample_frames(all_frames)
        H, W = frames[0].shape[:2]
        self.principal_point = (W / 2, H / 2)
        
        # 词汇表
        if target_objects is None:
            target_objects = []
        vocab = list(set(target_objects + EXTENDED_VOCABULARY))
        text_prompt = " . ".join(vocab) + " ."
        
        # 决定是否使用尺度校准
        # room_size 任务不使用尺度校准（避免回归）
        use_calibration = (question_type != 'room_size_estimation')
        
        # 存储检测
        self.all_detections = defaultdict(list)
        frame_detection_counts = defaultdict(lambda: defaultdict(int))
        
        for frame_idx, frame in enumerate(frames):
            original_idx = self.frame_indices[frame_idx]
            
            # 深度估计
            try:
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                if depth_map.shape[:2] != (H, W):
                    depth_map = cv2.resize(depth_map, (W, H))
            except Exception as e:
                logger.warning(f"深度估计失败: {e}")
                depth_map = np.ones((H, W), dtype=np.float32) * 2.5
            
            # 保存原始深度图（用于 room_size）
            median_depth = np.median(depth_map)
            default_scale = 2.5 / median_depth if median_depth > 0 else 1.0
            self.depth_maps_original.append(depth_map * default_scale)
            
            # 物体检测
            results = self._labeler.detect(frame, text_prompt)
            
            # 预处理检测
            raw_detections = []
            for det in results:
                x1, y1, x2, y2 = [int(v) for v in det.bbox_pixels]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    raw_detections.append({
                        'label': det.label.lower(),
                        'bbox': (x1, y1, x2, y2),
                        'confidence': det.confidence,
                    })
            
            # 尺度校准（条件性）
            if use_calibration:
                scale = self._calibrate_scale(raw_detections, depth_map)
            else:
                scale = default_scale
            
            depth_map_scaled = depth_map * scale
            self.depth_maps_calibrated.append(depth_map_scaled)
            
            # 处理检测
            for det in raw_detections:
                label = det['label']
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                
                frame_detection_counts[original_idx][label] += 1
                
                depth_roi = depth_map_scaled[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                depth_median = float(np.median(depth_roi))
                
                # 3D 位置
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                px, py = self.principal_point
                
                x_3d = (cx - px) / self.focal_length * depth_median
                y_3d = (cy - py) / self.focal_length * depth_median
                z_3d = depth_median
                
                # 3D 尺寸
                w_3d = (x2 - x1) / self.focal_length * depth_median
                h_3d = (y2 - y1) / self.focal_length * depth_median
                d_3d = min(w_3d, h_3d) * 0.5
                
                self.all_detections[label].append({
                    'frame_idx': original_idx,
                    'bbox': det['bbox'],
                    'confidence': confidence,
                    'position_3d': np.array([x_3d, y_3d, z_3d]),
                    'size_3d': np.array([w_3d, h_3d, d_3d]),
                    'depth_median': depth_median,
                })
        
        return self._aggregate(frame_detection_counts)
    
    def _aggregate(self, frame_detection_counts) -> Dict[str, MindMapEntity3D]:
        """聚合"""
        entities = {}
        
        for category, dets in self.all_detections.items():
            if not dets:
                continue
            
            # 最大单帧检测数
            max_count = 0
            for frame_idx, counts in frame_detection_counts.items():
                if category in counts:
                    max_count = max(max_count, counts[category])
            if max_count == 0:
                max_count = 1
            
            avg_conf = np.mean([d['confidence'] for d in dets])
            first_frame = min(d['frame_idx'] for d in dets)
            last_frame = max(d['frame_idx'] for d in dets)
            
            positions = np.array([d['position_3d'] for d in dets])
            sizes = np.array([d['size_3d'] for d in dets])
            depths = [d['depth_median'] for d in dets]
            
            entity = MindMapEntity3D(
                entity_id=f"entity_{category}",
                label=category,
                count=max_count,
                avg_confidence=float(avg_conf),
                first_seen_frame=first_frame,
                last_seen_frame=last_frame,
                position_3d=np.median(positions, axis=0),
                size_3d=np.median(sizes, axis=0),
                depth_median=float(np.median(depths)),
            )
            entities[category] = entity
        
        return entities


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], result_queue: mp.Queue):
    """GPU Worker"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    builder = MindMapBuilderOptimal(device='cuda', num_frames=32, box_threshold=0.25)
    
    results = []
    total = len(samples)
    
    for i, sample in enumerate(samples):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            target_objects = []
            if question_type == 'object_counting':
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            # 传入 question_type，让构建器决定是否使用尺度校准
            mind_map = builder.build_from_video(video_path, target_objects, question_type)
            
            # 回答问题
            if question_type == 'object_counting':
                pred = DirectQA.answer_counting(mind_map, question)
            elif question_type == 'object_size_estimation':
                pred = DirectQA.answer_object_size(mind_map, question)
            elif question_type == 'room_size_estimation':
                pred = DirectQA.answer_room_size(mind_map, question)
            elif question_type == 'object_abs_distance':
                pred = DirectQA.answer_abs_distance(mind_map, question)
            elif question_type.startswith('object_rel_direction'):
                difficulty = question_type.split('_')[-1]
                pred = DirectQA.answer_rel_direction(mind_map, question, options, difficulty)
            elif question_type == 'object_rel_distance':
                pred = DirectQA.answer_rel_distance(mind_map, question, options)
            elif question_type == 'obj_appearance_order':
                pred = DirectQA.answer_appearance_order(mind_map, question, options)
            elif question_type == 'route_planning':
                pred = DirectQA.answer_route_planning(mind_map, question, options)
            else:
                pred = str(options[0]) if options else "0"
            
            # 计算指标
            if question_type in NUMERICAL_TASKS:
                pred_num = normalize_number(pred)
                gt_num = normalize_number(str(gt))
                score = mean_relative_accuracy(pred_num, gt_num) if pred_num and gt_num else 0.0
                correct = score > 0.5
            else:
                pred_letter = None
                gt_letter = str(gt).strip().upper()
                
                if len(gt_letter) > 1:
                    for idx, opt in enumerate(options):
                        opt_content = re.sub(r'^[A-D]\.\s*', '', opt).strip()
                        if gt_letter.lower() == opt_content.lower():
                            gt_letter = chr(65 + idx)
                            break
                
                if pred:
                    letter_match = re.match(r'^([A-D])[\.\s]', pred.strip().upper())
                    if letter_match:
                        pred_letter = letter_match.group(1)
                    else:
                        pred_clean = re.sub(r'^[A-D]\.\s*', '', pred).lower().strip()
                        for idx, opt in enumerate(options):
                            opt_content = re.sub(r'^[A-D]\.\s*', '', opt).lower().strip()
                            if pred_clean == opt_content or pred_clean in opt_content or opt_content in pred_clean:
                                pred_letter = chr(65 + idx)
                                break
                
                correct = pred_letter == gt_letter if pred_letter else False
                score = 1.0 if correct else 0.0
            
            result = {
                'sample_id': sample['id'],
                'question_type': question_type,
                'question': question,
                'prediction': pred,
                'ground_truth': gt,
                'score': score,
                'correct': bool(correct),
            }
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} 样本 {sample['id']} 错误: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'sample_id': sample['id'],
                'question_type': question_type,
                'question': question,
                'prediction': '',
                'ground_truth': gt,
                'score': 0.0,
                'correct': False,
                'error': str(e),
            }
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"GPU {gpu_id}: {i+1}/{total} 完成")
    
    builder.unload()
    result_queue.put((gpu_id, results))


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--task-type', type=str, default='all')
    args = parser.parse_args()
    
    print("=" * 70)
    print("VSIBench 最优配置测试")
    print("策略: 对 room_size 外的任务使用尺度校准")
    print("=" * 70)
    print(f"GPU数量: {args.num_gpus}")
    
    from datasets import load_dataset
    dataset = load_dataset(
        "nyu-visionx/VSI-Bench",
        split="test",
        cache_dir="/home/tione/notebook/tianjungu/hf_cache/vsibench"
    )
    
    # 准备样本
    samples = []
    for idx, item in enumerate(dataset):
        scene_name = item['scene_name']
        question_type = item['question_type']
        
        if args.task_type != 'all' and question_type != args.task_type:
            continue
        
        video_path = find_video_path(scene_name)
        if not video_path:
            continue
        
        samples.append({
            'id': idx,
            'scene_name': scene_name,
            'source': get_scene_source(scene_name),
            'video_path': video_path,
            'question': item['question'],
            'question_type': question_type,
            'ground_truth': item['ground_truth'],
            'options': item.get('options', []),
        })
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"总样本数: {len(samples)}")
    
    type_counts = defaultdict(int)
    for s in samples:
        type_counts[s['question_type']] += 1
    print("\n任务类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    
    # 分配到 GPU
    num_gpus = min(args.num_gpus, len(samples))
    samples_per_gpu = len(samples) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])
    
    # 启动多进程
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    
    start_time = datetime.now()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_samples[gpu_id], result_queue))
        p.start()
        processes.append(p)
        logger.info(f"启动 GPU {gpu_id}: {len(gpu_samples[gpu_id])} 样本")
    
    # 收集结果
    all_results = []
    for _ in range(num_gpus):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        logger.info(f"GPU {gpu_id} 完成: {len(results)} 结果")
    
    for p in processes:
        p.join()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n总耗时: {duration:.1f}秒")
    
    # 统计
    type_scores = defaultdict(list)
    for r in all_results:
        type_scores[r['question_type']].append(r['score'])
    
    print("\n" + "="*60)
    print("最优配置测试结果")
    print("="*60)
    
    total_score = 0
    total_count = 0
    
    for q_type in sorted(type_scores.keys()):
        scores = type_scores[q_type]
        if q_type in NUMERICAL_TASKS:
            avg = np.mean(scores) * 100
            metric = "MRA"
        else:
            avg = np.mean(scores) * 100
            metric = "Acc"
        
        print(f"{q_type:30s}: {avg:6.2f}% {metric} ({len(scores)} 样本)")
        total_score += sum(scores)
        total_count += len(scores)
    
    overall = total_score / total_count * 100 if total_count > 0 else 0
    print("-"*60)
    print(f"{'Overall':30s}: {overall:6.2f}% ({total_count} 样本)")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/optimal_config_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    cleaned_results = []
    for r in all_results:
        cleaned = {k: convert_numpy(v) for k, v in r.items()}
        cleaned_results.append(cleaned)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'config': 'optimal: scale_calibration for non-room_size tasks',
            'summary': {q: {'mean': float(np.mean(s)), 'count': len(s)} for q, s in type_scores.items()},
            'overall': float(overall),
            'details': cleaned_results,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果保存到: {output_dir}")
