#!/usr/bin/env python3
"""
公平对比测试：DINO+SAM3 方法 vs 心智地图方法

目的：找出心智地图方法效果变差的原因

测试配置：
- 8 张 GPU 并行
- 完整 VSIBench object_counting 任务 (565 样本)
- 对比两种方法的 MRA

作者: tianjungu
日期: 2026-01-24
"""

import os
import sys
import json
import time
import gc
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import re

import numpy as np
import cv2
import torch
from sklearn.cluster import DBSCAN

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/GroundingDINO')

# ============================================================================
# VSIBench 评测指标
# ============================================================================

def normalize_number(text: str) -> Optional[float]:
    """提取数字"""
    if text is None:
        return None
    text = re.sub(r'(meters?|m|cm|feet|ft|inches?|in)\b', '', text.lower())
    match = re.search(r'[-+]?\d*\.?\d+', text)
    if match:
        try:
            return float(match.group())
        except:
            return None
    return None


def mean_relative_accuracy(pred: float, target: float, start=0.05, end=0.5, interval=0.05) -> float:
    """VSIBench 官方 MRA 指标"""
    if pred is None or target is None:
        return 0.0
    epsilon = 1e-8
    rel_error = abs(pred - target) / (abs(target) + epsilon)
    thresholds = np.arange(start, end + interval / 2, interval)
    conditions = rel_error < (1 - thresholds)
    return conditions.astype(float).mean()


# ============================================================================
# 物体类型自适应阈值 (来自 eval_vsibench_fast_v4.py 的最佳配置)
# ============================================================================

OBJECT_TYPE_THRESHOLDS = {
    'chair': 0.35,
    'table': 0.45,
    'sofa': 0.45,
    'bed': 0.45,
    'stool': 0.40,
    'door': 0.45,
    'window': 0.40,
    'washer': 0.35,
    'lamp': 0.40,
    'tv': 0.45,
    'monitor': 0.45,
    'pillow': 0.45,
    'toilet': 0.45,
    'bathtub': 0.45,
    'refrigerator': 0.45,
    'sink': 0.40,
    'mirror': 0.40,
    'towel': 0.35,
    'backpack': 0.35,
    'trash bin': 0.35,
    'default': 0.45
}


# ============================================================================
# 方法1: DINO + DA3 + 3D聚类 (之前效果好的方法)
# ============================================================================

class Method1_DINO_DA3:
    """
    DINO + DA3 方法
    
    核心流程:
    1. 帧采样
    2. DA3 深度估计
    3. GroundingDINO 检测 (自适应阈值)
    4. 2D→3D 投影
    5. 尺寸过滤
    6. 3D NMS
    7. DBSCAN 聚类
    8. 保守计数策略
    """
    
    def __init__(self, device: str = 'cuda', distance_threshold: float = 0.5):
        self.device = device
        self.distance_threshold = distance_threshold
        self.num_frames = 16
        
        # 延迟加载模型
        self._da3_model = None
        self._grounding_model = None
        self._grounding_processor = None
        
    def _load_da3(self):
        if self._da3_model is None:
            print(f"[{self.device}] Loading DA3...")
            from depth_anything_3.api import DepthAnything3
            self._da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
            self._da3_model = self._da3_model.to(self.device).eval()
            
    def _load_grounding(self):
        if self._grounding_model is None:
            print(f"[{self.device}] Loading GroundingDINO...")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            self._grounding_processor = AutoProcessor.from_pretrained(model_id)
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self._grounding_model = self._grounding_model.to(self.device)
            
    def _unload_models(self):
        """释放模型内存"""
        if self._da3_model is not None:
            del self._da3_model
            self._da3_model = None
        if self._grounding_model is not None:
            del self._grounding_model
            self._grounding_model = None
        if self._grounding_processor is not None:
            del self._grounding_processor
            self._grounding_processor = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def count_objects(self, video_path: str, target_object: str) -> Dict:
        """
        计数物体
        
        Returns:
            {'count': int, 'num_detections': int, 'debug': dict}
        """
        start_time = time.time()
        
        # 1. 提取帧
        frames = self._extract_frames(video_path)
        if not frames:
            return {'count': 0, 'num_detections': 0, 'error': 'No frames'}
        
        # 2. DA3 深度估计
        self._load_da3()
        depths, intrinsics, extrinsics, scale_factor = self._run_da3(frames)
        
        # 释放 DA3 显存
        del self._da3_model
        self._da3_model = None
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. GroundingDINO 检测
        self._load_grounding()
        threshold = OBJECT_TYPE_THRESHOLDS.get(target_object.lower(), OBJECT_TYPE_THRESHOLDS['default'])
        detections = self._detect_objects(frames, target_object, threshold)
        
        # 4. 2D → 3D 投影 + 去重
        result = self._deduplicate_3d(
            detections, depths, extrinsics, intrinsics, frames, scale_factor
        )
        
        elapsed = time.time() - start_time
        
        return {
            'count': result['count'],
            'num_detections': result['num_detections'],
            'num_clusters': result.get('num_clusters', 0),
            'num_noise': result.get('num_noise', 0),
            'threshold': threshold,
            'time': elapsed,
        }
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """提取帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def _run_da3(self, frames: List[np.ndarray]) -> Tuple:
        """运行 DA3 深度估计"""
        from PIL import Image
        
        # 转换为 PIL
        pil_images = [Image.fromarray(f) for f in frames]
        
        with torch.no_grad():
            predictions = self._da3_model.infer(pil_images, reference_selection="saddle_balanced")
        
        depths = []
        intrinsics = []
        extrinsics = []
        
        for pred in predictions:
            depths.append(pred['depth'].cpu().numpy())
            intrinsics.append(pred['intrinsics'].cpu().numpy())
            extrinsics.append(pred['extrinsics'].cpu().numpy())
        
        # 尺度估计 (假设相机高度 1.6m)
        camera_height = 1.6
        median_depth = np.median([d.mean() for d in depths])
        scale_factor = camera_height / median_depth if median_depth > 0 else 1.0
        
        return depths, intrinsics, extrinsics, scale_factor
    
    def _detect_objects(self, frames: List[np.ndarray], target_object: str, threshold: float) -> List[Dict]:
        """检测物体"""
        from PIL import Image
        
        text_prompt = f"{target_object} ."
        all_detections = []
        
        for frame_idx, frame in enumerate(frames):
            image = Image.fromarray(frame)
            
            inputs = self._grounding_processor(images=image, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._grounding_model(**inputs)
            
            results = self._grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=threshold,
                text_threshold=0.25,
                target_sizes=[image.size[::-1]]
            )
            
            for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
                all_detections.append({
                    'frame': frame_idx,
                    'label': label,
                    'confidence': score.item(),
                    'bbox': box.cpu().tolist(),
                })
        
        return all_detections
    
    def _deduplicate_3d(self, detections, depths, extrinsics, intrinsics, frames, scale_factor) -> Dict:
        """3D 去重 + 聚类计数"""
        if not detections:
            return {'count': 0, 'num_detections': 0}
        
        # 2D → 3D 投影
        objects_3d = []
        for det in detections:
            frame_idx = det['frame']
            if frame_idx >= len(depths):
                continue
            
            depth = depths[frame_idx]
            K = intrinsics[frame_idx]
            extrinsic = extrinsics[frame_idx]
            
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            h, w = depth.shape[:2]
            px = int(np.clip(cx, 0, w - 1))
            py = int(np.clip(cy, 0, h - 1))
            
            z = depth[py, px] * scale_factor
            if z <= 0 or z > 20:
                continue
            
            fx, fy = K[0, 0], K[1, 1]
            cx_cam, cy_cam = K[0, 2], K[1, 2]
            
            X = (cx - cx_cam) * z / fx
            Y = (cy - cy_cam) * z / fy
            Z = z
            
            point_cam = np.array([X, Y, Z, 1.0])
            point_world = extrinsic @ point_cam
            
            # 3D 尺寸
            width_3d = (x2 - x1) * z / fx
            height_3d = (y2 - y1) * z / fy
            
            # 尺寸过滤
            if width_3d < 0.05 or width_3d > 10 or height_3d < 0.05 or height_3d > 10:
                continue
            
            objects_3d.append({
                'position': point_world[:3],
                'confidence': det['confidence'],
                'size': [width_3d, height_3d],
            })
        
        if not objects_3d:
            return {'count': 0, 'num_detections': len(detections)}
        
        # 3D NMS
        objects_3d = self._apply_3d_nms(objects_3d)
        
        # DBSCAN 聚类
        if len(objects_3d) < 2:
            return {
                'count': len(objects_3d),
                'num_detections': len(detections),
                'num_clusters': len(objects_3d),
                'num_noise': 0,
            }
        
        positions = np.array([obj['position'] for obj in objects_3d])
        clustering = DBSCAN(eps=self.distance_threshold, min_samples=2).fit(positions)
        labels = clustering.labels_
        
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = list(labels).count(-1)
        
        # 保守计数策略
        total_count = num_clusters + max(0, (num_noise - 2)) // 4
        if total_count == 0 and num_noise > 0:
            total_count = max(1, num_noise // 5)
        
        return {
            'count': total_count,
            'num_detections': len(detections),
            'num_clusters': num_clusters,
            'num_noise': num_noise,
        }
    
    def _apply_3d_nms(self, objects, distance_threshold=0.3):
        """3D NMS"""
        if len(objects) <= 1:
            return objects
        
        objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        for obj in objects:
            should_keep = True
            for kept in keep:
                dist = np.linalg.norm(obj['position'] - kept['position'])
                if dist < distance_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(obj)
        
        return keep


# ============================================================================
# 方法2: 心智地图方法 (简单帧间最大计数)
# ============================================================================

class Method2_MindMap:
    """
    心智地图方法
    
    核心流程:
    1. 帧采样
    2. GroundingDINO 检测 (固定阈值)
    3. 按帧分组
    4. 取每帧检测数量的最大值作为计数
    
    问题: 没有 3D 投影、没有聚类去重、没有自适应阈值
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.num_frames = 15
        self.threshold = 0.25  # 心智地图使用的固定阈值
        
        self._grounding_model = None
        self._grounding_processor = None
        
    def _load_grounding(self):
        if self._grounding_model is None:
            print(f"[{self.device}] Loading GroundingDINO for MindMap...")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            self._grounding_processor = AutoProcessor.from_pretrained(model_id)
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self._grounding_model = self._grounding_model.to(self.device)
            
    def count_objects(self, video_path: str, target_object: str) -> Dict:
        """计数物体"""
        start_time = time.time()
        
        # 1. 提取帧
        frames = self._extract_frames(video_path)
        if not frames:
            return {'count': 0, 'num_detections': 0, 'error': 'No frames'}
        
        # 2. 检测
        self._load_grounding()
        detections_per_frame = self._detect_objects(frames, target_object)
        
        # 3. 取每帧最大检测数
        max_count = max(detections_per_frame) if detections_per_frame else 0
        total_detections = sum(detections_per_frame)
        
        elapsed = time.time() - start_time
        
        return {
            'count': max_count,
            'num_detections': total_detections,
            'detections_per_frame': detections_per_frame,
            'time': elapsed,
        }
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """提取帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def _detect_objects(self, frames: List[np.ndarray], target_object: str) -> List[int]:
        """检测并返回每帧的检测数"""
        from PIL import Image
        
        text_prompt = f"{target_object} ."
        detections_per_frame = []
        
        for frame in frames:
            image = Image.fromarray(frame)
            
            inputs = self._grounding_processor(images=image, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._grounding_model(**inputs)
            
            results = self._grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.threshold,
                text_threshold=0.25,
                target_sizes=[image.size[::-1]]
            )
            
            detections_per_frame.append(len(results[0]["scores"]))
        
        return detections_per_frame


# ============================================================================
# 方法3: 心智地图 + 3D去重 (改进的心智地图方法)
# ============================================================================

class Method3_MindMap_3D:
    """
    改进的心智地图方法: 加入 3D 投影和聚类
    
    这应该能达到接近 Method1 的效果
    """
    
    def __init__(self, device: str = 'cuda', distance_threshold: float = 0.5):
        self.device = device
        self.distance_threshold = distance_threshold
        self.num_frames = 16
        
        self._da3_model = None
        self._grounding_model = None
        self._grounding_processor = None
        
    def _load_da3(self):
        if self._da3_model is None:
            print(f"[{self.device}] Loading DA3...")
            from depth_anything_3.api import DepthAnything3
            self._da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
            self._da3_model = self._da3_model.to(self.device).eval()
            
    def _load_grounding(self):
        if self._grounding_model is None:
            print(f"[{self.device}] Loading GroundingDINO for MindMap+3D...")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-base"
            self._grounding_processor = AutoProcessor.from_pretrained(model_id)
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self._grounding_model = self._grounding_model.to(self.device)
    
    def count_objects(self, video_path: str, target_object: str) -> Dict:
        """计数物体 - 使用3D聚类"""
        # 复用 Method1 的逻辑，但使用自适应阈值
        method1 = Method1_DINO_DA3(device=self.device, distance_threshold=self.distance_threshold)
        return method1.count_objects(video_path, target_object)


# ============================================================================
# 单 GPU Worker
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], method: str, result_queue: mp.Queue):
    """单 GPU 处理 worker"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda'
    
    print(f"[GPU {gpu_id}] Starting with {len(samples)} samples, method={method}")
    
    # 初始化方法
    if method == 'dino_da3':
        evaluator = Method1_DINO_DA3(device=device)
    elif method == 'mindmap':
        evaluator = Method2_MindMap(device=device)
    elif method == 'mindmap_3d':
        evaluator = Method3_MindMap_3D(device=device)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    results = []
    
    for i, sample in enumerate(samples):
        video_path = sample['video_path']
        target_object = sample['target_object']
        gt = sample['ground_truth']
        
        try:
            result = evaluator.count_objects(video_path, target_object)
            pred = result['count']
            mra = mean_relative_accuracy(pred, gt)
            
            results.append({
                'sample_id': sample['id'],
                'video_path': video_path,
                'target_object': target_object,
                'ground_truth': gt,
                'prediction': pred,
                'mra': mra,
                'num_detections': result.get('num_detections', 0),
                'time': result.get('time', 0),
            })
            
            if (i + 1) % 10 == 0:
                avg_mra = np.mean([r['mra'] for r in results])
                print(f"[GPU {gpu_id}] Progress: {i+1}/{len(samples)}, Avg MRA: {avg_mra:.4f}")
                
        except Exception as e:
            print(f"[GPU {gpu_id}] Error on sample {sample['id']}: {e}")
            results.append({
                'sample_id': sample['id'],
                'error': str(e),
                'mra': 0.0,
            })
    
    result_queue.put((gpu_id, results))
    print(f"[GPU {gpu_id}] Finished")


# ============================================================================
# 主测试类
# ============================================================================

class FairComparisonTester:
    """公平对比测试器"""
    
    def __init__(self, num_gpus: int = 8):
        self.num_gpus = num_gpus
        self.output_dir = str(PROJECT_ROOT / "outputs" / f"fair_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.dataset = None
        
    def load_dataset(self):
        """加载 VSIBench 数据集"""
        print("Loading VSIBench dataset...")
        from datasets import load_dataset
        
        self.dataset = load_dataset(
            "nyu-visionx/VSI-Bench",
            split="test",
            cache_dir="/home/tione/notebook/tianjungu/hf_cache/vsibench"
        )
        print(f"Dataset size: {len(self.dataset)}")
        
    def get_counting_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """获取 object_counting 样本"""
        samples = []
        
        # 视频路径配置
        video_dirs = {
            'ARKitScenes': '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
            'ScanNet': '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
            'HM3D': '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/hm3d',
        }
        
        for item in self.dataset:
            if item['question_type'] != 'object_counting':
                continue
            
            scene_name = item['scene_name']
            
            # 判断来源
            if scene_name.startswith('scene'):
                source = 'ScanNet'
            elif scene_name.isdigit():
                source = 'ARKitScenes'
            else:
                source = 'HM3D'
            
            video_path = os.path.join(video_dirs[source], f"{scene_name}.mp4")
            
            if not os.path.exists(video_path):
                continue
            
            # 从问题中提取目标物体
            question = item['question']
            match = re.search(r'How many (\w+)\(s\)', question)
            if not match:
                continue
            
            target_object = match.group(1)
            gt = normalize_number(item['ground_truth'])
            
            if gt is None:
                continue
            
            samples.append({
                'id': item['id'],
                'scene_name': scene_name,
                'source': source,
                'video_path': video_path,
                'question': question,
                'target_object': target_object,
                'ground_truth': gt,
            })
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"Found {len(samples)} object_counting samples")
        return samples
    
    def run_comparison(self, samples: List[Dict], methods: List[str] = None):
        """运行对比测试"""
        if methods is None:
            methods = ['dino_da3', 'mindmap']
        
        all_results = {}
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Testing method: {method}")
            print(f"{'='*60}")
            
            # 分配样本到各 GPU
            samples_per_gpu = len(samples) // self.num_gpus
            gpu_samples = []
            for i in range(self.num_gpus):
                start = i * samples_per_gpu
                end = start + samples_per_gpu if i < self.num_gpus - 1 else len(samples)
                gpu_samples.append(samples[start:end])
            
            # 启动多进程
            result_queue = mp.Queue()
            processes = []
            
            for gpu_id, gpu_sample_list in enumerate(gpu_samples):
                if not gpu_sample_list:
                    continue
                p = mp.Process(target=worker_process, args=(gpu_id, gpu_sample_list, method, result_queue))
                p.start()
                processes.append(p)
            
            # 收集结果
            gpu_results = {}
            for _ in processes:
                gpu_id, results = result_queue.get()
                gpu_results[gpu_id] = results
            
            # 等待所有进程结束
            for p in processes:
                p.join()
            
            # 合并结果
            method_results = []
            for gpu_id in sorted(gpu_results.keys()):
                method_results.extend(gpu_results[gpu_id])
            
            all_results[method] = method_results
            
            # 计算统计
            mras = [r['mra'] for r in method_results if 'mra' in r]
            avg_mra = np.mean(mras) if mras else 0
            
            print(f"\n{method} Results:")
            print(f"  Samples: {len(method_results)}")
            print(f"  MRA: {avg_mra:.4f}")
        
        # 保存结果
        self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, all_results: Dict):
        """保存结果"""
        # 详细结果
        output_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 汇总报告
        summary = {
            'timestamp': datetime.now().isoformat(),
            'methods': {}
        }
        
        for method, results in all_results.items():
            mras = [r['mra'] for r in results if 'mra' in r]
            
            summary['methods'][method] = {
                'total_samples': len(results),
                'avg_mra': np.mean(mras) if mras else 0,
                'std_mra': np.std(mras) if mras else 0,
                'over_counting': sum(1 for r in results if r.get('prediction', 0) > r.get('ground_truth', 0)),
                'under_counting': sum(1 for r in results if r.get('prediction', 0) < r.get('ground_truth', 0)),
                'exact_match': sum(1 for r in results if r.get('prediction', 0) == r.get('ground_truth', 0)),
            }
        
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印对比报告
        self._print_report(summary)
        
    def _print_report(self, summary: Dict):
        """打印报告"""
        print("\n" + "="*80)
        print("Fair Comparison Results")
        print("="*80)
        
        for method, stats in summary['methods'].items():
            print(f"\n{method}:")
            print(f"  MRA: {stats['avg_mra']:.4f} ± {stats['std_mra']:.4f}")
            print(f"  Over-counting: {stats['over_counting']} ({stats['over_counting']/stats['total_samples']*100:.1f}%)")
            print(f"  Under-counting: {stats['under_counting']} ({stats['under_counting']/stats['total_samples']*100:.1f}%)")
            print(f"  Exact match: {stats['exact_match']} ({stats['exact_match']/stats['total_samples']*100:.1f}%)")
        
        print(f"\nResults saved to: {self.output_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fair Comparison Test')
    parser.add_argument('--num-gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to test')
    parser.add_argument('--methods', type=str, default='dino_da3,mindmap', help='Methods to compare')
    args = parser.parse_args()
    
    methods = args.methods.split(',')
    
    tester = FairComparisonTester(num_gpus=args.num_gpus)
    tester.load_dataset()
    
    samples = tester.get_counting_samples(max_samples=args.max_samples)
    
    if not samples:
        print("No samples found!")
        return
    
    tester.run_comparison(samples, methods)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
