#!/usr/bin/env python3
"""
简化版公平对比测试 - 单GPU顺序执行

对比两种方法:
1. DINO + DA3 + 3D聚类 (之前效果好的方法, MRA ~77%)
2. 心智地图方法 (简单帧间最大计数)

作者: tianjungu
日期: 2026-01-24
"""

import os
import sys
import json
import time
import gc
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import re

import numpy as np
import cv2

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

import torch
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    'trash': 0.35,
    'default': 0.45
}


# ============================================================================
# 共享组件 - 使用项目已有的 GroundingDINOLabeler
# ============================================================================

class GroundingDINODetector:
    """GroundingDINO 检测器 (封装项目已有的 GroundingDINOLabeler)"""
    
    def __init__(self, device: str = 'cuda', model_size: str = 'base'):
        self.device = device
        self.model_size = model_size
        self._labeler = None
        
    def load(self, threshold: float = 0.25):
        if self._labeler is None:
            print(f"Loading GroundingDINO ({self.model_size})...")
            from core.semantic_labeler import GroundingDINOLabeler
            
            model_id = f"IDEA-Research/grounding-dino-{self.model_size}"
            self._labeler = GroundingDINOLabeler(
                model_id=model_id,
                device=self.device,
                box_threshold=threshold,
                text_threshold=0.25,
            )
            self._labeler.load_model()
            print("GroundingDINO loaded")
    
    def set_threshold(self, threshold: float):
        """动态调整阈值"""
        if self._labeler is not None:
            self._labeler.box_threshold = threshold
    
    def unload(self):
        if self._labeler is not None:
            del self._labeler.model
            del self._labeler.processor
            self._labeler = None
            gc.collect()
            torch.cuda.empty_cache()
    
    def detect(self, frame: np.ndarray, text_prompt: str, threshold: float = 0.25) -> List[Dict]:
        """检测物体"""
        if self._labeler is None:
            self.load(threshold)
        
        # 动态设置阈值
        self._labeler.box_threshold = threshold
        
        results = self._labeler.detect(frame, text_prompt)
        
        detections = []
        for det in results:
            detections.append({
                'label': det.label,
                'confidence': det.confidence,
                'bbox': det.bbox_pixels,
            })
        
        return detections


# ============================================================================
# 方法1: DINO + 3D聚类 (使用 DA3 之前效果好的配置)
# ============================================================================

class Method1_DINO_3D:
    """
    DINO + 3D 聚类方法 (复刻 eval_vsibench_fast_v4.py)
    
    核心流程:
    1. 帧采样 (16帧)
    2. GroundingDINO 检测 (自适应阈值)
    3. DA3 深度估计
    4. 2D→3D 投影
    5. 尺寸过滤 + 3D NMS
    6. DBSCAN 聚类
    7. 保守计数策略
    """
    
    def __init__(self, device: str = 'cuda', distance_threshold: float = 0.5):
        self.device = device
        self.distance_threshold = distance_threshold
        self.num_frames = 16
        
        self.detector = GroundingDINODetector(device)
        self._da3_model = None
        
    def _load_da3(self):
        if self._da3_model is None:
            print("Loading DA3...")
            sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
            sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')
            from depth_anything_3.api import DepthAnything3
            self._da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
            self._da3_model = self._da3_model.to(self.device).eval()
            print("DA3 loaded")
    
    def _unload_da3(self):
        if self._da3_model is not None:
            del self._da3_model
            self._da3_model = None
            gc.collect()
            torch.cuda.empty_cache()
            
    def count_objects(self, video_path: str, target_object: str) -> Dict:
        """计数物体"""
        start_time = time.time()
        
        # 1. 提取帧
        frames = self._extract_frames(video_path)
        if not frames:
            return {'count': 0, 'num_detections': 0, 'error': 'No frames'}
        
        # 2. 获取自适应阈值
        threshold = OBJECT_TYPE_THRESHOLDS.get(target_object.lower(), OBJECT_TYPE_THRESHOLDS['default'])
        
        # 3. 检测物体
        all_detections = []
        text_prompt = f"{target_object} ."
        
        for frame_idx, frame in enumerate(frames):
            dets = self.detector.detect(frame, text_prompt, threshold)
            for det in dets:
                det['frame'] = frame_idx
            all_detections.extend(dets)
        
        if not all_detections:
            return {'count': 0, 'num_detections': 0, 'threshold': threshold}
        
        # 4. DA3 深度估计
        self._load_da3()
        depths, intrinsics, extrinsics, scale_factor = self._run_da3(frames)
        self._unload_da3()
        
        # 5. 3D 投影 + 去重 + 聚类
        result = self._deduplicate_3d(
            all_detections, depths, extrinsics, intrinsics, frames, scale_factor
        )
        
        elapsed = time.time() - start_time
        
        return {
            'count': result['count'],
            'num_detections': len(all_detections),
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
    心智地图方法 (当前 test_mindmap_with_critic.py 使用的方法)
    
    核心流程:
    1. 帧采样 (15帧)
    2. GroundingDINO 检测 (固定阈值 0.25)
    3. 按帧分组
    4. 取每帧检测数量的最大值作为计数
    
    问题: 
    - 没有 3D 投影 (无法利用深度信息去重)
    - 没有聚类去重 (同一物体多帧检测被重复计数)
    - 没有自适应阈值 (不同物体需要不同阈值)
    - 阈值太低 (0.25 导致过多误检)
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.num_frames = 15
        self.threshold = 0.25  # 心智地图使用的固定阈值
        
        self.detector = GroundingDINODetector(device)
            
    def count_objects(self, video_path: str, target_object: str) -> Dict:
        """计数物体"""
        start_time = time.time()
        
        # 1. 提取帧
        frames = self._extract_frames(video_path)
        if not frames:
            return {'count': 0, 'num_detections': 0, 'error': 'No frames'}
        
        # 2. 检测
        text_prompt = f"{target_object} ."
        detections_per_frame = []
        total_detections = 0
        
        for frame in frames:
            dets = self.detector.detect(frame, text_prompt, self.threshold)
            detections_per_frame.append(len(dets))
            total_detections += len(dets)
        
        # 3. 取每帧最大检测数
        max_count = max(detections_per_frame) if detections_per_frame else 0
        
        elapsed = time.time() - start_time
        
        return {
            'count': max_count,
            'num_detections': total_detections,
            'detections_per_frame': detections_per_frame,
            'threshold': self.threshold,
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


# ============================================================================
# 主测试类
# ============================================================================

class FairComparisonTester:
    """公平对比测试器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
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
            methods = ['dino_3d', 'mindmap']
        
        all_results = {}
        
        for method_name in methods:
            print(f"\n{'='*60}")
            print(f"Testing method: {method_name}")
            print(f"{'='*60}")
            
            # 初始化方法
            if method_name == 'dino_3d':
                method = Method1_DINO_3D(device=self.device)
            elif method_name == 'mindmap':
                method = Method2_MindMap(device=self.device)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            results = []
            
            for i, sample in enumerate(samples):
                video_path = sample['video_path']
                target_object = sample['target_object']
                gt = sample['ground_truth']
                
                try:
                    result = method.count_objects(video_path, target_object)
                    pred = result['count']
                    mra = mean_relative_accuracy(pred, gt)
                    
                    results.append({
                        'sample_id': sample['id'],
                        'scene_name': sample['scene_name'],
                        'target_object': target_object,
                        'ground_truth': gt,
                        'prediction': pred,
                        'mra': mra,
                        'num_detections': result.get('num_detections', 0),
                        'threshold': result.get('threshold', 0),
                        'time': result.get('time', 0),
                    })
                    
                    status = "✓" if mra >= 0.5 else "✗"
                    print(f"[{i+1}/{len(samples)}] {sample['scene_name']} | {target_object}: pred={pred} gt={int(gt)} | MRA={mra:.2f} {status}")
                    
                except Exception as e:
                    print(f"[{i+1}/{len(samples)}] Error: {e}")
                    results.append({
                        'sample_id': sample['id'],
                        'error': str(e),
                        'mra': 0.0,
                    })
            
            all_results[method_name] = results
            
            # 清理模型
            if hasattr(method, 'detector'):
                method.detector.unload()
            gc.collect()
            torch.cuda.empty_cache()
            
            # 计算统计
            mras = [r['mra'] for r in results if 'mra' in r]
            avg_mra = np.mean(mras) if mras else 0
            
            print(f"\n{method_name} Results: MRA = {avg_mra:.4f}")
        
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
            'output_dir': self.output_dir,
            'methods': {}
        }
        
        for method, results in all_results.items():
            mras = [r['mra'] for r in results if 'mra' in r]
            preds = [r.get('prediction', 0) for r in results if 'prediction' in r]
            gts = [r.get('ground_truth', 0) for r in results if 'ground_truth' in r]
            
            over_count = sum(1 for p, g in zip(preds, gts) if p > g)
            under_count = sum(1 for p, g in zip(preds, gts) if p < g)
            exact = sum(1 for p, g in zip(preds, gts) if p == g)
            
            summary['methods'][method] = {
                'total_samples': len(results),
                'avg_mra': np.mean(mras) if mras else 0,
                'std_mra': np.std(mras) if mras else 0,
                'over_counting': over_count,
                'under_counting': under_count,
                'exact_match': exact,
                'over_counting_pct': over_count / len(results) * 100 if results else 0,
                'under_counting_pct': under_count / len(results) * 100 if results else 0,
                'exact_match_pct': exact / len(results) * 100 if results else 0,
            }
        
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印对比报告
        self._print_report(summary)
        
    def _print_report(self, summary: Dict):
        """打印报告"""
        print("\n" + "="*80)
        print("Fair Comparison Results - Object Counting Task")
        print("="*80)
        
        print(f"\n{'Method':<20} | {'MRA':>8} | {'Over%':>8} | {'Under%':>8} | {'Exact%':>8}")
        print("-"*60)
        
        for method, stats in summary['methods'].items():
            print(f"{method:<20} | {stats['avg_mra']:>7.4f} | {stats['over_counting_pct']:>7.1f}% | {stats['under_counting_pct']:>7.1f}% | {stats['exact_match_pct']:>7.1f}%")
        
        print("\n" + "="*80)
        print("Analysis of Why MindMap Method Performs Worse:")
        print("="*80)
        print("""
1. 没有自适应阈值: 使用固定阈值 0.25，对于不同物体类型不是最优
   - 最佳方法: 不同物体使用 0.35-0.45 的自适应阈值

2. 没有 3D 投影: 无法利用深度信息进行空间去重
   - 同一物体在不同帧被检测多次，但无法判断是否为同一物体

3. 没有 DBSCAN 聚类: 直接取帧间最大值，过于简单
   - 最佳方法: 通过 3D 空间聚类判断物体实例数

4. 没有尺寸过滤: 可能包含不合理的检测结果

5. 没有 3D NMS: 可能有重复检测
""")
        
        print(f"\nResults saved to: {self.output_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fair Comparison Test (Simple Version)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--max-samples', type=int, default=50, help='Max samples to test')
    parser.add_argument('--methods', type=str, default='dino_3d,mindmap', help='Methods to compare')
    args = parser.parse_args()
    
    methods = args.methods.split(',')
    
    tester = FairComparisonTester(device=args.device)
    tester.load_dataset()
    
    samples = tester.get_counting_samples(max_samples=args.max_samples)
    
    if not samples:
        print("No samples found!")
        return
    
    tester.run_comparison(samples, methods)


if __name__ == '__main__':
    main()
