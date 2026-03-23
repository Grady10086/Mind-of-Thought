#!/usr/bin/env python3
"""
VSIBench 测试 - 心智地图 + DA3 深度估计 (真正的 3D 重建)

不依赖 GT meta_info，完全通过检测 + 深度估计获取 3D 信息

关键改进：
1. GroundingDINO 检测物体 → 2D bbox
2. DA3 深度估计 → 深度图
3. 2D bbox + 深度 → 3D 位置和尺寸
4. 基于 3D 信息回答所有问题

作者: tianjungu
日期: 2026-01-26
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
from dataclasses import dataclass, field
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


# ============================================================================
# VSIBench 评测指标
# ============================================================================

def normalize_number(text: str) -> Optional[float]:
    """提取数字"""
    if text is None:
        return None
    text = re.sub(r'(meters?|m|cm|feet|ft|inches?|in|square)\b', '', text.lower())
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


NUMERICAL_TASKS = {'object_counting', 'object_size_estimation', 'object_abs_distance', 'room_size_estimation'}
CHOICE_TASKS = {'object_rel_distance', 'object_rel_direction_easy', 'object_rel_direction_medium', 
                'object_rel_direction_hard', 'obj_appearance_order', 'route_planning'}


# ============================================================================
# 3D 心智地图实体
# ============================================================================

@dataclass
class MindMapEntity3D:
    """带 3D 信息的心智地图实体"""
    entity_id: str
    label: str
    
    # 2D 检测信息
    detections_2d: List[Dict] = field(default_factory=list)
    count: int = 1
    avg_confidence: float = 0.0
    first_seen_frame: int = 0
    
    # 3D 重建信息 (来自深度估计)
    position_3d: Optional[np.ndarray] = None  # (3,) 中心位置
    size_3d: Optional[np.ndarray] = None  # (3,) 尺寸 (宽,高,深)
    depth_median: float = 0.0  # 中值深度
    depth_range: Tuple[float, float] = (0.0, 0.0)  # 深度范围
    
    def to_dict(self) -> Dict:
        return {
            'id': self.entity_id,
            'label': self.label,
            'count': self.count,
            'confidence': round(self.avg_confidence, 3),
            'first_frame': self.first_seen_frame,
            'position_3d': self.position_3d.tolist() if self.position_3d is not None else None,
            'size_3d': self.size_3d.tolist() if self.size_3d is not None else None,
            'depth_median': round(self.depth_median, 3),
        }


# ============================================================================
# 心智地图构建器 (集成深度)
# ============================================================================

class MindMapBuilderWithDepth:
    """集成 DA3 深度估计的心智地图构建器"""
    
    # 典型物体尺寸先验 (米)，用于深度尺度校准
    OBJECT_SIZE_PRIOR = {
        'chair': 0.5, 'table': 1.0, 'sofa': 1.8, 'couch': 1.8,
        'bed': 2.0, 'tv': 1.0, 'television': 1.0, 'monitor': 0.5,
        'door': 2.0, 'window': 1.2, 'refrigerator': 1.8, 'fridge': 1.8,
        'toilet': 0.6, 'sink': 0.6, 'bathtub': 1.5, 'cabinet': 1.0,
        'desk': 1.2, 'shelf': 1.0, 'lamp': 0.5, 'pillow': 0.4,
        'microwave': 0.5, 'stove': 0.7, 'oven': 0.7,
        'trash can': 0.4, 'trash bin': 0.4, 'plant': 0.5,
        'nightstand': 0.5, 'closet': 1.5, 'washer': 0.6,
    }
    
    def __init__(self, device: str = 'cuda', num_frames: int = 16, 
                 box_threshold: float = 0.25, use_metric_depth: bool = True):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        self.use_metric_depth = use_metric_depth
        
        self._labeler = None
        self._depth_estimator = None
        
        # 相机内参估计 (假设标准相机)
        self.focal_length = 500  # 像素
        self.principal_point = None  # 将在推理时设置
        
    def _load_models(self):
        """加载检测器和深度估计器"""
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
            logger.info("GroundingDINO 加载完成")
        
        if self._depth_estimator is None:
            from core.perception import DepthEstimator
            self._depth_estimator = DepthEstimator(
                model_name="depth-anything/DA3-Large",
                device=self.device,
                half_precision=True,
            )
            logger.info("DA3 深度估计器加载完成")
    
    def unload(self):
        """释放模型内存"""
        import torch
        
        if self._labeler is not None:
            del self._labeler.model
            del self._labeler.processor
            self._labeler = None
            
        if self._depth_estimator is not None:
            del self._depth_estimator.model
            self._depth_estimator = None
            
        gc.collect()
        torch.cuda.empty_cache()
    
    def build_from_video(self, video_path: str, target_objects: List[str] = None) -> Dict[str, MindMapEntity3D]:
        """从视频构建带 3D 信息的心智地图"""
        self._load_models()
        
        # 提取视频帧
        frames = self._extract_frames(video_path)
        if not frames:
            return {}
        
        H, W = frames[0].shape[:2]
        self.principal_point = (W / 2, H / 2)
        
        # 准备检测词汇
        if target_objects is None:
            target_objects = []
        
        extended_vocab = list(set(target_objects + [
            "chair", "table", "sofa", "couch", "stove", "tv", "television",
            "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator",
            "sink", "toilet", "bathtub", "door", "window", "picture",
            "pillow", "cushion", "monitor", "backpack", "bag",
            "trash can", "trash bin", "mirror", "towel", "plant",
            "nightstand", "closet", "microwave", "printer", "washer",
        ]))
        text_prompt = " . ".join(extended_vocab) + " ."
        
        # 采样帧
        num_sample = min(self.num_frames, len(frames))
        frame_indices = np.linspace(0, len(frames) - 1, num_sample).astype(int)
        sampled_frames = [frames[i] for i in frame_indices]
        
        # 批量深度估计
        logger.info(f"深度估计 {len(sampled_frames)} 帧...")
        depth_maps = self._estimate_depth_batch(sampled_frames)
        
        # 逐帧检测并关联深度
        all_detections: Dict[str, List[Dict]] = defaultdict(list)
        
        for frame_idx, (frame, depth_map) in enumerate(zip(sampled_frames, depth_maps)):
            original_frame_idx = frame_indices[frame_idx]
            
            # 检测物体
            results = self._labeler.detect(frame, text_prompt)
            
            for det in results:
                # DetectionResult 是 dataclass，使用属性访问
                label = det.label.lower()
                bbox = det.bbox_pixels  # [x1, y1, x2, y2] in pixels
                confidence = det.confidence
                
                # 获取 bbox 内的深度信息
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                depth_roi = depth_map[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                
                # 计算深度统计
                depth_median = float(np.median(depth_roi))
                depth_min = float(np.min(depth_roi))
                depth_max = float(np.max(depth_roi))
                
                # 估算 3D 位置
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                position_3d = self._pixel_to_3d(cx, cy, depth_median, W, H)
                
                # 估算 3D 尺寸
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                size_3d = self._estimate_3d_size(bbox_width, bbox_height, depth_median, label, W, H)
                
                all_detections[label].append({
                    'frame_idx': original_frame_idx,
                    'bbox': bbox,
                    'confidence': confidence,
                    'depth_median': depth_median,
                    'depth_range': (depth_min, depth_max),
                    'position_3d': position_3d,
                    'size_3d': size_3d,
                })
        
        # 聚合为实体
        entities = self._aggregate_to_entities(all_detections)
        
        return entities
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """提取视频帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    
    def _estimate_depth_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """批量深度估计"""
        import torch
        
        depth_maps = []
        for frame in frames:
            try:
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                # depth_tensor: (1, H, W)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                
                # 如果需要 metric depth，进行尺度校准
                if self.use_metric_depth:
                    # DA3 输出的是相对深度，需要校准
                    # 简单方法：假设中值深度为 2.5 米
                    median_depth = np.median(depth_map)
                    if median_depth > 0:
                        scale = 2.5 / median_depth
                        depth_map = depth_map * scale
                
                depth_maps.append(depth_map)
            except Exception as e:
                logger.warning(f"深度估计失败: {e}")
                # 返回默认深度图
                depth_maps.append(np.ones(frame.shape[:2], dtype=np.float32) * 2.5)
        
        return depth_maps
    
    def _pixel_to_3d(self, px: float, py: float, depth: float, W: int, H: int) -> np.ndarray:
        """像素坐标 + 深度 → 3D 坐标"""
        cx, cy = W / 2, H / 2
        fx = fy = min(W, H)  # 假设 focal length 约等于图像尺寸
        
        x = (px - cx) * depth / fx
        y = (py - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])
    
    def _estimate_3d_size(self, bbox_w: float, bbox_h: float, depth: float, 
                          label: str, W: int, H: int) -> np.ndarray:
        """估算物体 3D 尺寸"""
        fx = fy = min(W, H)
        
        # 从 2D bbox 尺寸 + 深度估算 3D 尺寸
        width_3d = bbox_w * depth / fx
        height_3d = bbox_h * depth / fy
        
        # 深度方向尺寸：使用先验或假设为宽度的一半
        prior_size = self.OBJECT_SIZE_PRIOR.get(label, 0.5)
        depth_3d = min(width_3d, prior_size)  # 保守估计
        
        return np.array([width_3d, height_3d, depth_3d])
    
    def _aggregate_to_entities(self, all_detections: Dict[str, List[Dict]]) -> Dict[str, MindMapEntity3D]:
        """将检测结果聚合为实体"""
        # 标签标准化映射
        label_to_category = {
            'chair': 'chair', 'seat': 'chair', 'armchair': 'chair',
            'table': 'table', 'dining table': 'table', 'coffee table': 'table',
            'sofa': 'sofa', 'couch': 'sofa',
            'stove': 'stove', 'oven': 'stove',
            'tv': 'tv', 'television': 'tv',
            'monitor': 'monitor', 'screen': 'monitor',
            'pillow': 'pillow', 'cushion': 'pillow',
            'trash can': 'trash bin', 'trash bin': 'trash bin', 'garbage can': 'trash bin',
            'nightstand': 'nightstand', 'bedside table': 'nightstand',
        }
        
        category_detections: Dict[str, List[Dict]] = defaultdict(list)
        for label, dets in all_detections.items():
            category = label
            for key, cat in label_to_category.items():
                if key in label.lower():
                    category = cat
                    break
            category_detections[category].extend(dets)
        
        entities = {}
        for category, dets in category_detections.items():
            if not dets:
                continue
            
            # 按帧分组
            frame_dets = defaultdict(list)
            for det in dets:
                frame_dets[det['frame_idx']].append(det)
            
            # 帧间最大计数
            max_count = max(len(fd) for fd in frame_dets.values())
            avg_confidence = np.mean([d['confidence'] for d in dets])
            first_frame = min(d['frame_idx'] for d in dets)
            
            # 聚合 3D 信息 (使用所有检测的中位数)
            positions = np.array([d['position_3d'] for d in dets])
            sizes = np.array([d['size_3d'] for d in dets])
            depths = [d['depth_median'] for d in dets]
            
            median_position = np.median(positions, axis=0)
            median_size = np.median(sizes, axis=0)
            median_depth = float(np.median(depths))
            
            entity = MindMapEntity3D(
                entity_id=f"entity_{category}",
                label=category,
                detections_2d=dets,
                count=max_count,
                avg_confidence=float(avg_confidence),
                first_seen_frame=first_frame,
                position_3d=median_position,
                size_3d=median_size,
                depth_median=median_depth,
                depth_range=(min(depths), max(depths)),
            )
            entities[category] = entity
        
        return entities


# ============================================================================
# 问题回答器 (基于 3D 心智地图，不用 GT)
# ============================================================================

class MindMapQA3D:
    """基于 3D 心智地图回答问题 (不依赖 GT)"""
    
    @staticmethod
    def answer_counting(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        """回答计数问题"""
        match = re.search(r'How many (\w+)\(s\)', question)
        if not match:
            match = re.search(r'How many (\w+)', question)
        
        if not match:
            return "0"
        
        target = match.group(1).lower()
        
        for label, entity in mind_map.items():
            if target in label.lower() or label.lower() in target:
                return str(entity.count)
        
        return "0"
    
    @staticmethod
    def answer_object_size(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        """回答物体尺寸问题 (使用 3D 重建尺寸)"""
        q_lower = question.lower()
        
        for label, entity in mind_map.items():
            if label in q_lower:
                if entity.size_3d is not None:
                    # 返回最大尺寸 (厘米)
                    max_dim = float(np.max(entity.size_3d)) * 100
                    return str(int(max_dim))
        
        return "50"  # 默认值
    
    @staticmethod
    def answer_room_size(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        """估算房间面积 (基于物体分布)"""
        if not mind_map:
            return "25"
        
        # 收集所有物体的 3D 位置
        positions = []
        for entity in mind_map.values():
            if entity.position_3d is not None:
                positions.append(entity.position_3d)
        
        if len(positions) < 2:
            # 基于物体数量估计
            return str(15 + len(mind_map) * 2)
        
        positions = np.array(positions)
        
        # 计算 XY 平面的包围盒
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        # 房间面积 = 包围盒 * 扩展系数
        estimated_area = (x_range + 2) * (y_range + 2)  # +2 米边界
        estimated_area = max(10, min(100, estimated_area))  # 限制范围
        
        return f"{estimated_area:.1f}"
    
    @staticmethod
    def answer_abs_distance(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        """回答物体间距离问题 (使用 3D 位置)"""
        q_lower = question.lower()
        
        # 收集所有物体位置
        positions = {}
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                positions[label] = entity.position_3d
        
        # 找到问题中的两个物体
        objs_in_q = []
        for obj_name in positions.keys():
            if obj_name in q_lower:
                objs_in_q.append(obj_name)
        
        if len(objs_in_q) >= 2:
            dist = float(np.linalg.norm(positions[objs_in_q[0]] - positions[objs_in_q[1]]))
            return f"{dist:.1f}"
        
        # 如果只找到一个物体，估算到相机的距离
        if len(objs_in_q) == 1:
            pos = positions[objs_in_q[0]]
            dist = float(np.linalg.norm(pos))
            return f"{dist:.1f}"
        
        return "2.0"  # 默认值
    
    @staticmethod
    def answer_rel_direction(mind_map: Dict[str, MindMapEntity3D], question: str, 
                             options: List[str]) -> str:
        """回答相对方向问题"""
        if not options:
            return "left"
        
        q_lower = question.lower()
        
        # 收集物体位置
        positions = {}
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                positions[label] = entity.position_3d
        
        # 找到问题中的两个物体
        objs_in_q = []
        for obj_name in positions.keys():
            if obj_name in q_lower:
                objs_in_q.append(obj_name)
        
        if len(objs_in_q) >= 2:
            pos1 = positions[objs_in_q[0]]
            pos2 = positions[objs_in_q[1]]
            
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dz = pos2[2] - pos1[2]
            
            # 判断方向
            if abs(dx) > abs(dy) and abs(dx) > abs(dz):
                direction = "right" if dx > 0 else "left"
            elif abs(dy) > abs(dz):
                direction = "behind" if dy > 0 else "in front of"
            else:
                direction = "above" if dz > 0 else "below"
            
            # 在选项中找最匹配的
            for opt in options:
                if direction in opt.lower():
                    return opt
        
        return options[0]
    
    @staticmethod
    def answer_appearance_order(mind_map: Dict[str, MindMapEntity3D], question: str, 
                                 options: List[str]) -> str:
        """回答出现顺序问题"""
        if not options:
            return ""
        
        # 按 first_seen_frame 排序
        sorted_entities = sorted(
            mind_map.values(), 
            key=lambda e: e.first_seen_frame
        )
        
        # 提取标签顺序
        order = [e.label for e in sorted_entities]
        
        # 匹配选项
        for opt in options:
            opt_objs = re.findall(r'[A-Za-z]+', opt.lower())
            if len(opt_objs) >= 2:
                # 检查顺序是否正确
                try:
                    indices = [order.index(o) for o in opt_objs if o in order]
                    if indices == sorted(indices):
                        return opt
                except ValueError:
                    continue
        
        return options[0]
    
    @staticmethod
    def answer_rel_distance(mind_map: Dict[str, MindMapEntity3D], question: str, 
                            options: List[str]) -> str:
        """回答相对距离问题"""
        if not options:
            return ""
        
        q_lower = question.lower()
        
        # 收集物体深度
        depths = {}
        for label, entity in mind_map.items():
            depths[label] = entity.depth_median
        
        # 找到问题中的物体
        objs_in_q = [o for o in depths.keys() if o in q_lower]
        
        if len(objs_in_q) >= 2:
            d1, d2 = depths[objs_in_q[0]], depths[objs_in_q[1]]
            
            if "closer" in q_lower or "nearer" in q_lower:
                answer_obj = objs_in_q[0] if d1 < d2 else objs_in_q[1]
            else:  # farther
                answer_obj = objs_in_q[0] if d1 > d2 else objs_in_q[1]
            
            for opt in options:
                if answer_obj in opt.lower():
                    return opt
        
        return options[0]


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], result_queue: mp.Queue):
    """GPU Worker 进程"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    builder = MindMapBuilderWithDepth(device='cuda', num_frames=16, box_threshold=0.25)
    
    results = []
    total = len(samples)
    
    for i, sample in enumerate(samples):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 构建 3D 心智地图
            target_objects = []
            if question_type == 'object_counting':
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map = builder.build_from_video(video_path, target_objects)
            
            # 根据问题类型回答 (全部基于 3D 心智地图，不用 GT)
            if question_type == 'object_counting':
                pred = MindMapQA3D.answer_counting(mind_map, question)
            elif question_type == 'object_size_estimation':
                pred = MindMapQA3D.answer_object_size(mind_map, question)
            elif question_type == 'room_size_estimation':
                pred = MindMapQA3D.answer_room_size(mind_map, question)
            elif question_type == 'object_abs_distance':
                pred = MindMapQA3D.answer_abs_distance(mind_map, question)
            elif 'direction' in question_type:
                pred = MindMapQA3D.answer_rel_direction(mind_map, question, options)
            elif question_type == 'obj_appearance_order':
                pred = MindMapQA3D.answer_appearance_order(mind_map, question, options)
            elif question_type == 'object_rel_distance':
                pred = MindMapQA3D.answer_rel_distance(mind_map, question, options)
            else:
                pred = str(options[0]) if options else "0"
            
            # 计算指标
            if question_type in NUMERICAL_TASKS:
                pred_num = normalize_number(pred)
                gt_num = normalize_number(str(gt))
                score = mean_relative_accuracy(pred_num, gt_num) if pred_num is not None and gt_num is not None else 0.0
                correct = score > 0.5
            else:
                pred_lower = pred.lower().strip() if pred else ""
                gt_lower = str(gt).lower().strip() if gt else ""
                correct = pred_lower == gt_lower
                score = 1.0 if correct else 0.0
            
            result = {
                'sample_id': sample['id'],
                'question_type': question_type,
                'question': question,
                'prediction': pred,
                'ground_truth': gt,
                'score': score,
                'correct': correct,
            }
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} 样本 {sample['id']} 错误: {e}")
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
# 视频路径查找
# ============================================================================

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]


def find_video_path(scene_name: str) -> Optional[str]:
    """查找视频路径"""
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def get_scene_source(scene_name: str) -> str:
    """判断场景来源"""
    if scene_name.startswith('scene'):
        return 'ScanNet'
    elif scene_name.isdigit() or scene_name.startswith('4'):
        return 'ARKitScenes'
    return 'ScanNetPP'


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
    print("VSIBench 测试 - 心智地图 + DA3 深度 (真正的 3D 重建)")
    print("不依赖 GT meta_info，完全通过检测 + 深度估计")
    print(f"GPU数量: {args.num_gpus}")
    print("=" * 70)
    
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
    
    # 分配到各 GPU
    num_gpus = min(args.num_gpus, len(samples))
    samples_per_gpu = len(samples) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])
    
    print(f"每 GPU 样本数: ~{samples_per_gpu}")
    
    # 启动多进程
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    
    processes = []
    start_time = time.time()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_samples[gpu_id], result_queue))
        p.start()
        processes.append(p)
        print(f"启动 GPU {gpu_id} worker")
    
    # 收集结果
    all_results = []
    for _ in range(num_gpus):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        print(f"GPU {gpu_id} 完成，{len(results)} 个结果")
    
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}秒")
    
    # 统计结果
    task_results = defaultdict(list)
    for r in all_results:
        task_results[r['question_type']].append(r)
    
    print("\n" + "=" * 70)
    print("各任务结果 (基于 3D 心智地图，不依赖 GT)")
    print("=" * 70)
    
    summary = {'tasks': {}, 'total_samples': len(all_results)}
    
    for task, results in sorted(task_results.items()):
        if task in NUMERICAL_TASKS:
            scores = [r['score'] for r in results]
            avg_score = np.mean(scores) * 100
            metric = "MRA"
        else:
            correct = sum(1 for r in results if r.get('correct', False))
            avg_score = correct / len(results) * 100 if results else 0
            metric = "Accuracy"
        
        print(f"{task:35s}: {avg_score:6.2f}% {metric} ({len(results)} samples)")
        summary['tasks'][task] = {
            'score': avg_score,
            'metric': metric,
            'samples': len(results),
        }
    
    # 总体得分
    overall_scores = [r['score'] for r in all_results]
    overall_avg = np.mean(overall_scores) * 100
    print(f"\n{'Overall':35s}: {overall_avg:6.2f}%")
    summary['overall'] = overall_avg
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / f"mindmap_depth_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果保存到: {output_dir}")
