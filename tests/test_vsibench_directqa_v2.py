#!/usr/bin/env python3
"""
VSIBench 测试 - DirectQA V2 版本 (改进版心智地图)

改进点：
1. 动态尺度校准 - 利用已知物体尺寸
2. 实例级追踪 - 区分多个同类物体  
3. 概率性推理 - 支持不确定性估计

设计原则：
- 保守改进，确保不变差
- 新旧逻辑可切换对比

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
from dataclasses import dataclass, field
import re
import logging
import random

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

# 导入改进的心智地图模块
from core.mind_map_v3 import (
    Detection, CameraPose, TrackedInstance, ProbabilisticEntity,
    DynamicScaleCalibrator, InstanceTracker, ProbabilisticMindMap,
    KNOWN_OBJECT_SIZES, convert_to_legacy_mindmap
)


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
# 扩展词汇表
# ============================================================================

EXTENDED_VOCABULARY = [
    # 基础家具
    "chair", "table", "sofa", "couch", "bed", "desk", "shelf", "cabinet",
    "drawer", "closet", "wardrobe", "dresser", "nightstand", "bedside table",
    
    # 座椅类
    "stool", "bar stool", "step stool", "bench", "ottoman", "armchair",
    
    # 电器
    "tv", "television", "monitor", "screen", "display", "computer screen",
    "refrigerator", "fridge", "microwave", "stove", "oven", "washer",
    "dryer", "dishwasher", "air conditioner", "fan",
    
    # 采暖/照明
    "lamp", "light", "ceiling light", "heater", "radiator", "heating unit",
    
    # 浴室
    "toilet", "sink", "bathtub", "shower", "towel", "bath towel", "mirror",
    
    # 容器
    "trash can", "trash bin", "garbage bin", "bucket", "pail", "basket",
    "box", "container", "bin",
    
    # 小物品
    "cup", "mug", "bottle", "vase", "clock", "book", "pillow", "cushion",
    "blanket", "rug", "carpet", "mat", "curtain", "blind",
    
    # 装饰
    "picture", "painting", "poster", "frame", "plant", "flower",
    
    # 门窗
    "door", "window", "doorframe", "windowsill",
    
    # 办公
    "printer", "scanner", "keyboard", "mouse",
    
    # 其他
    "backpack", "bag", "suitcase", "shoes", "coat", "jacket",
    "counter", "countertop", "fireplace", "column", "beam", "wall",
]

# 同义词映射
SYNONYM_MAP = {
    'monitor': ['monitor', 'screen', 'display', 'computer screen'],
    'screen': ['monitor', 'screen', 'display', 'computer screen'],
    'display': ['monitor', 'screen', 'display', 'computer screen'],
    'heater': ['heater', 'radiator', 'heating unit'],
    'radiator': ['heater', 'radiator', 'heating unit'],
    'sofa': ['sofa', 'couch'],
    'couch': ['sofa', 'couch'],
    'tv': ['tv', 'television'],
    'television': ['tv', 'television'],
    'nightstand': ['nightstand', 'bedside table', 'night table', 'night stand'],
    'bedside table': ['nightstand', 'bedside table', 'night table'],
    'trash can': ['trash can', 'trash bin', 'garbage bin', 'bin'],
    'trash bin': ['trash can', 'trash bin', 'garbage bin', 'bin'],
    'lamp': ['lamp', 'light', 'ceiling light'],
    'light': ['lamp', 'light', 'ceiling light'],
    'ceiling light': ['lamp', 'light', 'ceiling light'],
    'refrigerator': ['refrigerator', 'fridge'],
    'fridge': ['refrigerator', 'fridge'],
    'stool': ['stool', 'bar stool', 'step stool'],
    'bucket': ['bucket', 'pail'],
    'basket': ['basket', 'bin'],
}


def get_synonyms(word: str) -> List[str]:
    """获取同义词列表"""
    word_lower = word.lower()
    return SYNONYM_MAP.get(word_lower, [word_lower])


def match_object_name(target: str, label: str) -> bool:
    """检查目标物体名称是否匹配"""
    target_lower = target.lower().strip()
    label_lower = label.lower().strip()
    
    if target_lower in label_lower or label_lower in target_lower:
        return True
    
    target_synonyms = get_synonyms(target_lower)
    label_synonyms = get_synonyms(label_lower)
    
    for ts in target_synonyms:
        for ls in label_synonyms:
            if ts in ls or ls in ts:
                return True
    
    target_words = set(target_lower.split())
    label_words = set(label_lower.split())
    if target_words & label_words:
        return True
    
    return False


# ============================================================================
# 改进版心智地图构建器
# ============================================================================

class MindMapBuilderV2:
    """心智地图构建器 V2 - 使用改进的模块"""
    
    def __init__(self, device: str = 'cuda', num_frames: int = 32,
                 box_threshold: float = 0.25, use_scale_calibration: bool = False,
                 use_instance_tracking: bool = True):
        """
        Args:
            use_scale_calibration: 是否使用动态尺度校准（默认关闭，因为可能影响 room_size）
            use_instance_tracking: 是否使用实例追踪（用于多实例位置查询）
        """
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        self.use_scale_calibration = use_scale_calibration
        self.use_instance_tracking = use_instance_tracking
        
        self._labeler = None
        self._depth_estimator = None
        
        # 相机内参
        self.focal_length = 500
        self.principal_point = None
        
        # 新模块
        self.scale_calibrator = DynamicScaleCalibrator(focal_length=self.focal_length)
        self.instance_tracker = InstanceTracker(distance_threshold=1.5)
        self.prob_mind_map = ProbabilisticMindMap()
    
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
        """提取视频帧"""
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
        """均匀采样帧"""
        num_sample = min(self.num_frames, len(frames))
        indices = np.linspace(0, len(frames) - 1, num_sample).astype(int)
        return [frames[i] for i in indices], indices.tolist()
    
    def _estimate_depth_with_calibration(self, frames: List[np.ndarray], 
                                          all_detections_per_frame: List[List[Detection]]) -> List[np.ndarray]:
        """带校准的深度估计"""
        depth_maps = []
        
        for i, frame in enumerate(frames):
            try:
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                
                # 总是先计算默认尺度
                median_depth = np.median(depth_map)
                default_scale = 2.5 / median_depth if median_depth > 0 else 1.0
                
                # 尝试动态尺度校准
                if self.use_scale_calibration and i < len(all_detections_per_frame):
                    calibrated_scale = self.scale_calibrator.calibrate_from_detections(
                        all_detections_per_frame[i], depth_map, frame.shape[0]
                    )
                    # 只有当校准成功（有历史记录）才使用校准尺度
                    if self.scale_calibrator.scale_history:
                        scale = calibrated_scale
                    else:
                        scale = default_scale
                else:
                    scale = default_scale
                
                depth_map = depth_map * scale
                depth_maps.append(depth_map)
                
            except Exception as e:
                logger.warning(f"深度估计失败: {e}")
                H, W = frame.shape[:2]
                depth_maps.append(np.ones((H, W), dtype=np.float32) * 2.5)
        
        return depth_maps
    
    def build_from_video(self, video_path: str, target_objects: List[str] = None) -> Dict:
        """
        从视频构建心智地图
        
        Returns:
            Dict[label, entity] - 兼容旧格式的心智地图
        """
        self._load_models()
        
        # 重置追踪器
        self.instance_tracker = InstanceTracker(distance_threshold=1.5)
        self.scale_calibrator = DynamicScaleCalibrator(focal_length=self.focal_length)
        
        # 提取和采样帧
        all_frames = self._extract_frames(video_path)
        if not all_frames:
            return {}
        
        frames, frame_indices = self._sample_frames(all_frames)
        H, W = frames[0].shape[:2]
        self.principal_point = (W / 2, H / 2)
        
        # 构建检测词汇表
        if target_objects is None:
            target_objects = []
        vocab = list(set(target_objects + EXTENDED_VOCABULARY))
        text_prompt = " . ".join(vocab) + " ."
        
        # 第一遍：检测所有帧（用于尺度校准）
        all_raw_detections: List[List[Detection]] = []
        
        for frame_idx, frame in enumerate(frames):
            results = self._labeler.detect(frame, text_prompt)
            frame_detections = []
            
            for det in results:
                detection = Detection(
                    frame_idx=frame_indices[frame_idx],
                    label=det.label.lower(),
                    bbox=tuple(int(v) for v in det.bbox_pixels),
                    confidence=det.confidence,
                )
                frame_detections.append(detection)
            
            all_raw_detections.append(frame_detections)
        
        # 深度估计（带校准）
        logger.info(f"深度估计 {len(frames)} 帧...")
        depth_maps = self._estimate_depth_with_calibration(frames, all_raw_detections)
        
        # 第二遍：计算 3D 位置并追踪
        for frame_idx, (frame, depth_map, raw_dets) in enumerate(zip(frames, depth_maps, all_raw_detections)):
            original_frame_idx = frame_indices[frame_idx]
            
            if depth_map.shape[:2] != (H, W):
                depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
            
            detections_with_3d = []
            
            for det in raw_dets:
                x1, y1, x2, y2 = det.bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 获取深度
                depth_roi = depth_map[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                
                depth_median = float(np.median(depth_roi))
                if depth_median <= 0:
                    continue
                
                # 计算 3D 位置（相机坐标系）
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                px, py = self.principal_point
                
                x_3d = (cx - px) / self.focal_length * depth_median
                y_3d = (cy - py) / self.focal_length * depth_median
                z_3d = depth_median
                
                # 计算 3D 尺寸
                width_pixels = x2 - x1
                height_pixels = y2 - y1
                
                width_3d = width_pixels / self.focal_length * depth_median
                height_3d = height_pixels / self.focal_length * depth_median
                depth_3d = min(width_3d, height_3d) * 0.5
                
                det.position_camera = np.array([x_3d, y_3d, z_3d])
                det.size_estimated = np.array([width_3d, height_3d, depth_3d])
                det.depth_median = depth_median
                
                detections_with_3d.append(det)
            
            # 更新追踪器
            if self.use_instance_tracking:
                self.instance_tracker.update(original_frame_idx, detections_with_3d)
        
        # 构建概率性心智地图
        self.prob_mind_map.build_from_tracker(self.instance_tracker)
        
        # 转换为旧格式（兼容）
        legacy_map = self._convert_to_legacy_format()
        
        return legacy_map
    
    def _convert_to_legacy_format(self) -> Dict:
        """转换为旧格式的心智地图"""
        @dataclass
        class LegacyEntity:
            entity_id: str
            label: str
            count: int
            avg_confidence: float
            first_seen_frame: int
            last_seen_frame: int
            position_3d: np.ndarray
            size_3d: np.ndarray
            depth_median: float
        
        legacy_map = {}
        
        # 按类别分组
        entities_by_label: Dict[str, List[ProbabilisticEntity]] = defaultdict(list)
        for entity in self.prob_mind_map.entities.values():
            entities_by_label[entity.label].append(entity)
        
        for label, entities in entities_by_label.items():
            # 计数：统计所有实例在各帧的检测数，取最大值
            # 这模拟原来的逻辑：单帧最大检测数
            frame_total_counts = defaultdict(int)
            
            for entity in entities:
                track = self.instance_tracker.active_tracks.get(entity.instance_id)
                if track:
                    for det in track.detections:
                        frame_total_counts[det.frame_idx] += 1
            
            if frame_total_counts:
                count = max(frame_total_counts.values())
            else:
                count = len(entities)  # 回退到实例数
            
            # 位置：如果只有 1 个实例用其位置，多个实例用最置信的
            if len(entities) == 1:
                best_entity = entities[0]
            else:
                best_entity = max(entities, key=lambda e: e.observation_count)
            
            position_3d = best_entity.position_mean
            size_3d = best_entity.size_mean
            
            # 置信度
            avg_conf = np.mean([e.existence_probability for e in entities])
            
            # 时序
            first_frame = min(e.first_seen_frame for e in entities)
            last_frame = max(e.last_seen_frame for e in entities)
            
            legacy_entity = LegacyEntity(
                entity_id=f"entity_{label}",
                label=label,
                count=count,
                avg_confidence=float(avg_conf),
                first_seen_frame=first_frame,
                last_seen_frame=last_frame,
                position_3d=position_3d,
                size_3d=size_3d if size_3d is not None else np.array([0.5, 0.5, 0.5]),
                depth_median=float(position_3d[2]) if position_3d is not None else 2.5,
            )
            
            legacy_map[label] = legacy_entity
        
        return legacy_map
    
    def get_instances_by_label(self, label: str) -> List[TrackedInstance]:
        """获取指定类别的所有实例（新接口）"""
        return self.instance_tracker.get_instances_by_label(label)
    
    def get_probabilistic_entity(self, label: str) -> Optional[ProbabilisticEntity]:
        """获取概率性实体（新接口）"""
        return self.prob_mind_map.get_entity_by_label(label)


# ============================================================================
# DirectQA V2 问题回答器
# ============================================================================

class DirectQAV2:
    """DirectQA V2 - 使用改进的心智地图"""
    
    # -------------------------------------------------------------------------
    # 数值任务 (保持原有逻辑，只优化匹配)
    # -------------------------------------------------------------------------
    
    @staticmethod
    def answer_counting(mind_map: Dict, question: str, builder: MindMapBuilderV2 = None) -> str:
        """回答计数问题
        
        注意：不使用实例追踪的实例数，因为没有相机位姿会导致过度分割。
        而是使用 legacy_map 中的 count（基于单帧最大检测数）。
        """
        match = re.search(r'How many (\w+)\(s\)', question)
        if not match:
            match = re.search(r'How many (\w+)', question)
        
        if not match:
            return "0"
        
        target = match.group(1).lower()
        
        # 直接使用 mind_map 中的 count（已在 _convert_to_legacy_format 中计算）
        for label, entity in mind_map.items():
            if match_object_name(target, label):
                return str(entity.count)
        
        return "0"
    
    @staticmethod
    def answer_object_size(mind_map: Dict, question: str) -> str:
        """回答物体尺寸问题"""
        q_lower = question.lower()
        
        for label, entity in mind_map.items():
            if label.lower() in q_lower:
                if entity.size_3d is not None:
                    max_dim = float(np.max(entity.size_3d)) * 100
                    return str(int(max_dim))
            else:
                for syn in get_synonyms(label.lower()):
                    if syn in q_lower:
                        if entity.size_3d is not None:
                            max_dim = float(np.max(entity.size_3d)) * 100
                            return str(int(max_dim))
                        break
        
        return "50"
    
    @staticmethod
    def answer_room_size(mind_map: Dict, question: str) -> str:
        """估算房间面积"""
        if not mind_map:
            return "20"
        
        positions = []
        for entity in mind_map.values():
            if entity.position_3d is not None:
                positions.append(entity.position_3d)
        
        if len(positions) < 2:
            return str(12 + len(mind_map) * 2)
        
        positions = np.array(positions)
        
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        estimated_area = (x_range + 1.5) * (y_range + 1.5)
        estimated_area = max(8, min(80, estimated_area))
        
        return f"{estimated_area:.1f}"
    
    @staticmethod
    def answer_abs_distance(mind_map: Dict, question: str) -> str:
        """回答物体间距离问题"""
        q_lower = question.lower()
        
        between_match = re.search(r'between (?:the )?([a-z]+(?:\s+[a-z]+)*)\s+and\s+(?:the )?([a-z]+(?:\s+[a-z]+)*)', q_lower)
        
        if between_match:
            obj1_name = between_match.group(1).strip()
            obj2_name = between_match.group(2).strip()
        else:
            obj1_name = obj2_name = None
            for label in mind_map.keys():
                if label.lower() in q_lower:
                    if obj1_name is None:
                        obj1_name = label
                    else:
                        obj2_name = label
                        break
        
        if not obj1_name or not obj2_name:
            return "2.0"
        
        def find_position_fuzzy(name: str) -> Optional[np.ndarray]:
            name_lower = name.lower().strip()
            
            for label, entity in mind_map.items():
                if label.lower() == name_lower and entity.position_3d is not None:
                    return entity.position_3d
            
            for label, entity in mind_map.items():
                if (name_lower in label.lower() or label.lower() in name_lower) and entity.position_3d is not None:
                    return entity.position_3d
            
            for label, entity in mind_map.items():
                if match_object_name(name_lower, label) and entity.position_3d is not None:
                    return entity.position_3d
            
            special_map = {
                'washing machine': 'washer',
                'trash bin': 'trash can',
                'garbage bin': 'trash can',
            }
            if name_lower in special_map:
                alt_name = special_map[name_lower]
                for label, entity in mind_map.items():
                    if alt_name in label.lower() and entity.position_3d is not None:
                        return entity.position_3d
            
            return None
        
        pos1 = find_position_fuzzy(obj1_name)
        pos2 = find_position_fuzzy(obj2_name)
        
        if pos1 is not None and pos2 is not None:
            dist = float(np.linalg.norm(pos1 - pos2))
            return f"{dist:.1f}"
        
        if pos1 is not None:
            return f"{float(np.linalg.norm(pos1)):.1f}"
        
        if pos2 is not None:
            return f"{float(np.linalg.norm(pos2)):.1f}"
        
        return "2.0"
    
    # -------------------------------------------------------------------------
    # 选择题任务 - 相对方向（使用概率性推理）
    # -------------------------------------------------------------------------
    
    @staticmethod
    def answer_rel_direction(mind_map: Dict, question: str, options: List[str],
                             difficulty: str = 'easy', builder: MindMapBuilderV2 = None) -> str:
        """回答相对方向问题"""
        if not options:
            return "left"
        
        q_lower = question.lower()
        
        # 解析问题
        standing_match = re.search(r'standing by (?:the )?([a-z]+(?:\s+[a-z]+)*?)(?:\s+and\s+facing|\s*,)', q_lower)
        facing_match = re.search(r'facing (?:the )?([a-z]+(?:\s+[a-z]+)*?)(?:\s*,|\s+is\b)', q_lower)
        
        target_match_my = re.search(r'is (?:the )?([a-z]+(?:\s+[a-z]+)*?)\s+to\s+my', q_lower)
        target_match_of = re.search(r'is (?:the )?([a-z]+(?:\s+[a-z]+)*?)\s+to\s+the\s+(?:left|right|back|front)', q_lower)
        
        if target_match_my:
            target_match = target_match_my
            relative_to_observer = True
        elif target_match_of:
            target_match = target_match_of
            relative_to_observer = False
        else:
            return options[0]
        
        if not all([standing_match, facing_match, target_match]):
            return options[0]
        
        standing_name = standing_match.group(1)
        facing_name = facing_match.group(1)
        target_name = target_match.group(1)
        
        def find_position(name: str) -> Optional[np.ndarray]:
            for label, entity in mind_map.items():
                if match_object_name(name, label) and entity.position_3d is not None:
                    return entity.position_3d
            return None
        
        standing_pos = find_position(standing_name)
        facing_pos = find_position(facing_name)
        target_pos = find_position(target_name)
        
        if standing_pos is None or facing_pos is None or target_pos is None:
            return options[0]
        
        # 计算方向向量
        forward = np.array([facing_pos[0] - standing_pos[0], facing_pos[2] - standing_pos[2]])
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return options[0]
        forward = forward / forward_norm
        
        right = np.array([forward[1], -forward[0]])
        
        if relative_to_observer:
            to_target = np.array([target_pos[0] - standing_pos[0], target_pos[2] - standing_pos[2]])
        else:
            to_target = np.array([target_pos[0] - facing_pos[0], target_pos[2] - facing_pos[2]])
        
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm < 1e-6:
            return options[0]
        to_target = to_target / to_target_norm
        
        front_dot = np.dot(to_target, forward)
        right_dot = np.dot(to_target, right)
        
        if 'left or' in q_lower and 'right' in q_lower and 'back' not in q_lower:
            direction = 'right' if right_dot > 0 else 'left'
        elif 'left, right, or back' in q_lower or 'left, right or back' in q_lower:
            if front_dot < -0.5:
                direction = 'back'
            elif right_dot > 0:
                direction = 'right'
            else:
                direction = 'left'
        elif 'front-left' in q_lower or 'front-right' in q_lower:
            if front_dot > 0:
                direction = 'front-right' if right_dot > 0 else 'front-left'
            else:
                direction = 'back-right' if right_dot > 0 else 'back-left'
        else:
            direction = 'right' if right_dot > 0 else 'left'
        
        for opt in options:
            opt_lower = opt.lower()
            if direction in opt_lower:
                return opt
        
        return options[0]
    
    # -------------------------------------------------------------------------
    # 选择题任务 - 相对距离（使用实例级追踪）
    # -------------------------------------------------------------------------
    
    @staticmethod
    def answer_rel_distance(mind_map: Dict, question: str, options: List[str],
                            builder: MindMapBuilderV2 = None) -> str:
        """回答相对距离问题"""
        if not options:
            return ""
        
        q_lower = question.lower()
        
        find_closest = 'closest' in q_lower or 'nearest' in q_lower or 'closer' in q_lower
        find_farthest = 'farthest' in q_lower or 'furthest' in q_lower or 'farther' in q_lower
        
        if not find_closest and not find_farthest:
            find_closest = True
        
        ref_match = re.search(r'(?:to|from) (?:the )?([a-z]+(?:\s+[a-z]+)*)\??', q_lower)
        if not ref_match:
            return random.choice(options)
        
        reference_name = ref_match.group(1)
        
        candidates_match = re.search(r'\(([^)]+)\)', q_lower)
        if candidates_match:
            candidates_text = candidates_match.group(1)
            candidate_names = [c.strip() for c in candidates_text.split(',')]
        else:
            candidate_names = []
            for opt in options:
                opt_content = re.sub(r'^[A-D]\.\s*', '', opt)
                candidate_names.append(opt_content.strip().lower())
        
        def find_position_fuzzy(name: str) -> Optional[np.ndarray]:
            name_lower = name.lower().strip()
            
            for label, entity in mind_map.items():
                if label.lower() == name_lower and entity.position_3d is not None:
                    return entity.position_3d
            
            for label, entity in mind_map.items():
                if (name_lower in label.lower() or label.lower() in name_lower) and entity.position_3d is not None:
                    return entity.position_3d
            
            for label, entity in mind_map.items():
                if match_object_name(name_lower, label) and entity.position_3d is not None:
                    return entity.position_3d
            
            name_words = set(name_lower.split())
            for label, entity in mind_map.items():
                label_words = set(label.lower().split())
                if name_words & label_words and entity.position_3d is not None:
                    return entity.position_3d
            
            return None
        
        ref_pos = find_position_fuzzy(reference_name)
        
        if ref_pos is None:
            alt_names = get_synonyms(reference_name)
            for alt in alt_names:
                ref_pos = find_position_fuzzy(alt)
                if ref_pos is not None:
                    break
        
        if ref_pos is None:
            return random.choice(options)
        
        distances = {}
        option_map = {}
        
        for i, cand in enumerate(candidate_names):
            cand_pos = find_position_fuzzy(cand)
            if cand_pos is not None:
                dist = float(np.linalg.norm(cand_pos - ref_pos))
                distances[cand] = dist
                if i < len(options):
                    option_map[cand] = options[i]
        
        if not distances:
            return random.choice(options)
        
        if find_closest:
            best_cand = min(distances.keys(), key=lambda k: distances[k])
        else:
            best_cand = max(distances.keys(), key=lambda k: distances[k])
        
        if best_cand in option_map:
            return option_map[best_cand]
        
        for opt in options:
            opt_lower = opt.lower()
            if best_cand in opt_lower or any(s in opt_lower for s in get_synonyms(best_cand)):
                return opt
        
        return options[0]
    
    # -------------------------------------------------------------------------
    # 选择题任务 - 出现顺序
    # -------------------------------------------------------------------------
    
    @staticmethod
    def answer_appearance_order(mind_map: Dict, question: str, options: List[str],
                                 builder: MindMapBuilderV2 = None) -> str:
        """回答出现顺序问题"""
        if not options:
            return ""
        
        match = re.search(r'following categories.*?:\s*(.+?)\?', question, re.IGNORECASE)
        if match:
            objects_text = match.group(1)
            target_objects = [obj.strip().lower() for obj in objects_text.split(',')]
        else:
            opt_content = re.sub(r'^[A-D]\.\s*', '', options[0])
            target_objects = [obj.strip().lower() for obj in opt_content.split(',')]
        
        def find_entity_fuzzy(target: str):
            target_lower = target.lower().strip()
            
            for label, entity in mind_map.items():
                if label.lower() == target_lower:
                    return entity
            
            for label, entity in mind_map.items():
                if target_lower in label.lower() or label.lower() in target_lower:
                    return entity
            
            for label, entity in mind_map.items():
                if match_object_name(target_lower, label):
                    return entity
            
            target_words = set(target_lower.split())
            for label, entity in mind_map.items():
                label_words = set(label.lower().split())
                if target_words & label_words:
                    return entity
            
            return None
        
        object_frames = {}
        for target in target_objects:
            entity = find_entity_fuzzy(target)
            if entity is not None:
                if target not in object_frames:
                    object_frames[target] = entity.first_seen_frame
                else:
                    object_frames[target] = min(object_frames[target], entity.first_seen_frame)
        
        if len(object_frames) < 2:
            return random.choice(options)
        
        sorted_objects = sorted(object_frames.items(), key=lambda x: x[1])
        predicted_order = [obj for obj, _ in sorted_objects]
        
        missing_objects = [t for t in target_objects if t not in object_frames]
        predicted_order.extend(missing_objects)
        
        best_option = options[0]
        best_score = -1
        
        for opt in options:
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt)
            opt_objects = [o.strip().lower() for o in opt_content.split(',')]
            
            score = 0
            for i, pred_obj in enumerate(predicted_order):
                for j, opt_obj in enumerate(opt_objects):
                    if match_object_name(pred_obj, opt_obj):
                        score += max(0, len(target_objects) - abs(i - j))
                        break
            
            if score > best_score:
                best_score = score
                best_option = opt
        
        return best_option
    
    # -------------------------------------------------------------------------
    # 选择题任务 - 路径规划
    # -------------------------------------------------------------------------
    
    @staticmethod
    def answer_route_planning(mind_map: Dict, question: str, options: List[str]) -> str:
        """回答路径规划问题"""
        if not options:
            return ""
        
        q_lower = question.lower()
        
        start_match = re.search(r'beginning at (?:the )?(\w+(?:\s+\w+)?)', q_lower)
        facing_match = re.search(r'facing (?:the )?(\w+)', q_lower)
        
        if not start_match or not facing_match:
            return options[0]
        
        steps = re.findall(r'go forward until (?:the )?(\w+(?:\s+\w+)?)', q_lower)
        fill_count = q_lower.count('[please fill in]')
        
        if fill_count == 0:
            return options[0]
        
        def find_position(name: str) -> Optional[np.ndarray]:
            for label, entity in mind_map.items():
                if match_object_name(name, label) and entity.position_3d is not None:
                    return entity.position_3d
            return None
        
        path_positions = []
        start_pos = find_position(start_match.group(1))
        if start_pos is not None:
            path_positions.append(start_pos)
        
        for step in steps:
            pos = find_position(step)
            if pos is not None:
                path_positions.append(pos)
        
        if len(path_positions) < 3:
            return options[0]
        
        turns = []
        for i in range(1, len(path_positions) - 1):
            prev_dir = path_positions[i] - path_positions[i-1]
            next_dir = path_positions[i+1] - path_positions[i]
            
            prev_dir = prev_dir[:2] / (np.linalg.norm(prev_dir[:2]) + 1e-8)
            next_dir = next_dir[:2] / (np.linalg.norm(next_dir[:2]) + 1e-8)
            
            cross = prev_dir[0] * next_dir[1] - prev_dir[1] * next_dir[0]
            dot = np.dot(prev_dir, next_dir)
            
            if dot < -0.5:
                turns.append('turn back')
            elif cross > 0.3:
                turns.append('turn right')
            elif cross < -0.3:
                turns.append('turn left')
            else:
                turns.append('go forward')
        
        for opt in options:
            opt_lower = opt.lower()
            match_count = 0
            for turn in turns:
                if turn in opt_lower:
                    match_count += 1
            
            if match_count == len(turns):
                return opt
        
        return options[0]


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], result_queue: mp.Queue,
                   use_v2_features: bool = True):
    """GPU Worker 进程"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    builder = MindMapBuilderV2(
        device='cuda', 
        num_frames=32, 
        box_threshold=0.25,
        use_scale_calibration=use_v2_features,
        use_instance_tracking=use_v2_features,
    )
    
    results = []
    total = len(samples)
    
    for i, sample in enumerate(samples):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 构建心智地图
            target_objects = []
            if question_type == 'object_counting':
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map = builder.build_from_video(video_path, target_objects)
            
            # 根据问题类型回答
            if question_type == 'object_counting':
                pred = DirectQAV2.answer_counting(mind_map, question, builder)
            elif question_type == 'object_size_estimation':
                pred = DirectQAV2.answer_object_size(mind_map, question)
            elif question_type == 'room_size_estimation':
                pred = DirectQAV2.answer_room_size(mind_map, question)
            elif question_type == 'object_abs_distance':
                pred = DirectQAV2.answer_abs_distance(mind_map, question)
            elif question_type == 'object_rel_direction_easy':
                pred = DirectQAV2.answer_rel_direction(mind_map, question, options, 'easy', builder)
            elif question_type == 'object_rel_direction_medium':
                pred = DirectQAV2.answer_rel_direction(mind_map, question, options, 'medium', builder)
            elif question_type == 'object_rel_direction_hard':
                pred = DirectQAV2.answer_rel_direction(mind_map, question, options, 'hard', builder)
            elif question_type == 'object_rel_distance':
                pred = DirectQAV2.answer_rel_distance(mind_map, question, options, builder)
            elif question_type == 'obj_appearance_order':
                pred = DirectQAV2.answer_appearance_order(mind_map, question, options, builder)
            elif question_type == 'route_planning':
                pred = DirectQAV2.answer_route_planning(mind_map, question, options)
            else:
                pred = str(options[0]) if options else "0"
            
            # 计算指标
            if question_type in NUMERICAL_TASKS:
                pred_num = normalize_number(pred)
                gt_num = normalize_number(str(gt))
                score = mean_relative_accuracy(pred_num, gt_num) if pred_num is not None and gt_num is not None else 0.0
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
                            if pred_clean == opt_content:
                                pred_letter = chr(65 + idx)
                                break
                        
                        if pred_letter is None:
                            for idx, opt in enumerate(options):
                                opt_content = re.sub(r'^[A-D]\.\s*', '', opt).lower().strip()
                                if pred_clean in opt_content or opt_content in pred_clean:
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
    parser.add_argument('--use-v2', type=bool, default=True, help='使用 V2 改进特性')
    args = parser.parse_args()
    
    print("=" * 70)
    print("VSIBench 测试 - DirectQA V2 版本 (改进版心智地图)")
    print(f"改进特性: 动态尺度校准={args.use_v2}, 实例追踪={args.use_v2}")
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
    
    # 任务类型统计
    type_counts = defaultdict(int)
    for s in samples:
        type_counts[s['question_type']] += 1
    print("\n任务类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    
    # 分配到各 GPU
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
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_samples[gpu_id], result_queue, args.use_v2))
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
    
    # 输出结果
    print("\n" + "="*60)
    print("DirectQA V2 版本测试结果")
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
    output_dir = f"/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/directqa_v2_test_{timestamp}"
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
            'summary': {q: {'mean': float(np.mean(s)), 'count': len(s)} for q, s in type_scores.items()},
            'overall': float(overall),
            'details': cleaned_results,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果保存到: {output_dir}")
