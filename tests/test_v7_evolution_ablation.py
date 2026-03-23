#!/usr/bin/env python3
"""
V7 Evolution消融实验

目的：测试规则Evolution对V7框架是否真的有帮助
对比：
1. 无Evolution：感知系统输出 → 直接推理
2. 有Evolution：感知系统输出 → 任务特定Evolution → 推理

使用V7的基准设置，只改变Evolution开关
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
from tqdm import tqdm

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
# 标定物配置 - 已知标准尺寸 (单位: 米)
# ============================================================================

CALIBRATION_OBJECTS = {
    'door': {'height': 2.0, 'range': (1.8, 2.2)},
    'chair': {'height': 0.80, 'range': (0.70, 0.95)},
    'dining chair': {'height': 0.80, 'range': (0.70, 0.95)},
    'office chair': {'height': 1.0, 'range': (0.9, 1.2)},
    'bed': {'length': 2.0, 'range': (1.8, 2.2)},
    'toilet': {'height': 0.40, 'range': (0.35, 0.45)},
    'refrigerator': {'height': 1.75, 'range': (1.5, 2.0)},
    'fridge': {'height': 1.75, 'range': (1.5, 2.0)},
    'sofa': {'height': 0.85, 'range': (0.7, 1.0)},
    'couch': {'height': 0.85, 'range': (0.7, 1.0)},
    'table': {'height': 0.75, 'range': (0.65, 0.85)},
    'dining table': {'height': 0.75, 'range': (0.65, 0.85)},
    'desk': {'height': 0.75, 'range': (0.65, 0.85)},
    'tv': {'diagonal': 1.0, 'range': (0.5, 1.5)},
    'window': {'height': 1.2, 'range': (0.8, 1.8)},
    'sink': {'height': 0.85, 'range': (0.75, 0.95)},
    'stove': {'height': 0.90, 'range': (0.80, 1.0)},
    'oven': {'height': 0.60, 'range': (0.50, 0.70)},
    'microwave': {'height': 0.30, 'range': (0.25, 0.40)},
    'nightstand': {'height': 0.55, 'range': (0.45, 0.65)},
    'lamp': {'height': 0.50, 'range': (0.30, 0.80)},
}

# 扩展词汇表
EXTENDED_VOCABULARY = [
    "chair", "table", "sofa", "couch", "stove", "tv", "television",
    "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator", "fridge",
    "sink", "toilet", "bathtub", "door", "window", "picture", "painting",
    "pillow", "cushion", "monitor", "backpack", "bag", "heater",
    "trash can", "trash bin", "mirror", "towel", "plant", "cup",
    "nightstand", "closet", "microwave", "printer", "washer", "dryer",
    "oven", "counter", "drawer", "curtain", "rug", "carpet", "clock",
    "fan", "air conditioner", "bookshelf", "armchair", "stool",
]

# 同义词映射
SYNONYM_MAP = {
    'sofa': ['couch', 'settee'],
    'tv': ['television', 'tv screen'],
    'refrigerator': ['fridge'],
    'trash bin': ['trash can', 'garbage can', 'dustbin'],
    'couch': ['sofa'],
    'nightstand': ['bedside table', 'night stand'],
    'lamp': ['light', 'table lamp'],
    'monitor': ['screen', 'display'],
}

# 任务分类
NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 'room_size_estimation', 'object_abs_distance']
CHOICE_TASKS = ['object_rel_distance', 'object_rel_direction_easy', 'object_rel_direction_medium', 
                'object_rel_direction_hard', 'obj_appearance_order', 'route_planning']


def get_synonyms(obj: str) -> List[str]:
    """获取物体的同义词"""
    obj_lower = obj.lower()
    synonyms = [obj_lower]
    if obj_lower in SYNONYM_MAP:
        synonyms.extend(SYNONYM_MAP[obj_lower])
    for key, values in SYNONYM_MAP.items():
        if obj_lower in values:
            synonyms.append(key)
            synonyms.extend([v for v in values if v != obj_lower])
    return list(set(synonyms))


def match_object_name(query: str, label: str) -> bool:
    """匹配物体名称"""
    query_lower = query.lower().strip()
    label_lower = label.lower().strip()
    
    if query_lower == label_lower or query_lower in label_lower or label_lower in query_lower:
        return True
    
    query_syns = get_synonyms(query_lower)
    label_syns = get_synonyms(label_lower)
    
    return bool(set(query_syns) & set(label_syns))


def normalize_number(text: str) -> Optional[float]:
    """提取数字"""
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
    """VSIBench MRA 指标"""
    if pred is None or target is None:
        return 0.0
    epsilon = 1e-8
    rel_error = abs(pred - target) / (abs(target) + epsilon)
    thresholds = np.arange(start, end + interval / 2, interval)
    conditions = rel_error < (1 - thresholds)
    return conditions.astype(float).mean()


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Detection:
    """单次检测记录"""
    frame_idx: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    depth: float
    position_3d: np.ndarray
    estimated_height: float = 0.0
    estimated_width: float = 0.0


@dataclass
class MindMapEntity:
    """心智地图实体"""
    label: str
    detections: List[Detection] = field(default_factory=list)
    count: int = 0
    avg_confidence: float = 0.0
    position_3d: np.ndarray = None
    size_3d: np.ndarray = None
    first_seen_frame: int = -1
    calibrated: bool = False
    
    def get_frame_indices(self) -> List[int]:
        return sorted(set(d.frame_idx for d in self.detections))
    
    def get_estimated_size(self) -> float:
        if not self.detections:
            return 0.0
        best = max(self.detections, key=lambda x: x.confidence)
        return max(best.estimated_height, best.estimated_width)
    
    def to_text(self) -> str:
        text = f"- {self.label}: count={self.count}"
        if self.position_3d is not None:
            pos = self.position_3d
            text += f", position=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m"
        if self.size_3d is not None:
            size = self.size_3d * 100
            text += f", size≈{max(size):.0f}cm"
        text += f", confidence={self.avg_confidence:.2f}"
        text += f", frames={len(self.get_frame_indices())}"
        if self.first_seen_frame >= 0:
            text += f", first_seen_frame={self.first_seen_frame}"
        return text


@dataclass
class CalibrationResult:
    """校准结果"""
    calibration_object: str
    estimated_size: float
    standard_size: float
    scale_factor: float
    confidence: float


@dataclass
class EvolutionAction:
    """演化动作"""
    action_type: str
    target_entity: str
    old_value: Any
    new_value: Any
    reasoning: str
    confidence: float


# ============================================================================
# 感知模块 - 构建初始心智地图
# ============================================================================

class MindMapBuilder:
    """心智地图构建器"""
    
    def __init__(self, device: str = 'cuda', num_frames: int = 32, box_threshold: float = 0.25):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        self._labeler = None
        self._depth_estimator = None
        self.focal_length = 500
    
    def load_models(self):
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
        if self._labeler is not None:
            try:
                del self._labeler.model
                del self._labeler.processor
            except:
                pass
            self._labeler = None
        if self._depth_estimator is not None:
            self._depth_estimator = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def build_from_video(self, video_path: str, target_objects: List[str] = None) -> Tuple[Dict[str, MindMapEntity], int, List[np.ndarray]]:
        """构建心智地图"""
        self.load_models()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, 0, []
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        vocab = list(set((target_objects or []) + EXTENDED_VOCABULARY))
        prompt = " . ".join(vocab) + " ."
        
        all_detections = defaultdict(list)
        sampled_frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(frame_rgb)
            h, w = frame_rgb.shape[:2]
            
            depth_result = self._depth_estimator.infer_single(frame_rgb)
            if isinstance(depth_result, tuple):
                depth_tensor = depth_result[0]
            else:
                depth_tensor = depth_result
            
            if depth_tensor is not None:
                depth_map = depth_tensor.cpu().numpy()
                if depth_map.ndim == 3:
                    depth_map = depth_map.squeeze()
                elif depth_map.ndim == 1:
                    depth_map = depth_map.reshape(h, w)
                
                if depth_map.shape[0] != h or depth_map.shape[1] != w:
                    depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                depth_map = np.ones((h, w), dtype=np.float32) * 5.0
            
            detections = self._labeler.detect(frame_rgb, prompt)
            
            for det in detections:
                raw_label = det.label.strip().lower()
                if raw_label.startswith('##'):
                    continue
                
                label = raw_label
                box = det.bbox_pixels
                conf = det.confidence
                
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                cx = min(max(cx, 0), w - 1)
                cy = min(max(cy, 0), h - 1)
                depth = float(depth_map[cy, cx])
                
                pos_3d = np.array([
                    (cx - w / 2) * depth / self.focal_length,
                    (cy - h / 2) * depth / self.focal_length,
                    depth
                ])
                
                box_h_px = box[3] - box[1]
                box_w_px = box[2] - box[0]
                est_height = box_h_px * depth / self.focal_length
                est_width = box_w_px * depth / self.focal_length
                
                detection = Detection(
                    frame_idx=int(frame_idx),
                    bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    confidence=float(conf),
                    depth=depth,
                    position_3d=pos_3d,
                    estimated_height=est_height,
                    estimated_width=est_width,
                )
                
                all_detections[label].append(detection)
        
        cap.release()
        
        mind_map = {}
        for label, dets in all_detections.items():
            entity = self._aggregate_detections(label, dets)
            mind_map[label] = entity
        
        return mind_map, total_frames, sampled_frames
    
    def _aggregate_detections(self, label: str, detections: List[Detection]) -> MindMapEntity:
        """聚合检测结果"""
        if not detections:
            return MindMapEntity(label=label)
        
        frame_dets = defaultdict(list)
        for det in detections:
            frame_dets[det.frame_idx].append(det)
        
        max_count = max(len(fd) for fd in frame_dets.values())
        avg_conf = np.mean([d.confidence for d in detections])
        
        positions = np.array([d.position_3d for d in detections])
        avg_pos = np.median(positions, axis=0)
        
        best_det = max(detections, key=lambda x: x.confidence)
        size_3d = np.array([best_det.estimated_width, best_det.estimated_height, 0.3])
        
        first_frame = min(d.frame_idx for d in detections)
        
        return MindMapEntity(
            label=label,
            detections=detections,
            count=max_count,
            avg_confidence=float(avg_conf),
            position_3d=avg_pos,
            size_3d=size_3d,
            first_seen_frame=first_frame,
        )


# ============================================================================
# 校准器 - 通过标定物校准尺度
# ============================================================================

class ScaleCalibrator:
    """尺度校准器"""
    
    def find_calibration_objects(self, mind_map: Dict[str, MindMapEntity]) -> List[Tuple[str, MindMapEntity, dict]]:
        """找到可用于校准的物体"""
        candidates = []
        
        for label, entity in mind_map.items():
            for cal_name, cal_info in CALIBRATION_OBJECTS.items():
                if match_object_name(label, cal_name):
                    if entity.avg_confidence >= 0.3:
                        candidates.append((cal_name, entity, cal_info))
                    break
        
        candidates.sort(key=lambda x: -x[1].avg_confidence)
        return candidates
    
    def calculate_scale_factor(self, mind_map: Dict[str, MindMapEntity]) -> CalibrationResult:
        """计算尺度校准系数"""
        candidates = self.find_calibration_objects(mind_map)
        
        if not candidates:
            return CalibrationResult(
                calibration_object="none",
                estimated_size=0,
                standard_size=0,
                scale_factor=1.0,
                confidence=0.0,
            )
        
        scale_factors = []
        
        for cal_name, entity, cal_info in candidates[:5]:
            est_size = entity.get_estimated_size()
            
            if est_size <= 0:
                continue
            
            std_size = cal_info.get('height') or cal_info.get('length') or cal_info.get('diagonal', 1.0)
            factor = std_size / est_size
            
            if 0.1 < factor < 10:
                scale_factors.append({
                    'object': cal_name,
                    'factor': factor,
                    'confidence': entity.avg_confidence,
                    'estimated': est_size,
                    'standard': std_size,
                })
        
        if not scale_factors:
            return CalibrationResult(
                calibration_object="none",
                estimated_size=0,
                standard_size=0,
                scale_factor=1.0,
                confidence=0.0,
            )
        
        total_weight = sum(sf['confidence'] for sf in scale_factors)
        weighted_factor = sum(sf['factor'] * sf['confidence'] for sf in scale_factors) / total_weight
        
        best = scale_factors[0]
        return CalibrationResult(
            calibration_object=best['object'],
            estimated_size=best['estimated'],
            standard_size=best['standard'],
            scale_factor=weighted_factor,
            confidence=total_weight / len(scale_factors),
        )
    
    def apply_calibration(self, mind_map: Dict[str, MindMapEntity], calibration: CalibrationResult) -> Dict[str, MindMapEntity]:
        """应用校准到心智地图"""
        if calibration.scale_factor == 1.0 or calibration.confidence < 0.1:
            return mind_map
        
        factor = calibration.scale_factor
        
        for label, entity in mind_map.items():
            if entity.size_3d is not None:
                entity.size_3d = entity.size_3d * factor
            
            if entity.position_3d is not None:
                entity.position_3d[2] = entity.position_3d[2] * factor
            
            for det in entity.detections:
                det.estimated_height *= factor
                det.estimated_width *= factor
                det.depth *= factor
                det.position_3d[2] *= factor
            
            entity.calibrated = True
        
        return mind_map


# ============================================================================
# 任务特定Evolution - 核心新增
# ============================================================================

class TaskSpecificEvolver:
    """任务特定的Evolution策略"""
    
    def __init__(self):
        self.calibration_refs = {
            'door': 2.0, 'chair': 0.8, 'table': 0.75, 'bed': 2.0,
            'toilet': 0.4, 'refrigerator': 1.7, 'fridge': 1.7,
            'sofa': 0.85, 'couch': 0.85, 'desk': 0.75, 'window': 1.2,
        }
        
        self.typical_sizes = {
            'chair': (0.4, 1.0), 'table': (0.6, 1.5), 'bed': (1.8, 2.2),
            'door': (1.8, 2.2), 'window': (0.6, 2.0), 'sofa': (0.6, 1.2),
            'desk': (0.6, 1.0), 'toilet': (0.3, 0.5), 'refrigerator': (1.5, 2.0),
        }
    
    def evolve_counting(self, mind_map: Dict[str, MindMapEntity], question: str) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """Counting任务Evolution"""
        actions = []
        
        match = re.search(r'How many (\w+)', question, re.IGNORECASE)
        target_obj = match.group(1).lower() if match else None
        
        for label, entity in mind_map.items():
            if target_obj and target_obj not in label.lower():
                continue
            
            if not entity.detections or len(entity.detections) < 2:
                continue
            
            frame_counts = defaultdict(int)
            for det in entity.detections:
                frame_counts[det.frame_idx] += 1
            
            counts = list(frame_counts.values())
            original_count = entity.count
            
            if len(counts) >= 3:
                median_count = int(np.median(counts))
                mode_count = max(set(counts), key=counts.count)
                
                if median_count == mode_count:
                    evolved_count = median_count
                    confidence = 0.9
                else:
                    evolved_count = min(median_count, mode_count)
                    confidence = 0.7
            else:
                evolved_count = int(np.median(counts)) if counts else original_count
                confidence = 0.6
            
            # 置信度过滤
            high_conf_dets = [d for d in entity.detections if d.confidence > 0.5]
            if high_conf_dets:
                high_conf_frame_counts = defaultdict(int)
                for det in high_conf_dets:
                    high_conf_frame_counts[det.frame_idx] += 1
                if high_conf_frame_counts:
                    high_conf_median = int(np.median(list(high_conf_frame_counts.values())))
                    if high_conf_median < evolved_count:
                        evolved_count = high_conf_median
            
            if evolved_count != original_count:
                entity.count = evolved_count
                actions.append(EvolutionAction(
                    action_type='count_correction',
                    target_entity=label,
                    old_value=original_count,
                    new_value=evolved_count,
                    reasoning=f"Multi-frame analysis: median={np.median(counts):.1f}",
                    confidence=confidence,
                ))
        
        return mind_map, actions
    
    def evolve_size(self, mind_map: Dict[str, MindMapEntity], question: str) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """Size任务Evolution"""
        actions = []
        
        # 寻找最佳标定物
        calibration_factor = 1.0
        calibration_source = None
        best_calibration_conf = 0
        
        for label, entity in mind_map.items():
            if entity.size_3d is None:
                continue
            
            for cal_obj, standard_size in self.calibration_refs.items():
                if cal_obj in label.lower():
                    estimated_size = max(entity.size_3d[0], entity.size_3d[1])
                    if estimated_size > 0.01:
                        factor = standard_size / estimated_size
                        if 0.2 < factor < 5.0 and entity.avg_confidence > best_calibration_conf:
                            calibration_factor = factor
                            calibration_source = cal_obj
                            best_calibration_conf = entity.avg_confidence
                    break
        
        if calibration_source:
            actions.append(EvolutionAction(
                action_type='calibration_applied',
                target_entity='global_scale',
                old_value=1.0,
                new_value=calibration_factor,
                reasoning=f"Using {calibration_source} as reference",
                confidence=best_calibration_conf,
            ))
        
        # 应用校准并检查物理约束
        for label, entity in mind_map.items():
            if entity.size_3d is None:
                continue
            
            if calibration_factor != 1.0:
                entity.size_3d = entity.size_3d * calibration_factor
                if entity.position_3d is not None:
                    entity.position_3d = entity.position_3d * calibration_factor
            
            # 物理约束检查
            for obj_type, (min_size, max_size) in self.typical_sizes.items():
                if obj_type in label.lower():
                    current_height = max(entity.size_3d[0], entity.size_3d[1])
                    
                    if current_height < min_size * 0.3 or current_height > max_size * 3:
                        target_size = (min_size + max_size) / 2
                        correction = target_size / current_height if current_height > 0 else 1
                        entity.size_3d = entity.size_3d * correction
                        
                        actions.append(EvolutionAction(
                            action_type='physics_constraint',
                            target_entity=label,
                            old_value=f"{current_height:.2f}m",
                            new_value=f"{max(entity.size_3d):.2f}m",
                            reasoning=f"Size outside typical range for {obj_type}",
                            confidence=0.7,
                        ))
                    break
        
        return mind_map, actions
    
    def evolve_distance(self, mind_map: Dict[str, MindMapEntity], question: str) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """Distance任务Evolution"""
        actions = []
        
        # 先应用Size的校准
        mind_map, size_actions = self.evolve_size(mind_map, question)
        actions.extend(size_actions)
        
        # 室内距离约束
        max_indoor_dist = 10.0
        
        positions = []
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                positions.append(entity.position_3d)
        
        if len(positions) >= 2:
            positions = np.array(positions)
            max_depth = np.max(positions[:, 2])
            
            if max_depth > max_indoor_dist:
                scale = max_indoor_dist / max_depth * 0.9
                for label, entity in mind_map.items():
                    if entity.position_3d is not None:
                        entity.position_3d = entity.position_3d * scale
                
                actions.append(EvolutionAction(
                    action_type='indoor_constraint',
                    target_entity='all_positions',
                    old_value=f"max_depth={max_depth:.2f}m",
                    new_value=f"max_depth={max_depth*scale:.2f}m",
                    reasoning="Scaled positions to fit indoor constraint",
                    confidence=0.75,
                ))
        
        return mind_map, actions
    
    def evolve_direction(self, mind_map: Dict[str, MindMapEntity], question: str) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """Direction任务Evolution"""
        actions = []
        
        # 先应用基础校准
        mind_map, _ = self.evolve_size(mind_map, question)
        
        # 多帧位置聚合
        for label, entity in mind_map.items():
            if not entity.detections or len(entity.detections) < 2:
                continue
            
            frame_positions = []
            for det in entity.detections:
                if hasattr(det, 'position_3d') and det.position_3d is not None:
                    frame_positions.append(det.position_3d)
            
            if len(frame_positions) >= 3:
                positions_array = np.array(frame_positions)
                median_position = np.median(positions_array, axis=0)
                original_position = entity.position_3d.copy() if entity.position_3d is not None else None
                
                position_std = np.std(positions_array, axis=0)
                stability = 1.0 / (1.0 + np.mean(position_std))
                
                entity.position_3d = median_position
                
                if original_position is not None:
                    position_change = np.linalg.norm(median_position - original_position)
                    if position_change > 0.1:
                        actions.append(EvolutionAction(
                            action_type='position_aggregation',
                            target_entity=label,
                            old_value=f"({original_position[0]:.2f}, {original_position[1]:.2f}, {original_position[2]:.2f})",
                            new_value=f"({median_position[0]:.2f}, {median_position[1]:.2f}, {median_position[2]:.2f})",
                            reasoning=f"Aggregated from {len(frame_positions)} frames",
                            confidence=stability,
                        ))
        
        return mind_map, actions
    
    def evolve_appearance_order(self, mind_map: Dict[str, MindMapEntity], question: str) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """Appearance Order任务Evolution"""
        actions = []
        
        for label, entity in mind_map.items():
            if not entity.detections:
                continue
            
            # 置信度加权的首现帧
            frame_conf = defaultdict(list)
            for det in entity.detections:
                frame_conf[det.frame_idx].append(det.confidence)
            
            first_high_conf_frame = None
            for frame_idx in sorted(frame_conf.keys()):
                max_conf = max(frame_conf[frame_idx])
                if max_conf > 0.4:
                    first_high_conf_frame = frame_idx
                    break
            
            if first_high_conf_frame is None:
                first_high_conf_frame = min(frame_conf.keys())
            
            # 检测连续性分析
            frames = sorted(frame_conf.keys())
            if len(frames) >= 2:
                gaps = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
                max_gap = max(gaps)
                avg_gap = np.mean(gaps)
                continuity = 1.0 if max_gap < avg_gap * 3 else 0.5
            else:
                continuity = 0.5
            
            original_first_frame = entity.first_seen_frame
            if first_high_conf_frame != original_first_frame:
                entity.first_seen_frame = first_high_conf_frame
                actions.append(EvolutionAction(
                    action_type='first_frame_correction',
                    target_entity=label,
                    old_value=original_first_frame,
                    new_value=first_high_conf_frame,
                    reasoning=f"Using first high-confidence detection, continuity={continuity:.2f}",
                    confidence=continuity * entity.avg_confidence,
                ))
        
        return mind_map, actions
    
    def evolve_route(self, mind_map: Dict[str, MindMapEntity], question: str) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """Route Planning任务Evolution"""
        actions = []
        
        # 先应用基础校准
        mind_map, _ = self.evolve_size(mind_map, question)
        
        # 功能区域识别
        kitchen_objects = ['stove', 'refrigerator', 'fridge', 'sink', 'microwave', 'oven']
        bedroom_objects = ['bed', 'nightstand', 'dresser', 'closet']
        living_objects = ['sofa', 'couch', 'tv', 'television', 'coffee table']
        bathroom_objects = ['toilet', 'bathtub', 'shower']
        
        detected_areas = defaultdict(list)
        for label in mind_map.keys():
            label_lower = label.lower()
            if any(k in label_lower for k in kitchen_objects):
                detected_areas['kitchen'].append(label)
            elif any(k in label_lower for k in bedroom_objects):
                detected_areas['bedroom'].append(label)
            elif any(k in label_lower for k in living_objects):
                detected_areas['living_room'].append(label)
            elif any(k in label_lower for k in bathroom_objects):
                detected_areas['bathroom'].append(label)
        
        if detected_areas:
            actions.append(EvolutionAction(
                action_type='area_detection',
                target_entity='scene_layout',
                old_value='unknown',
                new_value=dict(detected_areas),
                reasoning=f"Detected {len(detected_areas)} functional areas",
                confidence=0.7,
            ))
        
        return mind_map, actions
    
    def evolve(self, mind_map: Dict[str, MindMapEntity], question: str, question_type: str) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """根据任务类型选择合适的Evolution策略"""
        all_actions = []
        
        # 通用预处理：低置信度过滤
        low_conf_labels = [label for label, entity in mind_map.items() if entity.avg_confidence < 0.25]
        for label in low_conf_labels:
            all_actions.append(EvolutionAction(
                action_type='low_confidence_filter',
                target_entity=label,
                old_value=f"conf={mind_map[label].avg_confidence:.2f}",
                new_value='removed',
                reasoning="Confidence below threshold (0.25)",
                confidence=0.9,
            ))
            del mind_map[label]
        
        # 任务特定Evolution
        if 'counting' in question_type.lower():
            mind_map, actions = self.evolve_counting(mind_map, question)
        elif 'size' in question_type.lower():
            mind_map, actions = self.evolve_size(mind_map, question)
        elif 'distance' in question_type.lower():
            mind_map, actions = self.evolve_distance(mind_map, question)
        elif 'direction' in question_type.lower():
            mind_map, actions = self.evolve_direction(mind_map, question)
        elif 'appearance' in question_type.lower() or 'order' in question_type.lower():
            mind_map, actions = self.evolve_appearance_order(mind_map, question)
        elif 'route' in question_type.lower():
            mind_map, actions = self.evolve_route(mind_map, question)
        else:
            mind_map, actions = self.evolve_size(mind_map, question)
        
        all_actions.extend(actions)
        
        return mind_map, all_actions


# ============================================================================
# 问答器
# ============================================================================

class DualReasoningQA:
    """双推理模式问答器"""
    
    def __init__(self, vl_model=None, vl_processor=None, device='cuda'):
        self.vl_model = vl_model
        self.vl_processor = vl_processor
        self.device = device
    
    def load_vl_model(self, model_path: str):
        """加载VL模型"""
        if self.vl_model is None:
            # 优先尝试 Qwen3-VL
            if 'qwen3' in model_path.lower() or 'Qwen3' in model_path:
                try:
                    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
                    logger.info(f"Loading Qwen3-VL model: {model_path}")
                    
                    self.vl_processor = AutoProcessor.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                    self.vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map=self.device,
                        trust_remote_code=True,
                    )
                    logger.info("Qwen3-VL model loaded successfully")
                    return
                except ImportError:
                    logger.warning("Qwen3VLForConditionalGeneration not available, falling back to Qwen2.5-VL")
            
            # 回退到 Qwen2.5-VL
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            logger.info(f"Loading Qwen2.5-VL model: {model_path}")
            
            self.vl_processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            logger.info("Qwen2.5-VL model loaded successfully")
    
    def _build_mind_map_text(self, mind_map: Dict[str, MindMapEntity]) -> str:
        """构建心智地图的完整文本描述"""
        if not mind_map:
            return "No objects detected."
        
        lines = []
        for label, entity in sorted(mind_map.items(), key=lambda x: -x[1].count):
            lines.append(entity.to_text())
        
        return "\n".join(lines)
    
    def _call_vl_model(self, prompt: str, video_path: str) -> Tuple[str, str]:
        """调用VL模型"""
        try:
            from qwen_vl_utils import process_vision_info
            
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "nframes": 8,
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = self.vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.vl_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            
            response = self.vl_processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            return response.strip(), response.strip()
            
        except Exception as e:
            logger.warning(f"VL model call failed: {e}")
            return "", str(e)
    
    def _extract_choice(self, response: str, options: List[str]) -> str:
        """从响应中提取选择答案"""
        response_clean = response.split('[')[0].strip()
        
        choice_match = re.search(r'^([A-D])', response_clean.upper())
        if choice_match:
            return choice_match.group(1)
        
        for line in response.split('\n')[::-1]:
            line = line.strip()
            if line and line[0].upper() in 'ABCD':
                return line[0].upper()
        
        return options[0][0] if options else "A"
    
    def vl_answer(self, question: str, video_path: str, mind_map: Dict[str, MindMapEntity], 
                  question_type: str, options: List[str] = None, calibration: CalibrationResult = None,
                  total_frames: int = 0) -> Tuple[str, str]:
        """VL推理 - 使用与V7完全一致的prompt"""
        if self.vl_model is None:
            return "0" if question_type in NUMERICAL_TASKS else "A", "VL model not loaded"
        
        mind_map_text = self._build_mind_map_text(mind_map)
        
        if question_type == 'object_counting':
            prompt = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to count specific objects in the scene.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
The following objects have been detected by our perception system (GroundingDINO + Depth Anything V3):
{mind_map_text}

=== IMPORTANT NOTES ===
1. The perception system may have missed some objects or made errors.
2. Use BOTH the detected objects data AND what you observe in the video frames to answer.
3. The 'count' field represents the maximum number of that object detected in any single frame.
4. Pay attention to object positions - objects at different positions are different instances.

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
1. Carefully examine the video frames.
2. Cross-reference with the detected objects list.
3. If you see objects in the video that are not in the list, count them too.
4. If the perception count seems incorrect based on what you see, trust your visual observation.

Please respond with ONLY a single integer number representing the count.
Do not include any explanation or units, just the number.

Answer:"""

        elif question_type == 'object_size_estimation':
            cal_obj = calibration.calibration_object if calibration else "none"
            cal_factor = calibration.scale_factor if calibration else 1.0
            cal_conf = calibration.confidence if calibration else 0.0
            prompt = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to estimate the size of a specific object.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

=== CALIBRATION INFORMATION ===
Calibration object: {cal_obj}
Scale factor: {cal_factor:.2f}
Calibration confidence: {cal_conf:.2f}

=== REFERENCE SIZES (for context) ===
- Standard door height: ~200 cm
- Standard chair height: ~80 cm (seat height ~45 cm)
- Standard bed length: ~200 cm
- Standard table height: ~75 cm
- Standard refrigerator height: ~175 cm

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
1. Look at the target object in the video frames.
2. Compare it with known reference objects (doors, chairs, etc.) for scale.
3. Use the calibration information if available.
4. Consider the 'size' field in the detected objects list, but verify with visual observation.

Please respond with ONLY a single integer number representing the size in centimeters.
Do not include any explanation or units, just the number.

Answer:"""

        elif question_type == 'room_size_estimation':
            cal_obj = calibration.calibration_object if calibration else "none"
            cal_factor = calibration.scale_factor if calibration else 1.0
            prompt = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to estimate the size (floor area) of the room.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

=== CALIBRATION INFORMATION ===
Calibration object: {cal_obj}
Scale factor: {cal_factor:.2f}

=== REFERENCE SIZES (for context) ===
- Small bedroom: 10-15 m²
- Standard bedroom: 15-20 m²
- Living room: 20-40 m²
- Small bathroom: 3-5 m²
- Kitchen: 10-15 m²

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
1. Observe the room in the video frames.
2. Look for reference objects (doors, beds, sofas) to estimate scale.
3. Consider the positions of detected objects to estimate room extent.
4. Use the spread of object positions as a hint for room size.

Please respond with ONLY a single integer number representing the room area in square meters.
Do not include any explanation or units, just the number.

Answer:"""

        elif question_type == 'object_abs_distance':
            cal_obj = calibration.calibration_object if calibration else "none"
            cal_factor = calibration.scale_factor if calibration else 1.0
            prompt = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to estimate the distance between two objects.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

=== CALIBRATION INFORMATION ===
Calibration object: {cal_obj}
Scale factor: {cal_factor:.2f}

=== REFERENCE DISTANCES ===
- Arm's length: ~0.6 m
- Standard door width: ~0.9 m
- Dining table width: ~0.9-1.2 m
- Sofa length: ~2-3 m

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
1. Locate both objects in the video frames.
2. Estimate the distance using visual cues and reference objects.
3. Consider the 3D positions from the detected objects list.
4. Account for depth (objects farther from camera may appear closer together).

Please respond with ONLY a decimal number representing the distance in meters.
Do not include any explanation or units, just the number (e.g., "2.5" or "1.8").

Answer:"""

        else:
            # 选择题 (direction, appearance_order, rel_distance, route_planning)
            options_text = "\n".join(options) if options else ""
            
            if 'appearance_order' in question_type:
                task_specific = """=== TASK: APPEARANCE ORDER ===
You need to determine the order in which objects FIRST APPEAR in the video.
- Pay attention to the FIRST FRAME where each object becomes visible.
- The 'first_seen_frame' in detected objects provides initial estimates but verify with the video.
- Objects may appear as the camera moves through the scene."""
            elif 'direction' in question_type:
                task_specific = """=== TASK: RELATIVE DIRECTION ===
You need to determine the spatial relationship (direction) between objects.
- Consider the viewpoint/camera position when determining directions.
- Use terms like: left/right, above/below, in front/behind, closer/farther.
- The position data in detected objects can help, but visual verification is important."""
            elif 'rel_distance' in question_type:
                task_specific = """=== TASK: RELATIVE DISTANCE COMPARISON ===
You need to compare distances between objects.
- Determine which pair of objects is closer/farther from each other.
- Use the 3D positions from detected objects as a reference.
- Visual depth cues in the video can help verify."""
            elif 'route' in question_type:
                task_specific = """=== TASK: ROUTE PLANNING ===
You need to determine the best path or sequence of objects to visit.
- Consider the spatial layout of objects in the room.
- Think about efficient navigation (avoiding backtracking).
- Use the positions from detected objects to understand the layout."""
            else:
                task_specific = """=== TASK: GENERAL SPATIAL REASONING ===
Analyze the spatial relationships in the scene to answer the question."""
            
            prompt = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene.

{task_specific}

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

=== VIDEO INFORMATION ===
Total frames in video: {total_frames}
The perception system sampled 32 frames uniformly from the video.

=== QUESTION ===
{question}

=== OPTIONS ===
{options_text}

=== INSTRUCTIONS ===
1. Carefully watch the video frames from beginning to end.
2. Use the detected objects information as reference.
3. Consider spatial relationships, camera movement, and object positions.
4. Choose the option that best matches your observation.

IMPORTANT: Answer with ONLY the option letter (A, B, C, or D).
You may add [Confidence: X%] at the end to indicate your confidence level.

Answer:"""
        
        response, raw_response = self._call_vl_model(prompt, video_path)
        
        # 提取答案
        if question_type in NUMERICAL_TASKS:
            match = re.search(r'[\d.]+', response)
            if match:
                return match.group(), f"VL response: {raw_response}"
            return "0", f"Failed to extract number: {raw_response}"
        else:
            answer = self._extract_choice(response, options or [])
            return answer, f"VL response: {raw_response}"


# ============================================================================
# Worker进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], vl_model_path: str, 
                   enable_evolution: bool, result_queue: mp.Queue):
    """GPU Worker进程"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    builder = MindMapBuilder(device='cuda', num_frames=32, box_threshold=0.25)
    calibrator = ScaleCalibrator()
    evolver = TaskSpecificEvolver() if enable_evolution else None
    qa = DualReasoningQA(device='cuda')
    
    vl_loaded = False
    try:
        qa.load_vl_model(vl_model_path)
        vl_loaded = True
        logger.info(f"GPU {gpu_id}: VL model loaded successfully")
    except Exception as e:
        logger.warning(f"GPU {gpu_id}: Failed to load VL model: {e}")
    
    results = []
    
    evolution_mode = "enabled" if enable_evolution else "disabled"
    for sample in tqdm(samples, desc=f"GPU {gpu_id} (Evolution: {evolution_mode})"):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 1. 构建心智地图
            target_objects = []
            if 'counting' in question_type:
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map, total_frames, sampled_frames = builder.build_from_video(video_path, target_objects)
            
            # 2. 基础校准（总是执行）
            calibration = calibrator.calculate_scale_factor(mind_map)
            mind_map = calibrator.apply_calibration(mind_map, calibration)
            
            # 3. 任务特定Evolution（根据开关）
            evolution_actions = []
            if enable_evolution and evolver:
                mind_map, actions = evolver.evolve(mind_map, question, question_type)
                evolution_actions.extend(actions)
            
            # 4. VL推理
            vl_pred, vl_reasoning = "", ""
            if vl_loaded:
                vl_pred, vl_reasoning = qa.vl_answer(
                    question, video_path, mind_map, question_type, options, calibration, total_frames
                )
            
            # 5. 评估
            is_numerical = question_type in NUMERICAL_TASKS
            if is_numerical:
                pred_val = normalize_number(vl_pred)
                gt_val = normalize_number(gt)
                vl_score = mean_relative_accuracy(pred_val, gt_val) if pred_val and gt_val else 0.0
            else:
                pred_norm = vl_pred.strip().upper() if vl_pred else ""
                gt_norm = gt.strip().upper()
                if len(pred_norm) > 1 and pred_norm[1] in '.、':
                    pred_norm = pred_norm[0]
                if len(gt_norm) > 1 and gt_norm[1] in '.、':
                    gt_norm = gt_norm[0]
                vl_score = 1.0 if pred_norm == gt_norm else 0.0
            
            results.append({
                'scene_name': sample['scene_name'],
                'question': question,
                'question_type': question_type,
                'ground_truth': gt,
                'options': options,
                'vl_prediction': vl_pred,
                'vl_reasoning': vl_reasoning,
                'vl_score': vl_score,
                'enable_evolution': enable_evolution,
                'evolution_actions_count': len(evolution_actions),
                'calibration': {
                    'object': calibration.calibration_object,
                    'scale_factor': calibration.scale_factor,
                },
            })
            
        except Exception as e:
            logger.error(f"Error processing {sample['scene_name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'scene_name': sample['scene_name'],
                'question': question,
                'question_type': question_type,
                'ground_truth': gt,
                'vl_prediction': '',
                'vl_score': 0.0,
                'enable_evolution': enable_evolution,
                'error': str(e),
            })
    
    builder.unload()
    if vl_loaded:
        del qa.vl_model
        del qa.vl_processor
    gc.collect()
    torch.cuda.empty_cache()
    
    result_queue.put(results)


# ============================================================================
# 数据加载
# ============================================================================

def load_vsibench_data() -> List[Dict]:
    """加载VSI-Bench数据集"""
    vsibench_json = "/home/tione/notebook/tianjungu/projects/Spatial-MLLM/evaluate/annotation/eval_vsibench.json"
    
    logger.info(f"加载VSI-Bench数据集从: {vsibench_json}")
    with open(vsibench_json, 'r') as f:
        dataset = json.load(f)
    
    samples = []
    vsibench_video_base = "/home/tione/notebook/tianjungu/hf_cache/vsibench"
    
    for item in dataset:
        rel_path = item.get('path', '')
        if rel_path.startswith('./'):
            rel_path = rel_path[2:]
        
        video_path = os.path.join(vsibench_video_base, rel_path)
        if not os.path.exists(video_path):
            continue
        
        scene_name = os.path.basename(rel_path).replace('.mp4', '')
        
        question = item['problem']
        solution = item.get('solution', '')
        
        match = re.search(r'<answer>(.*?)</answer>', solution)
        ground_truth = match.group(1) if match else solution
        
        samples.append({
            'scene_name': scene_name,
            'video_path': video_path,
            'question': question,
            'question_type': item.get('original_question_type', 'unknown'),
            'options': item.get('options', []),
            'ground_truth': ground_truth,
        })
    
    logger.info(f"加载了 {len(samples)} 个有效样本")
    return samples


# ============================================================================
# 主函数
# ============================================================================

def run_experiment(data: List[Dict], vl_model_path: str, enable_evolution: bool, 
                   num_gpus: int, output_dir: Path) -> Dict:
    """运行单次实验"""
    exp_name = "with_evolution" if enable_evolution else "no_evolution"
    logger.info(f"\n{'='*80}")
    logger.info(f"开始实验: {exp_name}")
    logger.info(f"Evolution开关: {enable_evolution}")
    logger.info(f"{'='*80}")
    
    # 分配到各GPU
    samples_per_gpu = len(data) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(data)
        gpu_samples.append(data[start:end])
    
    # 启动多进程
    result_queue = mp.Queue()
    processes = []
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_samples[gpu_id], vl_model_path, enable_evolution, result_queue)
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
    
    # 统计
    type_stats = defaultdict(lambda: {'total': 0, 'score_sum': 0})
    
    for r in all_results:
        qtype = r['question_type']
        type_stats[qtype]['total'] += 1
        type_stats[qtype]['score_sum'] += r.get('vl_score', 0)
    
    overall_score = sum(ts['score_sum'] for ts in type_stats.values())
    overall_total = sum(ts['total'] for ts in type_stats.values())
    
    summary = {
        'experiment': exp_name,
        'enable_evolution': enable_evolution,
        'results_by_type': {
            qtype: {
                'accuracy': stats['score_sum'] / stats['total'] if stats['total'] > 0 else 0,
                'samples': stats['total'],
            }
            for qtype, stats in type_stats.items()
        },
        'overall': {
            'accuracy': overall_score / overall_total if overall_total > 0 else 0,
            'samples': overall_total,
        }
    }
    
    # 保存结果
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='V7 Evolution消融实验')
    parser.add_argument('--vl-model', type=str, 
                       default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct',
                       help='VL model path')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--run-both', action='store_true', default=True,
                       help='Run both experiments (with and without evolution)')
    parser.add_argument('--enable-evolution', type=str, choices=['true', 'false'],
                       help='Run only one experiment')
    
    args = parser.parse_args()
    
    # 加载数据
    data = load_vsibench_data()
    logger.info(f"加载 {len(data)} 条数据")
    
    if args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"限制为: {len(data)} 条数据")
    
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"evolution_ablation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summaries = {}
    
    if args.enable_evolution:
        # 只运行指定的实验
        enable = args.enable_evolution == 'true'
        summary = run_experiment(data, args.vl_model, enable, num_gpus, output_dir)
        summaries['with_evolution' if enable else 'no_evolution'] = summary
    else:
        # 运行两个实验
        for enable_evolution in [False, True]:
            summary = run_experiment(data, args.vl_model, enable_evolution, num_gpus, output_dir)
            summaries['with_evolution' if enable_evolution else 'no_evolution'] = summary
    
    # 打印对比结果
    print("\n" + "=" * 100)
    print("V7 Evolution消融实验 - 结果对比")
    print("=" * 100)
    print(f"VL Model: {args.vl_model}")
    print("-" * 100)
    
    if len(summaries) == 2:
        no_evo = summaries['no_evolution']
        with_evo = summaries['with_evolution']
        
        print(f"{'任务类型':<35} {'无Evolution':>15} {'有Evolution':>15} {'差异':>15}")
        print("-" * 100)
        
        all_types = set(no_evo['results_by_type'].keys()) | set(with_evo['results_by_type'].keys())
        
        for qtype in sorted(all_types):
            no_evo_acc = no_evo['results_by_type'].get(qtype, {}).get('accuracy', 0) * 100
            with_evo_acc = with_evo['results_by_type'].get(qtype, {}).get('accuracy', 0) * 100
            diff = with_evo_acc - no_evo_acc
            diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
            
            print(f"{qtype:<35} {no_evo_acc:>14.2f}% {with_evo_acc:>14.2f}% {diff_str:>15}")
        
        print("-" * 100)
        no_evo_overall = no_evo['overall']['accuracy'] * 100
        with_evo_overall = with_evo['overall']['accuracy'] * 100
        overall_diff = with_evo_overall - no_evo_overall
        overall_diff_str = f"+{overall_diff:.2f}%" if overall_diff >= 0 else f"{overall_diff:.2f}%"
        
        print(f"{'Overall':<35} {no_evo_overall:>14.2f}% {with_evo_overall:>14.2f}% {overall_diff_str:>15}")
        print("=" * 100)
        
        print(f"\n结论: Evolution {'有帮助' if overall_diff > 0 else '没有帮助或有负面影响'} ({overall_diff_str})")
    else:
        for name, summary in summaries.items():
            print(f"\n{name}:")
            print(f"  Overall: {summary['overall']['accuracy']*100:.2f}%")
    
    # 保存对比总结
    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    
    print(f"\n结果保存到: {output_dir}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
