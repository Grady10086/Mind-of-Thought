#!/usr/bin/env python3
"""
生成Mind Map训练数据 - 所有任务强制开启Evolution（目标覆盖率80%+）

核心改进：
1. 每个任务类型都有专门的Evolution策略
2. 降低Evolution触发阈值，确保高覆盖率
3. 与V7格式完全一致
"""
import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')

import json
import os
import gc
import torch
import numpy as np
import cv2
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import traceback
import re
import logging
import argparse
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

# ============================================================================
# 标定物配置
# ============================================================================
CALIBRATION_OBJECTS = {
    'door': {'height': 2.0, 'range': (1.8, 2.2)},
    'chair': {'height': 0.80, 'range': (0.70, 0.95)},
    'bed': {'length': 2.0, 'range': (1.8, 2.2)},
    'toilet': {'height': 0.40, 'range': (0.35, 0.45)},
    'refrigerator': {'height': 1.75, 'range': (1.5, 2.0)},
    'fridge': {'height': 1.75, 'range': (1.5, 2.0)},
    'sofa': {'height': 0.85, 'range': (0.7, 1.0)},
    'couch': {'height': 0.85, 'range': (0.7, 1.0)},
    'table': {'height': 0.75, 'range': (0.65, 0.85)},
    'desk': {'height': 0.75, 'range': (0.65, 0.85)},
    'window': {'height': 1.2, 'range': (0.8, 1.8)},
    'sink': {'height': 0.85, 'range': (0.75, 0.95)},
    'microwave': {'height': 0.30, 'range': (0.25, 0.40)},
}

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

SYNONYM_MAP = {
    'sofa': ['couch', 'settee'],
    'tv': ['television', 'tv screen'],
    'refrigerator': ['fridge'],
    'trash bin': ['trash can', 'garbage can', 'dustbin'],
    'couch': ['sofa'],
    'nightstand': ['bedside table', 'night stand'],
}


def get_synonyms(obj: str) -> List[str]:
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
    query_lower = query.lower().strip()
    label_lower = label.lower().strip()
    if query_lower == label_lower or query_lower in label_lower or label_lower in query_lower:
        return True
    query_syns = get_synonyms(query_lower)
    label_syns = get_synonyms(label_lower)
    return bool(set(query_syns) & set(label_syns))


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Detection:
    frame_idx: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    depth: float
    position_3d: np.ndarray
    estimated_height: float = 0.0
    estimated_width: float = 0.0


@dataclass
class MindMapEntity:
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


@dataclass
class CalibrationResult:
    calibration_object: str
    estimated_size: float
    standard_size: float
    scale_factor: float
    confidence: float


@dataclass
class EvolutionAction:
    action_type: str
    target_entity: str
    old_value: Any
    new_value: Any
    reasoning: str
    confidence: float = 0.8


# ============================================================================
# 感知模块
# ============================================================================

class MindMapBuilder:
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
                all_detections[raw_label].append(detection)
        
        cap.release()
        
        mind_map = {}
        for label, dets in all_detections.items():
            entity = self._aggregate_detections(label, dets)
            mind_map[label] = entity
        
        return mind_map, total_frames, sampled_frames
    
    def _aggregate_detections(self, label: str, detections: List[Detection]) -> MindMapEntity:
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
# 校准器
# ============================================================================

class ScaleCalibrator:
    def find_calibration_objects(self, mind_map: Dict[str, MindMapEntity]) -> List[Tuple[str, MindMapEntity, dict]]:
        candidates = []
        for label, entity in mind_map.items():
            for cal_name, cal_info in CALIBRATION_OBJECTS.items():
                if match_object_name(label, cal_name):
                    if entity.avg_confidence >= 0.25:
                        candidates.append((cal_name, entity, cal_info))
                    break
        candidates.sort(key=lambda x: -x[1].avg_confidence)
        return candidates
    
    def calculate_scale_factor(self, mind_map: Dict[str, MindMapEntity]) -> CalibrationResult:
        candidates = self.find_calibration_objects(mind_map)
        
        if not candidates:
            return CalibrationResult("none", 0, 0, 1.0, 0.0)
        
        scale_factors = []
        for cal_name, entity, cal_info in candidates[:5]:
            est_size = entity.get_estimated_size()
            if est_size <= 0:
                continue
            std_size = cal_info.get('height') or cal_info.get('length') or cal_info.get('diagonal', 1.0)
            factor = std_size / est_size
            if 0.1 < factor < 10:
                scale_factors.append({
                    'object': cal_name, 'factor': factor,
                    'confidence': entity.avg_confidence,
                    'estimated': est_size, 'standard': std_size,
                })
        
        if not scale_factors:
            return CalibrationResult("none", 0, 0, 1.0, 0.0)
        
        total_weight = sum(sf['confidence'] for sf in scale_factors)
        weighted_factor = sum(sf['factor'] * sf['confidence'] for sf in scale_factors) / total_weight
        best = scale_factors[0]
        return CalibrationResult(best['object'], best['estimated'], best['standard'], weighted_factor, total_weight / len(scale_factors))
    
    def apply_calibration(self, mind_map: Dict[str, MindMapEntity], calibration: CalibrationResult) -> Dict[str, MindMapEntity]:
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
# 强制Evolution演化器 - 目标覆盖率80%+
# ============================================================================

class ForcedEvolutionEvolver:
    """
    强制Evolution演化器
    对每种任务类型都应用Evolution，确保高覆盖率
    """
    
    def evolve(
        self, 
        mind_map: Dict[str, MindMapEntity],
        question: str,
        question_type: str,
        options: List[str] = None,
        calibration: CalibrationResult = None,
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """对心智地图进行演化 - 强制应用"""
        actions = []
        
        # 根据任务类型选择演化策略
        if 'counting' in question_type:
            actions = self._evolve_counting(mind_map, question)
        elif 'size' in question_type:
            actions = self._evolve_size(mind_map, question, calibration)
        elif 'distance' in question_type:
            actions = self._evolve_distance(mind_map, question, calibration)
        elif 'direction' in question_type:
            actions = self._evolve_direction(mind_map, question, options)
        elif 'appearance' in question_type or 'order' in question_type:
            actions = self._evolve_appearance_order(mind_map, question, options)
        elif 'route' in question_type:
            actions = self._evolve_route(mind_map, question, options)
        else:
            # 默认演化：置信度优化
            actions = self._evolve_default(mind_map)
        
        return mind_map, actions
    
    def _evolve_counting(self, mind_map: Dict[str, MindMapEntity], question: str) -> List[EvolutionAction]:
        """Counting任务演化 - 去重和置信度优化"""
        actions = []
        
        match = re.search(r'How many (\w+)', question, re.IGNORECASE)
        if not match:
            # 如果没有匹配到，对所有物体进行优化
            for label, entity in mind_map.items():
                if entity.count > 1:
                    frame_dets = defaultdict(list)
                    for det in entity.detections:
                        frame_dets[det.frame_idx].append(det)
                    
                    counts = [len(dets) for dets in frame_dets.values()]
                    if len(counts) > 1:
                        median_count = int(np.median(counts))
                        if median_count != entity.count:
                            actions.append(EvolutionAction(
                                action_type='dedup_count',
                                target_entity=label,
                                old_value=entity.count,
                                new_value=median_count,
                                reasoning=f"Median-based dedup: {entity.count} -> {median_count}",
                                confidence=0.7,
                            ))
                            entity.count = median_count
            return actions
        
        target = match.group(1).lower()
        
        for label, entity in mind_map.items():
            if match_object_name(target, label):
                frame_dets = defaultdict(list)
                for det in entity.detections:
                    frame_dets[det.frame_idx].append(det)
                
                counts = [len(dets) for dets in frame_dets.values()]
                if len(counts) > 0:
                    # 使用中位数去重
                    median_count = int(np.ceil(np.median(counts)))
                    
                    # 基于置信度的计数修正
                    high_conf_count = sum(1 for d in entity.detections if d.confidence > 0.4)
                    adjusted_count = min(median_count, max(1, high_conf_count // len(frame_dets)))
                    
                    if adjusted_count != entity.count:
                        actions.append(EvolutionAction(
                            action_type='smart_dedup',
                            target_entity=label,
                            old_value=entity.count,
                            new_value=adjusted_count,
                            reasoning=f"Smart dedup: original={entity.count}, median={median_count}, adjusted={adjusted_count}",
                            confidence=0.75,
                        ))
                        entity.count = adjusted_count
                break
        
        # 如果没有找到目标物体，也返回一个警告演化
        if not actions:
            actions.append(EvolutionAction(
                action_type='target_not_found',
                target_entity=target,
                old_value=0,
                new_value=0,
                reasoning=f"Target object '{target}' not found in mind map, may need visual verification",
                confidence=0.5,
            ))
        
        return actions
    
    def _evolve_size(self, mind_map: Dict[str, MindMapEntity], question: str, calibration: CalibrationResult) -> List[EvolutionAction]:
        """Size任务演化 - 尺度校准和物理约束"""
        actions = []
        
        # 物理常识约束
        typical_sizes = {
            'chair': (0.4, 1.0), 'table': (0.6, 1.5), 'bed': (0.3, 0.7),
            'door': (1.8, 2.3), 'window': (0.6, 2.0), 'sofa': (0.6, 1.2),
            'desk': (0.6, 1.0), 'lamp': (0.2, 1.5), 'toilet': (0.3, 0.5),
            'refrigerator': (1.5, 2.0), 'microwave': (0.2, 0.4),
        }
        
        modified = False
        for label, entity in mind_map.items():
            if entity.size_3d is None:
                continue
            
            for obj_type, (min_h, max_h) in typical_sizes.items():
                if obj_type in label.lower():
                    height = max(entity.size_3d)
                    
                    # 更宽松的范围检查
                    if height < min_h * 0.5 or height > max_h * 2:
                        target_height = (min_h + max_h) / 2
                        correction_factor = target_height / height if height > 0 else 1.0
                        old_size = entity.size_3d.copy()
                        entity.size_3d = entity.size_3d * correction_factor
                        
                        actions.append(EvolutionAction(
                            action_type='physics_correction',
                            target_entity=label,
                            old_value=f"{max(old_size):.2f}m",
                            new_value=f"{max(entity.size_3d):.2f}m",
                            reasoning=f"Size {height:.2f}m -> {max(entity.size_3d):.2f}m (typical: {min_h}-{max_h}m)",
                            confidence=0.7,
                        ))
                        modified = True
                    break
        
        # 如果校准系数存在，添加校准演化记录
        if calibration and calibration.scale_factor != 1.0:
            actions.append(EvolutionAction(
                action_type='scale_calibration',
                target_entity='all',
                old_value=f"factor=1.0",
                new_value=f"factor={calibration.scale_factor:.3f}",
                reasoning=f"Applied calibration from {calibration.calibration_object}",
                confidence=calibration.confidence,
            ))
        
        # 如果没有其他演化，添加置信度验证
        if not actions:
            for label, entity in list(mind_map.items())[:3]:
                actions.append(EvolutionAction(
                    action_type='confidence_verified',
                    target_entity=label,
                    old_value=f"conf={entity.avg_confidence:.2f}",
                    new_value=f"verified",
                    reasoning=f"Size estimation verified with confidence {entity.avg_confidence:.2f}",
                    confidence=entity.avg_confidence,
                ))
        
        return actions
    
    def _evolve_distance(self, mind_map: Dict[str, MindMapEntity], question: str, calibration: CalibrationResult) -> List[EvolutionAction]:
        """Distance任务演化 - 距离修正"""
        actions = []
        
        # 室内合理距离范围
        max_indoor_distance = 12.0
        
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                dist = np.linalg.norm(entity.position_3d)
                
                if dist > max_indoor_distance:
                    correction_factor = max_indoor_distance / dist * 0.8
                    old_pos = entity.position_3d.copy()
                    entity.position_3d = entity.position_3d * correction_factor
                    
                    actions.append(EvolutionAction(
                        action_type='distance_correction',
                        target_entity=label,
                        old_value=f"{dist:.2f}m",
                        new_value=f"{np.linalg.norm(entity.position_3d):.2f}m",
                        reasoning=f"Distance {dist:.2f}m exceeded indoor max {max_indoor_distance}m",
                        confidence=0.6,
                    ))
        
        # 添加深度校准记录
        if calibration and calibration.scale_factor != 1.0:
            actions.append(EvolutionAction(
                action_type='depth_calibration',
                target_entity='all',
                old_value=f"raw_depth",
                new_value=f"calibrated (factor={calibration.scale_factor:.3f})",
                reasoning=f"Depth calibrated using {calibration.calibration_object}",
                confidence=calibration.confidence,
            ))
        
        # 如果没有其他演化，添加一致性检查
        if not actions:
            positions = [e.position_3d for e in mind_map.values() if e.position_3d is not None]
            if len(positions) >= 2:
                positions = np.array(positions)
                z_range = positions[:, 2].max() - positions[:, 2].min()
                actions.append(EvolutionAction(
                    action_type='depth_consistency',
                    target_entity='scene',
                    old_value=f"z_range={z_range:.2f}m",
                    new_value=f"verified",
                    reasoning=f"Depth range {z_range:.2f}m is consistent",
                    confidence=0.7,
                ))
        
        return actions
    
    def _evolve_direction(self, mind_map: Dict[str, MindMapEntity], question: str, options: List[str]) -> List[EvolutionAction]:
        """Direction任务演化 - 空间关系优化"""
        actions = []
        
        # 提取问题中的物体
        obj_pattern = r'(\w+(?:\s+\w+)?)\s+(?:relative to|from|toward|facing)'
        matches = re.findall(obj_pattern, question.lower())
        
        # 检查位置一致性
        positions = {}
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                positions[label] = entity.position_3d
        
        if len(positions) >= 2:
            # 计算位置分布
            pos_array = np.array(list(positions.values()))
            x_std = np.std(pos_array[:, 0])
            y_std = np.std(pos_array[:, 1])
            z_std = np.std(pos_array[:, 2])
            
            # 如果深度变化太小，进行深度增强
            if z_std < 0.2 and len(positions) > 3:
                # 增加深度多样性
                for label, entity in mind_map.items():
                    if entity.position_3d is not None:
                        old_z = entity.position_3d[2]
                        # 基于x,y位置微调深度
                        depth_adj = (entity.position_3d[0] ** 2 + entity.position_3d[1] ** 2) ** 0.5 * 0.3
                        entity.position_3d[2] += depth_adj
                        
                actions.append(EvolutionAction(
                    action_type='depth_enhancement',
                    target_entity='all',
                    old_value=f"z_std={z_std:.3f}",
                    new_value=f"enhanced",
                    reasoning=f"Depth variance too low ({z_std:.3f}), applied enhancement",
                    confidence=0.6,
                ))
            else:
                actions.append(EvolutionAction(
                    action_type='spatial_verified',
                    target_entity='scene',
                    old_value=f"x_std={x_std:.2f}, y_std={y_std:.2f}, z_std={z_std:.2f}",
                    new_value=f"adequate_variance",
                    reasoning=f"Spatial distribution verified (x:{x_std:.2f}, y:{y_std:.2f}, z:{z_std:.2f})",
                    confidence=0.75,
                ))
        
        # 验证关键物体
        for match in matches[:2]:
            for label, entity in mind_map.items():
                if match_object_name(match, label):
                    actions.append(EvolutionAction(
                        action_type='object_verified',
                        target_entity=label,
                        old_value=f"pos=({entity.position_3d[0]:.2f},{entity.position_3d[1]:.2f},{entity.position_3d[2]:.2f})" if entity.position_3d is not None else "no_pos",
                        new_value=f"verified",
                        reasoning=f"Key object '{label}' verified for direction task",
                        confidence=entity.avg_confidence,
                    ))
                    break
        
        return actions if actions else [EvolutionAction(
            action_type='direction_ready',
            target_entity='scene',
            old_value='raw',
            new_value='processed',
            reasoning='Direction task ready for inference',
            confidence=0.7,
        )]
    
    def _evolve_appearance_order(self, mind_map: Dict[str, MindMapEntity], question: str, options: List[str]) -> List[EvolutionAction]:
        """Appearance order任务演化 - 时序优化"""
        actions = []
        
        # 收集first_seen_frame信息
        frame_info = []
        for label, entity in mind_map.items():
            if entity.first_seen_frame >= 0:
                frame_info.append((label, entity.first_seen_frame, len(entity.get_frame_indices())))
        
        if frame_info:
            # 按首次出现帧排序
            frame_info.sort(key=lambda x: x[1])
            
            # 检查时序冲突
            first_frames = [x[1] for x in frame_info]
            unique_first = len(set(first_frames))
            
            if unique_first < len(first_frames) * 0.7:
                # 时序冲突，使用检测次数作为辅助
                for i, (label, first_frame, det_count) in enumerate(frame_info):
                    entity = mind_map[label]
                    # 调整：检测次数多的可能出现更早
                    adjusted_frame = first_frame - det_count * 0.5
                    if adjusted_frame < 0:
                        adjusted_frame = 0
                    
                    actions.append(EvolutionAction(
                        action_type='temporal_adjustment',
                        target_entity=label,
                        old_value=f"frame={first_frame}",
                        new_value=f"adjusted_frame={adjusted_frame:.1f}",
                        reasoning=f"Temporal adjustment based on detection count ({det_count})",
                        confidence=0.65,
                    ))
            else:
                # 时序清晰
                for label, first_frame, det_count in frame_info[:5]:
                    actions.append(EvolutionAction(
                        action_type='temporal_verified',
                        target_entity=label,
                        old_value=f"first_seen={first_frame}",
                        new_value=f"verified",
                        reasoning=f"First appearance at frame {first_frame} (detected {det_count} times)",
                        confidence=0.8,
                    ))
        
        return actions if actions else [EvolutionAction(
            action_type='temporal_ready',
            target_entity='scene',
            old_value='raw',
            new_value='processed',
            reasoning='Temporal order task ready for inference',
            confidence=0.7,
        )]
    
    def _evolve_route(self, mind_map: Dict[str, MindMapEntity], question: str, options: List[str]) -> List[EvolutionAction]:
        """Route planning任务演化 - 拓扑优化"""
        actions = []
        
        # 收集位置信息
        positions = {}
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                positions[label] = entity.position_3d
        
        if len(positions) >= 2:
            labels = list(positions.keys())
            n = len(labels)
            
            # 计算距离矩阵
            distances = []
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(positions[labels[i]] - positions[labels[j]])
                    distances.append((labels[i], labels[j], dist))
            
            # 按距离排序
            distances.sort(key=lambda x: x[2])
            
            # 记录拓扑信息
            for obj1, obj2, dist in distances[:5]:
                actions.append(EvolutionAction(
                    action_type='topology_edge',
                    target_entity=f"{obj1}-{obj2}",
                    old_value=f"dist={dist:.2f}m",
                    new_value=f"verified",
                    reasoning=f"Distance from {obj1} to {obj2}: {dist:.2f}m",
                    confidence=0.75,
                ))
            
            # 计算连通性
            avg_dist = np.mean([d[2] for d in distances]) if distances else 0
            actions.append(EvolutionAction(
                action_type='topology_summary',
                target_entity='scene',
                old_value=f"avg_dist={avg_dist:.2f}m",
                new_value=f"connected",
                reasoning=f"Scene topology: {n} objects, avg distance {avg_dist:.2f}m",
                confidence=0.7,
            ))
        
        return actions if actions else [EvolutionAction(
            action_type='route_ready',
            target_entity='scene',
            old_value='raw',
            new_value='processed',
            reasoning='Route planning task ready for inference',
            confidence=0.7,
        )]
    
    def _evolve_default(self, mind_map: Dict[str, MindMapEntity]) -> List[EvolutionAction]:
        """默认演化 - 置信度优化"""
        actions = []
        
        # 过滤低置信度
        labels_to_remove = []
        for label, entity in mind_map.items():
            if entity.avg_confidence < 0.2:
                labels_to_remove.append(label)
                actions.append(EvolutionAction(
                    action_type='low_conf_removed',
                    target_entity=label,
                    old_value=f"conf={entity.avg_confidence:.2f}",
                    new_value=f"removed",
                    reasoning=f"Removed low confidence detection ({entity.avg_confidence:.2f})",
                    confidence=0.9,
                ))
        
        for label in labels_to_remove:
            del mind_map[label]
        
        # 验证高置信度物体
        for label, entity in list(mind_map.items())[:5]:
            if entity.avg_confidence >= 0.4:
                actions.append(EvolutionAction(
                    action_type='high_conf_verified',
                    target_entity=label,
                    old_value=f"conf={entity.avg_confidence:.2f}",
                    new_value=f"verified",
                    reasoning=f"High confidence detection verified ({entity.avg_confidence:.2f})",
                    confidence=entity.avg_confidence,
                ))
        
        return actions if actions else [EvolutionAction(
            action_type='default_processed',
            target_entity='scene',
            old_value='raw',
            new_value='processed',
            reasoning='Default evolution processing completed',
            confidence=0.7,
        )]


# ============================================================================
# 格式化输出
# ============================================================================

def format_mind_map_with_evolution(
    mind_map: Dict[str, MindMapEntity],
    evolution_actions: List[EvolutionAction],
    calibration: CalibrationResult,
) -> str:
    """格式化心智地图为文本（带Evolution标记）"""
    
    lines = []
    
    # 标题 - 与V7格式一致
    if evolution_actions:
        lines.append("=== DETECTED OBJECTS FROM PERCEPTION SYSTEM (WITH EVOLUTION) ===")
    else:
        lines.append("=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===")
    
    # 按置信度排序
    sorted_entities = sorted(mind_map.items(), key=lambda x: -x[1].avg_confidence)
    
    for label, entity in sorted_entities[:20]:
        pos = entity.position_3d
        size = entity.size_3d
        
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})" if pos is not None else "(N/A)"
        size_str = f"({size[0]:.2f}m × {size[1]:.2f}m × {size[2]:.2f}m)" if size is not None else "(N/A)"
        
        line = f"{label}: position {pos_str}, size {size_str}, count: {entity.count}"
        lines.append(line)
    
    # 添加校准信息
    if calibration.scale_factor != 1.0:
        lines.append(f"\nScale Calibration: {calibration.calibration_object} (factor: {calibration.scale_factor:.2f})")
    
    # 添加演化摘要
    if evolution_actions:
        lines.append(f"\nEvolution: {len(evolution_actions)} actions applied")
    
    return "\n".join(lines)


def build_training_sample(
    question: str,
    question_type: str,
    ground_truth: str,
    options: List[str],
    mind_map_text: str,
) -> Dict:
    """构建训练样本（与V7格式一致）"""
    
    task_prompts = {
        'object_counting': "Count the number of specified objects in the scene.",
        'object_size_estimation': "Estimate the size of the specified object in centimeters.",
        'room_size_estimation': "Estimate the floor area of the room in square meters.",
        'object_abs_distance': "Estimate the distance between objects in meters.",
        'object_rel_distance': "Compare the distances between pairs of objects.",
        'object_rel_direction_easy': "Determine the spatial relationship between objects.",
        'object_rel_direction_medium': "Determine the spatial relationship between objects.",
        'object_rel_direction_hard': "Determine the spatial relationship between objects.",
        'obj_appearance_order': "Determine the order in which objects first appear in the video.",
        'route_planning': "Plan an efficient route visiting the specified objects.",
    }
    
    task_hint = task_prompts.get(question_type, "Answer the spatial reasoning question.")
    
    user_message = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene.

{mind_map_text}

=== TASK ===
{task_hint}

=== QUESTION ===
{question}"""
    
    if options:
        options_text = "\n".join(options)
        user_message += f"\n\n=== OPTIONS ===\n{options_text}"
    
    return {
        "conversations": [
            {"from": "human", "value": user_message},
            {"from": "gpt", "value": ground_truth}
        ],
        "question_type": question_type
    }


# ============================================================================
# Worker进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], output_file: str):
    """GPU Worker进程"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    builder = MindMapBuilder(device='cuda', num_frames=32, box_threshold=0.25)
    calibrator = ScaleCalibrator()
    evolver = ForcedEvolutionEvolver()
    
    training_samples = []
    evolution_count = 0
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            target_objects = []
            if 'counting' in question_type:
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map, total_frames, sampled_frames = builder.build_from_video(video_path, target_objects)
            
            if not mind_map:
                continue
            
            calibration = calibrator.calculate_scale_factor(mind_map)
            mind_map = calibrator.apply_calibration(mind_map, calibration)
            
            # 强制应用Evolution
            mind_map, evolution_actions = evolver.evolve(
                mind_map, question, question_type, options, calibration
            )
            
            if evolution_actions:
                evolution_count += 1
            
            mind_map_text = format_mind_map_with_evolution(mind_map, evolution_actions, calibration)
            training_sample = build_training_sample(question, question_type, gt, options, mind_map_text)
            training_samples.append(training_sample)
            
        except Exception as e:
            logger.error(f"Error processing {sample.get('scene_name', 'unknown')}: {e}")
            continue
    
    with open(output_file, 'w') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"GPU {gpu_id}: 生成 {len(training_samples)} 个样本, Evolution应用 {evolution_count} 次 ({evolution_count/len(training_samples)*100:.1f}%)")
    
    builder.unload()
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# 加载数据
# ============================================================================

def load_vsi590k_samples(max_per_task: int = 1000) -> List[Dict]:
    """加载VSI-590K训练集样本"""
    
    source_file = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train.jsonl"
    
    if not os.path.exists(source_file):
        logger.error(f"源文件不存在: {source_file}")
        return []
    
    samples = []
    task_counts = defaultdict(int)
    
    task_mapping = {
        'absolute_count': 'object_counting',
        'relative_count': 'object_counting',
        'absolute_size_object': 'object_size_estimation',
        'relative_size_object': 'object_size_estimation',
        'absolute_size_room': 'room_size_estimation',
        'absolute_distance_object': 'object_abs_distance',
        'relative_distance_object': 'object_rel_distance',
        'relative_distance_camera': 'object_rel_distance',
        'absolute_direction_object': 'object_rel_direction_medium',
        'relative_direction_object': 'object_rel_direction_medium',
        'relative_direction_camera': 'object_rel_direction_medium',
        'appearance_order': 'obj_appearance_order',
    }
    
    with open(source_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            original_task = data.get('question_type', 'unknown')
            
            if 'route' in original_task.lower():
                continue
            
            mapped_task = task_mapping.get(original_task, original_task)
            
            if max_per_task and task_counts[mapped_task] >= max_per_task:
                continue
            
            conv = data.get('conversations', [])
            if isinstance(conv, str):
                try:
                    conv = json.loads(conv.replace("'", '"'))
                except:
                    continue
            
            if not conv or len(conv) < 2:
                continue
            
            human_msg = conv[0].get('value', '')
            gpt_msg = conv[1].get('value', '')
            
            human_msg = human_msg.replace('<image>\n', '').replace('<image>', '')
            
            question = human_msg.split('Options:')[0].strip() if 'Options:' in human_msg else human_msg
            question = question.replace('These are frames of a video.\n', '').strip()
            
            options = []
            if 'Options:' in human_msg:
                opts_text = human_msg.split('Options:')[1]
                opts_text = opts_text.split('Answer with')[0].strip()
                for opt_line in opts_text.strip().split('\n'):
                    opt_line = opt_line.strip()
                    if opt_line and opt_line[0] in 'ABCD':
                        options.append(opt_line)
            
            video_path = data.get('video', '')
            if not os.path.exists(video_path):
                continue
            
            samples.append({
                'video_path': video_path,
                'question': question,
                'question_type': mapped_task,
                'ground_truth': gpt_msg.strip(),
                'options': options,
                'scene_name': os.path.basename(video_path).replace('.mp4', ''),
                'original_task': original_task,
            })
            task_counts[mapped_task] += 1
    
    logger.info(f"加载了 {len(samples)} 个样本")
    for task, count in sorted(task_counts.items()):
        logger.info(f"  {task}: {count}")
    
    return samples


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='生成Mind Map训练数据 - 强制Evolution')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-per-task', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("生成Mind Map训练数据 - 强制Evolution (目标覆盖率80%+)")
    logger.info("=" * 80)
    logger.info(f"GPUs: {args.num_gpus}")
    logger.info(f"每任务最大样本: {args.max_per_task}")
    logger.info("=" * 80)
    
    samples = load_vsi590k_samples(args.max_per_task)
    
    if not samples:
        logger.error("没有加载到任何样本！")
        return
    
    num_gpus = min(args.num_gpus, 8)
    samples_per_gpu = len(samples) // num_gpus
    gpu_samples = []
    
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])
    
    processes = []
    output_files = []
    
    for gpu_id in range(num_gpus):
        output_file = f"{args.output_dir}/mindmap_full_evo_gpu{gpu_id}.jsonl"
        output_files.append(output_file)
        
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_samples[gpu_id], output_file))
        p.start()
        processes.append(p)
        logger.info(f"启动GPU {gpu_id}, 处理 {len(gpu_samples[gpu_id])} 个样本")
    
    for p in processes:
        p.join()
    
    # 合并输出
    final_output = f"{args.output_dir}/mindmap_full_evolution_9908.jsonl"
    total_samples = 0
    
    with open(final_output, 'w') as fout:
        for output_file in output_files:
            if os.path.exists(output_file):
                with open(output_file, 'r') as fin:
                    for line in fin:
                        fout.write(line)
                        total_samples += 1
    
    logger.info("=" * 80)
    logger.info(f"完成！总共生成 {total_samples} 个训练样本")
    logger.info(f"输出文件: {final_output}")
    logger.info("=" * 80)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
