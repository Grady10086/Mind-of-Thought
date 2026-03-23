#!/usr/bin/env python3
"""
生成Mind Map训练数据 - 所有任务都开启Evolution

与V7保持一致：
1. 使用相同的感知系统 (GroundingDINO + DA3)
2. 使用相同的校准机制
3. 对所有任务类型应用Evolution
4. 输出格式与V7训练数据一致
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 环境设置
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

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
    'microwave': {'height': 0.30, 'range': (0.25, 0.40)},
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
        """获取估计的尺寸"""
        if not self.detections:
            return 0.0
        best = max(self.detections, key=lambda x: x.confidence)
        return max(best.estimated_height, best.estimated_width)


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
    confidence: float = 0.8


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
            
            # 深度估计
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
            
            # 物体检测
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
        
        # 聚合成心智地图实体
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
# 演化器 - 修正心智地图（对所有任务类型）
# ============================================================================

class UniversalMindMapEvolver:
    """通用心智地图演化器 - 对所有任务类型应用Evolution"""
    
    def __init__(self):
        self.evolution_history = []
    
    def evolve(
        self, 
        mind_map: Dict[str, MindMapEntity],
        question: str,
        question_type: str,
        options: List[str] = None,
        calibration: CalibrationResult = None,
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """
        对心智地图进行演化
        根据任务类型应用不同的演化策略
        """
        actions = []
        
        # 1. 通用演化：去除低置信度检测
        actions.extend(self._filter_low_confidence(mind_map))
        
        # 2. 根据任务类型应用特定演化
        if 'counting' in question_type:
            task_actions = self._evolve_for_counting(mind_map, question)
            actions.extend(task_actions)
            
        elif 'size_estimation' in question_type:
            task_actions = self._evolve_for_size(mind_map, question, calibration)
            actions.extend(task_actions)
            
        elif 'distance' in question_type:
            task_actions = self._evolve_for_distance(mind_map, question, calibration)
            actions.extend(task_actions)
            
        elif 'direction' in question_type:
            task_actions = self._evolve_for_direction(mind_map, question, options)
            actions.extend(task_actions)
            
        elif 'appearance_order' in question_type:
            task_actions = self._evolve_for_appearance_order(mind_map, question, options)
            actions.extend(task_actions)
            
        elif 'route' in question_type:
            task_actions = self._evolve_for_route(mind_map, question, options)
            actions.extend(task_actions)
        
        return mind_map, actions
    
    def _filter_low_confidence(self, mind_map: Dict[str, MindMapEntity]) -> List[EvolutionAction]:
        """过滤低置信度检测"""
        actions = []
        labels_to_remove = []
        
        for label, entity in mind_map.items():
            if entity.avg_confidence < 0.2 and len(entity.detections) < 3:
                labels_to_remove.append(label)
                actions.append(EvolutionAction(
                    action_type='remove_low_confidence',
                    target_entity=label,
                    old_value=entity.count,
                    new_value=0,
                    reasoning=f"Low confidence ({entity.avg_confidence:.2f}) and few detections ({len(entity.detections)})",
                    confidence=0.9,
                ))
        
        for label in labels_to_remove:
            del mind_map[label]
        
        return actions
    
    def _evolve_for_counting(self, mind_map: Dict[str, MindMapEntity], question: str) -> List[EvolutionAction]:
        """Counting任务的演化 - 去重"""
        actions = []
        
        # 提取目标物体
        match = re.search(r'How many (\w+)', question, re.IGNORECASE)
        if not match:
            return actions
        
        target = match.group(1).lower()
        
        for label, entity in mind_map.items():
            if match_object_name(target, label):
                # 检查重复计数
                frame_dets = defaultdict(list)
                for det in entity.detections:
                    frame_dets[det.frame_idx].append(det)
                
                counts = [len(dets) for dets in frame_dets.values()]
                if len(counts) > 1:
                    max_count = max(counts)
                    median_count = np.median(counts)
                    
                    # 如果最大值明显大于中位数，进行修正
                    if max_count > median_count * 1.5 and max_count > 2:
                        new_count = int(np.ceil(median_count))
                        if new_count != entity.count:
                            actions.append(EvolutionAction(
                                action_type='correct_count',
                                target_entity=label,
                                old_value=entity.count,
                                new_value=new_count,
                                reasoning=f"Dedup: max({max_count}) > median({median_count:.1f})*1.5",
                                confidence=0.7,
                            ))
                            entity.count = new_count
                break
        
        return actions
    
    def _evolve_for_size(self, mind_map: Dict[str, MindMapEntity], question: str, calibration: CalibrationResult) -> List[EvolutionAction]:
        """Size estimation任务的演化 - 尺度修正"""
        actions = []
        
        # 物理常识约束
        typical_sizes = {
            'chair': (0.4, 1.0),
            'table': (0.6, 1.5),
            'bed': (0.3, 0.7),
            'door': (1.8, 2.3),
            'window': (0.6, 2.0),
            'sofa': (0.6, 1.2),
            'desk': (0.6, 1.0),
            'lamp': (0.2, 1.5),
        }
        
        for label, entity in mind_map.items():
            if entity.size_3d is None:
                continue
                
            for obj_type, (min_h, max_h) in typical_sizes.items():
                if obj_type in label.lower():
                    height = max(entity.size_3d)
                    
                    if height < min_h * 0.3 or height > max_h * 3:
                        # 尺寸异常，进行修正
                        target_height = (min_h + max_h) / 2
                        correction_factor = target_height / height if height > 0 else 1.0
                        
                        old_size = entity.size_3d.copy()
                        entity.size_3d = entity.size_3d * correction_factor
                        
                        actions.append(EvolutionAction(
                            action_type='correct_size',
                            target_entity=label,
                            old_value=list(old_size),
                            new_value=list(entity.size_3d),
                            reasoning=f"Size {height:.2f}m outside range [{min_h}, {max_h}]m for {obj_type}",
                            confidence=0.6,
                        ))
                    break
        
        return actions
    
    def _evolve_for_distance(self, mind_map: Dict[str, MindMapEntity], question: str, calibration: CalibrationResult) -> List[EvolutionAction]:
        """Distance任务的演化 - 距离修正"""
        actions = []
        
        # 室内距离的合理范围检查
        max_indoor_distance = 15.0  # 米
        
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                dist_from_origin = np.linalg.norm(entity.position_3d)
                
                if dist_from_origin > max_indoor_distance:
                    # 距离异常，进行修正
                    correction_factor = max_indoor_distance / dist_from_origin
                    old_pos = entity.position_3d.copy()
                    entity.position_3d = entity.position_3d * correction_factor
                    
                    actions.append(EvolutionAction(
                        action_type='correct_position',
                        target_entity=label,
                        old_value=list(old_pos),
                        new_value=list(entity.position_3d),
                        reasoning=f"Distance {dist_from_origin:.2f}m > max indoor {max_indoor_distance}m",
                        confidence=0.5,
                    ))
        
        return actions
    
    def _evolve_for_direction(self, mind_map: Dict[str, MindMapEntity], question: str, options: List[str]) -> List[EvolutionAction]:
        """Direction任务的演化 - 空间关系修正"""
        actions = []
        
        # 检查位置数据一致性
        positions = []
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                positions.append(entity.position_3d)
        
        if len(positions) >= 2:
            positions = np.array(positions)
            
            # 检查是否所有物体都在同一平面
            z_std = np.std(positions[:, 2])
            if z_std < 0.1:
                # 深度变化太小，可能深度估计有问题
                actions.append(EvolutionAction(
                    action_type='depth_warning',
                    target_entity='all',
                    old_value=f"z_std={z_std:.3f}",
                    new_value="depth variation too low",
                    reasoning="All objects appear at similar depth, direction may be unreliable",
                    confidence=0.4,
                ))
        
        return actions
    
    def _evolve_for_appearance_order(self, mind_map: Dict[str, MindMapEntity], question: str, options: List[str]) -> List[EvolutionAction]:
        """Appearance order任务的演化 - 时序修正"""
        actions = []
        
        # 检查first_seen_frame的一致性
        frame_entities = []
        for label, entity in mind_map.items():
            if entity.first_seen_frame >= 0:
                frame_entities.append((label, entity.first_seen_frame, len(entity.get_frame_indices())))
        
        if len(frame_entities) >= 2:
            # 按first_seen_frame排序
            frame_entities.sort(key=lambda x: x[1])
            
            # 检查是否有多个物体在同一帧首次出现
            first_frames = [x[1] for x in frame_entities]
            if len(set(first_frames)) < len(first_frames) * 0.5:
                actions.append(EvolutionAction(
                    action_type='temporal_warning',
                    target_entity='all',
                    old_value=str(first_frames),
                    new_value="many objects appear simultaneously",
                    reasoning="Multiple objects first appear in same frame, order may be unreliable",
                    confidence=0.5,
                ))
        
        return actions
    
    def _evolve_for_route(self, mind_map: Dict[str, MindMapEntity], question: str, options: List[str]) -> List[EvolutionAction]:
        """Route planning任务的演化 - 拓扑修正"""
        actions = []
        
        # 检查物体间的连通性
        positions = {}
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                positions[label] = entity.position_3d
        
        if len(positions) >= 3:
            # 计算物体间的距离矩阵
            labels = list(positions.keys())
            n = len(labels)
            
            nearby_count = 0
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(positions[labels[i]] - positions[labels[j]])
                    if dist < 2.0:  # 2米内认为相近
                        nearby_count += 1
            
            connectivity = nearby_count / (n * (n-1) / 2) if n > 1 else 0
            
            if connectivity > 0.8:
                actions.append(EvolutionAction(
                    action_type='spatial_cluster',
                    target_entity='all',
                    old_value=f"connectivity={connectivity:.2f}",
                    new_value="objects are clustered",
                    reasoning="Most objects are close together, route planning may be trivial",
                    confidence=0.5,
                ))
        
        return actions


# ============================================================================
# 格式化输出
# ============================================================================

def format_mind_map_with_evolution(
    mind_map: Dict[str, MindMapEntity],
    evolution_actions: List[EvolutionAction],
    calibration: CalibrationResult,
    question_type: str,
) -> str:
    """格式化心智地图为文本（带Evolution标记）"""
    
    lines = []
    
    # 标题 - 表明这是经过Evolution处理的
    if evolution_actions:
        lines.append("=== DETECTED OBJECTS FROM PERCEPTION SYSTEM (WITH EVOLUTION) ===")
    else:
        lines.append("=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===")
    
    # 按置信度排序
    sorted_entities = sorted(mind_map.items(), key=lambda x: -x[1].avg_confidence)
    
    for label, entity in sorted_entities[:20]:  # 最多显示20个物体
        pos = entity.position_3d
        size = entity.size_3d
        
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})" if pos is not None else "N/A"
        size_str = f"({size[0]:.2f}m × {size[1]:.2f}m × {size[2]:.2f}m)" if size is not None else "N/A"
        
        line = f"{label}: position {pos_str}, size {size_str}, count: {entity.count}"
        
        # 如果这个实体被演化过，添加标记
        for action in evolution_actions:
            if action.target_entity == label:
                line += f" [EVOLVED: {action.action_type}]"
                break
        
        lines.append(line)
    
    # 添加校准信息
    if calibration.scale_factor != 1.0:
        lines.append(f"\nScale Calibration: {calibration.calibration_object} (factor: {calibration.scale_factor:.2f})")
    
    # 添加演化摘要
    if evolution_actions:
        lines.append(f"\nEvolution Summary: {len(evolution_actions)} actions applied")
    
    return "\n".join(lines)


def build_training_sample(
    question: str,
    question_type: str,
    ground_truth: str,
    options: List[str],
    mind_map_text: str,
) -> Dict:
    """构建训练样本（与V7格式一致）"""
    
    # 构建任务特定的提示
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
    
    # 构建完整的用户消息
    user_message = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene.

{mind_map_text}

=== TASK ===
{task_hint}

=== QUESTION ===
{question}"""
    
    # 如果是选择题，添加选项
    if options:
        options_text = "\n".join(options)
        user_message += f"\n\n=== OPTIONS ===\n{options_text}"
    
    # 构建训练样本
    sample = {
        "conversations": [
            {"from": "human", "value": user_message},
            {"from": "gpt", "value": ground_truth}
        ],
        "question_type": question_type
    }
    
    return sample


# ============================================================================
# Worker进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], output_file: str, enable_evolution: bool):
    """GPU Worker进程"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 创建组件
    builder = MindMapBuilder(device='cuda', num_frames=32, box_threshold=0.25)
    calibrator = ScaleCalibrator()
    evolver = UniversalMindMapEvolver() if enable_evolution else None
    
    training_samples = []
    evolution_count = 0
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
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
            
            if not mind_map:
                continue
            
            # 2. 尺度校准
            calibration = calibrator.calculate_scale_factor(mind_map)
            mind_map = calibrator.apply_calibration(mind_map, calibration)
            
            # 3. 演化（如果启用）
            evolution_actions = []
            if evolver:
                mind_map, evolution_actions = evolver.evolve(
                    mind_map, question, question_type, options, calibration
                )
                if evolution_actions:
                    evolution_count += 1
            
            # 4. 格式化心智地图
            mind_map_text = format_mind_map_with_evolution(
                mind_map, evolution_actions, calibration, question_type
            )
            
            # 5. 构建训练样本
            training_sample = build_training_sample(
                question, question_type, gt, options, mind_map_text
            )
            
            training_samples.append(training_sample)
            
        except Exception as e:
            logger.error(f"Error processing {sample.get('scene_name', 'unknown')}: {e}")
            traceback.print_exc()
            continue
    
    # 保存结果
    with open(output_file, 'w') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"GPU {gpu_id}: 生成 {len(training_samples)} 个样本, Evolution应用 {evolution_count} 次")
    
    # 清理
    builder.unload()
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# 主函数
# ============================================================================

def load_vsi590k_samples(max_per_task: int = 1000) -> List[Dict]:
    """加载VSI-590K训练集样本"""
    
    # 使用原始训练数据
    source_file = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train.jsonl"
    
    if not os.path.exists(source_file):
        logger.error(f"源文件不存在: {source_file}")
        return []
    
    samples = []
    task_counts = defaultdict(int)
    
    # 任务类型映射 (VSI-590K格式 -> V7格式)
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
            
            # 跳过route_planning (需要单独处理)
            if 'route' in original_task.lower():
                continue
            
            # 映射任务类型
            mapped_task = task_mapping.get(original_task, original_task)
            
            if max_per_task and task_counts[mapped_task] >= max_per_task:
                continue
            
            # 解析conversations
            conv = data.get('conversations', [])
            if isinstance(conv, str):
                try:
                    conv = json.loads(conv.replace("'", '"'))
                except:
                    continue
            
            if not conv or len(conv) < 2:
                continue
            
            # 提取问题和答案
            human_msg = conv[0].get('value', '')
            gpt_msg = conv[1].get('value', '')
            
            # 移除<image>标签
            human_msg = human_msg.replace('<image>\n', '').replace('<image>', '')
            
            # 解析问题和选项
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


def main():
    parser = argparse.ArgumentParser(description='生成Mind Map训练数据 - 所有任务开启Evolution')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-per-task', type=int, default=1000)
    parser.add_argument('--enable-evolution', action='store_true', default=True)
    parser.add_argument('--no-evolution', action='store_true', help='禁用Evolution')
    parser.add_argument('--output-dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    enable_evolution = not args.no_evolution
    
    logger.info("=" * 80)
    logger.info("生成Mind Map训练数据 - 所有任务开启Evolution")
    logger.info("=" * 80)
    logger.info(f"GPUs: {args.num_gpus}")
    logger.info(f"每任务最大样本: {args.max_per_task}")
    logger.info(f"Evolution: {'开启' if enable_evolution else '关闭'}")
    logger.info("=" * 80)
    
    # 加载数据
    samples = load_vsi590k_samples(args.max_per_task)
    
    if not samples:
        logger.error("没有加载到任何样本！")
        return
    
    # 分配到各GPU
    num_gpus = min(args.num_gpus, 8)  # 最多8卡
    samples_per_gpu = len(samples) // num_gpus
    gpu_samples = []
    
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])
    
    # 启动多进程
    processes = []
    output_files = []
    
    suffix = "all_evolution" if enable_evolution else "no_evolution"
    
    for gpu_id in range(num_gpus):
        output_file = f"{args.output_dir}/mindmap_{suffix}_gpu{gpu_id}.jsonl"
        output_files.append(output_file)
        
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_samples[gpu_id], output_file, enable_evolution)
        )
        p.start()
        processes.append(p)
        logger.info(f"启动GPU {gpu_id}, 处理 {len(gpu_samples[gpu_id])} 个样本")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 合并输出文件
    final_output = f"{args.output_dir}/mindmap_{suffix}_final.jsonl"
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
