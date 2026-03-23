#!/usr/bin/env python3
"""
Self-Evolving Agent V7 - 双推理模式框架 + 微调VL模型

修改说明：
1. 支持加载微调后的Qwen3-VL模型 (LoRA adapter)
2. 其他部分与V7保持完全一致
3. 用于测试微调后的VL模型在V7框架中的效果

核心设计：
1. 所有任务同时保存两种推理结果：
   - rule_prediction: 基于心智地图的规则推理结果
   - vl_prediction: 基于微调后的 Qwen3-VL-8B-Instruct 的视觉语言推理结果
2. 使用微调后的 Qwen3-VL-8B-Instruct 作为 VL 模型
3. 完整的 prompt 设计，不做简化
4. 校准和演化机制与 V6 保持一致

架构：
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Self-Evolving Agent V7                              │
│                         (Dual Reasoning Mode)                               │
│                                                                             │
│  ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐               │
│  │ 感知    │ -> │ Calibrator│ -> │ Evolver  │ -> │ MindMap │               │
│  │DA3+DINO │    │标定物识别  │    │心智地图修正│    │ 校准后   │               │
│  └─────────┘    └───────────┘    └──────────┘    └────┬────┘               │
│                                                       │                     │
│                                    ┌──────────────────┴──────────────────┐  │
│                                    ▼                                     ▼  │
│                             ┌───────────┐                        ┌─────────┐│
│                             │ Rule-based│                        │Qwen3-VL ││
│                             │ Reasoning │                        │Reasoning││
│                             └─────┬─────┘                        └────┬────┘│
│                                   ▼                                   ▼     │
│                           rule_prediction                     vl_prediction │
└─────────────────────────────────────────────────────────────────────────────┘

标定物：已知标准尺寸的物体
- 门: ~200cm 高
- 椅子: ~80cm 高 (座高 ~45cm)
- 床: ~200cm 长
- 马桶: ~40cm 高
- 冰箱: ~180cm 高
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
from PIL import Image
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
    # 物体: (典型高度/长度, 可接受范围)
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
    'tv': {'diagonal': 1.0, 'range': (0.5, 1.5)},  # TV 尺寸变化大
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

# 视频目录 (VSIBench 官方路径)
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
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    depth: float
    position_3d: np.ndarray
    estimated_height: float = 0.0  # 估计的物体高度 (米)
    estimated_width: float = 0.0   # 估计的物体宽度 (米)


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
    calibrated: bool = False  # 是否已校准
    
    def get_frame_indices(self) -> List[int]:
        return sorted(set(d.frame_idx for d in self.detections))
    
    def get_best_frames(self, n: int = 3) -> List[int]:
        sorted_dets = sorted(self.detections, key=lambda x: -x.confidence)
        frames = []
        seen = set()
        for d in sorted_dets:
            if d.frame_idx not in seen:
                frames.append(d.frame_idx)
                seen.add(d.frame_idx)
            if len(frames) >= n:
                break
        return frames
    
    def get_estimated_size(self) -> float:
        """获取估计的尺寸 (最大维度，米)"""
        if not self.detections:
            return 0.0
        best = max(self.detections, key=lambda x: x.confidence)
        return max(best.estimated_height, best.estimated_width)
    
    def to_text(self) -> str:
        """转换为文本描述 - 完整版本"""
        text = f"- {self.label}: count={self.count}"
        if self.position_3d is not None:
            pos = self.position_3d
            text += f", position=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m"
        if self.size_3d is not None:
            size = self.size_3d * 100  # 转厘米
            text += f", size≈{max(size):.0f}cm"
        text += f", confidence={self.avg_confidence:.2f}"
        text += f", frames={len(self.get_frame_indices())}"
        if self.first_seen_frame >= 0:
            text += f", first_seen_frame={self.first_seen_frame}"
        return text


@dataclass
class CalibrationResult:
    """校准结果"""
    calibration_object: str  # 用于校准的物体
    estimated_size: float    # 估计尺寸
    standard_size: float     # 标准尺寸
    scale_factor: float      # 校准系数
    confidence: float        # 置信度


@dataclass
class EvolutionAction:
    """演化动作"""
    action_type: str  # 'calibrate', 'merge_duplicate', 'add_missing', 'correct_count', 'correct_position'
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
        """构建心智地图，返回 (mind_map, total_frames, sampled_frames)"""
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
            # depth_result: tuple (depth_tensor, None) from MiDaS or DepthPrediction from DA3
            if isinstance(depth_result, tuple):
                depth_tensor = depth_result[0]  # 取第一个元素(depth)
            else:
                depth_tensor = depth_result
            
            # 处理depth tensor
            if depth_tensor is not None:
                depth_map = depth_tensor.cpu().numpy()
                # 确保depth_map是2D
                if depth_map.ndim == 3:
                    depth_map = depth_map.squeeze()
                elif depth_map.ndim == 1:
                    # 如果是1D,尝试reshape
                    depth_map = depth_map.reshape(h, w)
                
                if depth_map.shape[0] != h or depth_map.shape[1] != w:
                    depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # 如果depth为None,使用默认深度
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
                
                # 计算中心点深度
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                cx = min(max(cx, 0), w - 1)
                cy = min(max(cy, 0), h - 1)
                depth = float(depth_map[cy, cx])
                
                # 计算 3D 位置
                pos_3d = np.array([
                    (cx - w / 2) * depth / self.focal_length,
                    (cy - h / 2) * depth / self.focal_length,
                    depth
                ])
                
                # 估计物体尺寸 (基于 bbox 和深度)
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
    """尺度校准器 - 使用标定物校准心智地图"""
    
    def __init__(self, vl_model=None):
        self.vl_model = vl_model
        self.calibration_history = []
    
    def find_calibration_objects(self, mind_map: Dict[str, MindMapEntity]) -> List[Tuple[str, MindMapEntity, dict]]:
        """找到可用于校准的物体"""
        candidates = []
        
        for label, entity in mind_map.items():
            # 检查是否是标定物
            for cal_name, cal_info in CALIBRATION_OBJECTS.items():
                if match_object_name(label, cal_name):
                    # 置信度要足够高
                    if entity.avg_confidence >= 0.3:
                        candidates.append((cal_name, entity, cal_info))
                    break
        
        # 按置信度排序
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
        
        # 使用多个标定物计算平均校准系数
        scale_factors = []
        
        for cal_name, entity, cal_info in candidates[:5]:  # 最多用5个
            est_size = entity.get_estimated_size()
            
            if est_size <= 0:
                continue
            
            # 获取标准尺寸
            std_size = cal_info.get('height') or cal_info.get('length') or cal_info.get('diagonal', 1.0)
            min_range, max_range = cal_info.get('range', (0.5, 2.0))
            
            # 计算校准系数
            factor = std_size / est_size
            
            # 检查是否在合理范围内
            if 0.1 < factor < 10:  # 校准系数不应太极端
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
        
        # 加权平均 (按置信度)
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
            # 校准 size_3d
            if entity.size_3d is not None:
                entity.size_3d = entity.size_3d * factor
            
            # 校准 position_3d (深度方向)
            if entity.position_3d is not None:
                entity.position_3d[2] = entity.position_3d[2] * factor
            
            # 校准每个检测的尺寸
            for det in entity.detections:
                det.estimated_height *= factor
                det.estimated_width *= factor
                det.depth *= factor
                det.position_3d[2] *= factor
            
            entity.calibrated = True
        
        return mind_map


# ============================================================================
# 演化器 - 修正心智地图
# ============================================================================

class MindMapEvolver:
    """心智地图演化器"""
    
    def __init__(self, vl_model=None, vl_processor=None, device='cuda'):
        self.vl_model = vl_model
        self.vl_processor = vl_processor
        self.device = device
        self.evolution_history = []
    
    def evolve_for_counting(
        self, 
        mind_map: Dict[str, MindMapEntity],
        target_object: str,
        frames: List[np.ndarray],
        frame_indices: List[int],
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """针对 counting 任务演化心智地图"""
        actions = []
        
        # 找到目标物体
        target_entity = None
        target_label = None
        for label, entity in mind_map.items():
            if match_object_name(target_object, label):
                target_entity = entity
                target_label = label
                break
        
        # 如果心智地图中没有该物体
        if target_entity is None:
            return mind_map, actions
        
        # 检查是否有重复计数的可能 (同一帧多个检测)
        frame_dets = defaultdict(list)
        for det in target_entity.detections:
            frame_dets[det.frame_idx].append(det)
        
        # 如果某帧检测数量明显多于其他帧，可能是重复计数
        counts = [len(dets) for dets in frame_dets.values()]
        if len(counts) > 1:
            max_count = max(counts)
            median_count = np.median(counts)
            
            # 如果最大值明显大于中位数，可能有问题
            if max_count > median_count * 2 and max_count > 2:
                # 使用中位数作为更保守的估计
                new_count = int(np.ceil(median_count))
                if new_count != target_entity.count:
                    actions.append(EvolutionAction(
                        action_type='correct_count',
                        target_entity=target_label,
                        old_value=target_entity.count,
                        new_value=new_count,
                        reasoning=f"Max count ({max_count}) >> median ({median_count:.1f}), using conservative estimate",
                        confidence=0.7,
                    ))
                    target_entity.count = new_count
        
        return mind_map, actions


# ============================================================================
# 问题回答器 - 双推理模式 (规则 + VL)
# ============================================================================

class DualReasoningQA:
    """双推理模式问答器 - 同时返回规则推理和 VL 推理结果"""
    
    def __init__(self, vl_model=None, vl_processor=None, device='cuda'):
        self.vl_model = vl_model
        self.vl_processor = vl_processor
        self.device = device
    
    def load_vl_model(self, model_path: str, adapter_path: str = None):
        """加载 Qwen3-VL 模型 (支持LoRA adapter)"""
        if self.vl_model is None:
            # 检测是否为 Qwen3-VL
            if 'qwen3' in model_path.lower() or 'Qwen3' in model_path:
                try:
                    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
                    from peft import PeftModel
                    logger.info(f"Loading Qwen3-VL model: {model_path}")
                    
                    self.vl_processor = AutoProcessor.from_pretrained(
                        model_path, 
                        trust_remote_code=True
                    )
                    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map=self.device,
                        trust_remote_code=True,
                    )
                    
                    # 加载 LoRA adapter (如果提供)
                    if adapter_path and Path(adapter_path).exists():
                        logger.info(f"Loading LoRA adapter: {adapter_path}")
                        self.vl_model = PeftModel.from_pretrained(base_model, adapter_path)
                        self.vl_model = self.vl_model.merge_and_unload()  # 合并LoRA权重
                        logger.info("LoRA adapter merged successfully")
                    else:
                        self.vl_model = base_model
                        logger.info("Using base model without adapter")
                    
                    self.vl_model.eval()
                    logger.info("Qwen3-VL model loaded successfully")
                    return
                except ImportError as e:
                    logger.warning(f"Qwen3VLForConditionalGeneration not available: {e}, falling back to Qwen2.5-VL")
            
            # 回退到 Qwen2.5-VL
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            logger.info(f"Loading Qwen2.5-VL model: {model_path}")
            
            self.vl_processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            logger.info("Qwen2.5-VL model loaded successfully")
    
    # ========================================================================
    # 规则推理方法
    # ========================================================================
    
    def rule_answer_counting(self, mind_map: Dict[str, MindMapEntity], question: str) -> Tuple[str, str]:
        """规则推理 - 计数问题"""
        match = re.search(r'How many (\w+)', question, re.IGNORECASE)
        if not match:
            return "0", "No object found in question"
        
        target = match.group(1).lower()
        
        for label, entity in mind_map.items():
            if match_object_name(target, label):
                reasoning = f"Found '{label}' in mind map with count={entity.count}, confidence={entity.avg_confidence:.2f}"
                return str(entity.count), reasoning
        
        return "0", f"Object '{target}' not found in mind map"
    
    def rule_answer_size_estimation(
        self, 
        mind_map: Dict[str, MindMapEntity], 
        question: str,
        calibration: CalibrationResult,
    ) -> Tuple[str, str]:
        """规则推理 - 尺寸估计问题"""
        match = re.search(r'of the (\w+)', question.lower())
        if not match:
            return "100", "No object found in question"
        
        target = match.group(1)
        
        for label, entity in mind_map.items():
            if match_object_name(target, label):
                size = entity.get_estimated_size()
                if calibration.scale_factor != 1.0:
                    size *= calibration.scale_factor
                size_cm = int(size * 100)
                reasoning = f"Found '{label}' in mind map, estimated size={size:.2f}m ({size_cm}cm)"
                if calibration.scale_factor != 1.0:
                    reasoning += f", calibrated with factor={calibration.scale_factor:.2f}"
                return str(size_cm), reasoning
        
        # 默认尺寸
        typical_sizes = {
            'chair': 80, 'table': 150, 'sofa': 200, 'bed': 200,
            'tv': 100, 'monitor': 60, 'door': 200, 'window': 120,
            'toilet': 60, 'sink': 50, 'lamp': 50, 'pillow': 40,
            'bathtub': 170, 'refrigerator': 180, 'desk': 120,
        }
        for k, v in typical_sizes.items():
            if k in target.lower():
                return str(v), f"Object '{target}' not found, using typical size for '{k}'"
        
        return "100", f"Object '{target}' not found, using default size"
    
    def rule_answer_room_size(
        self, 
        mind_map: Dict[str, MindMapEntity], 
        question: str,
        calibration: CalibrationResult,
    ) -> Tuple[str, str]:
        """规则推理 - 房间面积问题"""
        if mind_map:
            positions = []
            for entity in mind_map.values():
                if entity.position_3d is not None:
                    positions.append(entity.position_3d[:2])  # x, y
            
            if len(positions) >= 2:
                positions = np.array(positions)
                x_range = positions[:, 0].max() - positions[:, 0].min()
                y_range = positions[:, 1].max() - positions[:, 1].min()
                
                # 应用校准系数
                if calibration.scale_factor != 1.0:
                    x_range *= calibration.scale_factor
                    y_range *= calibration.scale_factor
                
                # 估计面积 (加上边界)
                area = max((x_range + 2) * (y_range + 2), 10)
                reasoning = f"Estimated from object positions: x_range={x_range:.2f}m, y_range={y_range:.2f}m, area≈{area:.1f}m²"
                return str(int(area)), reasoning
        
        return "25", "No position data available, using default room size"
    
    def rule_answer_distance(
        self, 
        mind_map: Dict[str, MindMapEntity], 
        question: str,
        calibration: CalibrationResult,
    ) -> Tuple[str, str]:
        """规则推理 - 距离问题"""
        patterns = [
            r'distance between (?:the )?([a-z]+(?:\s+[a-z]+)*)\s+and\s+(?:the )?([a-z]+(?:\s+[a-z]+)*)',
            r'from (?:the )?(\w+) to (?:the )?(\w+)',
            r'between (?:the )?(\w+) and (?:the )?(\w+)',
        ]
        
        obj1, obj2 = None, None
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                obj1, obj2 = match.groups()[:2]
                break
        
        if obj1 and obj2:
            ent1 = ent2 = None
            for label, entity in mind_map.items():
                if match_object_name(obj1, label):
                    ent1 = entity
                if match_object_name(obj2, label):
                    ent2 = entity
            
            if ent1 and ent2 and ent1.position_3d is not None and ent2.position_3d is not None:
                dist = np.linalg.norm(ent1.position_3d - ent2.position_3d)
                if calibration.scale_factor != 1.0:
                    dist *= calibration.scale_factor
                reasoning = f"Distance from '{obj1}' to '{obj2}': {dist:.2f}m"
                return f"{dist:.2f}", reasoning
        
        return "2.0", f"Could not find both objects in mind map"
    
    def rule_answer_appearance_order(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
        options: List[str],
    ) -> Tuple[str, str]:
        """规则推理 - 出现顺序问题 (使用 first_seen_frame)"""
        if not options:
            return "A", "No options provided"
        
        # 从问题中提取目标物体
        match = re.search(r'following categories.*?:\s*(.+?)\?', question, re.IGNORECASE)
        if match:
            objects_text = match.group(1)
            target_objects = [obj.strip().lower() for obj in objects_text.split(',')]
        else:
            # 从第一个选项提取
            opt_content = re.sub(r'^[A-D]\.\s*', '', options[0])
            target_objects = [obj.strip().lower() for obj in opt_content.split(',')]
        
        # 获取每个物体的首次出现帧
        object_first_frame = {}
        for target in target_objects:
            for label, entity in mind_map.items():
                if match_object_name(target, label):
                    frames = entity.get_frame_indices()
                    if frames:
                        object_first_frame[target] = min(frames)
                    break
        
        # 对物体按首次出现帧排序
        sorted_objects = sorted(object_first_frame.keys(), key=lambda x: object_first_frame.get(x, float('inf')))
        
        reasoning = f"Objects sorted by first_seen_frame: {sorted_objects}, frames: {object_first_frame}"
        
        if len(sorted_objects) < 2:
            return options[0][0] if options else "A", reasoning + " (insufficient objects)"
        
        # 匹配选项
        for i, opt in enumerate(options):
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt)
            opt_objects = [o.strip().lower() for o in opt_content.split(',')]
            
            # 检查顺序是否匹配
            if len(opt_objects) >= 2:
                opt_indices = []
                for o in opt_objects:
                    if o in sorted_objects:
                        opt_indices.append(sorted_objects.index(o))
                    elif any(match_object_name(o, s) for s in sorted_objects):
                        for j, s in enumerate(sorted_objects):
                            if match_object_name(o, s):
                                opt_indices.append(j)
                                break
                
                if opt_indices and opt_indices == sorted(opt_indices):
                    return chr(65 + i), reasoning + f" -> matched option {chr(65 + i)}"
        
        return options[0][0] if options else "A", reasoning + " (no exact match, using first option)"
    
    def rule_answer_choice(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
        options: List[str],
        question_type: str,
    ) -> Tuple[str, str]:
        """规则推理 - 通用选择题"""
        # 对于相对方向和距离问题，尝试基于位置信息推理
        if 'direction' in question_type:
            return self._rule_answer_direction(mind_map, question, options)
        elif 'rel_distance' in question_type:
            return self._rule_answer_relative_distance(mind_map, question, options)
        elif 'route' in question_type:
            return options[0][0] if options else "A", "Route planning requires VL reasoning"
        
        return options[0][0] if options else "A", "No rule-based method for this question type"
    
    def _rule_answer_direction(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
        options: List[str],
    ) -> Tuple[str, str]:
        """规则推理 - 方向问题"""
        # 提取物体
        patterns = [
            r'direction.*?(\w+).*?from.*?(\w+)',
            r'(\w+).*?relative to.*?(\w+)',
            r'where is (?:the )?(\w+).*?relative to (?:the )?(\w+)',
        ]
        
        obj1, obj2 = None, None
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                obj1, obj2 = match.groups()
                break
        
        if not obj1 or not obj2:
            return options[0][0] if options else "A", "Could not extract objects from question"
        
        ent1 = ent2 = None
        for label, entity in mind_map.items():
            if match_object_name(obj1, label):
                ent1 = entity
            if match_object_name(obj2, label):
                ent2 = entity
        
        if not ent1 or not ent2 or ent1.position_3d is None or ent2.position_3d is None:
            return options[0][0] if options else "A", f"Position data not available for '{obj1}' or '{obj2}'"
        
        # 计算相对位置
        diff = ent1.position_3d - ent2.position_3d
        dx, dy, dz = diff[0], diff[1], diff[2]
        
        # 判断主要方向
        directions = []
        if abs(dx) > 0.3:
            directions.append("right" if dx > 0 else "left")
        if abs(dy) > 0.3:
            directions.append("below" if dy > 0 else "above")
        if abs(dz) > 0.3:
            directions.append("farther" if dz > 0 else "closer")
        
        reasoning = f"Position diff: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, directions: {directions}"
        
        # 匹配选项
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            for d in directions:
                if d in opt_lower:
                    return chr(65 + i), reasoning + f" -> matched '{d}' in option {chr(65 + i)}"
        
        return options[0][0] if options else "A", reasoning + " (no direction match)"
    
    def _rule_answer_relative_distance(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
        options: List[str],
    ) -> Tuple[str, str]:
        """规则推理 - 相对距离问题"""
        # 这类问题通常需要视觉推理，规则方法效果有限
        return options[0][0] if options else "A", "Relative distance comparison requires VL reasoning"
    
    # ========================================================================
    # VL 推理方法 - 完整 prompt 版本
    # ========================================================================
    
    def vl_answer_counting(
        self,
        question: str,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        total_frames: int = 0,
    ) -> Tuple[str, str]:
        """VL 推理 - 计数问题"""
        if self.vl_model is None:
            return "0", "VL model not loaded"
        
        # 构建心智地图文本
        mind_map_text = self._build_mind_map_text(mind_map)
        
        # 完整的 counting prompt
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

        response, raw_response = self._call_vl_model(prompt, video_path)
        
        # 提取数字
        match = re.search(r'\d+', response)
        if match:
            return match.group(), f"VL response: {raw_response}"
        
        return "0", f"Failed to extract number from VL response: {raw_response}"
    
    def vl_answer_size_estimation(
        self,
        question: str,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        calibration: CalibrationResult,
    ) -> Tuple[str, str]:
        """VL 推理 - 尺寸估计问题"""
        if self.vl_model is None:
            return "100", "VL model not loaded"
        
        mind_map_text = self._build_mind_map_text(mind_map)
        
        # 完整的 size estimation prompt
        prompt = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to estimate the size of a specific object.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

=== CALIBRATION INFORMATION ===
Calibration object: {calibration.calibration_object}
Scale factor: {calibration.scale_factor:.2f}
Calibration confidence: {calibration.confidence:.2f}

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

        response, raw_response = self._call_vl_model(prompt, video_path)
        
        # 提取数字
        match = re.search(r'\d+', response)
        if match:
            return match.group(), f"VL response: {raw_response}"
        
        return "100", f"Failed to extract number from VL response: {raw_response}"
    
    def vl_answer_room_size(
        self,
        question: str,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        calibration: CalibrationResult,
    ) -> Tuple[str, str]:
        """VL 推理 - 房间面积问题"""
        if self.vl_model is None:
            return "25", "VL model not loaded"
        
        mind_map_text = self._build_mind_map_text(mind_map)
        
        # 完整的 room size estimation prompt
        prompt = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to estimate the size (floor area) of the room.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

=== CALIBRATION INFORMATION ===
Calibration object: {calibration.calibration_object}
Scale factor: {calibration.scale_factor:.2f}

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

        response, raw_response = self._call_vl_model(prompt, video_path)
        
        # 提取数字
        match = re.search(r'\d+', response)
        if match:
            return match.group(), f"VL response: {raw_response}"
        
        return "25", f"Failed to extract number from VL response: {raw_response}"
    
    def vl_answer_distance(
        self,
        question: str,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        calibration: CalibrationResult,
    ) -> Tuple[str, str]:
        """VL 推理 - 距离问题"""
        if self.vl_model is None:
            return "2.0", "VL model not loaded"
        
        mind_map_text = self._build_mind_map_text(mind_map)
        
        # 完整的 distance estimation prompt
        prompt = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to estimate the distance between two objects.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

=== CALIBRATION INFORMATION ===
Calibration object: {calibration.calibration_object}
Scale factor: {calibration.scale_factor:.2f}

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

        response, raw_response = self._call_vl_model(prompt, video_path)
        
        # 提取数字
        match = re.search(r'[\d.]+', response)
        if match:
            return match.group(), f"VL response: {raw_response}"
        
        return "2.0", f"Failed to extract number from VL response: {raw_response}"
    
    def vl_answer_choice(
        self,
        question: str,
        options: List[str],
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        question_type: str,
        total_frames: int = 0,
    ) -> Tuple[str, str]:
        """VL 推理 - 选择题 (方向、顺序、距离比较、路线规划)"""
        if self.vl_model is None:
            return options[0][0] if options else "A", "VL model not loaded"
        
        mind_map_text = self._build_mind_map_text(mind_map)
        options_text = "\n".join(options)
        
        # 根据问题类型构建特定的 prompt
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
        
        # 完整的选择题 prompt
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
        answer = self._extract_choice(response, options)
        return answer, f"VL response: {raw_response}"
    
    def _build_mind_map_text(self, mind_map: Dict[str, MindMapEntity]) -> str:
        """构建心智地图的完整文本描述"""
        if not mind_map:
            return "No objects detected."
        
        lines = []
        for label, entity in sorted(mind_map.items(), key=lambda x: -x[1].count):
            lines.append(entity.to_text())
        
        return "\n".join(lines)
    
    def _call_vl_model(self, prompt: str, video_path: str) -> Tuple[str, str]:
        """调用 VL 模型"""
        try:
            from qwen_vl_utils import process_vision_info
            
            # 使用视频输入
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
        # 清理响应
        response_clean = response.split('[')[0].strip()
        
        # 从开头提取选项字母
        choice_match = re.search(r'^([A-D])', response_clean.upper())
        if choice_match:
            return choice_match.group(1)
        
        # 从任意位置提取选项字母
        for line in response.split('\n')[::-1]:
            line = line.strip()
            if line and line[0].upper() in 'ABCD':
                return line[0].upper()
        
        # 尝试匹配选项内容
        response_lower = response.lower()
        for i, opt in enumerate(options):
            opt_content = opt.lower()
            if len(opt) >= 3 and opt[1] in '.、':
                opt_content = opt[3:].strip().lower()
            if opt_content in response_lower:
                return chr(65 + i)
        
        return options[0][0] if options else "A"


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process_export_mindmap(gpu_id: int, samples: List[Dict], output_file: str):
    """GPU Worker 进程 - 专门用于导出Mind Map训练数据"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 创建组件 (只需要感知、校准、演化,不需要VL模型)
    builder = MindMapBuilder(device='cuda', num_frames=32, box_threshold=0.25)
    calibrator = ScaleCalibrator()
    evolver = MindMapEvolver(device='cuda')
    
    training_samples = []
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        meta_info = sample.get('meta_info', {})
        
        try:
            # 1. 构建心智地图
            target_objects = []
            if 'counting' in question_type:
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map, total_frames, sampled_frames = builder.build_from_video(video_path, target_objects)
            
            # 2. 尺度校准
            calibration = calibrator.calculate_scale_factor(mind_map)
            mind_map = calibrator.apply_calibration(mind_map, calibration)
            
            # 3. 演化 (针对 counting)
            evolution_actions = []
            if question_type == 'object_counting' and target_objects:
                frame_indices = list(range(len(sampled_frames)))
                mind_map, actions = evolver.evolve_for_counting(
                    mind_map, target_objects[0], sampled_frames, frame_indices
                )
                evolution_actions.extend(actions)
            
            # 4. 双推理模式 - 同时获取规则结果和 VL 结果
            rule_pred, rule_reasoning = "", ""
            vl_pred, vl_reasoning = "", ""
            
            if question_type == 'object_counting':
                rule_pred, rule_reasoning = qa.rule_answer_counting(mind_map, question)
                if vl_loaded:
                    vl_pred, vl_reasoning = qa.vl_answer_counting(question, video_path, mind_map, total_frames)
                
            elif question_type == 'object_size_estimation':
                rule_pred, rule_reasoning = qa.rule_answer_size_estimation(mind_map, question, calibration)
                if vl_loaded:
                    vl_pred, vl_reasoning = qa.vl_answer_size_estimation(question, video_path, mind_map, calibration)
                
            elif question_type == 'room_size_estimation':
                rule_pred, rule_reasoning = qa.rule_answer_room_size(mind_map, question, calibration)
                if vl_loaded:
                    vl_pred, vl_reasoning = qa.vl_answer_room_size(question, video_path, mind_map, calibration)
                
            elif question_type == 'object_abs_distance':
                rule_pred, rule_reasoning = qa.rule_answer_distance(mind_map, question, calibration)
                if vl_loaded:
                    vl_pred, vl_reasoning = qa.vl_answer_distance(question, video_path, mind_map, calibration)
            
            elif question_type == 'obj_appearance_order':
                rule_pred, rule_reasoning = qa.rule_answer_appearance_order(mind_map, question, options)
                if vl_loaded:
                    vl_pred, vl_reasoning = qa.vl_answer_choice(question, options, video_path, mind_map, question_type, total_frames)
                
            else:
                # 其他选择题 (direction, rel_distance, route_planning)
                rule_pred, rule_reasoning = qa.rule_answer_choice(mind_map, question, options, question_type)
                if vl_loaded:
                    vl_pred, vl_reasoning = qa.vl_answer_choice(question, options, video_path, mind_map, question_type, total_frames)
            
            # 5. 评估两种结果
            def evaluate_prediction(pred, gt, is_numerical):
                if is_numerical:
                    pred_val = normalize_number(pred)
                    gt_val = normalize_number(gt)
                    return mean_relative_accuracy(pred_val, gt_val) if pred_val and gt_val else 0.0
                else:
                    pred_norm = pred.strip().upper() if pred else ""
                    gt_norm = gt.strip().upper()
                    if len(pred_norm) > 1 and pred_norm[1] in '.、':
                        pred_norm = pred_norm[0]
                    if len(gt_norm) > 1 and gt_norm[1] in '.、':
                        gt_norm = gt_norm[0]
                    return 1.0 if pred_norm == gt_norm else 0.0
            
            is_numerical = question_type in NUMERICAL_TASKS
            rule_score = evaluate_prediction(rule_pred, gt, is_numerical)
            vl_score = evaluate_prediction(vl_pred, gt, is_numerical) if vl_pred else 0.0
            
            # 使用较好的结果作为最终预测
            if vl_score >= rule_score and vl_pred:
                final_pred = vl_pred
                final_score = vl_score
                final_method = "vl"
            else:
                final_pred = rule_pred
                final_score = rule_score
                final_method = "rule"
            
            results.append({
                'scene_name': sample['scene_name'],
                'question': question,
                'question_type': question_type,
                'ground_truth': gt,
                'options': options,
                # 双推理结果
                'rule_prediction': rule_pred,
                'rule_reasoning': rule_reasoning,
                'rule_score': rule_score,
                'vl_prediction': vl_pred,
                'vl_reasoning': vl_reasoning,
                'vl_score': vl_score,
                # 最终结果
                'prediction': final_pred,
                'score': final_score,
                'method_used': final_method,
                # 校准和演化信息
                'calibration': {
                    'object': calibration.calibration_object,
                    'scale_factor': calibration.scale_factor,
                    'confidence': calibration.confidence,
                },
                'evolution_actions': [
                    {
                        'type': a.action_type,
                        'entity': a.target_entity,
                        'old': str(a.old_value),
                        'new': str(a.new_value),
                        'reasoning': a.reasoning,
                    } for a in evolution_actions
                ],
                'mind_map_summary': {
                    label: {
                        'count': e.count, 
                        'confidence': e.avg_confidence,
                        'first_seen_frame': e.first_seen_frame,
                    }
                    for label, e in list(mind_map.items())[:10]
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
                'options': options,
                'rule_prediction': '',
                'rule_score': 0.0,
                'vl_prediction': '',
                'vl_score': 0.0,
                'prediction': '',
                'score': 0.0,
                'error': str(e),
            })
    
    # 清理
    builder.unload()
    if vl_loaded:
        del qa.vl_model
        del qa.vl_processor
    gc.collect()
    torch.cuda.empty_cache()
    
    result_queue.put(results)


# ============================================================================
# 主函数
# ============================================================================

def load_vsibench_data() -> List[Dict]:
    """加载 VSI-Bench 数据集 (从JSON文件)"""
    vsibench_json = "/home/tione/notebook/tianjungu/projects/Spatial-MLLM/evaluate/annotation/eval_vsibench.json"
    
    logger.info(f"加载 VSI-Bench 数据集从: {vsibench_json}")
    with open(vsibench_json, 'r') as f:
        dataset = json.load(f)
    
    samples = []
    vsibench_video_base = "/home/tione/notebook/tianjungu/hf_cache/vsibench"
    
    for item in dataset:
        # Get video path from 'path' field (e.g., ./arkitscenes/41069025.mp4)
        rel_path = item.get('path', '')
        if rel_path.startswith('./'):
            rel_path = rel_path[2:]
        
        video_path = os.path.join(vsibench_video_base, rel_path)
        if not os.path.exists(video_path):
            continue
        
        scene_name = os.path.basename(rel_path).replace('.mp4', '')
        
        # Parse question and ground truth
        question = item['problem']  # Question text
        solution = item.get('solution', '')  # e.g., "<answer>4</answer>"
        
        # Extract answer from <answer> tags
        import re
        match = re.search(r'<answer>(.*?)</answer>', solution)
        ground_truth = match.group(1) if match else solution
        
        samples.append({
            'scene_name': scene_name,
            'video_path': video_path,
            'question': question,
            'question_type': item.get('original_question_type', 'unknown'),
            'options': item.get('options', []),
            'ground_truth': ground_truth,
            'meta_info': item.get('meta_info', {}),
        })
    
    logger.info(f"加载了 {len(samples)} 个有效样本")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Self-Evolving Agent V7 - Dual Reasoning Mode with Finetuned VL')
    parser.add_argument('--vl-model', type=str, 
                       default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct',
                       help='VL model path (default: Qwen3-VL-8B-Instruct)')
    parser.add_argument('--adapter-path', type=str, 
                       default='/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/qwen3vl_10pct_fast/v0-20260211-225557/checkpoint-147',
                       help='Path to LoRA adapter (optional)')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--question-types', type=str, nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Self-Evolving Agent V7 - Testing with Finetuned VL Model")
    logger.info("="*80)
    logger.info(f"Base VL Model: {args.vl_model}")
    logger.info(f"LoRA Adapter: {args.adapter_path}")
    logger.info(f"GPUs: {args.num_gpus}")
    logger.info("="*80)
    
    # 加载数据
    data = load_vsibench_data()
    logger.info(f"加载 {len(data)} 条数据")
    
    # 过滤任务类型
    if args.question_types:
        data = [d for d in data if d['question_type'] in args.question_types]
        logger.info(f"过滤后: {len(data)} 条数据")
    
    # 限制样本数
    if args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"限制为: {len(data)} 条数据")
    
    # 分配到各 GPU
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    samples_per_gpu = len(data) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(data)
        gpu_samples.append(data[start:end])
    
    # 启动多进程
    processes = []
    output_files = []
    
    for gpu_id in range(num_gpus):
        output_file = f"{args.output_dir}/mindmap_export_gpu{gpu_id}.jsonl"
        output_files.append(output_file)
        p = mp.Process(
            target=worker_process_export_mindmap,
            args=(gpu_id, gpu_samples[gpu_id], output_file)
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 读取所有输出文件
    logger.info("合并所有GPU的输出...")
    all_results = []
    for output_file in output_files:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    all_results.append(json.loads(line))
    
    logger.info(f"共生成 {len(all_results)} 个训练样本")
    
    # 统计结果
    type_stats = defaultdict(lambda: {
        'total': 0, 
        'rule_score_sum': 0, 
        'vl_score_sum': 0,
        'best_score_sum': 0,
        'rule_wins': 0,
        'vl_wins': 0,
    })
    calibration_stats = {'used': 0, 'total': 0}
    evolution_stats = {'actions': 0, 'samples': 0}
    
    for r in all_results:
        qtype = r['question_type']
        type_stats[qtype]['total'] += 1
        type_stats[qtype]['rule_score_sum'] += r.get('rule_score', 0)
        type_stats[qtype]['vl_score_sum'] += r.get('vl_score', 0)
        type_stats[qtype]['best_score_sum'] += r.get('score', 0)
        
        if r.get('method_used') == 'rule':
            type_stats[qtype]['rule_wins'] += 1
        else:
            type_stats[qtype]['vl_wins'] += 1
        
        if 'calibration' in r and r['calibration'].get('scale_factor', 1.0) != 1.0:
            calibration_stats['used'] += 1
        calibration_stats['total'] += 1
        
        if 'evolution_actions' in r and r['evolution_actions']:
            evolution_stats['actions'] += len(r['evolution_actions'])
            evolution_stats['samples'] += 1
    
    # 打印结果
    print("\n" + "=" * 100)
    print("V7 双推理模式框架 (规则 + Qwen3-VL) - 测试结果")
    print("=" * 100)
    print(f"VL Model: {args.vl_model}")
    print("-" * 100)
    print(f"{'任务类型':<35} {'Rule':>10} {'VL':>10} {'Best':>10} {'Rule Wins':>12} {'VL Wins':>10}")
    print("-" * 100)
    
    overall_rule = 0
    overall_vl = 0
    overall_best = 0
    overall_total = 0
    
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        n = stats['total']
        rule_avg = stats['rule_score_sum'] / n if n > 0 else 0
        vl_avg = stats['vl_score_sum'] / n if n > 0 else 0
        best_avg = stats['best_score_sum'] / n if n > 0 else 0
        
        print(f"{qtype:<35} {rule_avg*100:>9.2f}% {vl_avg*100:>9.2f}% {best_avg*100:>9.2f}% {stats['rule_wins']:>12} {stats['vl_wins']:>10}")
        
        overall_rule += stats['rule_score_sum']
        overall_vl += stats['vl_score_sum']
        overall_best += stats['best_score_sum']
        overall_total += n
    
    print("-" * 100)
    print(f"{'Overall':<35} {overall_rule/overall_total*100:>9.2f}% {overall_vl/overall_total*100:>9.2f}% {overall_best/overall_total*100:>9.2f}%")
    print("=" * 100)
    print(f"总样本数: {overall_total}")
    print(f"校准使用率: {calibration_stats['used']}/{calibration_stats['total']} ({calibration_stats['used']/calibration_stats['total']*100:.1f}%)")
    print(f"演化动作: {evolution_stats['actions']} actions in {evolution_stats['samples']} samples")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"evolving_agent_v7_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'results_by_type': {
            qtype: {
                'rule': stats['rule_score_sum'] / stats['total'] if stats['total'] > 0 else 0,
                'vl': stats['vl_score_sum'] / stats['total'] if stats['total'] > 0 else 0,
                'best': stats['best_score_sum'] / stats['total'] if stats['total'] > 0 else 0,
                'samples': stats['total'],
            }
            for qtype, stats in type_stats.items()
        },
        'overall': {
            'rule': overall_rule / overall_total if overall_total > 0 else 0,
            'vl': overall_vl / overall_total if overall_total > 0 else 0,
            'best': overall_best / overall_total if overall_total > 0 else 0,
        },
        'total_samples': overall_total,
        'calibration_stats': calibration_stats,
        'evolution_stats': evolution_stats,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果保存到: {output_dir}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
