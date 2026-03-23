#!/usr/bin/env python3
"""
VSIBench 测试 - DirectQA 版本 (纯规则推理，不依赖 VLM)

特点：
1. 所有任务都用规则/计算方法，不使用 Qwen 等 VLM 推理
2. 扩展词汇表：添加 stool, bucket, screen, display 等
3. rel_direction/rel_distance 基于 3D 位置计算
4. appearance_order 纯依靠 entity 的 first_seen_frame 排序

作者: tianjungu
日期: 2026-01-27
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
    """VSIBench 官方 MRA 指标 (多阈值版本)"""
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
    
    # 检测信息
    count: int = 1
    avg_confidence: float = 0.0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    
    # 3D 重建信息
    position_3d: Optional[np.ndarray] = None  # (3,) 中心位置 [x, y, z]
    size_3d: Optional[np.ndarray] = None  # (3,) 尺寸 (宽,高,深) in meters
    depth_median: float = 0.0  # 中值深度
    
    def to_dict(self) -> Dict:
        return {
            'id': self.entity_id,
            'label': self.label,
            'count': self.count,
            'confidence': round(self.avg_confidence, 3),
            'first_frame': self.first_seen_frame,
            'last_frame': self.last_seen_frame,
            'position_3d': self.position_3d.tolist() if self.position_3d is not None else None,
            'size_3d': self.size_3d.tolist() if self.size_3d is not None else None,
            'depth_median': round(self.depth_median, 3),
        }


# ============================================================================
# 扩展词汇表（关键改进！）
# ============================================================================

EXTENDED_VOCABULARY = [
    # 基础家具
    "chair", "table", "sofa", "couch", "bed", "desk", "shelf", "cabinet",
    "drawer", "closet", "wardrobe", "dresser", "nightstand", "bedside table",
    
    # 座椅类（扩展）
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

# 同义词映射（用于物体匹配）
SYNONYM_MAP = {
    # 显示器
    'monitor': ['monitor', 'screen', 'display', 'computer screen'],
    'screen': ['monitor', 'screen', 'display', 'computer screen'],
    'display': ['monitor', 'screen', 'display', 'computer screen'],
    
    # 加热器
    'heater': ['heater', 'radiator', 'heating unit'],
    'radiator': ['heater', 'radiator', 'heating unit'],
    
    # 沙发
    'sofa': ['sofa', 'couch'],
    'couch': ['sofa', 'couch'],
    
    # 电视
    'tv': ['tv', 'television'],
    'television': ['tv', 'television'],
    
    # 床头柜
    'nightstand': ['nightstand', 'bedside table', 'night table', 'night stand'],
    'bedside table': ['nightstand', 'bedside table', 'night table'],
    
    # 垃圾桶
    'trash can': ['trash can', 'trash bin', 'garbage bin', 'bin'],
    'trash bin': ['trash can', 'trash bin', 'garbage bin', 'bin'],
    
    # 灯
    'lamp': ['lamp', 'light', 'ceiling light'],
    'light': ['lamp', 'light', 'ceiling light'],
    'ceiling light': ['lamp', 'light', 'ceiling light'],
    
    # 冰箱
    'refrigerator': ['refrigerator', 'fridge'],
    'fridge': ['refrigerator', 'fridge'],
    
    # 凳子
    'stool': ['stool', 'bar stool', 'step stool'],
    
    # 容器
    'bucket': ['bucket', 'pail'],
    'basket': ['basket', 'bin'],
}


def get_synonyms(word: str) -> List[str]:
    """获取单词的同义词列表"""
    word_lower = word.lower()
    return SYNONYM_MAP.get(word_lower, [word_lower])


def match_object_name(target: str, label: str) -> bool:
    """检查目标物体名称是否与检测标签匹配（支持同义词）"""
    target_lower = target.lower().strip()
    label_lower = label.lower().strip()
    
    # 直接匹配
    if target_lower in label_lower or label_lower in target_lower:
        return True
    
    # 同义词匹配
    target_synonyms = get_synonyms(target_lower)
    label_synonyms = get_synonyms(label_lower)
    
    for ts in target_synonyms:
        for ls in label_synonyms:
            if ts in ls or ls in ts:
                return True
    
    # 单词级别匹配
    target_words = set(target_lower.split())
    label_words = set(label_lower.split())
    if target_words & label_words:
        return True
    
    return False


# ============================================================================
# 心智地图构建器
# ============================================================================

class MindMapBuilderDirectQA:
    """心智地图构建器 - DirectQA 版本"""
    
    def __init__(self, device: str = 'cuda', num_frames: int = 32, 
                 box_threshold: float = 0.25):
        self.device = device
        self.num_frames = num_frames  # 增加到 32 帧以提高 appearance_order 精度
        self.box_threshold = box_threshold
        
        self._labeler = None
        self._depth_estimator = None
        
        # 相机内参估计
        self.focal_length = 500
        self.principal_point = None
        
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
        """释放模型内存"""
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
        """从视频提取帧"""
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
    
    def _estimate_depth_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """批量深度估计"""
        depth_maps = []
        for frame in frames:
            try:
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                
                # 校准：假设中值深度为 2.5 米
                median_depth = np.median(depth_map)
                if median_depth > 0:
                    scale = 2.5 / median_depth
                    depth_map = depth_map * scale
                
                depth_maps.append(depth_map)
            except Exception as e:
                logger.warning(f"深度估计失败: {e}")
                H, W = frame.shape[:2]
                depth_maps.append(np.ones((H, W), dtype=np.float32) * 2.5)
        
        return depth_maps
    
    def build_from_video(self, video_path: str, target_objects: List[str] = None) -> Dict[str, MindMapEntity3D]:
        """从视频构建 3D 心智地图"""
        self._load_models()
        
        # 提取和采样帧
        all_frames = self._extract_frames(video_path)
        if not all_frames:
            return {}
        
        frames, frame_indices = self._sample_frames(all_frames)
        H, W = frames[0].shape[:2]
        self.principal_point = (W / 2, H / 2)
        
        # 构建检测词汇表（使用扩展词汇）
        if target_objects is None:
            target_objects = []
        
        vocab = list(set(target_objects + EXTENDED_VOCABULARY))
        text_prompt = " . ".join(vocab) + " ."
        
        # 深度估计
        logger.info(f"深度估计 {len(frames)} 帧...")
        depth_maps = self._estimate_depth_batch(frames)
        
        # 逐帧检测
        all_detections: Dict[str, List[Dict]] = defaultdict(list)
        
        for frame_idx, (frame, depth_map) in enumerate(zip(frames, depth_maps)):
            original_frame_idx = frame_indices[frame_idx]
            
            # 确保深度图尺寸匹配
            if depth_map.shape[:2] != (H, W):
                depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # 检测物体
            results = self._labeler.detect(frame, text_prompt)
            
            for det in results:
                label = det.label.lower()
                bbox = det.bbox_pixels  # [x1, y1, x2, y2]
                confidence = det.confidence
                
                # 获取 bbox 边界
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 获取深度信息
                depth_roi = depth_map[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                
                depth_median = float(np.median(depth_roi))
                
                # 计算 3D 位置
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
                depth_3d = min(width_3d, height_3d) * 0.5  # 假设深度是宽高较小值的一半
                
                all_detections[label].append({
                    'frame_idx': original_frame_idx,
                    'bbox': bbox,
                    'confidence': confidence,
                    'position_3d': np.array([x_3d, y_3d, z_3d]),
                    'size_3d': np.array([width_3d, height_3d, depth_3d]),
                    'depth_median': depth_median,
                })
        
        # 聚合成实体
        entities = {}
        for category, dets in all_detections.items():
            if not dets:
                continue
            
            # 按帧分组计数
            frame_dets = defaultdict(list)
            for d in dets:
                frame_dets[d['frame_idx']].append(d)
            
            max_count = max(len(fd) for fd in frame_dets.values())
            avg_conf = np.mean([d['confidence'] for d in dets])
            first_frame = min(d['frame_idx'] for d in dets)
            last_frame = max(d['frame_idx'] for d in dets)
            
            # 聚合 3D 信息
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
# DirectQA 问题回答器（核心！）
# ============================================================================

class DirectQA:
    """直接从心智地图回答问题（纯规则，不用 VLM）"""
    
    # -------------------------------------------------------------------------
    # 数值任务
    # -------------------------------------------------------------------------
    
    @staticmethod
    def answer_counting(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        """回答计数问题"""
        # 提取目标物体名称
        match = re.search(r'How many (\w+)\(s\)', question)
        if not match:
            match = re.search(r'How many (\w+)', question)
        
        if not match:
            return "0"
        
        target = match.group(1).lower()
        
        # 在心智地图中查找匹配的物体
        for label, entity in mind_map.items():
            if match_object_name(target, label):
                return str(entity.count)
        
        return "0"
    
    @staticmethod
    def answer_object_size(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        """回答物体尺寸问题"""
        q_lower = question.lower()
        
        # 在问题中找到目标物体
        for label, entity in mind_map.items():
            # 直接匹配
            if label.lower() in q_lower:
                if entity.size_3d is not None:
                    max_dim = float(np.max(entity.size_3d)) * 100
                    return str(int(max_dim))
            # 同义词匹配
            else:
                for syn in get_synonyms(label.lower()):
                    if syn in q_lower:
                        if entity.size_3d is not None:
                            max_dim = float(np.max(entity.size_3d)) * 100
                            return str(int(max_dim))
                        break
        
        return "50"  # 默认值
    
    @staticmethod
    def answer_room_size(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        """估算房间面积"""
        if not mind_map:
            return "20"
        
        # 收集所有物体的 3D 位置
        positions = []
        for entity in mind_map.values():
            if entity.position_3d is not None:
                positions.append(entity.position_3d)
        
        if len(positions) < 2:
            return str(12 + len(mind_map) * 2)
        
        positions = np.array(positions)
        
        # 计算 XY 平面的包围盒（保持原来的逻辑）
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        # 恢复原来的扩展系数，但略微缩小
        # 原来是 +2，现在改为 +1.5（对小房间影响更小）
        estimated_area = (x_range + 1.5) * (y_range + 1.5)
        estimated_area = max(8, min(80, estimated_area))
        
        return f"{estimated_area:.1f}"
    
    @staticmethod
    def answer_abs_distance(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        """回答物体间距离问题
        
        问题格式: "what is the distance between the X and the Y (in meters)?"
        """
        q_lower = question.lower()
        
        # 从问题中提取物体名（更可靠）
        # 格式: "between the X and the Y"
        between_match = re.search(r'between (?:the )?([a-z]+(?:\s+[a-z]+)*)\s+and\s+(?:the )?([a-z]+(?:\s+[a-z]+)*)', q_lower)
        
        if between_match:
            obj1_name = between_match.group(1).strip()
            obj2_name = between_match.group(2).strip()
        else:
            # 备选：从心智地图中找问题中出现的物体
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
        
        # 改进的位置查找
        def find_position_fuzzy(name: str) -> Optional[np.ndarray]:
            name_lower = name.lower().strip()
            
            # 1. 精确匹配
            for label, entity in mind_map.items():
                if label.lower() == name_lower and entity.position_3d is not None:
                    return entity.position_3d
            
            # 2. 子串匹配
            for label, entity in mind_map.items():
                if (name_lower in label.lower() or label.lower() in name_lower) and entity.position_3d is not None:
                    return entity.position_3d
            
            # 3. 同义词匹配
            for label, entity in mind_map.items():
                if match_object_name(name_lower, label) and entity.position_3d is not None:
                    return entity.position_3d
            
            # 4. 特殊处理：washing machine -> washer
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
            # 使用 3D 欧氏距离
            dist = float(np.linalg.norm(pos1 - pos2))
            return f"{dist:.1f}"
        
        if pos1 is not None:
            # 只找到一个物体，返回到原点的距离
            return f"{float(np.linalg.norm(pos1)):.1f}"
        
        if pos2 is not None:
            return f"{float(np.linalg.norm(pos2)):.1f}"
        
        return "2.0"
    
    # -------------------------------------------------------------------------
    # 选择题任务 - 相对方向（关键改进！）
    # -------------------------------------------------------------------------
    
    @staticmethod
    def answer_rel_direction(mind_map: Dict[str, MindMapEntity3D], question: str, 
                             options: List[str], difficulty: str = 'easy') -> str:
        """
        回答相对方向问题 - 基于 3D 位置计算
        
        问题格式有两种：
        1. "If I am standing by X and facing Y, is Z to my left or my right?"
           → 相对于观察者（站在X面向Y）判断Z的方向
        2. "If I am standing by X and facing Y, is Z to the left or the right of Y?"
           → 相对于物体Y判断Z的方向（从观察者视角看）
        """
        if not options:
            return "left"
        
        q_lower = question.lower()
        
        # 解析问题：提取 standing_obj, facing_obj, target_obj
        standing_match = re.search(r'standing by (?:the )?([a-z]+(?:\s+[a-z]+)*?)(?:\s+and\s+facing|\s*,)', q_lower)
        facing_match = re.search(r'facing (?:the )?([a-z]+(?:\s+[a-z]+)*?)(?:\s*,|\s+is\b)', q_lower)
        
        # 判断问题格式：
        # 格式1: "is Z to my left" - 相对于观察者
        # 格式2: "is Z to the left of Y" - 相对于物体Y
        target_match_my = re.search(r'is (?:the )?([a-z]+(?:\s+[a-z]+)*?)\s+to\s+my', q_lower)
        target_match_of = re.search(r'is (?:the )?([a-z]+(?:\s+[a-z]+)*?)\s+to\s+the\s+(?:left|right|back|front)', q_lower)
        
        # 确定使用哪种格式
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
        
        # 在心智地图中查找物体位置
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
        # position_3d = [x, y, z] 其中：
        # x = 水平位置（正方向向右）
        # y = 垂直位置（正方向向下，在图像坐标系中）
        # z = 深度（距离相机的距离）
        # 
        # 我们在 X-Y 平面（俯视图）工作，但需要注意 Y 轴方向
        # 在俯视图中：X 向右，Y（这里用 Z 表示深度）向前
        forward = np.array([facing_pos[0] - standing_pos[0], facing_pos[2] - standing_pos[2]])
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return options[0]
        forward = forward / forward_norm
        
        # 计算右方向
        # 在 2D 平面中，将向量 (a, b) 顺时针旋转 90 度得到 (b, -a)
        # 但这取决于坐标系约定
        # 尝试反转：逆时针旋转 90 度得到 (-b, a)
        right = np.array([forward[1], -forward[0]])
        
        if relative_to_observer:
            to_target = np.array([target_pos[0] - standing_pos[0], target_pos[2] - standing_pos[2]])
        else:
            to_target = np.array([target_pos[0] - facing_pos[0], target_pos[2] - facing_pos[2]])
        
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm < 1e-6:
            return options[0]
        to_target = to_target / to_target_norm
        
        # 计算点积
        front_dot = np.dot(to_target, forward)  # >0 前方, <0 后方
        right_dot = np.dot(to_target, right)    # >0 右边, <0 左边
        
        # 根据问题类型确定答案
        if 'left or' in q_lower and 'right' in q_lower and 'back' not in q_lower:
            # Easy: left / right (二选一)
            direction = 'right' if right_dot > 0 else 'left'
        elif 'left, right, or back' in q_lower or 'left, right or back' in q_lower:
            # Medium: left / right / back (三选一)
            if front_dot < -0.5:  # 后方阈值放宽一点
                direction = 'back'
            elif right_dot > 0:
                direction = 'right'
            else:
                direction = 'left'
        elif 'front-left' in q_lower or 'front-right' in q_lower:
            # Hard: quadrants (四象限)
            if front_dot > 0:
                direction = 'front-right' if right_dot > 0 else 'front-left'
            else:
                direction = 'back-right' if right_dot > 0 else 'back-left'
        else:
            direction = 'right' if right_dot > 0 else 'left'
        
        # 在选项中匹配
        for opt in options:
            opt_lower = opt.lower()
            if direction in opt_lower:
                return opt
        
        return options[0]
    
    # -------------------------------------------------------------------------
    # 选择题任务 - 相对距离（关键改进！）
    # -------------------------------------------------------------------------
    
    @staticmethod
    def answer_rel_distance(mind_map: Dict[str, MindMapEntity3D], question: str, 
                            options: List[str]) -> str:
        """
        回答相对距离问题 - 基于 3D 位置计算
        
        问题格式:
        - "which of these objects (A, B, C, D) is the closest to the X?"
        - "which of these objects (A, B, C, D) is the farthest from the X?"
        """
        if not options:
            return ""
        
        q_lower = question.lower()
        
        # 判断是找最近还是最远
        find_closest = 'closest' in q_lower or 'nearest' in q_lower or 'closer' in q_lower
        find_farthest = 'farthest' in q_lower or 'furthest' in q_lower or 'farther' in q_lower
        
        if not find_closest and not find_farthest:
            find_closest = True
        
        # 改进正则：支持多词物体名称
        ref_match = re.search(r'(?:to|from) (?:the )?([a-z]+(?:\s+[a-z]+)*)\??', q_lower)
        if not ref_match:
            return options[0]
        
        reference_name = ref_match.group(1)
        
        # 提取候选物体（括号内的物体列表）
        candidates_match = re.search(r'\(([^)]+)\)', q_lower)
        if candidates_match:
            candidates_text = candidates_match.group(1)
            candidate_names = [c.strip() for c in candidates_text.split(',')]
        else:
            candidate_names = []
            for opt in options:
                opt_content = re.sub(r'^[A-D]\.\s*', '', opt)
                candidate_names.append(opt_content.strip().lower())
        
        # 改进的位置查找：支持模糊匹配和多种尝试
        def find_position_fuzzy(name: str) -> Optional[np.ndarray]:
            name_lower = name.lower().strip()
            
            # 1. 精确匹配
            for label, entity in mind_map.items():
                if label.lower() == name_lower and entity.position_3d is not None:
                    return entity.position_3d
            
            # 2. 子串匹配
            for label, entity in mind_map.items():
                if (name_lower in label.lower() or label.lower() in name_lower) and entity.position_3d is not None:
                    return entity.position_3d
            
            # 3. 同义词匹配
            for label, entity in mind_map.items():
                if match_object_name(name_lower, label) and entity.position_3d is not None:
                    return entity.position_3d
            
            # 4. 单词级别匹配（如 "washing machine" 匹配 "washer"）
            name_words = set(name_lower.split())
            for label, entity in mind_map.items():
                label_words = set(label.lower().split())
                if name_words & label_words and entity.position_3d is not None:
                    return entity.position_3d
            
            return None
        
        # 获取参考物体位置
        ref_pos = find_position_fuzzy(reference_name)
        
        # 如果参考物体找不到，尝试用问题中其他信息
        if ref_pos is None:
            # 尝试匹配常见的同义词
            alt_names = get_synonyms(reference_name)
            for alt in alt_names:
                ref_pos = find_position_fuzzy(alt)
                if ref_pos is not None:
                    break
        
        # 如果仍然找不到，返回随机选项而非第一个（避免偏向 A）
        if ref_pos is None:
            import random
            return random.choice(options)
        
        # 计算每个候选物体到参考物体的距离
        distances = {}
        option_map = {}  # 候选名 -> 对应选项
        
        for i, cand in enumerate(candidate_names):
            cand_pos = find_position_fuzzy(cand)
            if cand_pos is not None:
                dist = float(np.linalg.norm(cand_pos - ref_pos))
                distances[cand] = dist
                if i < len(options):
                    option_map[cand] = options[i]
        
        if not distances:
            # 没有任何候选物体被找到，返回随机选项
            import random
            return random.choice(options)
        
        # 找最近或最远
        if find_closest:
            best_cand = min(distances.keys(), key=lambda k: distances[k])
        else:
            best_cand = max(distances.keys(), key=lambda k: distances[k])
        
        # 返回对应的选项
        if best_cand in option_map:
            return option_map[best_cand]
        
        # 在选项中匹配
        for opt in options:
            opt_lower = opt.lower()
            if best_cand in opt_lower or any(s in opt_lower for s in get_synonyms(best_cand)):
                return opt
        
        return options[0]
    
    # -------------------------------------------------------------------------
    # 选择题任务 - 出现顺序（关键改进！）
    # -------------------------------------------------------------------------
    
    @staticmethod
    def answer_appearance_order(mind_map: Dict[str, MindMapEntity3D], question: str, 
                                 options: List[str]) -> str:
        """
        回答出现顺序问题 - 纯依靠 entity 的 first_seen_frame 排序
        
        问题格式:
        "What will be the first-time appearance order of the following categories 
         in the video: ceiling light, cup, heater, door?"
        """
        if not options:
            return ""
        
        # 从问题中提取需要排序的物体列表
        match = re.search(r'following categories.*?:\s*(.+?)\?', question, re.IGNORECASE)
        if match:
            objects_text = match.group(1)
            target_objects = [obj.strip().lower() for obj in objects_text.split(',')]
        else:
            opt_content = re.sub(r'^[A-D]\.\s*', '', options[0])
            target_objects = [obj.strip().lower() for obj in opt_content.split(',')]
        
        # 改进的物体匹配函数
        def find_entity_fuzzy(target: str) -> Optional[MindMapEntity3D]:
            target_lower = target.lower().strip()
            
            # 1. 精确匹配
            for label, entity in mind_map.items():
                if label.lower() == target_lower:
                    return entity
            
            # 2. 子串匹配
            for label, entity in mind_map.items():
                if target_lower in label.lower() or label.lower() in target_lower:
                    return entity
            
            # 3. 同义词匹配
            for label, entity in mind_map.items():
                if match_object_name(target_lower, label):
                    return entity
            
            # 4. 单词级别匹配
            target_words = set(target_lower.split())
            for label, entity in mind_map.items():
                label_words = set(label.lower().split())
                if target_words & label_words:
                    return entity
            
            return None
        
        # 获取每个目标物体的首次出现帧
        object_frames = {}
        for target in target_objects:
            entity = find_entity_fuzzy(target)
            if entity is not None:
                if target not in object_frames:
                    object_frames[target] = entity.first_seen_frame
                else:
                    object_frames[target] = min(object_frames[target], entity.first_seen_frame)
        
        # 如果检测到的物体太少，直接选择随机选项
        if len(object_frames) < 2:
            import random
            return random.choice(options)
        
        # 按首次出现帧排序
        sorted_objects = sorted(object_frames.items(), key=lambda x: x[1])
        predicted_order = [obj for obj, _ in sorted_objects]
        
        # 对于未检测到的物体，假设它们在最后出现
        missing_objects = [t for t in target_objects if t not in object_frames]
        predicted_order.extend(missing_objects)
        
        # 计算每个选项与预测顺序的匹配度
        best_option = options[0]
        best_score = -1
        
        for opt in options:
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt)
            opt_objects = [o.strip().lower() for o in opt_content.split(',')]
            
            # 计算匹配分数（位置相似度）
            score = 0
            for i, pred_obj in enumerate(predicted_order):
                for j, opt_obj in enumerate(opt_objects):
                    if match_object_name(pred_obj, opt_obj):
                        # 位置越接近得分越高
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
    def answer_route_planning(mind_map: Dict[str, MindMapEntity3D], question: str, 
                              options: List[str]) -> str:
        """
        回答路径规划问题
        
        这是最复杂的任务，需要理解路径中的转向方向。
        当前实现：基于物体位置计算转向方向。
        """
        if not options:
            return ""
        
        q_lower = question.lower()
        
        # 解析起点、朝向、路径点
        # 格式: "You are a robot beginning at the X facing the Y. You want to navigate to the Z.
        #        1. Go forward until the A
        #        2. [please fill in]
        #        3. Go forward until the B
        #        ..."
        
        # 提取起始位置和朝向
        start_match = re.search(r'beginning at (?:the )?(\w+(?:\s+\w+)?)', q_lower)
        facing_match = re.search(r'facing (?:the )?(\w+)', q_lower)
        
        if not start_match or not facing_match:
            return options[0]
        
        # 提取路径点
        steps = re.findall(r'go forward until (?:the )?(\w+(?:\s+\w+)?)', q_lower)
        
        # 找到需要填写的转向点数量
        fill_count = q_lower.count('[please fill in]')
        
        if fill_count == 0:
            return options[0]
        
        # 获取物体位置的辅助函数
        def find_position(name: str) -> Optional[np.ndarray]:
            for label, entity in mind_map.items():
                if match_object_name(name, label) and entity.position_3d is not None:
                    return entity.position_3d
            return None
        
        # 构建路径位置列表
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
        
        # 计算每个转向点的方向
        turns = []
        for i in range(1, len(path_positions) - 1):
            prev_dir = path_positions[i] - path_positions[i-1]
            next_dir = path_positions[i+1] - path_positions[i]
            
            prev_dir = prev_dir[:2] / (np.linalg.norm(prev_dir[:2]) + 1e-8)
            next_dir = next_dir[:2] / (np.linalg.norm(next_dir[:2]) + 1e-8)
            
            # 计算叉积判断转向
            cross = prev_dir[0] * next_dir[1] - prev_dir[1] * next_dir[0]
            dot = np.dot(prev_dir, next_dir)
            
            if dot < -0.5:  # 大约 120 度以上
                turns.append('turn back')
            elif cross > 0.3:
                turns.append('turn right')
            elif cross < -0.3:
                turns.append('turn left')
            else:
                turns.append('go forward')
        
        # 匹配选项
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

def worker_process(gpu_id: int, samples: List[Dict], result_queue: mp.Queue):
    """GPU Worker 进程"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    builder = MindMapBuilderDirectQA(device='cuda', num_frames=16, box_threshold=0.25)
    
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
            
            # 根据问题类型用 DirectQA 回答
            if question_type == 'object_counting':
                pred = DirectQA.answer_counting(mind_map, question)
            elif question_type == 'object_size_estimation':
                pred = DirectQA.answer_object_size(mind_map, question)
            elif question_type == 'room_size_estimation':
                pred = DirectQA.answer_room_size(mind_map, question)
            elif question_type == 'object_abs_distance':
                pred = DirectQA.answer_abs_distance(mind_map, question)
            elif question_type == 'object_rel_direction_easy':
                pred = DirectQA.answer_rel_direction(mind_map, question, options, 'easy')
            elif question_type == 'object_rel_direction_medium':
                pred = DirectQA.answer_rel_direction(mind_map, question, options, 'medium')
            elif question_type == 'object_rel_direction_hard':
                pred = DirectQA.answer_rel_direction(mind_map, question, options, 'hard')
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
                score = mean_relative_accuracy(pred_num, gt_num) if pred_num is not None and gt_num is not None else 0.0
                correct = score > 0.5
            else:
                # 选择题：提取选项字母进行比较
                pred_letter = None
                gt_letter = str(gt).strip().upper()
                
                # 如果 GT 是选项内容而不是字母，找到对应字母
                if len(gt_letter) > 1:
                    for idx, opt in enumerate(options):
                        opt_content = re.sub(r'^[A-D]\.\s*', '', opt).strip()
                        if gt_letter.lower() == opt_content.lower():
                            gt_letter = chr(65 + idx)
                            break
                
                if pred:
                    # 方法1: 尝试从预测开头提取字母（如 "A. left"）
                    letter_match = re.match(r'^([A-D])[\.\s]', pred.strip().upper())
                    if letter_match:
                        pred_letter = letter_match.group(1)
                    else:
                        # 方法2: 找出预测内容对应的选项索引
                        pred_clean = re.sub(r'^[A-D]\.\s*', '', pred).lower().strip()
                        for idx, opt in enumerate(options):
                            opt_content = re.sub(r'^[A-D]\.\s*', '', opt).lower().strip()
                            if pred_clean == opt_content:
                                pred_letter = chr(65 + idx)
                                break
                        
                        # 方法3: 部分匹配
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
    args = parser.parse_args()
    
    print("=" * 70)
    print("VSIBench 测试 - DirectQA 版本 (纯规则推理)")
    print("不依赖 VLM，所有任务基于 3D 心智地图规则计算")
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
    
    # 输出结果
    print("\n" + "="*60)
    print("DirectQA 版本测试结果")
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
    output_dir = f"/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/directqa_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换 numpy 类型
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
