#!/usr/bin/env python3
"""
Self-Evolving Agent V5 - VL-Centric, Minimal Human Bias

核心设计理念：
1. 减少人为 Bias - 移除硬编码规则分支
2. VL-Centric - 所有决策由 Qwen3-VL 做出
3. 检测结果作为强参考 - 但最终答案由 VL 决定
4. VL-Based Critic - VL 观看视频判断是否需要演化
5. 动态尺度校准 - 使用已知物体或 VL 演化校准

架构：
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Self-Evolving Agent V5                               │
│                     "VL-Centric, Minimal Human Bias"                         │
│                                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │  Perception │ -> │   VL-Critic  │ -> │  VL-Reasoner │ -> │ VL-Evolver  │ │
│  │  DA3+DINO   │    │  检查+决策   │    │   推理答案   │    │  演化修正   │ │
│  │  +校准      │    │ (需要演化?)  │    │  (任何任务)  │    │ (VL验证)   │ │
│  └─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

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
# 扩展词汇表 V5 (200+ 词)
# ============================================================================

EXTENDED_VOCABULARY_V5 = [
    # === 基础家具 ===
    "chair", "table", "sofa", "couch", "bed", "desk", "shelf", "cabinet",
    "drawer", "closet", "wardrobe", "dresser", "nightstand", "bedside table",
    
    # === 座椅类（扩展）===
    "stool", "bar stool", "step stool", "bench", "ottoman", "armchair",
    "recliner", "rocking chair", "office chair", "dining chair", "folding chair",
    
    # === 电器 ===
    "tv", "television", "monitor", "screen", "display", "computer screen",
    "refrigerator", "fridge", "microwave", "stove", "oven", "washer",
    "dryer", "dishwasher", "air conditioner", "fan", "vacuum", "iron",
    
    # === 采暖/照明 ===
    "lamp", "light", "ceiling light", "floor lamp", "desk lamp", "chandelier",
    "heater", "radiator", "heating unit", "fireplace",
    
    # === 浴室 ===
    "toilet", "sink", "bathtub", "shower", "towel", "bath towel", "mirror",
    "soap", "shampoo", "toothbrush", "faucet", "toilet paper",
    
    # === 容器 ===
    "trash can", "trash bin", "garbage bin", "bucket", "pail", "basket",
    "box", "container", "bin", "crate", "barrel", "drum", "jar", "can",
    "carton", "package", "parcel",
    
    # === 小物品 ===
    "cup", "mug", "bottle", "vase", "clock", "book", "pillow", "cushion",
    "blanket", "rug", "carpet", "mat", "curtain", "blind", "tissue",
    
    # === 装饰 ===
    "picture", "painting", "poster", "frame", "plant", "flower", "sculpture",
    "candle", "photo", "artwork",
    
    # === 门窗 ===
    "door", "window", "doorframe", "windowsill", "glass door", "sliding door",
    
    # === 办公/电子设备（扩展）===
    "laptop", "computer", "keyboard", "mouse", "telephone", "phone",
    "printer", "scanner", "copier", "projector", "whiteboard", "blackboard",
    "notebook", "pen", "pencil", "stapler", "calculator",
    
    # === 厨房用品（新增）===
    "bowl", "plate", "pan", "pot", "kettle", "toaster", "blender",
    "cutting board", "knife", "fork", "spoon", "spatula", "cup", "glass",
    "dish", "tray", "pitcher",
    
    # === 工业/杂物（新增）===
    "pallet", "tank", "pipe", "valve", "exhaust", "vent", "duct",
    "grate", "drain", "cable", "wire", "hose", "rope",
    
    # === 建筑元素（新增）===
    "ceiling", "floor", "wall", "pillar", "column", "beam", "railing",
    "stairs", "step", "ladder", "elevator", "escalator", "handrail",
    
    # === 户外/运动（新增）===
    "bicycle", "bike", "scooter", "skateboard", "ball", "racket",
    "yoga mat", "dumbbell", "treadmill",
    
    # === 服饰/配件（新增）===
    "helmet", "hat", "glasses", "umbrella", "briefcase", "backpack",
    "bag", "suitcase", "shoes", "coat", "jacket", "purse", "wallet",
    
    # === 其他 ===
    "counter", "countertop", "island", "rack", "stand", "hook",
    "speaker", "radio", "alarm", "clock", "scale", "thermometer",
]

# 同义词映射（扩展版）
SYNONYM_MAP_V5 = {
    # 显示器
    'monitor': ['monitor', 'screen', 'display', 'computer screen', 'computer monitor'],
    'screen': ['monitor', 'screen', 'display', 'computer screen'],
    'display': ['monitor', 'screen', 'display', 'computer screen'],
    
    # 加热器
    'heater': ['heater', 'radiator', 'heating unit', 'space heater'],
    'radiator': ['heater', 'radiator', 'heating unit'],
    
    # 沙发
    'sofa': ['sofa', 'couch', 'loveseat'],
    'couch': ['sofa', 'couch', 'loveseat'],
    
    # 电视
    'tv': ['tv', 'television', 'television set'],
    'television': ['tv', 'television', 'television set'],
    
    # 床头柜
    'nightstand': ['nightstand', 'bedside table', 'night table', 'night stand', 'end table'],
    'bedside table': ['nightstand', 'bedside table', 'night table'],
    
    # 垃圾桶
    'trash can': ['trash can', 'trash bin', 'garbage bin', 'bin', 'waste bin', 'dustbin'],
    'trash bin': ['trash can', 'trash bin', 'garbage bin', 'bin'],
    'garbage': ['trash can', 'trash bin', 'garbage bin'],
    
    # 灯
    'lamp': ['lamp', 'light', 'lighting', 'light fixture'],
    'light': ['lamp', 'light', 'lighting', 'light fixture', 'ceiling light'],
    'ceiling light': ['lamp', 'light', 'ceiling light', 'overhead light'],
    
    # 冰箱
    'refrigerator': ['refrigerator', 'fridge', 'freezer'],
    'fridge': ['refrigerator', 'fridge'],
    
    # 凳子
    'stool': ['stool', 'bar stool', 'step stool', 'footstool'],
    
    # 容器
    'bucket': ['bucket', 'pail', 'container'],
    'basket': ['basket', 'bin', 'hamper'],
    
    # 电脑
    'computer': ['computer', 'pc', 'desktop', 'laptop'],
    'laptop': ['laptop', 'notebook', 'portable computer'],
    
    # 电话
    'telephone': ['telephone', 'phone', 'landline'],
    'phone': ['telephone', 'phone', 'mobile', 'cell phone'],
    
    # 椅子
    'chair': ['chair', 'seat', 'armchair', 'office chair', 'dining chair'],
    'armchair': ['armchair', 'chair', 'arm chair'],
    
    # 桌子
    'table': ['table', 'desk', 'dining table', 'coffee table'],
    'desk': ['desk', 'table', 'work desk', 'writing desk'],
    
    # 白板
    'whiteboard': ['whiteboard', 'white board', 'dry erase board', 'board'],
    
    # 投影仪
    'projector': ['projector', 'video projector'],
}


def get_synonyms_v5(word: str) -> List[str]:
    """获取单词的同义词列表"""
    word_lower = word.lower()
    return SYNONYM_MAP_V5.get(word_lower, [word_lower])


def match_object_name_v5(target: str, label: str) -> bool:
    """检查目标物体名称是否与检测标签匹配（支持同义词）"""
    target_lower = target.lower().strip()
    label_lower = label.lower().strip()
    
    # 直接匹配
    if target_lower == label_lower:
        return True
    if target_lower in label_lower or label_lower in target_lower:
        return True
    
    # 同义词匹配
    target_synonyms = get_synonyms_v5(target_lower)
    label_synonyms = get_synonyms_v5(label_lower)
    
    for t_syn in target_synonyms:
        for l_syn in label_synonyms:
            if t_syn == l_syn or t_syn in l_syn or l_syn in t_syn:
                return True
    
    # 复数形式
    if target_lower.endswith('s') and target_lower[:-1] == label_lower:
        return True
    if label_lower.endswith('s') and label_lower[:-1] == target_lower:
        return True
    
    return False


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


@dataclass
class MindMapEntity:
    """心智地图实体"""
    label: str
    detections: List[Detection] = field(default_factory=list)
    count: int = 0
    avg_confidence: float = 0.0
    position_3d: np.ndarray = None
    size_3d: np.ndarray = None
    
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
        return text


@dataclass 
class VLCriticFeedback:
    """VL Critic 反馈"""
    needs_evolution: bool
    confidence: float
    issues: List[str]
    objects_to_verify: List[str]
    reasoning: str


# ============================================================================
# 工具函数
# ============================================================================

NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 
                   'room_size_estimation', 'object_abs_distance']


def normalize_number(text: str) -> Optional[float]:
    """从文本中提取数字"""
    if text is None:
        return None
    text = str(text).strip().lower()
    match = re.search(r'[-+]?(?:\d+\.?\d*|\.\d+)', text)
    if match:
        try:
            return float(match.group())
        except:
            return None
    return None


def mean_relative_accuracy(pred: float, gt: float) -> float:
    """计算 MRA"""
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return min(pred / gt, gt / pred)


def evaluate_answer(pred: str, gt: str, question_type: str, options: List[str] = None) -> Tuple[float, bool]:
    if question_type in NUMERICAL_TASKS:
        pred_val = normalize_number(pred)
        gt_val = normalize_number(gt)
        if pred_val is None or gt_val is None:
            return 0.0, False
        score = mean_relative_accuracy(pred_val, gt_val)
        return score, score > 0.5
    else:
        pred = pred.strip().upper()
        gt = gt.strip().upper()
        if len(pred) >= 1 and pred[0] in 'ABCD':
            pred = pred[0]
        if len(gt) >= 1 and gt[0] in 'ABCD':
            gt = gt[0]
        correct = pred == gt
        return float(correct), correct


# ============================================================================
# 视频路径查找
# ============================================================================

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]


def find_video_path(scene_name: str) -> Optional[str]:
    """查找视频文件路径"""
    for base_dir in VIDEO_DIRS:
        video_path = os.path.join(base_dir, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


# ============================================================================
# 1. Perception - 感知模块 (带动态尺度校准)
# ============================================================================

class MindMapBuilderV5:
    """心智地图构建器 V5 - 带动态尺度校准"""
    
    # 已知物体的真实世界尺寸 (米)
    REFERENCE_SIZES = {
        'door': {'height': 2.0, 'width': 0.9},
        'chair': {'height': 0.85, 'width': 0.5, 'depth': 0.5},
        'bed': {'length': 2.0, 'width': 1.5, 'height': 0.5},
        'table': {'height': 0.75, 'width': 1.0},
        'desk': {'height': 0.75, 'width': 1.2},
        'sofa': {'height': 0.85, 'width': 1.8, 'depth': 0.9},
        'refrigerator': {'height': 1.7, 'width': 0.6, 'depth': 0.6},
        'toilet': {'height': 0.4, 'width': 0.4},
        'tv': {'height': 0.5, 'width': 0.9},
        'window': {'height': 1.2, 'width': 1.0},
        'monitor': {'height': 0.35, 'width': 0.5},
        'laptop': {'height': 0.02, 'width': 0.35},
        'microwave': {'height': 0.3, 'width': 0.5},
    }
    
    def __init__(self, device: str = 'cuda', num_frames: int = 32, box_threshold: float = 0.25):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        
        self._labeler = None
        self._depth_estimator = None
        self.focal_length = 500
        self.scale_factor = 1.0  # 动态校准因子
        self.scale_calibrated = False
    
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
    
    def _calibrate_scale(self, all_detections: Dict[str, List[Detection]], 
                         depth_maps: List[np.ndarray], frame_shapes: List[Tuple[int, int]]) -> float:
        """
        动态尺度校准 - 使用已知物体尺寸
        
        Returns:
            scale_factor: 校准因子
        """
        scale_factors = []
        
        for label, dets in all_detections.items():
            label_lower = label.lower()
            
            # 检查是否为参考物体
            for ref_name, ref_dims in self.REFERENCE_SIZES.items():
                if ref_name in label_lower or label_lower in ref_name:
                    # 使用置信度最高的检测
                    best_det = max(dets, key=lambda x: x.confidence)
                    
                    x1, y1, x2, y2 = best_det.bbox
                    pixel_height = y2 - y1
                    pixel_width = x2 - x1
                    depth = best_det.depth
                    
                    # 使用深度估计实际尺寸
                    estimated_height = pixel_height * depth / self.focal_length
                    estimated_width = pixel_width * depth / self.focal_length
                    
                    # 计算校准因子
                    if 'height' in ref_dims and estimated_height > 0:
                        factor = ref_dims['height'] / estimated_height
                        if 0.1 < factor < 10:  # 合理范围
                            scale_factors.append(factor)
                    if 'width' in ref_dims and estimated_width > 0:
                        factor = ref_dims['width'] / estimated_width
                        if 0.1 < factor < 10:
                            scale_factors.append(factor)
                    
                    break
        
        if scale_factors:
            self.scale_calibrated = True
            return float(np.median(scale_factors))
        else:
            self.scale_calibrated = False
            return 1.0  # 没有参考物体，使用默认值
    
    def build_from_video(self, video_path: str) -> Tuple[Dict[str, MindMapEntity], int]:
        """从视频构建心智地图"""
        self.load_models()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, 0
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        # 使用核心词汇表（避免超过 GroundingDINO 的 256 token 限制）
        # 这是最常见的室内物体，覆盖 VSIBench 大部分场景
        core_vocabulary = [
            # 家具
            "chair", "table", "sofa", "couch", "bed", "desk", "shelf", "cabinet",
            "drawer", "closet", "wardrobe", "dresser", "nightstand", "bedside table",
            "stool", "bench", "ottoman", "armchair",
            # 电器
            "tv", "television", "monitor", "screen", "refrigerator", "fridge",
            "microwave", "stove", "oven", "washer", "dryer", "dishwasher", "fan",
            # 采暖/照明
            "lamp", "light", "ceiling light", "heater", "radiator", "fireplace",
            # 浴室
            "toilet", "sink", "bathtub", "shower", "towel", "mirror",
            # 容器
            "trash can", "trash bin", "bucket", "basket", "box", "container",
            # 小物品
            "cup", "bottle", "vase", "clock", "book", "pillow", "cushion",
            "blanket", "rug", "carpet", "curtain",
            # 装饰
            "picture", "painting", "poster", "plant", "flower",
            # 门窗
            "door", "window",
            # 办公
            "laptop", "computer", "keyboard", "mouse", "telephone", "phone",
            "printer", "whiteboard", "projector",
            # 厨房
            "bowl", "plate", "pan", "pot", "kettle",
            # 建筑
            "ceiling", "floor", "wall", "pillar", "column", "stairs",
            # 其他
            "backpack", "bag", "suitcase", "counter", "countertop",
        ]
        
        prompt = " . ".join(core_vocabulary) + " ."
        
        # 收集所有检测
        all_detections = defaultdict(list)
        depth_maps = []
        frame_shapes = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            frame_shapes.append((h, w))
            
            # 深度估计
            depth_result = self._depth_estimator.infer_single(frame_rgb)
            depth_map = depth_result[1].cpu().numpy() if isinstance(depth_result, tuple) else depth_result.cpu().numpy()
            
            if depth_map.shape[0] != h or depth_map.shape[1] != w:
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
            depth_maps.append(depth_map)
            
            # 物体检测
            try:
                detections = self._labeler.detect(frame_rgb, prompt)
            except Exception as e:
                logger.warning(f"检测失败 (frame {frame_idx}): {e}")
                continue
            
            for det in detections:
                raw_label = det.label.strip().lower()
                if raw_label.startswith('##'):
                    continue
                
                label = raw_label
                box = det.bbox_pixels
                conf = det.confidence
                
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                cx = min(max(cx, 0), depth_map.shape[1] - 1)
                cy = min(max(cy, 0), depth_map.shape[0] - 1)
                depth = float(depth_map[cy, cx])
                
                pos_3d = np.array([
                    (cx - w / 2) * depth / self.focal_length,
                    (cy - h / 2) * depth / self.focal_length,
                    depth
                ])
                
                detection = Detection(
                    frame_idx=int(frame_idx),
                    bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    confidence=float(conf),
                    depth=depth,
                    position_3d=pos_3d
                )
                
                all_detections[label].append(detection)
        
        cap.release()
        
        # 动态尺度校准
        self.scale_factor = self._calibrate_scale(all_detections, depth_maps, frame_shapes)
        logger.debug(f"尺度校准因子: {self.scale_factor:.2f}, 已校准: {self.scale_calibrated}")
        
        # 聚合成心智地图实体
        mind_map = {}
        for label, dets in all_detections.items():
            entity = self._aggregate_detections(label, dets)
            mind_map[label] = entity
        
        return mind_map, total_frames
    
    def _aggregate_detections(self, label: str, detections: List[Detection]) -> MindMapEntity:
        """聚合检测结果"""
        if not detections:
            return MindMapEntity(label=label)
        
        # 按帧分组
        frame_dets = defaultdict(list)
        for det in detections:
            frame_dets[det.frame_idx].append(det)
        
        # 取所有帧中检测到的最大数量
        max_count = max(len(fd) for fd in frame_dets.values())
        
        avg_conf = np.mean([d.confidence for d in detections])
        
        # 聚合位置（应用尺度校准）
        positions = np.array([d.position_3d for d in detections])
        avg_pos = np.median(positions, axis=0) * self.scale_factor
        
        # 估计尺寸
        best_det = max(detections, key=lambda x: x.confidence)
        box = best_det.bbox
        box_w = (box[2] - box[0]) / 640
        box_h = (box[3] - box[1]) / 480
        size_3d = np.array([box_w * best_det.depth, box_h * best_det.depth, 0.3]) * self.scale_factor
        
        return MindMapEntity(
            label=label,
            detections=detections,
            count=max_count,
            avg_confidence=float(avg_conf),
            position_3d=avg_pos,
            size_3d=size_3d
        )


# ============================================================================
# 2. VL-Based Critic - 由 VL 判断是否需要演化
# ============================================================================

class VLCritic:
    """VL-Based Critic - 让 Qwen3-VL 自主判断是否需要演化"""
    
    def __init__(self, vl_model_name: str, device: str = 'cuda'):
        self.vl_model_name = vl_model_name
        self.device = device
        self.model = None
        self.processor = None
    
    def load_model(self):
        if self.model is None:
            from transformers import AutoProcessor
            
            logger.info(f"加载 VL Critic 模型: {self.vl_model_name}")
            
            if 'Qwen3' in self.vl_model_name or 'qwen3' in self.vl_model_name.lower():
                from transformers import Qwen3VLForConditionalGeneration
                model_class = Qwen3VLForConditionalGeneration
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_class = Qwen2_5_VLForConditionalGeneration
            
            self.model = model_class.from_pretrained(
                self.vl_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.vl_model_name,
                trust_remote_code=True,
            )
    
    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def _format_mind_map(self, mind_map: Dict[str, MindMapEntity]) -> str:
        if not mind_map:
            return "No objects detected."
        lines = []
        for label, entity in sorted(mind_map.items(), key=lambda x: -x[1].count):
            lines.append(entity.to_text())
        return "\n".join(lines)
    
    def _format_options(self, options: List[str]) -> str:
        if not options:
            return ""
        return "\nOptions:\n" + "\n".join(options)
    
    def evaluate(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        question: str,
        question_type: str,
        initial_answer: str,
        options: List[str] = None,
        scale_calibrated: bool = True
    ) -> VLCriticFeedback:
        """
        VL 模型观看视频，判断检测结果和初始答案是否可靠
        """
        self.load_model()
        
        # 构建评估 Prompt
        scale_note = "" if scale_calibrated else """
Note: Scale calibration was NOT successful (no reference objects found).
Distance and size estimates may be inaccurate. Please verify with visual observation."""

        prompt = f"""You are a quality control agent. Your job is to verify if the detection results are correct by watching this video.

=== DETECTION RESULTS (Mind Map) ===
{self._format_mind_map(mind_map)}
{scale_note}

=== QUESTION ===
{question}
{self._format_options(options)}

=== PROPOSED ANSWER ===
{initial_answer}

=== YOUR TASK ===
Watch the video carefully and evaluate:

1. **Detection Accuracy**: Are all relevant objects detected? Are the counts correct?
   - Check for: missed objects, over-counting (same object counted multiple times across frames), wrong labels

2. **Answer Correctness**: Is the proposed answer likely correct based on what you see?

3. **Need for Evolution**: Should we re-verify the detection results?

Please respond in this EXACT format:
NEEDS_EVOLUTION: [YES/NO]
CONFIDENCE: [0-100]%
ISSUES: [List specific problems found, or "None"]
OBJECTS_TO_VERIFY: [Objects that need re-counting or verification, or "None"]
REASONING: [Brief explanation of your evaluation]
"""

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
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            
            response = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            return self._parse_response(response)
            
        except Exception as e:
            logger.warning(f"VL Critic 失败: {e}")
            return VLCriticFeedback(
                needs_evolution=False,
                confidence=0.5,
                issues=[str(e)],
                objects_to_verify=[],
                reasoning="Critic evaluation failed"
            )
    
    def _parse_response(self, response: str) -> VLCriticFeedback:
        """解析 VL Critic 响应"""
        # 解析 NEEDS_EVOLUTION
        needs_evolution = False
        evolution_match = re.search(r'NEEDS_EVOLUTION:\s*(YES|NO)', response, re.IGNORECASE)
        if evolution_match:
            needs_evolution = evolution_match.group(1).upper() == 'YES'
        
        # 解析置信度
        confidence = 0.5
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)%?', response, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1)) / 100
        
        # 解析 ISSUES
        issues = []
        issues_match = re.search(r'ISSUES:\s*(.+?)(?=OBJECTS_TO_VERIFY|REASONING|$)', response, re.IGNORECASE | re.DOTALL)
        if issues_match:
            issues_text = issues_match.group(1).strip()
            if issues_text.lower() != 'none':
                issues = [i.strip() for i in issues_text.split(',') if i.strip()]
        
        # 解析 OBJECTS_TO_VERIFY
        objects_to_verify = []
        objects_match = re.search(r'OBJECTS_TO_VERIFY:\s*(.+?)(?=REASONING|$)', response, re.IGNORECASE | re.DOTALL)
        if objects_match:
            objects_text = objects_match.group(1).strip()
            if objects_text.lower() != 'none':
                objects_to_verify = [o.strip() for o in objects_text.split(',') if o.strip()]
        
        # 解析 REASONING
        reasoning = ""
        reasoning_match = re.search(r'REASONING:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        return VLCriticFeedback(
            needs_evolution=needs_evolution,
            confidence=confidence,
            issues=issues,
            objects_to_verify=objects_to_verify,
            reasoning=reasoning
        )


# ============================================================================
# 3. VL-Reasoner - 统一由 VL 推理答案
# ============================================================================

class VLReasoner:
    """统一的 VL 推理器 - 所有任务类型都由 VL 回答"""
    
    def __init__(self, vl_model_name: str, device: str = 'cuda'):
        self.vl_model_name = vl_model_name
        self.device = device
        self.model = None
        self.processor = None
    
    def load_model(self):
        if self.model is None:
            from transformers import AutoProcessor
            
            logger.info(f"加载 VL Reasoner 模型: {self.vl_model_name}")
            
            if 'Qwen3' in self.vl_model_name or 'qwen3' in self.vl_model_name.lower():
                from transformers import Qwen3VLForConditionalGeneration
                model_class = Qwen3VLForConditionalGeneration
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_class = Qwen2_5_VLForConditionalGeneration
            
            self.model = model_class.from_pretrained(
                self.vl_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.vl_model_name,
                trust_remote_code=True,
            )
    
    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def _format_mind_map(self, mind_map: Dict[str, MindMapEntity]) -> str:
        if not mind_map:
            return "No objects detected."
        lines = []
        for label, entity in sorted(mind_map.items(), key=lambda x: -x[1].count):
            lines.append(entity.to_text())
        return "\n".join(lines)
    
    def _get_task_hint(self, question_type: str) -> str:
        """任务提示（帮助 VL 理解任务，但不是硬编码规则）"""
        hints = {
            'object_counting': """For COUNTING tasks:
- The detection results were computed from 32 video frames
- The 'count=' value represents the MAXIMUM number detected in any single frame
- You should TRUST the detection count as a strong baseline
- Only adjust if you see clear evidence of over/under-counting
- Remember: objects may appear in different frames, not all visible at once""",
            'object_size_estimation': "Estimate the size in centimeters. Use the detection results as reference, but verify visually.",
            'room_size_estimation': "Estimate the room area in square meters. Consider the spread of detected objects and your visual observation.",
            'object_abs_distance': "Estimate the distance between objects in meters. Use position data as reference.",
            'object_rel_distance': "Compare which object is closer to the camera based on visual observation.",
            'object_rel_direction_easy': "Determine the spatial relationship (left/right) based on what you see.",
            'object_rel_direction_medium': "Determine the spatial relationship based on visual observation.",
            'object_rel_direction_hard': "Carefully analyze the spatial relationship from the camera's perspective.",
            'obj_appearance_order': "Determine which object appears first in the video timeline.",
            'route_planning': "Plan a reasonable path between two locations in the scene.",
        }
        return hints.get(question_type, "Answer based on both the detection results and your visual observation.")
    
    def reason(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        question: str,
        question_type: str,
        options: List[str] = None,
        scale_calibrated: bool = True
    ) -> Tuple[str, float, str]:
        """
        所有任务都由 VL 模型回答，检测结果作为强参考
        """
        self.load_model()
        
        task_hint = self._get_task_hint(question_type)
        
        # 构建选项文本
        options_text = ""
        if options:
            options_text = "\nOptions:\n" + "\n".join(options)
        
        # 根据任务类型确定答案格式
        if question_type in NUMERICAL_TASKS:
            answer_format = "Your answer must be a NUMBER only (e.g., 5, 3.2, 25)."
        else:
            answer_format = "Your answer must be the OPTION LETTER only (A, B, C, or D)."
        
        # 尺度校准提示
        scale_note = ""
        if not scale_calibrated:
            scale_note = """
IMPORTANT: Scale calibration was not successful. The position and size values may be inaccurate.
For distance and size questions, rely more on your visual observation than the numerical values."""

        # 对于 counting 任务，特殊处理 prompt
        if question_type == 'object_counting':
            # 从问题中提取目标物体
            import re
            match = re.search(r'[Hh]ow many (\w+)', question)
            target_obj = match.group(1).lower() if match else ""
            
            # 从心智地图中找匹配的计数
            detected_count = 0
            matched_label = ""
            for label, entity in mind_map.items():
                if match_object_name_v5(target_obj, label):
                    if entity.count > detected_count:
                        detected_count = entity.count
                        matched_label = label
            
            count_info = f"\nDETECTED COUNT for '{target_obj}': {detected_count} (from label '{matched_label}')" if detected_count > 0 else f"\nWARNING: Object '{target_obj}' was NOT detected in the video."
            
            prompt = f"""You are analyzing a 3D indoor scene video for object counting.

=== DETECTED OBJECTS ===
{self._format_mind_map(mind_map)}
{count_info}

=== QUESTION ===
{question}

=== IMPORTANT INSTRUCTIONS ===
{task_hint}

Based on the detection results and your observation, answer with the count.
If the detected count seems reasonable, USE IT as your answer.
Only change if you have strong visual evidence of a different count.

Your answer must be a NUMBER only.
Add [Confidence: X%] at the end.

Answer:"""
        else:
            prompt = f"""You are analyzing a 3D indoor scene video. Use BOTH the detection results AND your visual observation to answer.

=== DETECTED OBJECTS (Strong Reference) ===
{self._format_mind_map(mind_map)}
{scale_note}

=== QUESTION ===
{question}{options_text}

=== TASK HINT ===
{task_hint}

=== INSTRUCTIONS ===
1. The detection results above are your PRIMARY reference
2. Watch the video to VERIFY and SUPPLEMENT the detection results
3. If you see something different from the detection, trust your observation
4. Give your final answer with confidence level

{answer_format}
Add [Confidence: X%] at the end.

Answer:"""

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
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
            response = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            return self._parse_response(response, question_type, options)
            
        except Exception as e:
            logger.warning(f"VL 推理失败: {e}")
            return "0" if question_type in NUMERICAL_TASKS else "A", 0.2, str(e)
    
    def _parse_response(self, response: str, question_type: str, options: List[str] = None) -> Tuple[str, float, str]:
        # 提取置信度
        conf_match = re.search(r'\[Confidence[：:]\s*(\d+)%?\]', response, re.IGNORECASE)
        confidence = float(conf_match.group(1)) / 100 if conf_match else 0.5
        
        response_clean = response.split('[')[0].strip()
        
        if question_type in NUMERICAL_TASKS:
            num_match = re.search(r'(\d+\.?\d*)', response_clean)
            answer = num_match.group(1) if num_match else "0"
        else:
            choice_match = re.search(r'^([A-D])', response_clean.upper())
            if choice_match:
                answer = choice_match.group(1)
            else:
                answer = "A"
        
        return answer, confidence, response_clean


# ============================================================================
# 4. VL-Evolver - VL 驱动的演化修正
# ============================================================================

class VLEvolver:
    """VL-Based 演化器 - 根据 Critic 反馈进行针对性修正"""
    
    def __init__(self, vl_model_name: str, device: str = 'cuda'):
        self.vl_model_name = vl_model_name
        self.device = device
        self.model = None
        self.processor = None
    
    def load_model(self):
        if self.model is None:
            from transformers import AutoProcessor
            
            logger.info(f"加载 VL Evolver 模型: {self.vl_model_name}")
            
            if 'Qwen3' in self.vl_model_name or 'qwen3' in self.vl_model_name.lower():
                from transformers import Qwen3VLForConditionalGeneration
                model_class = Qwen3VLForConditionalGeneration
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_class = Qwen2_5_VLForConditionalGeneration
            
            self.model = model_class.from_pretrained(
                self.vl_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.vl_model_name,
                trust_remote_code=True,
            )
    
    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def _format_mind_map(self, mind_map: Dict[str, MindMapEntity]) -> str:
        if not mind_map:
            return "No objects detected."
        lines = []
        for label, entity in sorted(mind_map.items(), key=lambda x: -x[1].count):
            lines.append(entity.to_text())
        return "\n".join(lines)
    
    def evolve(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        critic_feedback: VLCriticFeedback,
        question: str,
        question_type: str,
        options: List[str] = None,
        scale_calibrated: bool = True
    ) -> Tuple[Dict[str, MindMapEntity], str, float]:
        """
        根据 Critic 指出的问题，VL 进行针对性修正
        """
        self.load_model()
        
        issues_text = "\n".join(f"- {issue}" for issue in critic_feedback.issues) if critic_feedback.issues else "None specified"
        objects_text = ", ".join(critic_feedback.objects_to_verify) if critic_feedback.objects_to_verify else "All detected objects"
        
        # 尺度校准提示
        scale_note = ""
        if not scale_calibrated:
            scale_note = """
IMPORTANT: Scale calibration failed. Please also estimate:
- Approximate room size based on furniture layout
- Approximate distances based on typical object sizes (e.g., door height ~2m, chair height ~0.85m)"""

        # 根据任务类型确定答案格式
        if question_type in NUMERICAL_TASKS:
            answer_format = "FINAL_ANSWER must be a NUMBER only (e.g., 5, 3.2, 25)"
        else:
            answer_format = "FINAL_ANSWER must be the OPTION LETTER only (A, B, C, or D)"
        
        options_text_display = ""
        if options:
            options_text_display = "\nOptions:\n" + "\n".join(options)

        prompt = f"""You are correcting detection errors. The quality control found these issues:

=== ISSUES FOUND ===
{issues_text}
Critic reasoning: {critic_feedback.reasoning}

=== OBJECTS TO VERIFY ===
{objects_text}

=== CURRENT DETECTION ===
{self._format_mind_map(mind_map)}
{scale_note}

=== ORIGINAL QUESTION ===
{question}{options_text_display}

=== YOUR TASK ===
1. Watch the video carefully, focusing on the problematic objects
2. Provide CORRECTED counts and information
3. Give the correct answer to the question

Response format:
CORRECTIONS:
- [object_name]: COUNT=[number], REASON=[why original was wrong]
...

{answer_format}
FINAL_ANSWER: [your corrected answer]
CONFIDENCE: [0-100]%
"""

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "nframes": 12,  # 演化时用更多帧
                },
                {"type": "text", "text": prompt}
            ]
        }]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            
            response = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            return self._parse_and_update(response, mind_map, question_type)
            
        except Exception as e:
            logger.warning(f"演化失败: {e}")
            return mind_map, None, 0.0
    
    def _parse_and_update(
        self, 
        response: str, 
        mind_map: Dict[str, MindMapEntity],
        question_type: str
    ) -> Tuple[Dict[str, MindMapEntity], str, float]:
        """解析 VL 响应并更新心智地图"""
        # 创建副本
        updated_map = {}
        for k, v in mind_map.items():
            updated_map[k] = MindMapEntity(
                label=v.label,
                detections=v.detections.copy(),
                count=v.count,
                avg_confidence=v.avg_confidence,
                position_3d=v.position_3d.copy() if v.position_3d is not None else None,
                size_3d=v.size_3d.copy() if v.size_3d is not None else None,
            )
        
        # 解析 CORRECTIONS
        corrections_pattern = r'-\s*(\w+(?:\s+\w+)*):\s*COUNT=(\d+)'
        matches = re.findall(corrections_pattern, response, re.IGNORECASE)
        
        for match in matches:
            obj_name = match[0].lower()
            new_count = int(match[1])
            
            # 查找匹配的实体并更新
            for label in updated_map.keys():
                if match_object_name_v5(obj_name, label):
                    updated_map[label].count = new_count
                    updated_map[label].avg_confidence = 0.8  # 演化后提高置信度
                    break
        
        # 提取 FINAL_ANSWER
        final_match = re.search(r'FINAL_ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if final_match:
            answer_text = final_match.group(1).strip()
            if question_type in NUMERICAL_TASKS:
                num_match = re.search(r'(\d+\.?\d*)', answer_text)
                new_answer = num_match.group(1) if num_match else None
            else:
                choice_match = re.search(r'([A-D])', answer_text.upper())
                new_answer = choice_match.group(1) if choice_match else None
        else:
            new_answer = None
        
        # 提取置信度
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)%?', response, re.IGNORECASE)
        new_confidence = float(conf_match.group(1)) / 100 if conf_match else 0.5
        
        return updated_map, new_answer, new_confidence


# ============================================================================
# 5. Self-Evolving Agent V5 - 整合
# ============================================================================

class SelfEvolvingAgentV5:
    """自演化智能体 V5 - VL-Centric, Minimal Human Bias"""
    
    def __init__(
        self,
        device: str = 'cuda',
        vl_model_name: str = "/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct",
        num_frames: int = 32,
        max_evolution_rounds: int = 2,
    ):
        self.device = device
        self.vl_model_name = vl_model_name
        self.num_frames = num_frames
        self.max_evolution_rounds = max_evolution_rounds
        
        # 初始化组件
        self.perception = MindMapBuilderV5(device=device, num_frames=num_frames)
        self.critic = VLCritic(vl_model_name=vl_model_name, device=device)
        self.reasoner = VLReasoner(vl_model_name=vl_model_name, device=device)
        self.evolver = VLEvolver(vl_model_name=vl_model_name, device=device)
        
        # 共享 VL 模型（减少显存占用）
        self._vl_model_loaded = False
    
    def _share_vl_model(self):
        """在各组件间共享 VL 模型"""
        if not self._vl_model_loaded:
            # 只加载一次 VL 模型
            self.reasoner.load_model()
            
            # 共享给 critic 和 evolver
            self.critic.model = self.reasoner.model
            self.critic.processor = self.reasoner.processor
            self.evolver.model = self.reasoner.model
            self.evolver.processor = self.reasoner.processor
            
            self._vl_model_loaded = True
    
    def process(
        self,
        video_path: str,
        question: str,
        question_type: str,
        options: List[str] = None,
    ) -> Dict[str, Any]:
        """
        处理单个样本的完整流程
        
        流程:
        1. 感知 - 构建心智地图 (带动态尺度校准)
        2. VL-Reasoner - 初次推理
        3. VL-Critic - 判断是否需要演化
        4. VL-Evolver - 演化修正 (如需要)
        5. 返回最终答案
        """
        # 1. 感知 - 构建心智地图
        mind_map, total_frames = self.perception.build_from_video(video_path)
        scale_calibrated = self.perception.scale_calibrated
        
        if not mind_map:
            return {
                "answer": "0" if question_type in NUMERICAL_TASKS else "A",
                "confidence": 0.1,
                "evolved": False,
                "evolution_rounds": 0,
                "reasoning": "Failed to build mind map",
                "mind_map_summary": "Empty",
                "scale_calibrated": False,
            }
        
        # 共享 VL 模型
        self._share_vl_model()
        
        # 2. VL-Reasoner 初次推理
        answer, confidence, reasoning = self.reasoner.reason(
            video_path, mind_map, question, question_type, options, scale_calibrated
        )
        
        # 3. VL-Critic 评估
        critic_feedback = self.critic.evaluate(
            video_path, mind_map, question, question_type, answer, options, scale_calibrated
        )
        
        evolved = False
        evolution_rounds = 0
        
        # 4. 演化循环 (由 VL-Critic 决定)
        while critic_feedback.needs_evolution and evolution_rounds < self.max_evolution_rounds:
            evolved = True
            evolution_rounds += 1
            
            logger.debug(f"演化轮次 {evolution_rounds}: {critic_feedback.issues}")
            
            # VL-Evolver 演化
            evolved_mind_map, new_answer, new_confidence = self.evolver.evolve(
                video_path, mind_map, critic_feedback, question, question_type, options, scale_calibrated
            )
            
            # 如果演化成功，采用新结果
            if new_answer is not None and new_confidence > 0.3:
                mind_map = evolved_mind_map
                answer = new_answer
                confidence = new_confidence
                reasoning = f"Evolved: {critic_feedback.reasoning}"
                
                # 重新评估
                if evolution_rounds < self.max_evolution_rounds:
                    critic_feedback = self.critic.evaluate(
                        video_path, mind_map, question, question_type, answer, options, scale_calibrated
                    )
            else:
                # 演化失败，跳出循环
                logger.debug(f"演化失败，保留原结果")
                break
        
        # 构建心智地图摘要
        map_summary = "; ".join([f"{k}:{v.count}" for k, v in mind_map.items()])
        
        return {
            "answer": answer,
            "confidence": confidence,
            "evolved": evolved,
            "evolution_rounds": evolution_rounds,
            "reasoning": reasoning,
            "mind_map_summary": map_summary,
            "scale_calibrated": scale_calibrated,
            "critic_feedback": critic_feedback.reasoning if critic_feedback else "",
        }
    
    def unload_all(self):
        """释放所有模型"""
        self.perception.unload()
        self.reasoner.unload()
        # critic 和 evolver 共享模型，不需要单独释放
        self.critic.model = None
        self.critic.processor = None
        self.evolver.model = None
        self.evolver.processor = None
        self._vl_model_loaded = False
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# 数据加载
# ============================================================================

def load_vsibench_dataset(max_samples: int = None) -> List[Dict]:
    """从 HuggingFace 加载 VSI-Bench 数据集"""
    from datasets import load_dataset
    
    logger.info("加载 VSI-Bench 数据集...")
    ds = load_dataset(
        'nyu-visionx/VSI-Bench',
        split='test',
        cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench'
    )
    
    samples = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        
        scene_name = item['scene_name']
        video_path = find_video_path(scene_name)
        
        if video_path is None:
            continue
        
        samples.append({
            'scene_name': scene_name,
            'video_path': video_path,
            'question': item['question'],
            'question_type': item['question_type'],
            'answer': str(item['ground_truth']),
            'options': item.get('options'),
        })
    
    logger.info(f"找到 {len(samples)} 个有视频的样本")
    return samples


# ============================================================================
# GPU Worker
# ============================================================================

def process_sample_on_gpu(args):
    """在指定 GPU 上处理样本"""
    gpu_id, samples, vl_model_name, num_frames, max_evolution_rounds = args
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda'
    
    agent = SelfEvolvingAgentV5(
        device=device,
        vl_model_name=vl_model_name,
        num_frames=num_frames,
        max_evolution_rounds=max_evolution_rounds,
    )
    
    results = []
    
    for sample in tqdm(samples, desc=f"GPU-{gpu_id}", position=gpu_id, leave=True):
        try:
            result = agent.process(
                video_path=sample['video_path'],
                question=sample['question'],
                question_type=sample['question_type'],
                options=sample.get('options'),
            )
            
            # 评估
            score, correct = evaluate_answer(
                result['answer'],
                sample['answer'],
                sample['question_type'],
                sample.get('options')
            )
            
            results.append({
                'scene_name': sample['scene_name'],
                'question_type': sample['question_type'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'pred': result['answer'],
                'score': score,
                'correct': correct,
                'confidence': result['confidence'],
                'evolved': result['evolved'],
                'evolution_rounds': result['evolution_rounds'],
                'reasoning': result['reasoning'],
                'mind_map_summary': result['mind_map_summary'],
                'scale_calibrated': result.get('scale_calibrated', False),
                'critic_feedback': result.get('critic_feedback', ''),
            })
            
        except Exception as e:
            logger.error(f"处理失败 {sample['scene_name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'scene_name': sample['scene_name'],
                'question_type': sample['question_type'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'pred': '0',
                'score': 0.0,
                'correct': False,
                'confidence': 0.0,
                'evolved': False,
                'error': str(e),
            })
    
    agent.unload_all()
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Self-Evolving Agent V5 - VL-Centric")
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--vl-model', type=str, default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--max-evolution-rounds', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    # 加载数据
    logger.info("加载数据集...")
    samples = load_vsibench_dataset(max_samples=args.max_samples)
    logger.info(f"加载 {len(samples)} 个样本")
    
    # 分配到各 GPU
    samples_per_gpu = np.array_split(samples, args.num_gpus)
    
    # 准备参数
    worker_args = [
        (i, list(samples_per_gpu[i]), args.vl_model, args.num_frames, args.max_evolution_rounds)
        for i in range(args.num_gpus)
    ]
    
    # 多进程执行
    logger.info(f"启动 {args.num_gpus} GPU 并行处理...")
    
    all_results = []
    with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = [executor.submit(process_sample_on_gpu, arg) for arg in worker_args]
        
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Worker 失败: {e}")
    
    # 统计结果
    logger.info("\n" + "=" * 60)
    logger.info("=== Self-Evolving Agent V5 最终结果 ===")
    
    # 按任务类型统计
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r['question_type']].append(r)
    
    total_score = 0
    total_count = 0
    evolved_count = 0
    scale_calibrated_count = 0
    
    for qtype, results in sorted(by_type.items()):
        scores = [r['score'] for r in results]
        evolved = [r for r in results if r.get('evolved', False)]
        calibrated = [r for r in results if r.get('scale_calibrated', False)]
        
        avg_score = np.mean(scores) if scores else 0
        total_score += sum(scores)
        total_count += len(scores)
        evolved_count += len(evolved)
        scale_calibrated_count += len(calibrated)
        
        logger.info(f"{qtype:35}: {avg_score*100:6.2f}% MRA ({len(results)} 样本, {len(evolved)} 演化)")
    
    overall = total_score / total_count * 100 if total_count > 0 else 0
    logger.info(f"\n{'总体':35}: {overall:6.2f}% MRA ({total_count} 样本)")
    if total_count > 0:
        logger.info(f"演化触发: {evolved_count}/{total_count} ({evolved_count/total_count*100:.1f}%)")
        logger.info(f"尺度校准成功: {scale_calibrated_count}/{total_count} ({scale_calibrated_count/total_count*100:.1f}%)")
    
    # 演化效果分析
    evolved_results = [r for r in all_results if r.get('evolved', False)]
    non_evolved_results = [r for r in all_results if not r.get('evolved', False)]
    
    if evolved_results and non_evolved_results:
        evolved_avg = np.mean([r['score'] for r in evolved_results])
        non_evolved_avg = np.mean([r['score'] for r in non_evolved_results])
        logger.info(f"\n演化样本平均: {evolved_avg*100:.2f}%")
        logger.info(f"非演化样本平均: {non_evolved_avg*100:.2f}%")
        logger.info(f"演化提升: {(evolved_avg - non_evolved_avg)*100:+.2f}%")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"evolving_agent_v5_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存配置
    config = {
        'vl_model': args.vl_model,
        'num_frames': args.num_frames,
        'max_evolution_rounds': args.max_evolution_rounds,
        'num_gpus': args.num_gpus,
        'total_samples': total_count,
        'overall_score': overall,
        'evolved_count': evolved_count,
        'scale_calibrated_count': scale_calibrated_count,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
