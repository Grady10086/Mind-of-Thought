#!/usr/bin/env python3
"""
Self-Evolving Agent V5.1 - MindMap-Centric Evolution

核心设计理念：
1. 所有演化都针对心智地图本身，而不是直接输出答案
2. VL-Critic 通过分析具体检测帧来验证心智地图中的记录
3. VL-Evolver 修正心智地图（合并重复检测、修正计数、修正标签）
4. VL-Reasoner 基于演化后的心智地图给出答案

演化流程：
MindMap → VL-Critic(验证每个entity的检测帧) → VL-Evolver(修正MindMap) → VL-Reasoner(基于MindMap回答)
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
# 扩展词汇表 V5
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
    
    # === 办公/电子设备 ===
    "laptop", "computer", "keyboard", "mouse", "telephone", "phone",
    "printer", "scanner", "copier", "projector", "whiteboard",
    
    # === 厨房用品 ===
    "bowl", "plate", "pan", "pot", "kettle", "toaster", "blender",
    "cutting board", "knife", "fork", "spoon",
    
    # === 建筑元素 ===
    "ceiling", "floor", "wall", "pillar", "column", "beam", "railing",
    "stairs", "step", "ladder", "elevator",
    
    # === 其他 ===
    "backpack", "bag", "suitcase", "counter", "countertop",
    "bicycle", "bike", "scooter", "ball",
    "helmet", "hat", "glasses", "umbrella", "briefcase",
]

# 同义词映射
SYNONYM_MAP_V5 = {
    # 座椅类
    'chair': ['chair', 'seat', 'armchair', 'office chair', 'dining chair'],
    'armchair': ['armchair', 'chair', 'seat'],
    'stool': ['stool', 'bar stool', 'step stool'],
    'sofa': ['sofa', 'couch', 'settee', 'loveseat'],
    'couch': ['couch', 'sofa', 'settee'],
    'bench': ['bench', 'seat'],
    
    # 桌子类
    'table': ['table', 'desk', 'counter', 'countertop'],
    'desk': ['desk', 'table', 'workstation'],
    'counter': ['counter', 'countertop', 'table'],
    
    # 电器类
    'tv': ['tv', 'television', 'monitor', 'screen', 'display'],
    'television': ['television', 'tv', 'monitor', 'screen'],
    'refrigerator': ['refrigerator', 'fridge'],
    'fridge': ['fridge', 'refrigerator'],
    
    # 照明类
    'lamp': ['lamp', 'light', 'lighting'],
    'light': ['light', 'lamp', 'lighting', 'ceiling light'],
    
    # 容器类
    'trash can': ['trash can', 'trash bin', 'garbage bin', 'dustbin', 'wastebasket'],
    'bin': ['bin', 'trash bin', 'container', 'bucket'],
    'box': ['box', 'container', 'crate'],
    
    # 办公类
    'laptop': ['laptop', 'notebook', 'computer'],
    'computer': ['computer', 'laptop', 'pc', 'desktop'],
    'phone': ['phone', 'telephone', 'cellphone', 'mobile'],
    'whiteboard': ['whiteboard', 'white board', 'board'],
    
    # 床类
    'bed': ['bed', 'mattress'],
    'pillow': ['pillow', 'cushion'],
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
    """单次检测记录 - 记录每一帧中的检测"""
    frame_idx: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    depth: float
    position_3d: np.ndarray
    
    def get_center(self) -> Tuple[int, int]:
        """获取检测框中心"""
        return ((self.bbox[0] + self.bbox[2]) // 2, 
                (self.bbox[1] + self.bbox[3]) // 2)
    
    def get_area(self) -> int:
        """获取检测框面积"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class MindMapEntity:
    """心智地图实体 - 记录某类物体的所有检测"""
    label: str
    detections: List[Detection] = field(default_factory=list)
    count: int = 0  # 当前估计的物体数量
    avg_confidence: float = 0.0
    position_3d: np.ndarray = None
    size_3d: np.ndarray = None
    verified: bool = False  # 是否已通过 VL 验证
    evolution_history: List[str] = field(default_factory=list)  # 演化历史
    
    def get_frame_indices(self) -> List[int]:
        """获取该物体出现的所有帧"""
        return sorted(set(d.frame_idx for d in self.detections))
    
    def get_best_frames(self, n: int = 3) -> List[int]:
        """获取置信度最高的 n 帧"""
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
    
    def get_detections_for_frame(self, frame_idx: int) -> List[Detection]:
        """获取某帧中该物体的所有检测"""
        return [d for d in self.detections if d.frame_idx == frame_idx]
    
    def get_max_per_frame_count(self) -> int:
        """获取单帧最大检测数量"""
        if not self.detections:
            return 0
        frame_counts = defaultdict(int)
        for d in self.detections:
            frame_counts[d.frame_idx] += 1
        return max(frame_counts.values())
    
    def to_text(self) -> str:
        """转换为文本描述"""
        text = f"- {self.label}: count={self.count}"
        if self.position_3d is not None:
            pos = self.position_3d
            text += f", position=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m"
        if self.size_3d is not None:
            size = self.size_3d * 100
            text += f", size≈{max(size):.0f}cm"
        text += f", confidence={self.avg_confidence:.2f}"
        text += f", frames={len(self.get_frame_indices())}"
        text += f", verified={self.verified}"
        return text
    
    def to_detailed_text(self) -> str:
        """详细文本描述，包括每帧检测信息"""
        lines = [self.to_text()]
        frame_indices = self.get_frame_indices()
        lines.append(f"  Appears in frames: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")
        lines.append(f"  Max detections in single frame: {self.get_max_per_frame_count()}")
        return "\n".join(lines)


@dataclass
class EntityVerification:
    """单个 Entity 的验证结果"""
    label: str
    is_valid: bool
    issue_type: str  # 'none', 'over_counting', 'under_counting', 'wrong_label', 'duplicate_detection'
    suggested_count: int
    reasoning: str
    frames_checked: List[int]


@dataclass 
class VLCriticFeedback:
    """VL Critic 反馈 - 针对心智地图的验证结果"""
    needs_evolution: bool
    entity_verifications: Dict[str, EntityVerification]  # 每个 entity 的验证结果
    overall_confidence: float
    summary: str


# ============================================================================
# 视频路径和数据加载
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


def load_vsibench_data(max_samples: int = None) -> List[Dict]:
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
        
        if not video_path:
            continue
        
        samples.append({
            'scene_name': scene_name,
            'video': scene_name,
            'video_path': video_path,
            'question': item['question'],
            'question_type': item['question_type'],
            'options': item.get('options', None),
            'answer': item['ground_truth'],  # 字段名是 ground_truth
        })
    
    logger.info(f"加载了 {len(samples)} 个有效样本")
    return samples


# 任务类型定义
NUMERICAL_TASKS = {
    'object_counting', 
    'object_size_estimation', 
    'room_size_estimation',
    'object_abs_distance'
}


# ============================================================================
# 1. MindMap Builder - 构建心智地图（带详细帧记录）
# ============================================================================

class MindMapBuilderV5:
    """心智地图构建器 - 记录每帧的详细检测信息"""
    
    # 参考物体尺寸（米）
    REFERENCE_SIZES = {
        'door': {'height': 2.0, 'width': 0.9},
        'chair': {'height': 0.85, 'width': 0.5},
        'bed': {'length': 2.0, 'width': 1.5},
        'table': {'height': 0.75, 'width': 1.0},
        'desk': {'height': 0.75, 'width': 1.2},
        'sofa': {'height': 0.85, 'width': 1.8},
        'refrigerator': {'height': 1.7, 'width': 0.6},
        'toilet': {'height': 0.4, 'width': 0.4},
        'tv': {'height': 0.5, 'width': 0.9},
        'window': {'height': 1.2, 'width': 1.0},
    }
    
    def __init__(
        self,
        device: str = 'cuda',
        num_frames: int = 32,
        focal_length: float = 500.0
    ):
        self.device = device
        self.num_frames = num_frames
        self.focal_length = focal_length
        
        self._labeler = None
        self._depth_estimator = None
        
        self.scale_factor = 1.0
        self.scale_calibrated = False
    
    def load_models(self):
        """加载检测模型"""
        if self._labeler is None:
            from core.semantic_labeler import GroundingDINOLabeler
            logger.info("加载 GroundingDINO 检测器...")
            self._labeler = GroundingDINOLabeler(
                model_id="IDEA-Research/grounding-dino-base",
                device=self.device,
            )
            self._labeler.load_model()
        
        if self._depth_estimator is None:
            from core.perception import DepthEstimator
            self._depth_estimator = DepthEstimator(
                model_name="depth-anything/DA3-Large",
                device=self.device,
                half_precision=True,
            )
            logger.info("DA3 深度估计器加载完成")
    
    def unload_models(self):
        """卸载模型"""
        if self._labeler is not None:
            del self._labeler
            self._labeler = None
        if self._depth_estimator is not None:
            del self._depth_estimator
            self._depth_estimator = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def _calibrate_scale(
        self, 
        all_detections: Dict[str, List[Detection]], 
        depth_maps: List[np.ndarray],
        frame_shapes: List[Tuple[int, int]]
    ) -> float:
        """动态尺度校准"""
        scale_factors = []
        
        for label, dets in all_detections.items():
            label_lower = label.lower()
            
            for ref_name, ref_dims in self.REFERENCE_SIZES.items():
                if ref_name in label_lower:
                    for det in dets:
                        x1, y1, x2, y2 = det.bbox
                        pixel_height = y2 - y1
                        pixel_width = x2 - x1
                        
                        estimated_height = pixel_height * det.depth / self.focal_length
                        estimated_width = pixel_width * det.depth / self.focal_length
                        
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
            return 1.0
    
    def _aggregate_detections(self, label: str, dets: List[Detection]) -> MindMapEntity:
        """聚合检测结果，计算物体数量"""
        if not dets:
            return MindMapEntity(label=label, count=0)
        
        # 按帧分组
        frame_detections = defaultdict(list)
        for d in dets:
            frame_detections[d.frame_idx].append(d)
        
        # 使用单帧最大检测数作为估计数量
        max_per_frame = max(len(frame_dets) for frame_dets in frame_detections.values())
        
        # 计算平均位置和置信度
        positions = [d.position_3d for d in dets if d.position_3d is not None]
        avg_pos = np.mean(positions, axis=0) * self.scale_factor if positions else None
        avg_conf = np.mean([d.confidence for d in dets])
        
        # 估计尺寸
        bboxes = [d.bbox for d in dets]
        depths = [d.depth for d in dets]
        if bboxes and depths:
            avg_w = np.mean([b[2] - b[0] for b in bboxes])
            avg_h = np.mean([b[3] - b[1] for b in bboxes])
            avg_depth = np.mean(depths)
            size_x = avg_w * avg_depth / self.focal_length * self.scale_factor
            size_y = avg_h * avg_depth / self.focal_length * self.scale_factor
            size_3d = np.array([size_x, size_y, 0.1])
        else:
            size_3d = None
        
        return MindMapEntity(
            label=label,
            detections=dets,
            count=max_per_frame,
            avg_confidence=float(avg_conf),
            position_3d=avg_pos,
            size_3d=size_3d,
            verified=False,
        )
    
    def build_from_video(self, video_path: str) -> Tuple[Dict[str, MindMapEntity], int, str]:
        """从视频构建心智地图"""
        self.load_models()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, 0, video_path
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        # 核心词汇表（避免超过 GroundingDINO 限制）
        core_vocabulary = [
            "chair", "table", "sofa", "couch", "bed", "desk", "shelf", "cabinet",
            "drawer", "closet", "wardrobe", "dresser", "nightstand", "bedside table",
            "stool", "bench", "ottoman", "armchair",
            "tv", "television", "monitor", "screen", "refrigerator", "fridge",
            "microwave", "stove", "oven", "washer", "dryer", "dishwasher", "fan",
            "lamp", "light", "ceiling light", "heater", "radiator", "fireplace",
            "toilet", "sink", "bathtub", "shower", "towel", "mirror",
            "trash can", "trash bin", "bucket", "basket", "box", "container",
            "cup", "bottle", "vase", "clock", "book", "pillow", "cushion",
            "blanket", "rug", "carpet", "curtain",
            "picture", "painting", "poster", "plant", "flower",
            "door", "window",
            "laptop", "computer", "keyboard", "mouse", "telephone", "phone",
            "printer", "whiteboard", "projector",
            "bowl", "plate", "pan", "pot", "kettle",
            "ceiling", "floor", "wall", "pillar", "column", "stairs",
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
        
        return mind_map, total_frames, video_path


# ============================================================================
# 2. VL-Critic - 针对心智地图实体的验证
# ============================================================================

class VLCriticMindMapCentric:
    """
    VL-Critic: 通过分析具体检测帧来验证心智地图中的记录
    
    核心职责:
    - 对于每个待验证的 entity，提取其检测帧
    - VL 观看这些具体帧，判断检测是否正确
    - 识别: 重复计数、遗漏、错标等问题
    """
    
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
    
    def verify_entity(
        self,
        video_path: str,
        entity: MindMapEntity,
        total_frames: int
    ) -> EntityVerification:
        """
        验证单个 entity 的检测是否正确
        
        方法: 提取该 entity 检测到的帧，让 VL 观看并判断
        """
        self.load_model()
        
        # 获取该 entity 检测最好的几帧
        best_frames = entity.get_best_frames(n=4)
        max_per_frame = entity.get_max_per_frame_count()
        
        if not best_frames:
            return EntityVerification(
                label=entity.label,
                is_valid=False,
                issue_type='no_detection',
                suggested_count=0,
                reasoning="No frames with valid detection",
                frames_checked=[]
            )
        
        # 提取这些帧作为图片
        cap = cv2.VideoCapture(video_path)
        images = []
        for frame_idx in best_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 在帧上标注检测框
                dets = entity.get_detections_for_frame(frame_idx)
                for det in dets:
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_rgb, f"{entity.label}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                images.append(Image.fromarray(frame_rgb))
        cap.release()
        
        if not images:
            return EntityVerification(
                label=entity.label,
                is_valid=True,
                issue_type='none',
                suggested_count=entity.count,
                reasoning="Could not extract frames",
                frames_checked=best_frames
            )
        
        # 构建验证 prompt
        frame_info = ", ".join([f"frame {f}" for f in best_frames])
        prompt = f"""I'm verifying the detection of "{entity.label}" objects in a 3D indoor scene.

=== DETECTION INFO ===
Object: {entity.label}
Current count in MindMap: {entity.count}
Max detected in single frame: {max_per_frame}
These images show frames where "{entity.label}" was detected (blue boxes mark detections).

=== YOUR TASK ===
Look at the images carefully and answer:

1. Are the blue detection boxes correctly identifying "{entity.label}" objects?
2. Is the count of {entity.count} reasonable? Or are some detections:
   - Duplicate (same object detected multiple times)?
   - Incorrect (wrong object labeled)?
   - Missing (more objects exist but not detected)?

Respond in this EXACT format:
IS_VALID: [YES/NO]
ISSUE_TYPE: [none/over_counting/under_counting/wrong_label/duplicate_detection]
SUGGESTED_COUNT: [your count, a number]
REASONING: [brief explanation, why is the count correct or incorrect?]
"""
        
        # 构建多图消息
        content = []
        for i, img in enumerate(images):
            content.append({
                "type": "image",
                "image": img,
            })
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
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
                generated_ids = self.model.generate(**inputs, max_new_tokens=200)
            
            response = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            return self._parse_verification(response, entity.label, entity.count, best_frames)
            
        except Exception as e:
            logger.warning(f"Entity 验证失败 ({entity.label}): {e}")
            return EntityVerification(
                label=entity.label,
                is_valid=True,  # 默认信任检测结果
                issue_type='none',
                suggested_count=entity.count,
                reasoning=f"Verification failed: {str(e)}",
                frames_checked=best_frames
            )
    
    def _parse_verification(
        self, 
        response: str, 
        label: str, 
        original_count: int,
        frames_checked: List[int]
    ) -> EntityVerification:
        """解析验证结果"""
        # 解析 IS_VALID
        is_valid = True
        valid_match = re.search(r'IS_VALID:\s*(YES|NO)', response, re.IGNORECASE)
        if valid_match:
            is_valid = valid_match.group(1).upper() == 'YES'
        
        # 解析 ISSUE_TYPE
        issue_type = 'none'
        issue_match = re.search(r'ISSUE_TYPE:\s*(\w+(?:_\w+)*)', response, re.IGNORECASE)
        if issue_match:
            issue_type = issue_match.group(1).lower()
        
        # 解析 SUGGESTED_COUNT
        suggested_count = original_count
        count_match = re.search(r'SUGGESTED_COUNT:\s*(\d+)', response, re.IGNORECASE)
        if count_match:
            suggested_count = int(count_match.group(1))
        
        # 解析 REASONING
        reasoning = ""
        reason_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()
        
        return EntityVerification(
            label=label,
            is_valid=is_valid,
            issue_type=issue_type,
            suggested_count=suggested_count,
            reasoning=reasoning,
            frames_checked=frames_checked
        )
    
    def evaluate_mindmap(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        question: str,
        question_type: str,
        total_frames: int
    ) -> VLCriticFeedback:
        """
        评估整个心智地图
        
        对于 counting 任务，重点验证目标物体
        """
        entity_verifications = {}
        needs_evolution = False
        
        # 从问题中提取目标物体（如果是 counting 任务）
        target_objects = []
        if question_type == 'object_counting':
            match = re.search(r'[Hh]ow many (\w+)', question)
            if match:
                target_objects.append(match.group(1).lower())
        
        # 验证相关的 entities
        for label, entity in mind_map.items():
            # 对于 counting 任务，只验证目标物体
            if target_objects:
                is_target = any(match_object_name_v5(t, label) for t in target_objects)
                if not is_target:
                    continue
            
            # 只验证有检测结果的 entity
            if entity.count == 0:
                continue
            
            verification = self.verify_entity(video_path, entity, total_frames)
            entity_verifications[label] = verification
            
            if not verification.is_valid:
                needs_evolution = True
        
        # 计算整体置信度
        if entity_verifications:
            valid_count = sum(1 for v in entity_verifications.values() if v.is_valid)
            overall_conf = valid_count / len(entity_verifications)
        else:
            overall_conf = 0.5
        
        # 生成摘要
        issues = []
        for label, v in entity_verifications.items():
            if not v.is_valid:
                issues.append(f"{label}: {v.issue_type} (suggested: {v.suggested_count})")
        
        summary = "All verifications passed." if not issues else f"Issues found: {'; '.join(issues)}"
        
        return VLCriticFeedback(
            needs_evolution=needs_evolution,
            entity_verifications=entity_verifications,
            overall_confidence=overall_conf,
            summary=summary
        )


# ============================================================================
# 3. VL-Evolver - 针对心智地图的演化修正
# ============================================================================

class VLEvolverMindMapCentric:
    """
    VL-Evolver: 根据 Critic 反馈修正心智地图
    
    核心职责:
    - 根据验证结果修正 entity 的 count
    - 合并重复检测
    - 修正错误标签
    """
    
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
    
    def evolve_entity(
        self,
        video_path: str,
        entity: MindMapEntity,
        verification: EntityVerification,
        total_frames: int
    ) -> MindMapEntity:
        """
        演化单个 entity
        
        方法: 如果 Critic 发现问题，让 VL 重新观看更多帧进行精确计数
        """
        if verification.is_valid:
            # 验证通过，只标记为已验证
            entity.verified = True
            return entity
        
        self.load_model()
        
        # 获取更多帧进行重新计数
        all_frames = entity.get_frame_indices()
        # 均匀采样更多帧
        if len(all_frames) > 8:
            sample_indices = np.linspace(0, len(all_frames)-1, 8, dtype=int)
            check_frames = [all_frames[i] for i in sample_indices]
        else:
            check_frames = all_frames
        
        # 提取帧
        cap = cv2.VideoCapture(video_path)
        images = []
        for frame_idx in check_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 标注现有检测
                dets = entity.get_detections_for_frame(frame_idx)
                for det in dets:
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                images.append(Image.fromarray(frame_rgb))
        cap.release()
        
        if not images:
            return entity
        
        # 构建演化 prompt
        prompt = f"""The detection of "{entity.label}" needs correction.

=== CURRENT STATUS ===
Object: {entity.label}
Current count: {entity.count}
Issue identified: {verification.issue_type}
Critic reasoning: {verification.reasoning}
Suggested count by critic: {verification.suggested_count}

=== YOUR TASK ===
Look at these {len(images)} frames showing the scene.
Blue boxes show where "{entity.label}" was originally detected.

Please carefully count the ACTUAL number of distinct "{entity.label}" objects.
Consider:
- Same object appearing in multiple frames should only be counted ONCE
- If the same object is detected multiple times in different frames, it's still ONE object
- Only count DISTINCT physical objects

Answer in this format:
CORRECTED_COUNT: [number]
CONFIDENCE: [0-100]%
REASONING: [explain your count]
"""
        
        # 构建消息
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
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
                generated_ids = self.model.generate(**inputs, max_new_tokens=150)
            
            response = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            # 解析结果
            count_match = re.search(r'CORRECTED_COUNT:\s*(\d+)', response, re.IGNORECASE)
            if count_match:
                new_count = int(count_match.group(1))
                old_count = entity.count
                entity.count = new_count
                entity.verified = True
                entity.evolution_history.append(
                    f"Evolved from {old_count} to {new_count}: {verification.issue_type}"
                )
                logger.info(f"Entity '{entity.label}' evolved: {old_count} -> {new_count}")
            
            return entity
            
        except Exception as e:
            logger.warning(f"Entity 演化失败 ({entity.label}): {e}")
            # 使用 Critic 建议的计数
            entity.count = verification.suggested_count
            entity.verified = True
            entity.evolution_history.append(f"Used critic suggestion: {verification.suggested_count}")
            return entity
    
    def evolve_mindmap(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        critic_feedback: VLCriticFeedback,
        total_frames: int
    ) -> Dict[str, MindMapEntity]:
        """演化整个心智地图"""
        evolved_map = {}
        
        for label, entity in mind_map.items():
            if label in critic_feedback.entity_verifications:
                verification = critic_feedback.entity_verifications[label]
                evolved_entity = self.evolve_entity(
                    video_path, entity, verification, total_frames
                )
                evolved_map[label] = evolved_entity
            else:
                # 没有验证的 entity 保持不变
                evolved_map[label] = entity
        
        return evolved_map


# ============================================================================
# 4. VL-Reasoner - 基于心智地图的推理
# ============================================================================

class VLReasonerMindMapCentric:
    """
    VL-Reasoner: 基于演化后的心智地图回答问题
    
    核心原则:
    - 对于 counting: 直接使用心智地图中的 count
    - 对于其他任务: 结合心智地图和视觉观察
    """
    
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
        基于心智地图回答问题
        
        对于 counting 任务: 直接从心智地图获取 count
        """
        # 对于 counting 任务，直接从心智地图获取答案
        if question_type == 'object_counting':
            # 提取目标物体
            match = re.search(r'[Hh]ow many (\w+)', question)
            if match:
                target_obj = match.group(1).lower()
                
                # 从心智地图中找匹配的计数
                best_count = 0
                best_label = ""
                best_conf = 0.0
                
                for label, entity in mind_map.items():
                    if match_object_name_v5(target_obj, label):
                        if entity.count > best_count or (entity.count == best_count and entity.avg_confidence > best_conf):
                            best_count = entity.count
                            best_label = label
                            best_conf = entity.avg_confidence
                
                if best_count > 0:
                    reasoning = f"From MindMap: {best_label} has count={best_count} (verified={mind_map[best_label].verified})"
                    return str(best_count), best_conf, reasoning
                else:
                    # 目标物体未检测到，使用 VL 观察
                    return self._reason_with_vl(video_path, mind_map, question, question_type, options, scale_calibrated)
        
        # 其他任务使用 VL 推理
        return self._reason_with_vl(video_path, mind_map, question, question_type, options, scale_calibrated)
    
    def _reason_with_vl(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
        question: str,
        question_type: str,
        options: List[str] = None,
        scale_calibrated: bool = True
    ) -> Tuple[str, float, str]:
        """使用 VL 模型进行推理"""
        self.load_model()
        
        # 构建选项文本
        options_text = ""
        if options:
            options_text = "\nOptions:\n" + "\n".join(options)
        
        # 答案格式
        if question_type in NUMERICAL_TASKS:
            answer_format = "Your answer must be a NUMBER only."
        else:
            answer_format = "Your answer must be the OPTION LETTER only (A, B, C, or D)."
        
        # 尺度校准提示
        scale_note = ""
        if not scale_calibrated:
            scale_note = "\nNote: Scale calibration was not successful. Distance/size values may be inaccurate."
        
        prompt = f"""You are analyzing a 3D indoor scene. Use the MindMap information below to answer.

=== MIND MAP (Verified Object Information) ===
{self._format_mind_map(mind_map)}
{scale_note}

=== QUESTION ===
{question}{options_text}

=== INSTRUCTIONS ===
1. The MindMap above contains verified object counts and positions
2. Use this information as your PRIMARY reference
3. Watch the video only to supplement if needed

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
# 5. Self-Evolving Agent V5.1 - MindMap Centric
# ============================================================================

class SelfEvolvingAgentV5MindMapCentric:
    """
    MindMap-Centric Self-Evolving Agent
    
    流程:
    1. 构建心智地图（记录每帧检测）
    2. VL-Critic 验证心智地图中的实体（分析具体帧）
    3. VL-Evolver 修正心智地图（如果需要）
    4. VL-Reasoner 基于心智地图回答问题
    """
    
    def __init__(
        self,
        vl_model_name: str = "/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct",
        device: str = 'cuda',
        max_evolution_rounds: int = 2,
    ):
        self.device = device
        self.max_evolution_rounds = max_evolution_rounds
        
        # 组件
        self.mind_map_builder = MindMapBuilderV5(device=device)
        self.critic = VLCriticMindMapCentric(vl_model_name, device)
        self.evolver = VLEvolverMindMapCentric(vl_model_name, device)
        self.reasoner = VLReasonerMindMapCentric(vl_model_name, device)
    
    def unload_all(self):
        """卸载所有模型"""
        self.mind_map_builder.unload_models()
        self.critic.unload()
        self.evolver.unload()
        self.reasoner.unload()
    
    def process(
        self,
        video_path: str,
        question: str,
        question_type: str,
        options: List[str] = None
    ) -> Dict[str, Any]:
        """
        处理单个问题
        """
        result = {
            'question': question,
            'question_type': question_type,
            'evolution_rounds': 0,
            'evolved': False,
        }
        
        # 1. 构建心智地图
        logger.info(f"构建心智地图: {video_path}")
        mind_map, total_frames, _ = self.mind_map_builder.build_from_video(video_path)
        
        result['initial_mind_map'] = {k: v.to_text() for k, v in mind_map.items()}
        result['scale_calibrated'] = self.mind_map_builder.scale_calibrated
        
        if not mind_map:
            logger.warning("心智地图为空")
            result['answer'] = "0" if question_type in NUMERICAL_TASKS else "A"
            result['confidence'] = 0.1
            return result
        
        # 2. VL-Critic 验证心智地图
        logger.info("VL-Critic 验证心智地图...")
        critic_feedback = self.critic.evaluate_mindmap(
            video_path, mind_map, question, question_type, total_frames
        )
        
        result['critic_summary'] = critic_feedback.summary
        result['needs_evolution'] = critic_feedback.needs_evolution
        
        # 3. 如果需要演化，执行演化
        evolved_map = mind_map
        if critic_feedback.needs_evolution:
            for round_idx in range(self.max_evolution_rounds):
                logger.info(f"演化轮次 {round_idx + 1}...")
                
                evolved_map = self.evolver.evolve_mindmap(
                    video_path, evolved_map, critic_feedback, total_frames
                )
                
                result['evolution_rounds'] = round_idx + 1
                result['evolved'] = True
                
                # 重新验证
                critic_feedback = self.critic.evaluate_mindmap(
                    video_path, evolved_map, question, question_type, total_frames
                )
                
                if not critic_feedback.needs_evolution:
                    break
        
        result['final_mind_map'] = {k: v.to_text() for k, v in evolved_map.items()}
        
        # 4. VL-Reasoner 基于演化后的心智地图回答
        logger.info("VL-Reasoner 推理答案...")
        answer, confidence, reasoning = self.reasoner.reason(
            video_path, evolved_map, question, question_type, options,
            self.mind_map_builder.scale_calibrated
        )
        
        result['answer'] = answer
        result['confidence'] = confidence
        result['reasoning'] = reasoning
        
        return result


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_answer(pred: str, gt: str, question_type: str) -> float:
    """评估答案"""
    if question_type in NUMERICAL_TASKS:
        try:
            pred_val = float(pred)
            gt_val = float(gt)
            if gt_val == 0:
                return 1.0 if pred_val == 0 else 0.0
            error = abs(pred_val - gt_val) / gt_val
            return max(0, 1 - error)
        except:
            return 0.0
    else:
        return 1.0 if pred.upper().strip() == gt.upper().strip() else 0.0


def main():
    parser = argparse.ArgumentParser(description='Self-Evolving Agent V5.1 - MindMap Centric')
    parser.add_argument('--vl-model', type=str, 
                       default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-evolution-rounds', type=int, default=2)
    parser.add_argument('--question-types', type=str, nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    # 加载数据
    data = load_vsibench_data(max_samples=None)  # 先加载全部，后面再过滤
    logger.info(f"加载 {len(data)} 条数据")
    
    # 过滤任务类型
    if args.question_types:
        data = [d for d in data if d.get('question_type') in args.question_types]
        logger.info(f"过滤后: {len(data)} 条数据")
    
    # 限制样本数
    if args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"限制为: {len(data)} 条数据")
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'evolving_agent_v5.1_mindmap_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建 Agent
    agent = SelfEvolvingAgentV5MindMapCentric(
        vl_model_name=args.vl_model,
        device='cuda',
        max_evolution_rounds=args.max_evolution_rounds,
    )
    
    # 运行评估
    results = []
    scores_by_type = defaultdict(list)
    evolution_stats = {'triggered': 0, 'total': 0}
    
    for item in tqdm(data, desc="Processing"):
        scene_name = item.get('video', item.get('scene_name', ''))
        video_path = find_video_path(scene_name)
        
        if not video_path:
            logger.warning(f"视频未找到: {scene_name}")
            continue
        
        question = item['question']
        question_type = item.get('question_type', 'unknown')
        options = item.get('options', None)
        gt = str(item['answer'])
        
        try:
            result = agent.process(video_path, question, question_type, options)
            
            pred = result['answer']
            score = evaluate_answer(pred, gt, question_type)
            
            result['ground_truth'] = gt
            result['score'] = score
            result['scene'] = scene_name
            
            results.append(result)
            scores_by_type[question_type].append(score)
            
            evolution_stats['total'] += 1
            if result.get('evolved', False):
                evolution_stats['triggered'] += 1
            
            logger.info(f"[{question_type}] GT: {gt}, Pred: {pred}, Score: {score:.2f}, Evolved: {result.get('evolved', False)}")
            
        except Exception as e:
            logger.error(f"处理失败 {scene_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 卸载模型
    agent.unload_all()
    
    # 统计结果
    print("\n" + "="*60)
    print("V5.1 MindMap-Centric Results")
    print("="*60)
    
    for qtype, scores in sorted(scores_by_type.items()):
        avg = np.mean(scores)
        print(f"{qtype}: {avg:.4f} ({len(scores)} samples)")
    
    all_scores = [s for scores in scores_by_type.values() for s in scores]
    print(f"\nOverall: {np.mean(all_scores):.4f} ({len(all_scores)} samples)")
    
    evo_rate = evolution_stats['triggered'] / evolution_stats['total'] * 100 if evolution_stats['total'] > 0 else 0
    print(f"\nEvolution Rate: {evo_rate:.1f}% ({evolution_stats['triggered']}/{evolution_stats['total']})")
    
    # 保存结果
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'results_by_type': {k: np.mean(v) for k, v in scores_by_type.items()},
        'overall': np.mean(all_scores),
        'total_samples': len(all_scores),
        'evolution_stats': evolution_stats,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
