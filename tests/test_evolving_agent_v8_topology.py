#!/usr/bin/env python3
"""
Self-Evolving Agent V8 - 2D 拓扑图增强版

核心改进 (相比 V7)：
1. 将心智地图转换为 2D 俯视图拓扑图
2. 拓扑图作为图像输入给 Qwen3-VL，辅助空间推理
3. 特别针对 object_rel_direction 和 route_planning 任务优化
4. 三种推理模式：rule / vl / topo (拓扑图增强)

架构：
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Self-Evolving Agent V8                              │
│                         (2D Topology Enhanced)                              │
│                                                                             │
│  ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐               │
│  │ 感知    │ -> │ Calibrator│ -> │ Evolver  │ -> │ MindMap │               │
│  │DA3+DINO │    │标定物识别  │    │心智地图修正│    │ 校准后   │               │
│  └─────────┘    └───────────┘    └──────────┘    └────┬────┘               │
│                                                       │                     │
│                                    ┌──────────────────┼──────────────────┐  │
│                                    ▼                  ▼                  ▼  │
│                             ┌───────────┐    ┌───────────┐    ┌───────────┐│
│                             │ Rule-based│    │  VL+Video │    │VL+Topology││
│                             │ Reasoning │    │ (V7方式)   │    │ (V8新增)  ││
│                             └─────┬─────┘    └─────┬─────┘    └─────┬─────┘│
│                                   ▼                ▼                ▼      │
│                           rule_prediction   vl_prediction   topo_prediction│
└─────────────────────────────────────────────────────────────────────────────┘

2D 拓扑图特点：
- 俯视图视角 (Bird's Eye View)
- 物体用彩色圆点表示，标注名称
- 高亮问题中涉及的物体
- 显示物体间距离和连线
- 添加相机位置和方向指示
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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import io

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

# V8 新增: 需要拓扑图增强的任务
TOPOLOGY_TASKS = ['object_rel_direction_easy', 'object_rel_direction_medium', 
                  'object_rel_direction_hard', 'object_rel_distance', 'route_planning']

# 物体颜色映射 (用于拓扑图)
OBJECT_COLORS = {
    'chair': (255, 100, 100),    # 红
    'table': (100, 255, 100),    # 绿
    'sofa': (100, 100, 255),     # 蓝
    'couch': (100, 100, 255),    # 蓝
    'bed': (255, 200, 100),      # 橙
    'tv': (200, 100, 255),       # 紫
    'door': (150, 150, 150),     # 灰
    'window': (100, 200, 255),   # 浅蓝
    'toilet': (255, 255, 100),   # 黄
    'sink': (100, 255, 255),     # 青
    'refrigerator': (200, 200, 200),  # 浅灰
    'desk': (150, 255, 150),     # 浅绿
    'lamp': (255, 255, 200),     # 浅黄
    'default': (180, 180, 180),  # 默认灰色
}

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
# V8 新增: 2D 拓扑图生成器
# ============================================================================

class TopologyMapper:
    """将心智地图转换为 2D 俯视图拓扑图"""
    
    def __init__(self, image_size: int = 512, margin: int = 50):
        self.image_size = image_size
        self.margin = margin
        self.effective_size = image_size - 2 * margin
        
    def generate_topology_image(
        self,
        mind_map: Dict[str, 'MindMapEntity'],
        highlight_objects: List[str] = None,
        show_distances: bool = False,
        show_camera: bool = True,
        title: str = "Room Layout (Top View)",
    ) -> Image.Image:
        """生成 2D 俯视图拓扑图"""
        img = Image.new('RGB', (self.image_size, self.image_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        # 收集所有物体位置 (x, z 作为俯视图坐标)
        positions = {}
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                x, y, z = entity.position_3d
                positions[label] = (x, z, entity)
        
        if not positions:
            draw.text((self.image_size // 2 - 50, self.image_size // 2), 
                     "No position data", fill=(128, 128, 128), font=font)
            return img
        
        # 计算坐标范围
        xs = [p[0] for p in positions.values()]
        zs = [p[1] for p in positions.values()]
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(zs), max(zs)
        x_range = max(x_max - x_min, 1)
        z_range = max(z_max - z_min, 1)
        scale = self.effective_size / max(x_range, z_range)
        
        def to_pixel(x, z):
            px = int(self.margin + (x - x_min) * scale)
            pz = int(self.image_size - self.margin - (z - z_min) * scale)
            return max(self.margin, min(self.image_size - self.margin, px)), \
                   max(self.margin, min(self.image_size - self.margin, pz))
        
        # 绘制网格
        for i in range(self.margin, self.image_size - self.margin + 1, 50):
            draw.line([(i, self.margin), (i, self.image_size - self.margin)], fill=(230, 230, 230), width=1)
            draw.line([(self.margin, i), (self.image_size - self.margin, i)], fill=(230, 230, 230), width=1)
        
        # 绘制边框
        draw.rectangle([self.margin, self.margin, self.image_size - self.margin, self.image_size - self.margin],
                      outline=(200, 200, 200), width=2)
        
        # 绘制相机位置
        if show_camera:
            cam_x, cam_z = to_pixel(0, 0)
            cam_size = 12
            draw.polygon([
                (cam_x, cam_z - cam_size),
                (cam_x - cam_size // 2, cam_z + cam_size // 2),
                (cam_x + cam_size // 2, cam_z + cam_size // 2)
            ], fill=(0, 0, 0), outline=(0, 0, 0))
            draw.text((cam_x + 8, cam_z - 5), "Cam", fill=(0, 0, 0), font=font)
        
        # 收集像素位置
        pixel_positions = {}
        for label, (x, z, entity) in positions.items():
            px, pz = to_pixel(x, z)
            pixel_positions[label] = (px, pz)
        
        # 绘制高亮物体间的连线和距离
        if show_distances and highlight_objects and len(highlight_objects) >= 2:
            highlighted = []
            for h in highlight_objects:
                for label, (px, pz) in pixel_positions.items():
                    if match_object_name(h, label):
                        highlighted.append((label, px, pz, positions[label][0], positions[label][1]))
                        break
            
            for i in range(len(highlighted)):
                for j in range(i + 1, len(highlighted)):
                    p1, p2 = highlighted[i], highlighted[j]
                    dist = np.sqrt((p1[3] - p2[3])**2 + (p1[4] - p2[4])**2)
                    draw.line([(p1[1], p1[2]), (p2[1], p2[2])], fill=(255, 100, 100), width=2)
                    mid_x, mid_y = (p1[1] + p2[1]) // 2, (p1[2] + p2[2]) // 2
                    draw.text((mid_x - 15, mid_y - 8), f"{dist:.1f}m", fill=(200, 0, 0), font=font)
        
        # 绘制物体
        for label, (px, pz) in pixel_positions.items():
            color = OBJECT_COLORS.get('default')
            for key in OBJECT_COLORS:
                if key in label.lower():
                    color = OBJECT_COLORS[key]
                    break
            
            is_highlighted = highlight_objects and any(match_object_name(h, label) for h in highlight_objects)
            radius = 12 if is_highlighted else 8
            outline_color = (255, 0, 0) if is_highlighted else (0, 0, 0)
            outline_width = 3 if is_highlighted else 1
            
            draw.ellipse([px - radius, pz - radius, px + radius, pz + radius],
                        fill=color, outline=outline_color, width=outline_width)
            
            short_label = label[:12] + ".." if len(label) > 12 else label
            text_color = (200, 0, 0) if is_highlighted else (50, 50, 50)
            draw.text((px + radius + 2, pz - 5), short_label, fill=text_color, font=font)
        
        # 绘制标题和图例
        draw.text((10, 10), title, fill=(0, 0, 0), font=title_font)
        draw.text((self.image_size - 50, 15), "N ↑", fill=(0, 0, 0), font=font)
        
        # 比例尺
        scale_bar_px = 50
        scale_bar_m = scale_bar_px / scale if scale > 0 else 1
        draw.line([(self.margin, self.image_size - 15), (self.margin + scale_bar_px, self.image_size - 15)],
                 fill=(0, 0, 0), width=2)
        draw.text((self.margin, self.image_size - 30), f"{scale_bar_m:.1f}m", fill=(0, 0, 0), font=font)
        
        return img
    
    def generate_direction_analysis(
        self,
        mind_map: Dict[str, 'MindMapEntity'],
        obj1: str,
        obj2: str,
    ) -> Tuple[Image.Image, str]:
        """生成方向判断专用的拓扑图和分析"""
        ent1, ent2 = None, None
        label1, label2 = "", ""
        
        for label, entity in mind_map.items():
            if match_object_name(obj1, label):
                ent1, label1 = entity, label
            if match_object_name(obj2, label):
                ent2, label2 = entity, label
        
        img = self.generate_topology_image(
            mind_map,
            highlight_objects=[obj1, obj2],
            show_distances=True,
            show_camera=True,
            title=f"{obj1} vs {obj2}"
        )
        
        if not ent1 or not ent2 or ent1.position_3d is None or ent2.position_3d is None:
            return img, f"Position not available for '{obj1}' or '{obj2}'"
        
        diff = ent1.position_3d - ent2.position_3d
        dx, dy, dz = diff[0], diff[1], diff[2]
        
        directions = []
        if abs(dx) > 0.3:
            directions.append("to the RIGHT" if dx > 0 else "to the LEFT")
        if abs(dz) > 0.3:
            directions.append("in FRONT (farther)" if dz > 0 else "BEHIND (closer)")
        if abs(dy) > 0.3:
            directions.append("ABOVE" if dy < 0 else "BELOW")
        
        analysis = f"Spatial Analysis:\n"
        analysis += f"- {obj1} position: ({ent1.position_3d[0]:.2f}, {ent1.position_3d[2]:.2f})m\n"
        analysis += f"- {obj2} position: ({ent2.position_3d[0]:.2f}, {ent2.position_3d[2]:.2f})m\n"
        analysis += f"- Offset: dx={dx:.2f}m, dz={dz:.2f}m\n"
        analysis += f"- Distance: {np.linalg.norm(diff):.2f}m\n"
        if directions:
            analysis += f"- Direction: {obj1} is {', '.join(directions)} of {obj2}"
        
        return img, analysis


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
            depth_map = depth_result[1].cpu().numpy() if isinstance(depth_result, tuple) else depth_result.cpu().numpy()
            if depth_map.shape[0] != h or depth_map.shape[1] != w:
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
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
# 问题回答器 - V8 三推理模式 (规则 + VL + 拓扑图)
# ============================================================================

class DualReasoningQA:
    """V8 三推理模式问答器 - 规则 / VL / 拓扑图增强"""
    
    def __init__(self, vl_model=None, vl_processor=None, device='cuda'):
        self.vl_model = vl_model
        self.vl_processor = vl_processor
        self.device = device
        self.topology_mapper = TopologyMapper(image_size=512)  # V8 新增
    
    def load_vl_model(self, model_path: str):
        """加载 Qwen3-VL 模型"""
        if self.vl_model is None:
            # 检测是否为 Qwen3-VL
            if 'qwen3' in model_path.lower() or 'Qwen3' in model_path:
                try:
                    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
                    logger.info(f"Loading Qwen3-VL model: {model_path}")
                    
                    self.vl_processor = AutoProcessor.from_pretrained(
                        model_path, 
                        trust_remote_code=True
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
        response_clean = response.split('[')[0].strip()
        
        choice_match = re.search(r'^([A-D])', response_clean.upper())
        if choice_match:
            return choice_match.group(1)
        
        for line in response.split('\n')[::-1]:
            line = line.strip()
            if line and line[0].upper() in 'ABCD':
                return line[0].upper()
        
        response_lower = response.lower()
        for i, opt in enumerate(options):
            opt_content = opt.lower()
            if len(opt) >= 3 and opt[1] in '.、':
                opt_content = opt[3:].strip().lower()
            if opt_content in response_lower:
                return chr(65 + i)
        
        return options[0][0] if options else "A"
    
    # ========================================================================
    # V8 新增: 拓扑图增强推理方法
    # ========================================================================
    
    def topo_answer_direction(
        self,
        question: str,
        options: List[str],
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
    ) -> Tuple[str, str]:
        """V8 拓扑图增强 - 方向判断"""
        if self.vl_model is None:
            return options[0][0] if options else "A", "VL model not loaded"
        
        # 从问题中提取物体
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
            return options[0][0] if options else "A", "Could not extract objects"
        
        # 生成拓扑图和分析
        topo_img, direction_analysis = self.topology_mapper.generate_direction_analysis(mind_map, obj1, obj2)
        
        options_text = "\n".join(options)
        
        prompt = f"""Look at the TOP-DOWN VIEW (bird's eye view) of a room layout.

=== TOPOLOGY IMAGE ===
The image shows objects from above. Red highlighted circles are the objects in question.
- Camera position is marked as black triangle
- Left/Right in image = Left/Right from camera view  
- Top of image = farther from camera, Bottom = closer

=== SPATIAL ANALYSIS ===
{direction_analysis}

=== QUESTION ===
{question}

=== OPTIONS ===
{options_text}

Based on the topology image and spatial analysis, answer with ONLY the letter (A/B/C/D).
Answer:"""

        response, raw = self._call_vl_model_with_topology(prompt, video_path, topo_img)
        answer = self._extract_choice(response, options)
        return answer, f"Topo: {raw}\n{direction_analysis}"
    
    def topo_answer_rel_distance(
        self,
        question: str,
        options: List[str],
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
    ) -> Tuple[str, str]:
        """V8 拓扑图增强 - 相对距离比较"""
        if self.vl_model is None:
            return options[0][0] if options else "A", "VL model not loaded"
        
        # 提取所有可能的物体
        all_objects = set()
        text = question + " " + " ".join(options)
        for vocab in EXTENDED_VOCABULARY:
            if vocab in text.lower():
                all_objects.add(vocab)
        
        topo_img = self.topology_mapper.generate_topology_image(
            mind_map,
            highlight_objects=list(all_objects)[:6],
            show_distances=True,
            show_camera=True,
            title="Distance Comparison"
        )
        
        # 计算距离信息
        dist_info = "Measured distances:\n"
        for label1, ent1 in mind_map.items():
            for label2, ent2 in mind_map.items():
                if label1 < label2 and ent1.position_3d is not None and ent2.position_3d is not None:
                    if any(match_object_name(o, label1) for o in all_objects) and \
                       any(match_object_name(o, label2) for o in all_objects):
                        dist = np.linalg.norm(ent1.position_3d - ent2.position_3d)
                        dist_info += f"  {label1} <-> {label2}: {dist:.2f}m\n"
        
        options_text = "\n".join(options)
        
        prompt = f"""Look at the TOP-DOWN VIEW of a room showing object positions.

=== DISTANCE MEASUREMENTS ===
{dist_info}

=== QUESTION ===
{question}

=== OPTIONS ===
{options_text}

Compare the distances and answer with ONLY the letter (A/B/C/D).
Answer:"""

        response, raw = self._call_vl_model_with_topology(prompt, video_path, topo_img)
        answer = self._extract_choice(response, options)
        return answer, f"Topo: {raw}\n{dist_info}"
    
    def topo_answer_route(
        self,
        question: str,
        options: List[str],
        video_path: str,
        mind_map: Dict[str, MindMapEntity],
    ) -> Tuple[str, str]:
        """V8 拓扑图增强 - 路线规划"""
        if self.vl_model is None:
            return options[0][0] if options else "A", "VL model not loaded"
        
        # 提取物体
        all_objects = set()
        text = question + " " + " ".join(options)
        for vocab in EXTENDED_VOCABULARY:
            if vocab in text.lower():
                all_objects.add(vocab)
        
        topo_img = self.topology_mapper.generate_topology_image(
            mind_map,
            highlight_objects=list(all_objects)[:8],
            show_distances=True,
            show_camera=True,
            title="Route Planning"
        )
        
        # 计算位置和建议路线
        positions = {}
        for obj in all_objects:
            for label, ent in mind_map.items():
                if match_object_name(obj, label) and ent.position_3d is not None:
                    positions[obj] = ent.position_3d
                    break
        
        route_info = "Object positions from camera:\n"
        for obj, pos in sorted(positions.items(), key=lambda x: np.linalg.norm(x[1])):
            dist_from_cam = np.linalg.norm(pos)
            route_info += f"  {obj}: {dist_from_cam:.2f}m from camera\n"
        
        options_text = "\n".join(options)
        
        prompt = f"""Look at the TOP-DOWN VIEW showing room layout for route planning.

=== OBJECT DISTANCES FROM CAMERA ===
{route_info}

=== QUESTION ===
{question}

=== OPTIONS ===
{options_text}

Consider efficient routing (minimize backtracking). Answer with ONLY the letter (A/B/C/D).
Answer:"""

        response, raw = self._call_vl_model_with_topology(prompt, video_path, topo_img)
        answer = self._extract_choice(response, options)
        return answer, f"Topo: {raw}\n{route_info}"
    
    def _call_vl_model_with_topology(self, prompt: str, video_path: str, topo_image: Image.Image) -> Tuple[str, str]:
        """调用 VL 模型 (视频 + 拓扑图双输入)"""
        try:
            from qwen_vl_utils import process_vision_info
            
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "nframes": 4,  # 减少帧数，为拓扑图留空间
                    },
                    {
                        "type": "image",
                        "image": topo_image,
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
            logger.warning(f"VL with topology failed: {e}")
            return "", str(e)


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], vl_model_path: str, result_queue: mp.Queue):
    """GPU Worker 进程"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 创建组件
    builder = MindMapBuilder(device='cuda', num_frames=32, box_threshold=0.25)
    calibrator = ScaleCalibrator()
    evolver = MindMapEvolver(device='cuda')
    qa = DualReasoningQA(device='cuda')
    
    # 加载 VL 模型
    vl_loaded = False
    try:
        qa.load_vl_model(vl_model_path)
        vl_loaded = True
        logger.info(f"GPU {gpu_id}: VL model loaded successfully")
    except Exception as e:
        logger.warning(f"GPU {gpu_id}: Failed to load VL model: {e}")
    
    results = []
    
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
            
            # 4. V8 三推理模式 - 规则 / VL / 拓扑图增强
            rule_pred, rule_reasoning = "", ""
            vl_pred, vl_reasoning = "", ""
            topo_pred, topo_reasoning = "", ""
            topo_used = False
            
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
            
            elif question_type in TOPOLOGY_TASKS:
                # V8 新增: 使用拓扑图增强
                rule_pred, rule_reasoning = qa.rule_answer_choice(mind_map, question, options, question_type)
                if vl_loaded:
                    vl_pred, vl_reasoning = qa.vl_answer_choice(question, options, video_path, mind_map, question_type, total_frames)
                    # 拓扑图增强推理
                    if 'direction' in question_type:
                        topo_pred, topo_reasoning = qa.topo_answer_direction(question, options, video_path, mind_map)
                        topo_used = True
                    elif 'rel_distance' in question_type:
                        topo_pred, topo_reasoning = qa.topo_answer_rel_distance(question, options, video_path, mind_map)
                        topo_used = True
                    elif 'route' in question_type:
                        topo_pred, topo_reasoning = qa.topo_answer_route(question, options, video_path, mind_map)
                        topo_used = True
                
            else:
                rule_pred, rule_reasoning = qa.rule_answer_choice(mind_map, question, options, question_type)
                if vl_loaded:
                    vl_pred, vl_reasoning = qa.vl_answer_choice(question, options, video_path, mind_map, question_type, total_frames)
            
            # 5. 评估三种结果
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
            topo_score = evaluate_prediction(topo_pred, gt, is_numerical) if topo_pred else 0.0
            
            # 选择最好的结果
            scores = [('rule', rule_score, rule_pred), ('vl', vl_score, vl_pred)]
            if topo_used:
                scores.append(('topo', topo_score, topo_pred))
            
            best = max(scores, key=lambda x: x[1])
            final_method, final_score, final_pred = best
            
            results.append({
                'scene_name': sample['scene_name'],
                'question': question,
                'question_type': question_type,
                'ground_truth': gt,
                'options': options,
                # V8 三推理结果
                'rule_prediction': rule_pred,
                'rule_reasoning': rule_reasoning,
                'rule_score': rule_score,
                'vl_prediction': vl_pred,
                'vl_reasoning': vl_reasoning,
                'vl_score': vl_score,
                'topo_prediction': topo_pred,
                'topo_reasoning': topo_reasoning,
                'topo_score': topo_score,
                'topo_used': topo_used,
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
    """加载 VSI-Bench 数据集"""
    from datasets import load_dataset
    
    logger.info("加载 VSI-Bench 数据集...")
    ds = load_dataset(
        'nyu-visionx/VSI-Bench',
        split='test',
        cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench'
    )
    
    samples = []
    for item in ds:
        scene_name = item['scene_name']
        video_path = find_video_path(scene_name)
        
        if not video_path:
            continue
        
        samples.append({
            'scene_name': scene_name,
            'video_path': video_path,
            'question': item['question'],
            'question_type': item['question_type'],
            'options': item.get('options', []),
            'ground_truth': item['ground_truth'],
            'meta_info': item.get('meta_info', {}),
        })
    
    logger.info(f"加载了 {len(samples)} 个有效样本")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Self-Evolving Agent V8 - 2D Topology Enhanced')
    parser.add_argument('--vl-model', type=str, 
                       default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct',
                       help='VL model path (default: Qwen3-VL-8B-Instruct)')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--question-types', type=str, nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
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
    result_queue = mp.Queue()
    processes = []
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_samples[gpu_id], args.vl_model, result_queue)
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
    
    # 统计结果 - V8 三模式
    type_stats = defaultdict(lambda: {
        'total': 0, 
        'rule_score_sum': 0, 
        'vl_score_sum': 0,
        'topo_score_sum': 0,
        'topo_count': 0,
        'rule_wins': 0,
        'vl_wins': 0,
        'topo_wins': 0,
    })
    calibration_stats = {'used': 0, 'total': 0}
    evolution_stats = {'actions': 0, 'samples': 0}
    
    for r in all_results:
        qtype = r['question_type']
        type_stats[qtype]['total'] += 1
        type_stats[qtype]['rule_score_sum'] += r.get('rule_score', 0)
        type_stats[qtype]['vl_score_sum'] += r.get('vl_score', 0)
        type_stats[qtype]['topo_score_sum'] += r.get('topo_score', 0)
        if r.get('topo_used', False):
            type_stats[qtype]['topo_count'] += 1
        
        method = r.get('method_used', 'rule')
        if method == 'rule':
            type_stats[qtype]['rule_wins'] += 1
        elif method == 'vl':
            type_stats[qtype]['vl_wins'] += 1
        elif method == 'topo':
            type_stats[qtype]['topo_wins'] += 1
        
        if 'calibration' in r and r['calibration'].get('scale_factor', 1.0) != 1.0:
            calibration_stats['used'] += 1
        calibration_stats['total'] += 1
        
        if 'evolution_actions' in r and r['evolution_actions']:
            evolution_stats['actions'] += len(r['evolution_actions'])
            evolution_stats['samples'] += 1
    
    # 打印结果
    print("\n" + "=" * 120)
    print("V8 三推理模式框架 (规则 + VL + 拓扑图增强) - 测试结果")
    print("=" * 120)
    print(f"VL Model: {args.vl_model}")
    print("-" * 120)
    print(f"{'任务类型':<35} {'Rule':>10} {'VL':>10} {'Topo':>10} {'Topo样本':>10}")
    print("-" * 120)
    
    overall_rule = 0
    overall_vl = 0
    overall_topo = 0
    overall_topo_count = 0
    overall_total = 0
    
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        n = stats['total']
        tc = stats['topo_count']
        rule_avg = stats['rule_score_sum'] / n if n > 0 else 0
        vl_avg = stats['vl_score_sum'] / n if n > 0 else 0
        topo_avg = stats['topo_score_sum'] / tc if tc > 0 else 0
        
        topo_str = f"{topo_avg*100:>9.2f}%" if tc > 0 else "    N/A   "
        print(f"{qtype:<35} {rule_avg*100:>9.2f}% {vl_avg*100:>9.2f}% {topo_str} {tc:>10}")
        
        overall_rule += stats['rule_score_sum']
        overall_vl += stats['vl_score_sum']
        overall_topo += stats['topo_score_sum']
        overall_topo_count += tc
        overall_total += n
    
    print("-" * 120)
    topo_overall = overall_topo / overall_topo_count * 100 if overall_topo_count > 0 else 0
    print(f"{'Overall':<35} {overall_rule/overall_total*100:>9.2f}% {overall_vl/overall_total*100:>9.2f}% {topo_overall:>9.2f}% {overall_topo_count:>10}")
    print("=" * 120)
    print(f"总样本数: {overall_total}, 使用拓扑图: {overall_topo_count}")
    print(f"校准使用率: {calibration_stats['used']}/{calibration_stats['total']} ({calibration_stats['used']/calibration_stats['total']*100:.1f}%)")
    print(f"演化动作: {evolution_stats['actions']} actions in {evolution_stats['samples']} samples")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"evolving_agent_v8_{timestamp}"
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
                'topo': stats['topo_score_sum'] / stats['topo_count'] if stats['topo_count'] > 0 else 0,
                'samples': stats['total'],
                'topo_samples': stats['topo_count'],
            }
            for qtype, stats in type_stats.items()
        },
        'overall': {
            'rule': overall_rule / overall_total if overall_total > 0 else 0,
            'vl': overall_vl / overall_total if overall_total > 0 else 0,
            'topo': overall_topo / overall_topo_count if overall_topo_count > 0 else 0,
        },
        'total_samples': overall_total,
        'topo_samples': overall_topo_count,
        'calibration_stats': calibration_stats,
        'evolution_stats': evolution_stats,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果保存到: {output_dir}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
