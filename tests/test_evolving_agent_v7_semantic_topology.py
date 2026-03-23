#!/usr/bin/env python3
"""
Self-Evolving Agent V7.1 - 语义拓扑增强版本 (Semantic Topology Augmentation)

核心设计理念：
1. 不再使用 2D 俯视图，因为 Qwen3-VL 不擅长处理抽象拓扑
2. 通过"语言描述的空间锚点"增强心智地图
3. 使用 VoxelMap 射线检测进行物理验证
4. 实现 Chain-of-Spatial-Thought (CoST) 推理
5. 通过视觉回溯演化修正空间幻觉

架构：
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Self-Evolving Agent V7.1                                │
│                  (Semantic Topology Augmentation)                           │
│                                                                             │
│  ┌─────────┐    ┌───────────┐    ┌──────────────┐    ┌─────────┐          │
│  │ 感知    │ -> │ Manager   │ -> │ CoST_Reasoner│ -> │ Critic  │          │
│  │DA3+DINO │    │任务分析    │    │语义空间推理   │    │物理验证  │          │
│  └─────────┘    └───────────┘    └──────────────┘    └────┬────┘          │
│       │                                                    │               │
│       v                                                    v               │
│  ┌─────────┐    ┌───────────┐                       ┌─────────┐          │
│  │ VoxelMap│ -> │SpatialAnch│    冲突检测？ ───────>│ Evolver │          │
│  │占据地图  │    │空间锚点    │         │            │视觉回溯  │          │
│  └─────────┘    └───────────┘         │            └────┬────┘          │
│                                        └──── 重新推理 <──┘               │
└─────────────────────────────────────────────────────────────────────────────┘

作者: tianjungu
日期: 2026-02-04
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
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from datetime import datetime
from enum import Enum

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
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

# 导入 V7 基础版本的共享组件
from tests.test_evolving_agent_v7_dual_reasoning import (
    CALIBRATION_OBJECTS, EXTENDED_VOCABULARY, SYNONYM_MAP,
    NUMERICAL_TASKS, CHOICE_TASKS, VIDEO_DIRS,
    find_video_path, get_synonyms, match_object_name,
    normalize_number, mean_relative_accuracy,
    Detection, CalibrationResult, EvolutionAction,
    MindMapBuilder, ScaleCalibrator,
)


# ============================================================================
# V7.1 新增数据结构
# ============================================================================

class CriticVerdict(Enum):
    """Critic 判定结果"""
    PASS = "pass"
    SPATIAL_CONFLICT = "spatial_conflict"
    LOW_CONFIDENCE = "low_confidence"
    NEEDS_VISUAL_VERIFICATION = "needs_visual_verification"


@dataclass
class MindMapEntityV71:
    """心智地图实体 - V7.1 增强版"""
    label: str
    detections: List[Detection] = field(default_factory=list)
    count: int = 0
    avg_confidence: float = 0.0
    position_3d: np.ndarray = None
    size_3d: np.ndarray = None
    first_seen_frame: int = -1
    calibrated: bool = False
    
    # V7.1 新增字段
    connectivity: float = 1.0          # 周围通行性
    depth_certainty: float = 1.0       # 深度置信度
    spatial_relation_to: Dict[str, str] = field(default_factory=dict)
    
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
            text += f", pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})m"
        if self.size_3d is not None:
            text += f", size≈{max(self.size_3d)*100:.0f}cm"
        text += f", conf={self.avg_confidence:.2f}, depth_cert={self.depth_certainty:.2f}"
        if self.first_seen_frame >= 0:
            text += f", first={self.first_seen_frame}"
        return text


@dataclass
class SpatialAnchor:
    """空间锚点 - 语义方位描述"""
    entity_label: str
    position: np.ndarray
    relative_descriptions: List[str] = field(default_factory=list)
    distance_to_camera: float = 0.0
    
    def to_text(self) -> str:
        desc = f"[{self.entity_label}] at depth {self.distance_to_camera:.2f}m"
        if self.relative_descriptions:
            desc += ": " + ", ".join(self.relative_descriptions)
        return desc


@dataclass
class CriticFeedback:
    """Critic 反馈"""
    verdict: CriticVerdict
    conflicting_objects: List[Tuple[str, str]] = field(default_factory=list)
    occlusion_info: Dict[str, Any] = field(default_factory=dict)
    suggested_reobserve: List[str] = field(default_factory=list)
    confidence: float = 1.0


# ============================================================================
# 稀疏体素地图 - 用于射线检测
# ============================================================================

@dataclass
class VoxelInfo:
    semantic_label: str = ""
    occupancy_prob: float = 0.0
    observation_count: int = 0


class SparseVoxelMap:
    """稀疏体素占据地图"""
    
    def __init__(self, voxel_size: float = 0.1):
        self.voxel_size = voxel_size
        self.voxels: Dict[Tuple[int, int, int], VoxelInfo] = {}
        self.min_bounds = np.array([float('inf')] * 3)
        self.max_bounds = np.array([float('-inf')] * 3)
    
    def world_to_voxel(self, pos: np.ndarray) -> Tuple[int, int, int]:
        return tuple(np.floor(pos / self.voxel_size).astype(int))
    
    def add_observation(self, position: np.ndarray, label: str, 
                        confidence: float = 1.0, extent: np.ndarray = None):
        self.min_bounds = np.minimum(self.min_bounds, position)
        self.max_bounds = np.maximum(self.max_bounds, position)
        
        if extent is not None:
            half = extent / 2
            min_v = self.world_to_voxel(position - half)
            max_v = self.world_to_voxel(position + half)
            for vx in range(min_v[0], max_v[0] + 1):
                for vy in range(min_v[1], max_v[1] + 1):
                    for vz in range(min_v[2], max_v[2] + 1):
                        self._update((vx, vy, vz), label, confidence)
        else:
            self._update(self.world_to_voxel(position), label, confidence)
    
    def _update(self, coord: Tuple[int, int, int], label: str, conf: float):
        if coord not in self.voxels:
            self.voxels[coord] = VoxelInfo()
        v = self.voxels[coord]
        v.observation_count += 1
        prior = v.occupancy_prob if v.observation_count > 1 else 0.5
        v.occupancy_prob = (prior * (v.observation_count - 1) + conf) / v.observation_count
        if not v.semantic_label or conf > 0.5:
            v.semantic_label = label
    
    def ray_cast(self, origin: np.ndarray, target: np.ndarray, 
                 exclude: Set[str] = None) -> Optional[Tuple[np.ndarray, str, float]]:
        """射线检测遮挡"""
        if exclude is None:
            exclude = set()
        
        direction = target - origin
        total_dist = np.linalg.norm(direction)
        if total_dist < 0.1:
            return None
        
        direction = direction / total_dist
        step = self.voxel_size * 0.5
        
        for t in np.arange(step, total_dist - step, step):
            point = origin + direction * t
            coord = self.world_to_voxel(point)
            
            if coord in self.voxels:
                v = self.voxels[coord]
                if v.occupancy_prob > 0.5 and v.semantic_label not in exclude:
                    return (point, v.semantic_label, t)
        return None
    
    def check_path_clearance(self, a: np.ndarray, b: np.ndarray, 
                             radius: float = 0.3) -> Tuple[bool, List[str]]:
        """检查路径通畅性"""
        obstacles = []
        direction = b - a
        total = np.linalg.norm(direction)
        if total < 0.1:
            return True, []
        
        direction = direction / total
        step = self.voxel_size
        
        for t in np.arange(0, total, step):
            center = a + direction * t
            coord = self.world_to_voxel(center)
            if coord in self.voxels:
                v = self.voxels[coord]
                if v.occupancy_prob > 0.5 and v.semantic_label not in obstacles:
                    obstacles.append(v.semantic_label)
        
        return len(obstacles) == 0, obstacles


# ============================================================================
# V7.1 心智地图构建器
# ============================================================================

class MindMapBuilderV71(MindMapBuilder):
    """V7.1 心智地图构建器 - 增加体素地图和空间关系计算"""
    
    def build_from_video(self, video_path: str, target_objects: List[str] = None):
        """构建心智地图和体素地图"""
        self.load_models()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, SparseVoxelMap(), 0, [], []
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
        
        vocab = list(set((target_objects or []) + EXTENDED_VOCABULARY))
        prompt = " . ".join(vocab) + " ."
        
        all_detections = defaultdict(list)
        sampled_frames = []
        voxel_map = SparseVoxelMap(voxel_size=0.1)
        depth_stds = defaultdict(list)
        
        for idx, frame_idx in enumerate(frame_indices):
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
                
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                depth_roi = depth_map[y1:y2, x1:x2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                depth = float(np.median(depth_roi)) if depth_roi.size > 0 else float(depth_map[cy, cx])
                depth_std = float(np.std(depth_roi)) if depth_roi.size > 0 else 0.5
                depth_stds[label].append(depth_std)
                
                pos_3d = np.array([
                    (cx - w / 2) * depth / self.focal_length,
                    (cy - h / 2) * depth / self.focal_length,
                    depth
                ])
                
                est_h = (box[3] - box[1]) * depth / self.focal_length
                est_w = (box[2] - box[0]) * depth / self.focal_length
                
                detection = Detection(
                    frame_idx=int(frame_idx),
                    bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    confidence=float(conf),
                    depth=depth,
                    position_3d=pos_3d,
                    estimated_height=est_h,
                    estimated_width=est_w,
                )
                
                all_detections[label].append(detection)
                
                # 添加到体素地图
                extent = np.array([est_w, est_h, min(est_w, est_h) * 0.5])
                voxel_map.add_observation(pos_3d, label, conf, extent)
        
        cap.release()
        
        # 聚合成实体
        mind_map = {}
        for label, dets in all_detections.items():
            entity = self._aggregate_v71(label, dets)
            if label in depth_stds and depth_stds[label]:
                avg_std = np.mean(depth_stds[label])
                entity.depth_certainty = 1.0 / (1.0 + avg_std)
            mind_map[label] = entity
        
        # 计算空间关系
        self._compute_spatial_relations(mind_map)
        
        return mind_map, voxel_map, total_frames, sampled_frames, frame_indices
    
    def _aggregate_v71(self, label: str, detections: List[Detection]) -> MindMapEntityV71:
        if not detections:
            return MindMapEntityV71(label=label)
        
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
        
        return MindMapEntityV71(
            label=label,
            detections=detections,
            count=max_count,
            avg_confidence=float(avg_conf),
            position_3d=avg_pos,
            size_3d=size_3d,
            first_seen_frame=first_frame,
        )
    
    def _compute_spatial_relations(self, mind_map: Dict[str, MindMapEntityV71]):
        """计算实体间空间关系"""
        entities = list(mind_map.items())
        
        for i, (label1, ent1) in enumerate(entities):
            if ent1.position_3d is None:
                continue
            
            for j, (label2, ent2) in enumerate(entities):
                if i == j or ent2.position_3d is None:
                    continue
                
                diff = ent1.position_3d - ent2.position_3d
                dx, dy, dz = diff[0], diff[1], diff[2]
                
                relations = []
                if abs(dx) > 0.3:
                    relations.append("right of" if dx > 0 else "left of")
                if abs(dy) > 0.3:
                    relations.append("below" if dy > 0 else "above")
                if abs(dz) > 0.5:
                    relations.append("farther than" if dz > 0 else "closer than")
                
                dist = np.linalg.norm(diff)
                relations.append(f"({dist:.1f}m)")
                
                ent1.spatial_relation_to[label2] = " ".join(relations)


# ============================================================================
# CoST Reasoner - 空间链式思考推理
# ============================================================================

class CoSTReasoner:
    """Chain-of-Spatial-Thought 推理器"""
    
    def generate_spatial_anchors(self, mind_map: Dict[str, MindMapEntityV71], 
                                 targets: List[str]) -> List[SpatialAnchor]:
        """生成空间锚点"""
        anchors = []
        for target in targets:
            for label, ent in mind_map.items():
                if match_object_name(target, label) and ent.position_3d is not None:
                    descriptions = [f"{rel} the {other}" 
                                    for other, rel in list(ent.spatial_relation_to.items())[:3]]
                    anchors.append(SpatialAnchor(
                        entity_label=label,
                        position=ent.position_3d.copy(),
                        relative_descriptions=descriptions,
                        distance_to_camera=float(ent.position_3d[2]),
                    ))
                    break
        return anchors
    
    def build_cost_prompt(self, question: str, mind_map: Dict[str, MindMapEntityV71],
                          targets: List[str], question_type: str) -> str:
        """构建 CoST Prompt"""
        anchors = self.generate_spatial_anchors(mind_map, targets)
        anchors_text = "\n".join([a.to_text() for a in anchors]) if anchors else "No targets found."
        
        mind_map_text = "\n".join([e.to_text() for e in sorted(
            mind_map.values(), key=lambda x: -x.count)][:15])
        
        # 任务特定 CoST 步骤
        if 'counting' in question_type:
            steps = """[CoST Steps]:
1. Identify all instances of the target object
2. Verify uniqueness by checking positions
3. Cross-frame consistency check
4. Report maximum count in any single frame"""
        elif 'direction' in question_type:
            steps = """[CoST Steps]:
1. Locate observer position ("standing by X")
2. Determine facing direction (vector from X to Y)
3. Compute target vector (from observer to target)
4. Apply cross-product: (forward × to_target).Y > 0 means RIGHT
5. Conclude direction"""
        elif 'rel_distance' in question_type:
            steps = """[CoST Steps]:
1. Identify reference object
2. Measure Euclidean distance to each candidate
3. Consider depth_certainty for close distances
4. Report closest/farthest based on reliable measurements"""
        elif 'appearance_order' in question_type:
            steps = """[CoST Steps]:
1. Extract first_seen_frame for each object
2. Sort by frame number (smaller = earlier)
3. Verify order with video progression
4. Report objects in appearance order"""
        else:
            steps = """[CoST Steps]:
1. Identify relevant objects
2. Extract spatial information
3. Apply spatial reasoning
4. Provide answer"""
        
        return f"""You are a spatial intelligence expert. Use Chain-of-Spatial-Thought reasoning.

=== SPATIAL ANCHORS ===
{anchors_text}

=== DETECTED OBJECTS ===
{mind_map_text}

{steps}

=== QUESTION ===
{question}

Follow CoST steps, then provide:
[Reasoning]: <step-by-step>
[Answer]: <final answer>"""
    
    def extract_targets(self, question: str, options: List[str] = None) -> List[str]:
        """从问题提取目标物体"""
        targets = []
        patterns = [
            r'How many (\w+)', r'size of (?:the )?(\w+)',
            r'standing by (?:the )?(\w+)', r'facing (?:the )?(\w+)',
            r'is (?:the )?(\w+)', r'to (?:the )?(\w+)',
            r'from (?:the )?(\w+)', r'between (?:the )?(\w+)',
            r'closest to (?:the )?(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question.lower())
            targets.extend(matches)
        
        if options:
            for opt in options:
                words = re.findall(r'\b[a-z]+\b', opt.lower())
                targets.extend([w for w in words if w in EXTENDED_VOCABULARY])
        
        return list(set(targets))


# ============================================================================
# Physical Critic - 物理验证
# ============================================================================

class PhysicalCritic:
    """物理验证器"""
    
    def __init__(self, voxel_map: SparseVoxelMap):
        self.voxel_map = voxel_map
    
    def verify_direction(self, observer: np.ndarray, target: np.ndarray,
                         mind_map: Dict) -> CriticFeedback:
        """验证方向判断"""
        hit = self.voxel_map.ray_cast(observer, target)
        
        if hit is not None:
            hit_point, hit_label, hit_dist = hit
            return CriticFeedback(
                verdict=CriticVerdict.SPATIAL_CONFLICT,
                occlusion_info={'blocking_object': hit_label, 'hit_distance': hit_dist},
                suggested_reobserve=[hit_label],
                confidence=0.7,
            )
        
        return CriticFeedback(verdict=CriticVerdict.PASS, confidence=0.9)
    
    def verify_distance(self, ref_pos: np.ndarray,
                        candidates: List[Tuple[str, np.ndarray]],
                        claimed: str) -> CriticFeedback:
        """验证距离判断"""
        distances = [(label, np.linalg.norm(pos - ref_pos)) for label, pos in candidates]
        distances.sort(key=lambda x: x[1])
        
        actual_closest = distances[0][0] if distances else None
        
        if actual_closest and not match_object_name(claimed, actual_closest):
            if len(distances) >= 2 and distances[1][1] - distances[0][1] < 0.3:
                return CriticFeedback(verdict=CriticVerdict.LOW_CONFIDENCE, confidence=0.5)
            
            return CriticFeedback(
                verdict=CriticVerdict.SPATIAL_CONFLICT,
                occlusion_info={'claimed': claimed, 'actual': actual_closest},
                confidence=0.8,
            )
        
        return CriticFeedback(verdict=CriticVerdict.PASS, confidence=0.9)


# ============================================================================
# Visual Evolver - 视觉回溯演化
# ============================================================================

class VisualEvolver:
    """视觉回溯演化器"""
    
    def __init__(self, vl_model=None, vl_processor=None, device='cuda'):
        self.vl_model = vl_model
        self.vl_processor = vl_processor
        self.device = device
    
    def highlight_blocking(self, frame: np.ndarray, bbox: Tuple, label: str) -> np.ndarray:
        """高亮标注挡路物体"""
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, max(0, y1 - 20)), f"[BLOCKING: {label}]", fill='red')
        return np.array(frame_pil)
    
    def verify_with_frame(self, frame: np.ndarray, mind_map: Dict,
                          blocking: str, question: str, 
                          original_answer: str) -> Tuple[str, str]:
        """视觉验证"""
        if self.vl_model is None:
            return original_answer, "VL model not available"
        
        # 找到 bbox
        blocking_ent = None
        for label, ent in mind_map.items():
            if match_object_name(blocking, label):
                blocking_ent = ent
                break
        
        if blocking_ent is None or not blocking_ent.detections:
            return original_answer, f"Could not find {blocking}"
        
        best_det = max(blocking_ent.detections, key=lambda x: x.confidence)
        highlighted = self.highlight_blocking(frame, best_det.bbox, blocking)
        
        prompt = f"""Verify this spatial answer.
QUESTION: {question}
INITIAL ANSWER: {original_answer}

The object "{blocking}" (RED box) may be blocking the view/path.
Is this blocking relevant? If yes, correct the answer.

[Verification]: <analysis>
[Final Answer]: <answer>"""
        
        try:
            response = self._call_vl(highlighted, prompt)
            match = re.search(r'\[Final Answer\]:\s*(.+?)(?:\n|$)', response, re.I)
            if match:
                return match.group(1).strip(), f"Visual verification: {response}"
            return original_answer, f"Could not extract from: {response}"
        except Exception as e:
            return original_answer, f"Failed: {e}"
    
    def _call_vl(self, image: np.ndarray, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(image)},
            {"type": "text", "text": prompt}
        ]}]
        
        text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.vl_processor(text=[text], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vl_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        
        return self.vl_processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                               skip_special_tokens=True)[0].strip()


# ============================================================================
# 心智地图演化器 - 保持 Counting 演化
# ============================================================================

class MindMapEvolver:
    """心智地图演化器"""
    
    def evolve_for_counting(self, mind_map: Dict[str, MindMapEntityV71],
                            target: str, frames: List, indices: List) -> Tuple[Dict, List]:
        actions = []
        
        target_entity = None
        target_label = None
        for label, entity in mind_map.items():
            if match_object_name(target, label):
                target_entity = entity
                target_label = label
                break
        
        if target_entity is None:
            return mind_map, actions
        
        frame_dets = defaultdict(list)
        for det in target_entity.detections:
            frame_dets[det.frame_idx].append(det)
        
        counts = [len(dets) for dets in frame_dets.values()]
        if len(counts) > 1:
            max_count = max(counts)
            median_count = np.median(counts)
            
            if max_count > median_count * 2 and max_count > 2:
                new_count = int(np.ceil(median_count))
                if new_count != target_entity.count:
                    actions.append(EvolutionAction(
                        'correct_count', target_label,
                        target_entity.count, new_count,
                        f"Max({max_count}) >> median({median_count:.1f})",
                        0.7
                    ))
                    target_entity.count = new_count
        
        return mind_map, actions


# ============================================================================
# Semantic Topology QA - 集成 CoST + Critic + Evolver
# ============================================================================

class SemanticTopologyQA:
    """语义拓扑增强 QA 系统 - Agentic Loop"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.vl_model = None
        self.vl_processor = None
        self.cost_reasoner = CoSTReasoner()
        self.visual_evolver = VisualEvolver(device=device)
    
    def load_vl_model(self, model_path: str):
        if self.vl_model is not None:
            return
        
        if 'qwen3' in model_path.lower():
            try:
                from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
                self.vl_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                self.vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
                logger.info("Loaded Qwen3-VL")
                self.visual_evolver.vl_model = self.vl_model
                self.visual_evolver.vl_processor = self.vl_processor
                return
            except ImportError:
                pass
        
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.vl_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
        logger.info("Loaded Qwen2.5-VL")
        self.visual_evolver.vl_model = self.vl_model
        self.visual_evolver.vl_processor = self.vl_processor
    
    def agentic_solve(self, question: str, question_type: str, options: List[str],
                      mind_map: Dict, voxel_map: SparseVoxelMap,
                      video_path: str, frames: List[np.ndarray],
                      calibration: CalibrationResult) -> Tuple[str, str, Dict]:
        """Agentic 求解循环"""
        debug = {'visual_verification': False, 'evolution': False}
        
        # 1. 提取目标
        targets = self.cost_reasoner.extract_targets(question, options)
        
        # 2. 规则推理
        rule_ans, rule_reason = self._rule_answer(question, question_type, options, mind_map, calibration)
        
        # 3. VL 推理
        vl_ans, vl_reason = "", ""
        if self.vl_model:
            vl_ans, vl_reason = self._vl_answer(question, question_type, options, mind_map, targets, video_path, calibration)
        
        # 4. 物理验证 (方向/距离任务)
        critic = PhysicalCritic(voxel_map)
        feedback = None
        
        if 'direction' in question_type and vl_ans:
            feedback = self._verify_direction(question, mind_map, critic)
        elif 'rel_distance' in question_type and vl_ans:
            feedback = self._verify_distance(question, vl_ans, mind_map, critic)
        
        # 5. 视觉回溯
        final_ans = vl_ans if vl_ans else rule_ans
        final_reason = vl_reason if vl_reason else rule_reason
        
        if feedback and feedback.verdict == CriticVerdict.SPATIAL_CONFLICT:
            debug['visual_verification'] = True
            blocking = feedback.occlusion_info.get('blocking_object') or \
                       (feedback.suggested_reobserve[0] if feedback.suggested_reobserve else None)
            
            if blocking and frames:
                corrected, corr_reason = self.visual_evolver.verify_with_frame(
                    frames[len(frames)//2], mind_map, blocking, question, final_ans
                )
                if corrected != final_ans:
                    debug['evolution'] = True
                    final_ans = corrected
                    final_reason += f"\n[Correction]: {corr_reason}"
        
        return final_ans, f"Rule: {rule_reason}\nVL: {vl_reason}\nFinal: {final_reason}", debug
    
    def _rule_answer(self, question: str, qtype: str, options: List[str],
                     mind_map: Dict, calibration: CalibrationResult) -> Tuple[str, str]:
        """规则推理"""
        if qtype == 'object_counting':
            match = re.search(r'How many (\w+)', question, re.I)
            if match:
                target = match.group(1).lower()
                for label, ent in mind_map.items():
                    if match_object_name(target, label):
                        return str(ent.count), f"count={ent.count}"
            return "0", "not found"
        
        elif qtype == 'object_size_estimation':
            match = re.search(r'of the (\w+)', question.lower())
            if match:
                target = match.group(1)
                for label, ent in mind_map.items():
                    if match_object_name(target, label):
                        size = ent.get_estimated_size() * calibration.scale_factor * 100
                        return str(int(size)), f"size={size:.0f}cm"
            return "100", "default"
        
        elif qtype == 'room_size_estimation':
            positions = [e.position_3d[:2] for e in mind_map.values() if e.position_3d is not None]
            if len(positions) >= 2:
                positions = np.array(positions)
                x_range = (positions[:, 0].max() - positions[:, 0].min()) * calibration.scale_factor
                y_range = (positions[:, 1].max() - positions[:, 1].min()) * calibration.scale_factor
                area = max((x_range + 2) * (y_range + 2), 10)
                return str(int(area)), f"area={area:.1f}m²"
            return "25", "default"
        
        elif qtype == 'object_abs_distance':
            match = re.search(r'between (?:the )?(\w+)\s+and\s+(?:the )?(\w+)', question.lower())
            if match:
                obj1, obj2 = match.groups()
                ent1 = ent2 = None
                for label, ent in mind_map.items():
                    if match_object_name(obj1, label): ent1 = ent
                    if match_object_name(obj2, label): ent2 = ent
                if ent1 and ent2 and ent1.position_3d is not None and ent2.position_3d is not None:
                    dist = np.linalg.norm(ent1.position_3d - ent2.position_3d) * calibration.scale_factor
                    return f"{dist:.2f}", f"dist={dist:.2f}m"
            return "2.0", "default"
        
        elif qtype == 'obj_appearance_order':
            if not options:
                return "A", "no options"
            
            match = re.search(r'categories.*?:\s*(.+?)\?', question, re.I)
            targets = [o.strip().lower() for o in (match.group(1).split(',') if match else 
                      re.sub(r'^[A-D]\.\s*', '', options[0]).split(','))]
            
            obj_frames = {}
            for t in targets:
                for label, ent in mind_map.items():
                    if match_object_name(t, label) and ent.get_frame_indices():
                        obj_frames[t] = min(ent.get_frame_indices())
                        break
            
            sorted_objs = sorted(obj_frames.keys(), key=lambda x: obj_frames.get(x, float('inf')))
            
            for i, opt in enumerate(options):
                opt_objs = [o.strip().lower() for o in re.sub(r'^[A-D]\.\s*', '', opt).split(',')]
                indices = [sorted_objs.index(o) for o in opt_objs if o in sorted_objs]
                if indices == sorted(indices) and len(indices) >= 2:
                    return chr(65 + i), f"order={sorted_objs}"
            
            return options[0][0], f"sorted={sorted_objs}"
        
        elif 'direction' in qtype:
            return self._rule_direction(mind_map, question, options)
        
        elif 'rel_distance' in qtype:
            return self._rule_rel_distance(mind_map, question, options)
        
        return options[0][0] if options else "", "default"
    
    def _rule_direction(self, mind_map: Dict, question: str, options: List[str]) -> Tuple[str, str]:
        if not options:
            return "left", "no options"
        
        q = question.lower()
        standing = re.search(r'standing by (?:the )?(\w+)', q)
        facing = re.search(r'facing (?:the )?(\w+)', q)
        target = re.search(r'is (?:the )?(\w+)\s+to', q)
        
        if not all([standing, facing, target]):
            return options[0][0], "parse failed"
        
        def find(name):
            for label, ent in mind_map.items():
                if match_object_name(name, label): return ent
            return None
        
        s_ent, f_ent, t_ent = find(standing.group(1)), find(facing.group(1)), find(target.group(1))
        
        if not all([s_ent, f_ent, t_ent]) or any(e.position_3d is None for e in [s_ent, f_ent, t_ent]):
            return options[0][0], "missing entities"
        
        forward = np.array([f_ent.position_3d[0] - s_ent.position_3d[0],
                           f_ent.position_3d[2] - s_ent.position_3d[2]])
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.array([forward[1], -forward[0]])
        
        to_target = np.array([t_ent.position_3d[0] - s_ent.position_3d[0],
                             t_ent.position_3d[2] - s_ent.position_3d[2]])
        to_target = to_target / (np.linalg.norm(to_target) + 1e-8)
        
        front_dot = np.dot(to_target, forward)
        right_dot = np.dot(to_target, right)
        
        if 'back' in q and front_dot < -0.5:
            direction = 'back'
        else:
            direction = 'right' if right_dot > 0 else 'left'
        
        for opt in options:
            if direction in opt.lower():
                return opt[0] if len(opt) == 1 or opt[1] in '.、' else opt, f"dir={direction}"
        
        return options[0][0], f"dir={direction}"
    
    def _rule_rel_distance(self, mind_map: Dict, question: str, options: List[str]) -> Tuple[str, str]:
        if not options:
            return "A", "no options"
        
        q = question.lower()
        find_closest = 'closest' in q or 'nearest' in q
        
        ref_match = re.search(r'(?:to|from) (?:the )?(\w+)', q)
        if not ref_match:
            return options[0][0], "no ref"
        
        ref_name = ref_match.group(1)
        ref_ent = None
        for label, ent in mind_map.items():
            if match_object_name(ref_name, label):
                ref_ent = ent
                break
        
        if not ref_ent or ref_ent.position_3d is None:
            return options[0][0], "ref not found"
        
        distances = {}
        for i, opt in enumerate(options):
            cand = re.sub(r'^[A-D]\.\s*', '', opt).strip().lower()
            for label, ent in mind_map.items():
                if match_object_name(cand, label) and ent.position_3d is not None:
                    distances[chr(65 + i)] = np.linalg.norm(ent.position_3d - ref_ent.position_3d)
                    break
        
        if not distances:
            return options[0][0], "no distances"
        
        best = min(distances.keys(), key=lambda k: distances[k]) if find_closest else \
               max(distances.keys(), key=lambda k: distances[k])
        
        return best, f"distances={distances}"
    
    def _vl_answer(self, question: str, qtype: str, options: List[str],
                   mind_map: Dict, targets: List[str], video_path: str,
                   calibration: CalibrationResult) -> Tuple[str, str]:
        """VL 推理"""
        if self.vl_model is None:
            return "", "no model"
        
        prompt = self.cost_reasoner.build_cost_prompt(question, mind_map, targets, qtype)
        if options:
            prompt += f"\n\n=== OPTIONS ===\n" + "\n".join(options) + "\n\nAnswer with letter (A/B/C/D):"
        
        try:
            from qwen_vl_utils import process_vision_info
            
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video_path, "max_pixels": 360*420, "nframes": 8},
                {"type": "text", "text": prompt}
            ]}]
            
            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.vl_processor(text=[text], images=image_inputs, videos=video_inputs,
                                       padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.vl_model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
            response = self.vl_processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                                      skip_special_tokens=True)[0]
            
            if options:
                match = re.search(r'\[Answer\]:\s*([A-D])', response, re.I)
                if match:
                    return match.group(1).upper(), response
                for line in response.split('\n')[::-1]:
                    if line.strip() and line.strip()[0].upper() in 'ABCD':
                        return line.strip()[0].upper(), response
                return options[0][0], response
            else:
                match = re.search(r'\[Answer\]:\s*([\d.]+)', response, re.I)
                if match:
                    return match.group(1), response
                num_match = re.search(r'[\d.]+', response)
                return num_match.group() if num_match else "0", response
                
        except Exception as e:
            return "", str(e)
    
    def _verify_direction(self, question: str, mind_map: Dict, critic: PhysicalCritic) -> Optional[CriticFeedback]:
        q = question.lower()
        standing = re.search(r'standing by (?:the )?(\w+)', q)
        target = re.search(r'is (?:the )?(\w+)', q)
        
        if not standing or not target:
            return None
        
        s_ent = t_ent = None
        for label, ent in mind_map.items():
            if match_object_name(standing.group(1), label): s_ent = ent
            if match_object_name(target.group(1), label): t_ent = ent
        
        if not s_ent or not t_ent or s_ent.position_3d is None or t_ent.position_3d is None:
            return None
        
        return critic.verify_direction(s_ent.position_3d, t_ent.position_3d, mind_map)
    
    def _verify_distance(self, question: str, answer: str, mind_map: Dict, 
                         critic: PhysicalCritic) -> Optional[CriticFeedback]:
        ref_match = re.search(r'(?:to|from) (?:the )?(\w+)', question.lower())
        if not ref_match:
            return None
        
        ref_ent = None
        for label, ent in mind_map.items():
            if match_object_name(ref_match.group(1), label):
                ref_ent = ent
                break
        
        if not ref_ent or ref_ent.position_3d is None:
            return None
        
        candidates = [(label, ent.position_3d) for label, ent in mind_map.items()
                     if ent.position_3d is not None and not match_object_name(ref_match.group(1), label)]
        
        if not candidates:
            return None
        
        return critic.verify_distance(ref_ent.position_3d, candidates, answer)


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], vl_model_path: str, result_queue: mp.Queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    builder = MindMapBuilderV71(device='cuda', num_frames=32, box_threshold=0.25)
    calibrator = ScaleCalibrator()
    evolver = MindMapEvolver()
    qa = SemanticTopologyQA(device='cuda')
    
    vl_loaded = False
    try:
        qa.load_vl_model(vl_model_path)
        vl_loaded = True
        logger.info(f"GPU {gpu_id}: VL loaded")
    except Exception as e:
        logger.warning(f"GPU {gpu_id}: VL failed: {e}")
    
    results = []
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        video_path = sample['video_path']
        question = sample['question']
        qtype = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 1. 构建心智地图 + 体素地图
            target_objects = []
            if 'counting' in qtype:
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map, voxel_map, total_frames, frames, indices = builder.build_from_video(video_path, target_objects)
            
            # 2. 校准
            calibration = calibrator.calculate_scale_factor(mind_map)
            mind_map = calibrator.apply_calibration(mind_map, calibration)
            
            # 3. Counting 演化
            evo_actions = []
            if qtype == 'object_counting' and target_objects:
                mind_map, actions = evolver.evolve_for_counting(mind_map, target_objects[0], frames, indices)
                evo_actions.extend(actions)
            
            # 4. Agentic 求解
            pred, reasoning, debug = qa.agentic_solve(
                question, qtype, options, mind_map, voxel_map, video_path, frames, calibration
            )
            
            # 5. 评估
            if qtype in NUMERICAL_TASKS:
                pred_val = normalize_number(pred)
                gt_val = normalize_number(gt)
                score = mean_relative_accuracy(pred_val, gt_val) if pred_val and gt_val else 0.0
            else:
                p = pred.strip().upper() if pred else ""
                g = gt.strip().upper()
                if len(p) > 1 and p[1] in '.、': p = p[0]
                if len(g) > 1 and g[1] in '.、': g = g[0]
                score = 1.0 if p == g else 0.0
            
            results.append({
                'scene_name': sample['scene_name'],
                'question': question,
                'question_type': qtype,
                'ground_truth': gt,
                'options': options,
                'prediction': pred,
                'score': score,
                'reasoning': reasoning,
                'debug_info': debug,
                'calibration': {'object': calibration.calibration_object, 
                               'scale': calibration.scale_factor},
                'evolution_actions': [{'type': a.action_type, 'entity': a.target_entity,
                                       'old': str(a.old_value), 'new': str(a.new_value)} for a in evo_actions],
            })
            
        except Exception as e:
            logger.error(f"Error: {sample['scene_name']}: {e}")
            results.append({
                'scene_name': sample['scene_name'],
                'question': question,
                'question_type': qtype,
                'ground_truth': gt,
                'prediction': '',
                'score': 0.0,
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
# 主函数
# ============================================================================

def load_vsibench_data() -> List[Dict]:
    from datasets import load_dataset
    
    logger.info("Loading VSI-Bench...")
    ds = load_dataset('nyu-visionx/VSI-Bench', split='test',
                      cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench')
    
    samples = []
    for item in ds:
        video_path = find_video_path(item['scene_name'])
        if video_path:
            samples.append({
                'scene_name': item['scene_name'],
                'video_path': video_path,
                'question': item['question'],
                'question_type': item['question_type'],
                'options': item.get('options', []),
                'ground_truth': item['ground_truth'],
            })
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description='V7.1 Semantic Topology Augmentation')
    parser.add_argument('--vl-model', type=str, 
                       default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--question-types', type=str, nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    data = load_vsibench_data()
    
    if args.question_types:
        data = [d for d in data if d['question_type'] in args.question_types]
        logger.info(f"Filtered: {len(data)}")
    
    if args.max_samples:
        data = data[:args.max_samples]
    
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    samples_per_gpu = len(data) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(data)
        gpu_samples.append(data[start:end])
    
    result_queue = mp.Queue()
    processes = []
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_samples[gpu_id], args.vl_model, result_queue))
        p.start()
        processes.append(p)
    
    all_results = []
    for _ in range(num_gpus):
        all_results.extend(result_queue.get())
    
    for p in processes:
        p.join()
    
    # 统计
    type_stats = defaultdict(lambda: {'total': 0, 'score_sum': 0, 'visual_verifications': 0})
    
    for r in all_results:
        stats = type_stats[r['question_type']]
        stats['total'] += 1
        stats['score_sum'] += r.get('score', 0)
        if r.get('debug_info', {}).get('visual_verification'):
            stats['visual_verifications'] += 1
    
    print("\n" + "=" * 80)
    print("V7.1 Semantic Topology Augmentation - Results")
    print("=" * 80)
    print(f"{'Task Type':<40} {'Accuracy':>10} {'Visual Verify':>15}")
    print("-" * 80)
    
    overall_score = 0
    overall_total = 0
    overall_verify = 0
    
    for qtype in sorted(type_stats.keys()):
        s = type_stats[qtype]
        acc = s['score_sum'] / s['total'] if s['total'] > 0 else 0
        print(f"{qtype:<40} {acc*100:>9.2f}% {s['visual_verifications']:>15}")
        overall_score += s['score_sum']
        overall_total += s['total']
        overall_verify += s['visual_verifications']
    
    print("-" * 80)
    print(f"{'Overall':<40} {overall_score/overall_total*100:>9.2f}% {overall_verify:>15}")
    print("=" * 80)
    print(f"Total samples: {overall_total}")
    print(f"Visual verifications triggered: {overall_verify}")
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"v71_semantic_topology_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'results_by_type': {qtype: {'accuracy': s['score_sum']/s['total'] if s['total'] > 0 else 0,
                                    'samples': s['total'], 'visual_verifications': s['visual_verifications']}
                           for qtype, s in type_stats.items()},
        'overall_accuracy': overall_score / overall_total if overall_total > 0 else 0,
        'total_samples': overall_total,
        'visual_verifications': overall_verify,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
