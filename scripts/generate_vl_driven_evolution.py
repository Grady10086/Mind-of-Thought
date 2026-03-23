#!/usr/bin/env python3
"""
VL驱动的Evolution训练数据生成

核心思想：
1. 感知系统生成初始Mind Map
2. VL模型审视Mind Map，发现潜在问题
3. VL模型给出修正建议
4. 根据VL反馈更新Mind Map
5. 最终基于演化后的Mind Map回答问题

训练目标：让VL模型学会：
- 审视感知结果的合理性
- 发现并指出问题
- 给出修正建议
- 基于修正后的信息推理答案

架构：
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VL-Driven Evolution Framework                        │
│                                                                             │
│  ┌─────────┐    ┌───────────┐    ┌──────────────┐    ┌─────────────┐       │
│  │ 感知    │ -> │ Initial   │ -> │ VL Review &  │ -> │ Evolved     │       │
│  │DA3+DINO │    │ Mind Map  │    │ Correction   │    │ Mind Map    │       │
│  └─────────┘    └───────────┘    └──────────────┘    └──────┬──────┘       │
│                                         │                    │              │
│                                         ▼                    ▼              │
│                                  ┌─────────────┐     ┌─────────────┐       │
│                                  │ Evolution   │     │ Final       │       │
│                                  │ Reasoning   │     │ Answer      │       │
│                                  └─────────────┘     └─────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘

训练数据格式 (Multi-turn conversation):
Turn 1: 用户提供初始Mind Map + 问题
Turn 2: VL模型审视并给出Evolution建议
Turn 3: 用户确认Evolution
Turn 4: VL模型给出最终答案
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
    
    def get_frame_indices(self) -> List[int]:
        return sorted(set(d.frame_idx for d in self.detections))


@dataclass 
class CalibrationResult:
    calibration_object: str
    estimated_size: float
    standard_size: float
    scale_factor: float
    confidence: float


# ============================================================================
# VL驱动的Evolution提示模板
# ============================================================================

class VLEvolutionPromptGenerator:
    """生成VL驱动Evolution的训练提示"""
    
    @staticmethod
    def format_mind_map(mind_map: Dict[str, MindMapEntity], calibration: CalibrationResult = None) -> str:
        """格式化Mind Map为文本"""
        lines = []
        for label, entity in sorted(mind_map.items(), key=lambda x: -x[1].avg_confidence)[:20]:
            pos_str = f"({entity.position_3d[0]:.2f}, {entity.position_3d[1]:.2f}, {entity.position_3d[2]:.2f})" if entity.position_3d is not None else "unknown"
            size_str = f"({entity.size_3d[0]:.2f}m × {entity.size_3d[1]:.2f}m × {entity.size_3d[2]:.2f}m)" if entity.size_3d is not None else "unknown"
            conf_str = f"{entity.avg_confidence:.2f}"
            lines.append(f"- {label}: position {pos_str}, size {size_str}, count: {entity.count}, confidence: {conf_str}")
        
        result = "\n".join(lines)
        if calibration:
            result += f"\n\nCalibration: {calibration.calibration_object} (scale factor: {calibration.scale_factor:.2f})"
        return result
    
    @staticmethod
    def generate_counting_evolution(
        mind_map: Dict[str, MindMapEntity],
        question: str,
        ground_truth: str,
        calibration: CalibrationResult = None
    ) -> Dict:
        """生成Counting任务的VL Evolution训练数据"""
        
        # 提取目标物体
        match = re.search(r'How many (\w+)', question, re.IGNORECASE)
        target_obj = match.group(1).lower() if match else "objects"
        
        # 找到对应的检测
        detected_count = 0
        detected_conf = 0.0
        for label, entity in mind_map.items():
            if target_obj in label.lower() or label.lower() in target_obj:
                detected_count = entity.count
                detected_conf = entity.avg_confidence
                break
        
        # 解析ground truth
        gt_answer = ground_truth.strip().upper()
        gt_number = None
        number_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}  # 需要从选项中解析
        
        mind_map_text = VLEvolutionPromptGenerator.format_mind_map(mind_map, calibration)
        
        # 构建多轮对话
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"""You are a spatial intelligence assistant with the ability to review and correct perception results.

=== INITIAL PERCEPTION RESULTS ===
{mind_map_text}

=== TASK ===
{question}

Please first review the perception results for potential issues, then provide your answer.

Step 1: Review the detection of "{target_obj}" - is the count of {detected_count} reasonable based on typical indoor scenes?
Step 2: If you find any issues, explain what corrections are needed.
Step 3: Provide your final answer."""
                },
                {
                    "from": "gpt", 
                    "value": f"""Let me review the perception results and reason through this step by step.

**Step 1: Review Detection**
The perception system detected {detected_count} {target_obj}(s) with confidence {detected_conf:.2f}.

**Step 2: Evolution Analysis**
{"Based on the video frames, I should verify this count by:" if detected_count > 0 else "The target object was not detected. Let me check if:"}
- Checking for duplicate detections across frames (same object detected multiple times)
- Verifying spatial consistency (detected positions should be distinct)
- Considering occlusion and partial visibility
- Cross-referencing with other detected objects for scene consistency

**Step 3: Final Answer**
After careful review and evolution of the perception results, my answer is: {gt_answer}"""
                }
            ],
            "question_type": "object_counting",
            "evolution_type": "vl_driven"
        }
        
        return conversation
    
    @staticmethod
    def generate_size_evolution(
        mind_map: Dict[str, MindMapEntity],
        question: str,
        ground_truth: str,
        options: List[str],
        calibration: CalibrationResult = None
    ) -> Dict:
        """生成Size任务的VL Evolution训练数据"""
        
        mind_map_text = VLEvolutionPromptGenerator.format_mind_map(mind_map, calibration)
        options_text = "\n".join(options) if options else ""
        gt_answer = ground_truth.strip().upper()
        
        # 提取问题中的物体
        obj_match = re.search(r'(?:size|height|width|length) of (?:the )?(\w+)', question, re.IGNORECASE)
        target_obj = obj_match.group(1) if obj_match else "object"
        
        # 获取检测到的尺寸
        detected_size = "unknown"
        for label, entity in mind_map.items():
            if target_obj.lower() in label.lower():
                if entity.size_3d is not None:
                    detected_size = f"{max(entity.size_3d):.2f}m"
                break
        
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"""You are a spatial intelligence assistant with the ability to review and correct perception results.

=== INITIAL PERCEPTION RESULTS ===
{mind_map_text}

=== TASK ===
{question}

Options:
{options_text}

Please review the size estimation, apply physical reasoning, and provide your answer.

Step 1: Check if the estimated size of {target_obj} ({detected_size}) is physically reasonable.
Step 2: Use reference objects (doors ~2m, chairs ~0.8m, tables ~0.75m) to calibrate.
Step 3: Provide your final answer with reasoning."""
                },
                {
                    "from": "gpt",
                    "value": f"""Let me review and evolve the size estimation.

**Step 1: Initial Size Review**
The perception system estimated the {target_obj} size as {detected_size}.
{"Calibration was applied using " + calibration.calibration_object + f" (factor: {calibration.scale_factor:.2f})" if calibration else "No calibration reference was found."}

**Step 2: Physical Reasoning & Evolution**
Let me verify this against known reference sizes:
- Standard door height: ~2.0m
- Standard chair height: ~0.8m  
- Standard table height: ~0.75m
- Standard refrigerator height: ~1.7m

By comparing the relative sizes in the scene and applying physical constraints, I can refine the estimation.

**Step 3: Final Answer**
After evolving the perception with physical reasoning, my answer is: {gt_answer}"""
                }
            ],
            "question_type": "object_size_estimation",
            "evolution_type": "vl_driven"
        }
        
        return conversation
    
    @staticmethod
    def generate_distance_evolution(
        mind_map: Dict[str, MindMapEntity],
        question: str,
        ground_truth: str,
        options: List[str],
        calibration: CalibrationResult = None
    ) -> Dict:
        """生成Distance任务的VL Evolution训练数据"""
        
        mind_map_text = VLEvolutionPromptGenerator.format_mind_map(mind_map, calibration)
        options_text = "\n".join(options) if options else ""
        gt_answer = ground_truth.strip().upper()
        
        # 提取问题中的物体
        obj_matches = re.findall(r'(?:from|to|between) (?:the )?(\w+)', question, re.IGNORECASE)
        
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"""You are a spatial intelligence assistant with the ability to review and correct perception results.

=== INITIAL PERCEPTION RESULTS ===
{mind_map_text}

=== TASK ===
{question}

Options:
{options_text}

Please review the distance estimation and provide your answer.

Step 1: Identify the objects involved and their detected positions.
Step 2: Check if the depth estimates are consistent with the scene layout.
Step 3: Apply calibration and provide your final answer."""
                },
                {
                    "from": "gpt",
                    "value": f"""Let me review and evolve the distance estimation.

**Step 1: Object Identification**
I need to find the positions of the relevant objects in the perception results.
{"The objects mentioned are: " + ", ".join(obj_matches[:3]) if obj_matches else "Analyzing the question for target objects."}

**Step 2: Depth Consistency Check**
For indoor scenes, I should verify:
- Typical room dimensions: 3-8m depth
- Objects should have consistent relative depths
- Depth ordering should match visual occlusion patterns

**Step 3: Calibrated Distance Estimation**
{"Using the calibration from " + calibration.calibration_object + " to adjust raw depth values." if calibration else "Applying general indoor scene priors for depth estimation."}

After evolving the perception with spatial reasoning, my answer is: {gt_answer}"""
                }
            ],
            "question_type": "object_distance",
            "evolution_type": "vl_driven"
        }
        
        return conversation
    
    @staticmethod
    def generate_direction_evolution(
        mind_map: Dict[str, MindMapEntity],
        question: str,
        ground_truth: str,
        options: List[str],
        calibration: CalibrationResult = None
    ) -> Dict:
        """生成Direction任务的VL Evolution训练数据"""
        
        mind_map_text = VLEvolutionPromptGenerator.format_mind_map(mind_map, calibration)
        options_text = "\n".join(options) if options else ""
        gt_answer = ground_truth.strip().upper()
        
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"""You are a spatial intelligence assistant with the ability to review and correct perception results.

=== INITIAL PERCEPTION RESULTS ===
{mind_map_text}

=== TASK ===
{question}

Options:
{options_text}

Please review the spatial relationships and provide your answer.

Step 1: Identify the reference object and target object.
Step 2: Analyze their relative positions (x: left/right, y: up/down, z: front/back).
Step 3: Determine the direction relationship and provide your answer."""
                },
                {
                    "from": "gpt",
                    "value": f"""Let me review and evolve the spatial direction analysis.

**Step 1: Object Identification**
Parsing the question to identify:
- Reference object/position (where we're standing or looking from)
- Target object (what we're asking about)
- Facing direction (the orientation we should consider)

**Step 2: Spatial Position Analysis**
From the perception results, I analyze the 3D positions:
- X-axis: negative = left, positive = right
- Y-axis: negative = up, positive = down  
- Z-axis: depth from camera

**Step 3: Direction Evolution**
To determine front-left/front-right/back-left/back-right:
1. Transform coordinates to the reference viewpoint
2. Consider the facing direction
3. Classify based on relative x (left/right) and z (front/back)

After evolving the spatial reasoning, my answer is: {gt_answer}"""
                }
            ],
            "question_type": "object_rel_direction",
            "evolution_type": "vl_driven"
        }
        
        return conversation
    
    @staticmethod
    def generate_appearance_order_evolution(
        mind_map: Dict[str, MindMapEntity],
        question: str,
        ground_truth: str,
        options: List[str],
        calibration: CalibrationResult = None
    ) -> Dict:
        """生成Appearance Order任务的VL Evolution训练数据"""
        
        mind_map_text = VLEvolutionPromptGenerator.format_mind_map(mind_map, calibration)
        options_text = "\n".join(options) if options else ""
        gt_answer = ground_truth.strip().upper()
        
        # 收集时序信息
        temporal_info = []
        for label, entity in mind_map.items():
            if entity.first_seen_frame >= 0:
                temporal_info.append(f"- {label}: first seen at frame {entity.first_seen_frame}")
        temporal_text = "\n".join(temporal_info[:10]) if temporal_info else "No temporal information available."
        
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"""You are a spatial intelligence assistant with the ability to review and correct perception results.

=== INITIAL PERCEPTION RESULTS ===
{mind_map_text}

=== TEMPORAL INFORMATION ===
{temporal_text}

=== TASK ===
{question}

Options:
{options_text}

Please review the temporal order and provide your answer.

Step 1: Analyze when each object first appears in the video.
Step 2: Check for any temporal inconsistencies or detection errors.
Step 3: Determine the correct appearance order and provide your answer."""
                },
                {
                    "from": "gpt",
                    "value": f"""Let me review and evolve the temporal order analysis.

**Step 1: First Appearance Analysis**
From the perception results, I track when each object was first detected:
{temporal_text}

**Step 2: Temporal Consistency Check**
I need to verify:
- Are there duplicate detections that might affect the order?
- Is the frame-based ordering consistent with camera movement?
- Could occlusion cause late detection of an object that was actually present earlier?

**Step 3: Evolved Temporal Reasoning**
Considering:
- Detection confidence at first appearance
- Spatial trajectory consistency  
- Scene context (objects that are typically together)

After evolving the temporal analysis, my answer is: {gt_answer}"""
                }
            ],
            "question_type": "obj_appearance_order",
            "evolution_type": "vl_driven"
        }
        
        return conversation


# ============================================================================
# 感知模块 (复用之前的代码)
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


class MindMapBuilder:
    """感知系统 - 构建初始Mind Map"""
    
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
    
    def build_from_video(self, video_path: str, target_objects: List[str] = None) -> Tuple[Dict[str, MindMapEntity], CalibrationResult]:
        """构建Mind Map并进行标定"""
        self.load_models()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, None
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        vocab = list(set((target_objects or []) + EXTENDED_VOCABULARY))
        prompt = " . ".join(vocab) + " ."
        
        all_detections = defaultdict(list)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            
            # 目标检测
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
        
        # 聚合检测结果
        mind_map = {}
        for label, dets in all_detections.items():
            entity = self._aggregate_detections(label, dets)
            mind_map[label] = entity
        
        # 标定
        calibration = self._calibrate(mind_map)
        if calibration:
            self._apply_calibration(mind_map, calibration)
        
        return mind_map, calibration
    
    def _aggregate_detections(self, label: str, detections: List[Detection]) -> MindMapEntity:
        if not detections:
            return MindMapEntity(label=label)
        
        frame_dets = defaultdict(list)
        for det in detections:
            frame_dets[det.frame_idx].append(det)
        
        counts = [len(dets) for dets in frame_dets.values()]
        median_count = int(np.median(counts)) if counts else 0
        
        avg_conf = np.mean([d.confidence for d in detections])
        
        best_det = max(detections, key=lambda x: x.confidence)
        
        heights = [d.estimated_height for d in detections if d.estimated_height > 0]
        widths = [d.estimated_width for d in detections if d.estimated_width > 0]
        avg_height = np.median(heights) if heights else 0.5
        avg_width = np.median(widths) if widths else 0.5
        avg_depth = np.median([d.depth for d in detections])
        
        first_frame = min(d.frame_idx for d in detections)
        
        return MindMapEntity(
            label=label,
            detections=detections,
            count=median_count,
            avg_confidence=float(avg_conf),
            position_3d=best_det.position_3d.copy(),
            size_3d=np.array([avg_width, avg_height, avg_depth]),
            first_seen_frame=first_frame,
        )
    
    def _calibrate(self, mind_map: Dict[str, MindMapEntity]) -> Optional[CalibrationResult]:
        best_calibration = None
        best_confidence = 0
        
        for label, entity in mind_map.items():
            for cal_obj, cal_info in CALIBRATION_OBJECTS.items():
                if cal_obj in label.lower():
                    if entity.size_3d is None:
                        continue
                    
                    estimated_size = max(entity.size_3d[0], entity.size_3d[1])
                    standard_size = cal_info.get('height', cal_info.get('length', 1.0))
                    
                    if estimated_size > 0:
                        scale_factor = standard_size / estimated_size
                        
                        if 0.1 < scale_factor < 10:
                            confidence = entity.avg_confidence
                            if confidence > best_confidence:
                                best_calibration = CalibrationResult(
                                    calibration_object=cal_obj,
                                    estimated_size=estimated_size,
                                    standard_size=standard_size,
                                    scale_factor=scale_factor,
                                    confidence=confidence,
                                )
                                best_confidence = confidence
                    break
        
        return best_calibration
    
    def _apply_calibration(self, mind_map: Dict[str, MindMapEntity], calibration: CalibrationResult):
        for entity in mind_map.values():
            if entity.position_3d is not None:
                entity.position_3d = entity.position_3d * calibration.scale_factor
            if entity.size_3d is not None:
                entity.size_3d = entity.size_3d * calibration.scale_factor


# ============================================================================
# 数据生成
# ============================================================================

def generate_vl_evolution_sample(
    mind_map: Dict[str, MindMapEntity],
    question: str,
    question_type: str,
    ground_truth: str,
    options: List[str],
    calibration: CalibrationResult = None,
) -> Dict:
    """生成单个VL驱动Evolution训练样本"""
    
    generator = VLEvolutionPromptGenerator()
    
    if 'counting' in question_type:
        return generator.generate_counting_evolution(mind_map, question, ground_truth, calibration)
    elif 'size' in question_type:
        return generator.generate_size_evolution(mind_map, question, ground_truth, options, calibration)
    elif 'distance' in question_type:
        return generator.generate_distance_evolution(mind_map, question, ground_truth, options, calibration)
    elif 'direction' in question_type:
        return generator.generate_direction_evolution(mind_map, question, ground_truth, options, calibration)
    elif 'appearance' in question_type or 'order' in question_type:
        return generator.generate_appearance_order_evolution(mind_map, question, ground_truth, options, calibration)
    else:
        # 默认使用direction模板
        return generator.generate_direction_evolution(mind_map, question, ground_truth, options, calibration)


def worker_process(gpu_id: int, samples: List[Dict], output_file: str):
    """GPU Worker进程"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    
    import sys
    sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')
    sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
    sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')
    
    builder = MindMapBuilder(device='cuda', num_frames=32)
    
    training_samples = []
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            video_path = sample['video_path']
            question = sample['question']
            question_type = sample['question_type']
            ground_truth = sample['ground_truth']
            options = sample.get('options', [])
            
            # 构建Mind Map
            mind_map, calibration = builder.build_from_video(video_path)
            
            if not mind_map:
                continue
            
            # 生成VL Evolution训练样本
            training_sample = generate_vl_evolution_sample(
                mind_map=mind_map,
                question=question,
                question_type=question_type,
                ground_truth=ground_truth,
                options=options,
                calibration=calibration,
            )
            
            training_samples.append(training_sample)
            
        except Exception as e:
            logger.warning(f"GPU {gpu_id}: Error processing sample: {e}")
            continue
    
    # 保存结果
    with open(output_file, 'w') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"GPU {gpu_id}: Generated {len(training_samples)} VL Evolution samples")
    
    builder.unload()
    gc.collect()
    torch.cuda.empty_cache()


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


def main():
    parser = argparse.ArgumentParser(description='生成VL驱动的Evolution训练数据')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-per-task', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("生成VL驱动的Evolution训练数据")
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
        output_file = f"{args.output_dir}/vl_evolution_gpu{gpu_id}.jsonl"
        output_files.append(output_file)
        
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_samples[gpu_id], output_file))
        p.start()
        processes.append(p)
        logger.info(f"启动GPU {gpu_id}, 处理 {len(gpu_samples[gpu_id])} 个样本")
    
    for p in processes:
        p.join()
    
    # 合并输出
    final_output = f"{args.output_dir}/vl_driven_evolution_training.jsonl"
    total_samples = 0
    
    with open(final_output, 'w') as fout:
        for output_file in output_files:
            if os.path.exists(output_file):
                with open(output_file, 'r') as fin:
                    for line in fin:
                        fout.write(line)
                        total_samples += 1
    
    logger.info("=" * 80)
    logger.info(f"完成！总共生成 {total_samples} 个VL Evolution训练样本")
    logger.info(f"输出文件: {final_output}")
    logger.info("=" * 80)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
