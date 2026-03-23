#!/usr/bin/env python3
"""
Self-Evolving Agent V4 - 心智地图驱动的智能决策系统

核心设计：
1. 心智地图作为中央知识库，记录物体的帧索引
2. 根据问题类型，从心智地图决定读取哪些帧
3. VL 模型只在需要时被调用（验证/演化/推理）
4. 规则优先：counting/size 直接从心智地图读取

架构：
┌─────────────────────────────────────────────────────────────────────────┐
│                         Self-Evolving Agent V4                          │
│                                                                         │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────┐ │
│  │ 感知    │ -> │ Manager │ -> │ Reasoner │ -> │ Critic  │ -> │Evolver│ │
│  │DA3+DINO │    │任务分析  │    │规则/VL   │    │一致性检查│    │VL回溯 │ │
│  └─────────┘    └─────────┘    └──────────┘    └─────────┘    └──────┘ │
│       │              │              │               │             │     │
│       v              v              v               v             v     │
│  MindMapV5 ──> 提取策略 ──> 推理答案 ──> 验证/演化 ──> 更新地图   │
└─────────────────────────────────────────────────────────────────────────┘

心智地图记录：
- 每个物体的 count, position, size
- 物体出现的帧索引 (frame_indices)
- 每个检测的 bbox (用于回溯验证)
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

# 从 DirectQA 导入关键组件
from tests.test_vsibench_directqa import (
    EXTENDED_VOCABULARY, SYNONYM_MAP, 
    get_synonyms, match_object_name,
    mean_relative_accuracy, normalize_number,
    VIDEO_DIRS, find_video_path,
    DirectQA,  # 导入 DirectQA 类用于规则推理
)

# ============================================================================
# 数据结构 - 增强版心智地图
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
class MindMapEntityV5:
    """增强版心智地图实体 - 记录所有检测实例"""
    label: str
    detections: List[Detection] = field(default_factory=list)  # 所有检测
    count: int = 0  # 聚合后的数量
    avg_confidence: float = 0.0
    position_3d: np.ndarray = None  # 聚合后的位置
    size_3d: np.ndarray = None  # 聚合后的尺寸
    
    def get_frame_indices(self) -> List[int]:
        """获取所有出现的帧索引"""
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
    
    def get_detection_at_frame(self, frame_idx: int) -> Optional[Detection]:
        """获取指定帧的检测"""
        for d in self.detections:
            if d.frame_idx == frame_idx:
                return d
        return None
    
    def to_text(self) -> str:
        """转换为文本描述"""
        text = f"- {self.label}: count={self.count}"
        if self.position_3d is not None:
            pos = self.position_3d
            text += f", position=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m"
        if self.size_3d is not None:
            size = self.size_3d * 100  # 转厘米
            text += f", size≈{max(size):.0f}cm"
        text += f", confidence={self.avg_confidence:.2f}"
        text += f", frames={len(self.get_frame_indices())}"
        return text


@dataclass
class CriticFeedback:
    """Critic 反馈"""
    is_confident: bool
    issues: List[str]
    entities_to_verify: List[str]
    frames_to_check: List[int]  # 建议检查的帧
    suggested_action: str  # "accept", "verify", "recount"
    reasoning: str


# ============================================================================
# 配置（从 DirectQA 导入）
# ============================================================================

# 任务类型分类 - 扩展规则任务范围
RULE_BASED_TASKS = [
    'object_counting', 'object_size_estimation', 'room_size_estimation',
    'object_abs_distance',  # 距离可以用规则计算
    'obj_appearance_order',  # 出现顺序可以用 first_seen_frame
]
VL_REASONING_TASKS = [
    'object_rel_direction_easy', 'object_rel_direction_medium', 
    'object_rel_direction_hard', 'object_rel_distance',
    'route_planning',  # 只有这些真正需要 VL 推理
]

NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 
                   'room_size_estimation', 'object_abs_distance']


def normalize_choice_answer(pred: str, gt: str, options: List[str] = None) -> Tuple[str, str]:
    pred = pred.strip()
    gt = gt.strip()
    
    if len(gt) == 1 and gt.upper() in 'ABCD':
        gt_label = gt.upper()
        if len(pred) == 1 and pred.upper() in 'ABCD':
            return pred.upper(), gt_label
        if len(pred) >= 2 and pred[0].upper() in 'ABCD' and pred[1] in '.、':
            return pred[0].upper(), gt_label
        if options:
            pred_lower = pred.lower()
            for i, opt in enumerate(options):
                opt_content = opt.lower()
                if len(opt) >= 3 and opt[1] in '.、':
                    opt_content = opt[3:].strip().lower()
                if pred_lower in opt_content or opt_content in pred_lower:
                    return chr(65 + i), gt_label
        return pred.upper() if len(pred) == 1 else pred, gt_label
    return pred.lower(), gt.lower()


def evaluate_answer(pred: str, gt: str, question_type: str, options: List[str] = None) -> Tuple[float, bool]:
    if question_type in NUMERICAL_TASKS:
        pred_val = normalize_number(pred)
        gt_val = normalize_number(gt)
        if pred_val is None or gt_val is None:
            return 0.0, False
        score = mean_relative_accuracy(pred_val, gt_val)
        return score, score > 0.5
    else:
        pred_norm, gt_norm = normalize_choice_answer(pred, gt, options)
        correct = pred_norm == gt_norm
        return float(correct), correct


# ============================================================================
# 1. Perception - 感知模块 (构建心智地图)
# ============================================================================

class MindMapBuilderV5:
    """心智地图构建器 V5 - 记录详细的检测信息"""
    
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
    
    def build_from_video(
        self, 
        video_path: str, 
        target_objects: List[str] = None
    ) -> Tuple[Dict[str, MindMapEntityV5], int]:
        """从视频构建心智地图，返回 (mind_map, total_frames)"""
        self.load_models()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, 0
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        # 使用完整的扩展词汇表（与 DirectQA 一致）
        if target_objects is None:
            target_objects = []
        vocab = list(set(target_objects + EXTENDED_VOCABULARY))
        prompt = " . ".join(vocab) + " ."
        
        # 收集所有检测
        all_detections = defaultdict(list)  # label -> [Detection]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            
            # 深度估计 - 使用 infer_single 方法，返回 (batch_tensor, single_tensor)
            depth_result = self._depth_estimator.infer_single(frame_rgb)
            depth_map = depth_result[1].cpu().numpy() if isinstance(depth_result, tuple) else depth_result.cpu().numpy()
            
            # 确保深度图尺寸与原始图像匹配
            if depth_map.shape[0] != h or depth_map.shape[1] != w:
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 物体检测
            detections = self._labeler.detect(frame_rgb, prompt)
            
            for det in detections:
                raw_label = det.label.strip().lower()
                
                # 跳过 tokenizer 残留 (以 ## 开头)
                if raw_label.startswith('##'):
                    continue
                
                # 保持与 DirectQA 一致：只做基本清理
                label = raw_label
                
                box = det.bbox_pixels
                conf = det.confidence
                
                # 计算中心点深度
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                cx = min(max(cx, 0), depth_map.shape[1] - 1)
                cy = min(max(cy, 0), depth_map.shape[0] - 1)
                depth = float(depth_map[cy, cx])
                
                # 计算 3D 位置
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
        
        # 聚合成心智地图实体
        mind_map = {}
        for label, dets in all_detections.items():
            entity = self._aggregate_detections(label, dets)
            mind_map[label] = entity
        
        return mind_map, total_frames
    
    def _aggregate_detections(self, label: str, detections: List[Detection]) -> MindMapEntityV5:
        """聚合检测结果 - 使用最大帧计数法"""
        if not detections:
            return MindMapEntityV5(label=label)
        
        # 按帧分组（每帧可能有多个同类物体）
        frame_dets = defaultdict(list)
        for det in detections:
            frame_dets[det.frame_idx].append(det)
        
        # 取所有帧中检测到的最大数量作为 count
        max_count = max(len(fd) for fd in frame_dets.values())
        
        # 计算平均置信度
        avg_conf = np.mean([d.confidence for d in detections])
        
        # 聚合 3D 位置
        positions = np.array([d.position_3d for d in detections])
        avg_pos = np.median(positions, axis=0)
        
        # 估计尺寸（从最高置信度的检测）
        best_det = max(detections, key=lambda x: x.confidence)
        box = best_det.bbox
        # 假设视频宽度约 640 像素
        box_w = (box[2] - box[0]) / 640
        box_h = (box[3] - box[1]) / 480
        size_3d = np.array([box_w * best_det.depth, box_h * best_det.depth, 0.3])
        
        return MindMapEntityV5(
            label=label,
            detections=detections,  # 保留所有检测
            count=max_count,
            avg_confidence=float(avg_conf),
            position_3d=avg_pos,
            size_3d=size_3d
        )


# ============================================================================
# 2. Task Manager - 任务分析
# ============================================================================

class TaskManager:
    """任务管理器 - 分析问题并决定策略"""
    
    def __init__(self):
        self.task_patterns = {
            'counting': [r'how many', r'count', r'number of'],
            'size': [r'how (big|large|small|tall|wide)', r'size', r'dimension'],
            'direction': [r'left', r'right', r'front', r'behind', r'above', r'below'],
            'distance': [r'how far', r'distance', r'close', r'near'],
            'appearance': [r'which.*first', r'order', r'appear'],
            'route': [r'route', r'path', r'way to'],
        }
    
    def analyze_task(
        self, 
        question: str, 
        question_type: str,
        mind_map: Dict[str, MindMapEntityV5]
    ) -> Dict[str, Any]:
        """分析任务，返回执行策略"""
        question_lower = question.lower()
        
        # 提取问题中涉及的物体
        target_objects = self._extract_target_objects(question, mind_map)
        
        # 根据问题类型决定策略
        if question_type in RULE_BASED_TASKS:
            strategy = "rule_based"
            frames_needed = []  # 规则推理不需要额外帧
        else:
            strategy = "vl_reasoning"
            # 从心智地图中找到相关物体的帧
            frames_needed = self._get_relevant_frames(target_objects, mind_map)
        
        return {
            "strategy": strategy,
            "target_objects": target_objects,
            "frames_needed": frames_needed,
            "question_type": question_type,
        }
    
    def _extract_target_objects(self, question: str, mind_map: Dict[str, MindMapEntityV5]) -> List[str]:
        """从问题中提取目标物体"""
        targets = []
        question_lower = question.lower()
        
        # 检查心智地图中的物体是否在问题中提及
        for label in mind_map.keys():
            clean_label = label.replace('_', ' ')
            if clean_label in question_lower or label in question_lower:
                targets.append(label)
        
        # 额外模式匹配
        patterns = [
            r'how many (\w+)',
            r'the (\w+) is',
            r'(\w+) and (\w+)',
            r'between the (\w+)',
            r'from the (\w+)',
            r'to the (\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question_lower)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        for label in mind_map.keys():
                            if match_object_name(m, label):
                                targets.append(label)
                else:
                    for label in mind_map.keys():
                        if match_object_name(match, label):
                            targets.append(label)
        
        return list(set(targets))
    
    def _get_relevant_frames(
        self, 
        target_objects: List[str], 
        mind_map: Dict[str, MindMapEntityV5],
        max_frames: int = 8
    ) -> List[int]:
        """获取与目标物体相关的帧"""
        all_frames = set()
        
        for obj in target_objects:
            if obj in mind_map:
                # 获取该物体置信度最高的帧
                best_frames = mind_map[obj].get_best_frames(n=3)
                all_frames.update(best_frames)
        
        # 如果没有找到，返回所有实体的最佳帧
        if not all_frames:
            for entity in mind_map.values():
                all_frames.update(entity.get_best_frames(n=2))
        
        # 限制帧数
        frames_list = sorted(all_frames)
        if len(frames_list) > max_frames:
            # 均匀采样
            indices = np.linspace(0, len(frames_list) - 1, max_frames, dtype=int)
            frames_list = [frames_list[i] for i in indices]
        
        return frames_list


# ============================================================================
# 3. Reasoner - 推理模块
# ============================================================================

class Reasoner:
    """推理器 - 支持规则推理和 VL 推理"""
    
    def __init__(self, vl_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = 'cuda'):
        self.vl_model_name = vl_model_name
        self.device = device
        self.model = None
        self.processor = None
    
    def load_vl_model(self):
        if self.model is None:
            from transformers import AutoProcessor
            
            logger.info(f"加载 VL 模型: {self.vl_model_name}")
            
            # 根据模型名选择正确的类
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
            logger.info("VL 模型加载完成")
    
    def unload_vl_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def reason(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntityV5],
        question: str,
        question_type: str,
        options: List[str] = None,
        task_info: Dict[str, Any] = None,
        total_frames: int = 0
    ) -> Tuple[str, float, str]:
        """执行推理"""
        strategy = task_info.get("strategy", "rule_based") if task_info else "rule_based"
        
        if strategy == "rule_based":
            return self._rule_based_reason(mind_map, question, question_type, options)
        else:
            frames_needed = task_info.get("frames_needed", []) if task_info else []
            return self._vl_reason(video_path, mind_map, question, question_type, 
                                   options, frames_needed, total_frames)
    
    def _rule_based_reason(
        self,
        mind_map: Dict[str, MindMapEntityV5],
        question: str,
        question_type: str,
        options: List[str] = None
    ) -> Tuple[str, float, str]:
        """规则推理 - 直接从心智地图读取"""
        
        if question_type == 'object_counting':
            return self._answer_counting(mind_map, question)
        elif question_type == 'object_size_estimation':
            return self._answer_size(mind_map, question)
        elif question_type == 'room_size_estimation':
            return self._answer_room_size(mind_map, question)
        elif question_type == 'object_abs_distance':
            return self._answer_abs_distance(mind_map, question)
        elif question_type == 'obj_appearance_order':
            return self._answer_appearance_order(mind_map, question, options)
        else:
            return "unknown", 0.3, "Unsupported rule-based task"
    
    def _answer_counting(self, mind_map: Dict[str, MindMapEntityV5], question: str) -> Tuple[str, float, str]:
        """回答计数问题 - 合并所有匹配标签的计数"""
        match = re.search(r'[Hh]ow many (\w+)', question)
        if not match:
            return "0", 0.3, "Cannot parse counting question"
        
        target = match.group(1).lower()
        
        # 找到所有匹配的标签
        matched_labels = []
        for label, entity in mind_map.items():
            if match_object_name(target, label):
                matched_labels.append((label, entity))
        
        if not matched_labels:
            return "0", 0.5, f"Object '{target}' not found in mind map"
        
        # 策略：取最大计数（因为不同标签可能指向同一物体）
        # 例如 chair:5, armchair_chair:3 => 取 5
        best_match = max(matched_labels, key=lambda x: x[1].count)
        label, entity = best_match
        
        reasoning = f"Found {entity.count} {label}(s) in {len(entity.get_frame_indices())} frames"
        return str(entity.count), entity.avg_confidence, reasoning
    
    def _answer_size(self, mind_map: Dict[str, MindMapEntityV5], question: str) -> Tuple[str, float, str]:
        """回答尺寸问题"""
        for label, entity in mind_map.items():
            if label in question.lower() or label.replace('_', ' ') in question.lower():
                if entity.size_3d is not None:
                    max_size = max(entity.size_3d) * 100  # 转厘米
                    return f"{max_size:.0f}", entity.avg_confidence, f"Size of {label}"
        
        return "50", 0.3, "Default size estimate"
    
    def _answer_room_size(self, mind_map: Dict[str, MindMapEntityV5], question: str) -> Tuple[str, float, str]:
        """回答房间尺寸问题"""
        if not mind_map:
            return "25", 0.3, "No objects to estimate room size"
        
        all_positions = []
        for entity in mind_map.values():
            if entity.position_3d is not None:
                all_positions.append(entity.position_3d)
        
        if len(all_positions) < 2:
            return "25", 0.3, "Not enough objects"
        
        positions = np.array(all_positions)
        x_range = np.ptp(positions[:, 0])
        y_range = np.ptp(positions[:, 1])
        area = max(x_range * y_range * 1.5, 10)  # 至少 10 平米
        
        return f"{area:.0f}", 0.6, f"Estimated from {len(all_positions)} objects"
    
    def _answer_abs_distance(self, mind_map: Dict[str, MindMapEntityV5], question: str) -> Tuple[str, float, str]:
        """回答物体间距离问题"""
        q_lower = question.lower()
        
        # 从问题中提取两个物体
        between_match = re.search(r'between (?:the )?([a-z]+(?:\s+[a-z]+)*)\s+and\s+(?:the )?([a-z]+(?:\s+[a-z]+)*)', q_lower)
        
        if between_match:
            obj1_name = between_match.group(1).strip()
            obj2_name = between_match.group(2).strip()
        else:
            return "2.0", 0.3, "Cannot parse distance question"
        
        # 查找物体位置
        def find_position(name: str):
            name_lower = name.lower().strip()
            for label, entity in mind_map.items():
                if match_object_name(name_lower, label) and entity.position_3d is not None:
                    return entity.position_3d
            return None
        
        pos1 = find_position(obj1_name)
        pos2 = find_position(obj2_name)
        
        if pos1 is not None and pos2 is not None:
            distance = np.linalg.norm(pos1 - pos2)
            return f"{distance:.2f}", 0.7, f"Distance between {obj1_name} and {obj2_name}"
        
        return "2.0", 0.3, f"Cannot find positions for {obj1_name} and {obj2_name}"
    
    def _answer_appearance_order(self, mind_map: Dict[str, MindMapEntityV5], question: str, options: List[str]) -> Tuple[str, float, str]:
        """回答出现顺序问题 - 使用 first_seen_frame"""
        if not options:
            return "A", 0.3, "No options provided"
        
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
        
        if len(sorted_objects) < 2:
            return options[0], 0.3, "Not enough objects found"
        
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
                    return chr(65 + i), 0.7, f"Order matched: {sorted_objects}"
        
        return options[0], 0.4, f"Best guess based on: {sorted_objects}"
    
    def _vl_reason(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntityV5],
        question: str,
        question_type: str,
        options: List[str],
        frames_needed: List[int],
        total_frames: int
    ) -> Tuple[str, float, str]:
        """VL 推理 - 只读取相关帧"""
        self.load_vl_model()
        
        # 读取指定帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not frames_needed:
            # 如果没有指定帧，均匀采样
            if total_frames > 0:
                frames_needed = list(np.linspace(0, total_frames - 1, 8, dtype=int))
            else:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_needed = list(np.linspace(0, total - 1, 8, dtype=int))
        
        for frame_idx in frames_needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        
        if not frames:
            return "A", 0.2, "Cannot read frames"
        
        # 构建心智地图描述
        map_text = self._mind_map_to_text(mind_map)
        
        # 构建提示
        options_text = ""
        if options:
            options_text = "\nOptions:\n" + "\n".join(options)
        
        if question_type in NUMERICAL_TASKS:
            answer_format = "Answer with a NUMBER only."
        else:
            answer_format = "Answer with the OPTION LETTER only (A, B, C, or D)."
        
        prompt = f"""You are analyzing a scene. Use the detected object information AND the video frames to answer.

{map_text}

Question: {question}{options_text}

IMPORTANT: Base your answer on BOTH the mind map data AND what you see in the frames.
{answer_format}
Add [Confidence: X%] at the end.

Answer:"""
        
        # 使用 video 输入
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "nframes": min(8, len(frames_needed)),
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
            return "A", 0.2, str(e)
    
    def _mind_map_to_text(self, mind_map: Dict[str, MindMapEntityV5]) -> str:
        if not mind_map:
            return "No objects detected."
        
        lines = ["Detected Objects:"]
        for label, entity in sorted(mind_map.items(), key=lambda x: -x[1].count):
            lines.append(entity.to_text())
        return "\n".join(lines)
    
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
# 4. Critic - 验证模块
# ============================================================================

class Critic:
    """验证器 - 检查答案是否可靠，决定是否需要演化"""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
    
    def evaluate(
        self,
        answer: str,
        confidence: float,
        reasoning: str,
        mind_map: Dict[str, MindMapEntityV5],
        question: str,
        question_type: str,
        options: List[str] = None
    ) -> CriticFeedback:
        """评估答案，返回反馈
        
        优化策略：
        - 高置信度结果直接接受（避免过度演化）
        - 只对明显可疑的结果进行演化
        """
        issues = []
        entities_to_verify = []
        frames_to_check = []
        
        # 规则任务（counting/size/room_size）优先信任检测结果
        is_rule_task = question_type in RULE_BASED_TASKS
        
        # 1. 检查置信度 - 规则任务放宽要求
        conf_threshold = 0.4 if is_rule_task else self.confidence_threshold
        if confidence < conf_threshold:
            issues.append(f"Low confidence: {confidence:.2f}")
        
        # 2. 检查心智地图中的相关实体
        target_objects = self._extract_targets(question, mind_map)
        
        for obj in target_objects:
            if obj in mind_map:
                entity = mind_map[obj]
                
                # 只有检测非常不稳定时才标记
                frame_count = len(entity.get_frame_indices())
                if frame_count < 2 and entity.avg_confidence < 0.4:
                    issues.append(f"Object '{obj}' unreliable: {frame_count} frames, conf={entity.avg_confidence:.2f}")
                    entities_to_verify.append(obj)
                    frames_to_check.extend(entity.get_best_frames(3))
            else:
                # 物体完全未检测到 - 这是真正需要演化的情况
                issues.append(f"Object '{obj}' not found in mind map")
                entities_to_verify.append(obj)
        
        # 3. 检查计数问题的合理性
        if question_type == 'object_counting':
            count = normalize_number(answer)
            if count is not None:
                if count == 0 and target_objects:
                    issues.append(f"Count is 0 but question asks about {target_objects}")
                elif count > 30:  # 放宽阈值
                    issues.append(f"Unusually high count: {count}")
        
        # 决定是否需要演化 - 更保守的策略
        # 只有当有严重问题时才演化
        serious_issues = [i for i in issues if 'not found' in i or 'is 0' in i]
        is_confident = len(serious_issues) == 0 and confidence >= conf_threshold
        
        if not is_confident and entities_to_verify:
            action = "verify"
        elif not is_confident and serious_issues:
            action = "recount"
        else:
            action = "accept"
        
        return CriticFeedback(
            is_confident=is_confident,
            issues=issues,
            entities_to_verify=list(set(entities_to_verify)),
            frames_to_check=list(set(frames_to_check)),
            suggested_action=action,
            reasoning="; ".join(issues) if issues else "Answer looks reliable"
        )
    
    def _extract_targets(self, question: str, mind_map: Dict[str, MindMapEntityV5]) -> List[str]:
        """提取问题中的目标物体"""
        targets = []
        question_lower = question.lower()
        
        for label in mind_map.keys():
            if label in question_lower or label.replace('_', ' ') in question_lower:
                targets.append(label)
        
        # 常见模式
        match = re.search(r'how many (\w+)', question_lower)
        if match:
            target = match.group(1)
            for label in mind_map.keys():
                if match_object_name(target, label):
                    targets.append(label)
        
        return list(set(targets))


# ============================================================================
# 5. Evolver - 演化模块
# ============================================================================

class Evolver:
    """演化器 - 使用 VL 回溯验证并更新心智地图"""
    
    def __init__(self, vl_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = 'cuda'):
        self.vl_model_name = vl_model_name
        self.device = device
        self.model = None
        self.processor = None
    
    def load_model(self):
        if self.model is None:
            from transformers import AutoProcessor
            
            logger.info(f"加载演化 VL 模型: {self.vl_model_name}")
            
            # 根据模型名选择正确的类
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
    
    def evolve(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntityV5],
        feedback: CriticFeedback,
        question: str,
        question_type: str,
        total_frames: int
    ) -> Tuple[Dict[str, MindMapEntityV5], str, float]:
        """
        演化心智地图
        
        Returns:
            updated_mind_map: 更新后的心智地图
            new_answer: 新的答案
            new_confidence: 新的置信度
        """
        self.load_model()
        
        # 确定要检查的帧
        frames_to_check = feedback.frames_to_check
        if not frames_to_check:
            # 如果没有指定，检查相关物体的帧
            for obj in feedback.entities_to_verify:
                if obj in mind_map:
                    frames_to_check.extend(mind_map[obj].get_best_frames(3))
        
        # 去重并排序
        frames_to_check = sorted(set(frames_to_check))
        
        if not frames_to_check:
            # 均匀采样
            frames_to_check = list(np.linspace(0, total_frames - 1, 8, dtype=int))
        
        # 读取帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_indices = []
        
        for frame_idx in frames_to_check[:8]:  # 最多 8 帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                frame_indices.append(frame_idx)
        cap.release()
        
        if not frames:
            return mind_map, "0", 0.3
        
        # 构建验证提示
        entities_info = []
        for obj in feedback.entities_to_verify:
            if obj in mind_map:
                entity = mind_map[obj]
                entities_info.append(f"- {obj}: detected {entity.count} times, confidence={entity.avg_confidence:.2f}")
            else:
                entities_info.append(f"- {obj}: NOT detected")
        
        verification_prompt = f"""You are verifying object detection results. Look at these frames carefully.

Current detection results:
{chr(10).join(entities_info)}

Issues found: {'; '.join(feedback.issues)}

Question being answered: {question}

Please verify:
1. Are the detected counts correct?
2. Are there any missed objects?
3. What is the correct count for each object mentioned?

Respond in this format:
OBJECT: <name> | COUNT: <number> | CONFIDENCE: <0-100>%
...
FINAL_ANSWER: <your answer to the question>
"""
        
        # VL 验证
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "nframes": len(frames),
                },
                {"type": "text", "text": verification_prompt}
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
            
            # 解析响应并更新心智地图
            return self._parse_and_update(response, mind_map, question_type)
            
        except Exception as e:
            logger.warning(f"演化失败: {e}")
            return mind_map, "0", 0.3
    
    def _parse_and_update(
        self, 
        response: str, 
        mind_map: Dict[str, MindMapEntityV5],
        question_type: str
    ) -> Tuple[Dict[str, MindMapEntityV5], str, float]:
        """解析 VL 响应并更新心智地图"""
        updated_map = {k: MindMapEntityV5(
            label=v.label,
            detections=v.detections.copy(),
            count=v.count,
            avg_confidence=v.avg_confidence,
            position_3d=v.position_3d.copy() if v.position_3d is not None else None,
            size_3d=v.size_3d.copy() if v.size_3d is not None else None,
        ) for k, v in mind_map.items()}
        
        # 解析 OBJECT: xxx | COUNT: xxx 格式
        object_pattern = r'OBJECT:\s*(\w+)\s*\|\s*COUNT:\s*(\d+)\s*\|\s*CONFIDENCE:\s*(\d+)%?'
        matches = re.findall(object_pattern, response, re.IGNORECASE)
        
        for match in matches:
            obj_name = match[0].lower()
            new_count = int(match[1])
            new_conf = float(match[2]) / 100
            
            # 查找匹配的实体
            for label in updated_map.keys():
                if match_object_name(obj_name, label):
                    updated_map[label].count = new_count
                    updated_map[label].avg_confidence = new_conf
                    break
        
        # 提取最终答案
        final_match = re.search(r'FINAL_ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if final_match:
            answer_text = final_match.group(1).strip()
            if question_type in NUMERICAL_TASKS:
                num_match = re.search(r'(\d+)', answer_text)
                new_answer = num_match.group(1) if num_match else None
            else:
                choice_match = re.search(r'([A-D])', answer_text.upper())
                new_answer = choice_match.group(1) if choice_match else None
        else:
            # 没有找到 FINAL_ANSWER，返回 None 表示解析失败
            new_answer = None
        
        # 计算新置信度
        if matches:
            new_confidence = np.mean([float(m[2]) / 100 for m in matches])
        else:
            new_confidence = 0.0  # 解析失败，置信度为 0
        
        return updated_map, new_answer, new_confidence


# ============================================================================
# 6. Self-Evolving Agent - 整合
# ============================================================================

class SelfEvolvingAgentV4:
    """自演化智能体 V4 - 心智地图驱动"""
    
    def __init__(
        self,
        device: str = 'cuda',
        vl_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        num_frames: int = 32,
        max_evolution_rounds: int = 1,
        confidence_threshold: float = 0.6
    ):
        self.device = device
        self.vl_model_name = vl_model_name
        self.num_frames = num_frames
        self.max_evolution_rounds = max_evolution_rounds
        
        # 初始化组件
        self.perception = MindMapBuilderV5(device=device, num_frames=num_frames)
        self.task_manager = TaskManager()
        self.reasoner = Reasoner(vl_model_name=vl_model_name, device=device)
        self.critic = Critic(confidence_threshold=confidence_threshold)
        self.evolver = Evolver(vl_model_name=vl_model_name, device=device)
    
    def process(
        self,
        video_path: str,
        question: str,
        question_type: str,
        options: List[str] = None,
    ) -> Dict[str, Any]:
        """
        处理单个样本的完整流程
        
        Returns:
            {
                "answer": str,
                "confidence": float,
                "evolved": bool,
                "evolution_rounds": int,
                "reasoning": str,
                "mind_map_summary": str,
            }
        """
        # 1. 感知 - 构建心智地图
        mind_map, total_frames = self.perception.build_from_video(video_path)
        
        if not mind_map:
            return {
                "answer": "0" if question_type in NUMERICAL_TASKS else "A",
                "confidence": 0.1,
                "evolved": False,
                "evolution_rounds": 0,
                "reasoning": "Failed to build mind map",
                "mind_map_summary": "Empty",
            }
        
        # 2. 任务分析
        task_info = self.task_manager.analyze_task(question, question_type, mind_map)
        
        # 3. 初次推理
        answer, confidence, reasoning = self.reasoner.reason(
            video_path, mind_map, question, question_type, options, task_info, total_frames
        )
        
        # 4. Critic 评估
        feedback = self.critic.evaluate(
            answer, confidence, reasoning, mind_map, question, question_type, options
        )
        
        evolved = False
        evolution_rounds = 0
        
        # 5. 演化循环
        original_mind_map = mind_map  # 保留原始心智地图
        while not feedback.is_confident and evolution_rounds < self.max_evolution_rounds:
            if feedback.suggested_action in ["verify", "recount"]:
                evolved = True
                evolution_rounds += 1
                
                # 演化心智地图 - 返回更新后的版本
                evolved_mind_map, new_answer, new_confidence = self.evolver.evolve(
                    video_path, mind_map, feedback, question, question_type, total_frames
                )
                
                # 只有当演化成功（置信度 > 0）时才采用新的心智地图
                if new_confidence > 0.3 and new_answer is not None:
                    mind_map = evolved_mind_map
                    
                    # 重新推理（使用更新后的心智地图）
                    task_info = self.task_manager.analyze_task(question, question_type, mind_map)
                    answer, confidence, reasoning = self.reasoner.reason(
                        video_path, mind_map, question, question_type, options, task_info, total_frames
                    )
                    
                    # 如果演化给出的答案置信度更高，使用它
                    if new_confidence > confidence:
                        answer = new_answer
                        confidence = new_confidence
                else:
                    # 演化失败，保留原始结果，跳出循环
                    mind_map = original_mind_map
                    break
                
                # 重新评估
                feedback = self.critic.evaluate(
                    answer, confidence, reasoning, mind_map, question, question_type, options
                )
            else:
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
        }
    
    def unload_all(self):
        """释放所有模型"""
        self.perception.unload()
        self.reasoner.unload_vl_model()
        self.evolver.unload()
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
    
    agent = SelfEvolvingAgentV4(
        device=device,
        vl_model_name=vl_model_name,
        num_frames=num_frames,
        max_evolution_rounds=max_evolution_rounds,
    )
    
    results = []
    
    # 使用 tqdm 显示进度
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
            })
            
        except Exception as e:
            logger.error(f"处理失败 {sample['scene_name']}: {e}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--vl-model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--max-evolution-rounds', type=int, default=1)
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
    logger.info("=== 最终结果 ===")
    
    # 按任务类型统计
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r['question_type']].append(r)
    
    total_score = 0
    total_count = 0
    evolved_count = 0
    evolved_improved = 0
    
    for qtype, results in sorted(by_type.items()):
        scores = [r['score'] for r in results]
        evolved = [r for r in results if r.get('evolved', False)]
        
        avg_score = np.mean(scores) if scores else 0
        total_score += sum(scores)
        total_count += len(scores)
        evolved_count += len(evolved)
        
        logger.info(f"{qtype:35}: {avg_score*100:6.2f}% MRA ({len(results)} 样本, {len(evolved)} 演化)")
    
    overall = total_score / total_count * 100 if total_count > 0 else 0
    logger.info(f"\n{'总体':35}: {overall:6.2f}% MRA ({total_count} 样本)")
    if total_count > 0:
        logger.info(f"演化触发: {evolved_count}/{total_count} ({evolved_count/total_count*100:.1f}%)")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"evolving_agent_v4_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
