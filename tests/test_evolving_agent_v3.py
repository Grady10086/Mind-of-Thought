#!/usr/bin/env python3
"""
Self-Evolving Agent V3 - 真正的 VL 驱动演化系统

架构:
┌─────────────────────────────────────────────────────────────────────────┐
│                         Self-Evolving Agent Loop                        │
│                                                                         │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────┐ │
│  │ 感知    │ -> │ Manager │ -> │ Reasoner │ -> │ Critic  │ -> │Evolver│ │
│  │DA3+DINO │    │任务分析  │    │  VL推理  │    │一致性检查│    │VL回溯 │ │
│  └─────────┘    └─────────┘    └──────────┘    └─────────┘    └──────┘ │
│       │              │              │               │             │     │
│       v              v              v               v             v     │
│  MindMapV5 ──> 提取策略 ──> 推理答案 ──> 验证/演化 ──> 更新地图   │
└─────────────────────────────────────────────────────────────────────────┘

核心改进:
1. VL 模型作为大脑进行推理（不是规则）
2. Critic 分析心智地图与问题的一致性
3. Evolver 通过 VL 回溯视频修正地图
"""

import os
import sys
import json
import argparse
import logging
import re
import time
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image

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
# 数据结构
# ============================================================================

@dataclass
class MindMapEntity3D:
    """心智地图实体"""
    label: str
    count: int = 1
    position_3d: np.ndarray = None
    size_3d: np.ndarray = None
    avg_confidence: float = 0.5
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    frame_indices: List[int] = field(default_factory=list)
    
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
        return text


@dataclass
class CriticFeedback:
    """Critic 反馈"""
    is_confident: bool
    issues: List[str]
    entities_to_verify: List[str]
    suggested_action: str  # "accept", "verify"
    reasoning: str


# ============================================================================
# 视频路径配置
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


# ============================================================================
# 同义词映射
# ============================================================================

SYNONYM_MAP = {
    'couch': ['sofa', 'loveseat', 'settee'],
    'sofa': ['couch', 'loveseat', 'settee'],
    'tv': ['television', 'monitor', 'screen', 'tv_monitor'],
    'television': ['tv', 'monitor', 'screen'],
    'fridge': ['refrigerator', 'freezer'],
    'refrigerator': ['fridge', 'freezer'],
    'desk': ['table', 'worktable', 'computer_desk'],
    'table': ['desk', 'dining_table', 'coffee_table'],
    'chair': ['seat', 'armchair', 'office_chair'],
    'lamp': ['light', 'lighting', 'floor_lamp', 'table_lamp'],
    'cabinet': ['cupboard', 'closet', 'wardrobe', 'storage'],
    'bed': ['mattress', 'bedframe'],
    'plant': ['flower', 'potted_plant', 'houseplant'],
    'picture': ['painting', 'photo', 'artwork', 'poster'],
    'curtain': ['drape', 'blind', 'window_covering'],
    'rug': ['carpet', 'mat', 'floor_covering'],
    'stool': ['bar_stool', 'step_stool', 'footstool'],
    'washer': ['washing_machine', 'laundry_machine'],
    'dryer': ['clothes_dryer', 'tumble_dryer'],
}


def get_synonyms(word: str) -> List[str]:
    word = word.lower().replace(' ', '_')
    synonyms = [word]
    if word in SYNONYM_MAP:
        synonyms.extend(SYNONYM_MAP[word])
    return synonyms


def match_object_name(target: str, label: str) -> bool:
    target = target.lower().replace(' ', '_')
    label = label.lower().replace(' ', '_')
    if target in label or label in target:
        return True
    for syn in get_synonyms(target):
        if syn in label or label in syn:
            return True
    return False


# ============================================================================
# 评估工具
# ============================================================================

NUMERICAL_TASKS = [
    'object_counting', 'object_size_estimation', 
    'room_size_estimation', 'object_abs_distance'
]

CHOICE_TASKS = [
    'object_rel_direction_easy', 'object_rel_direction_medium', 
    'object_rel_direction_hard', 'object_rel_distance',
    'obj_appearance_order', 'route_planning'
]


def normalize_number(text: str) -> Optional[float]:
    if text is None:
        return None
    text = str(text).strip()
    match = re.search(r'[-+]?\d*\.?\d+', text)
    if match:
        return float(match.group())
    return None


def mean_relative_accuracy(pred: float, gt: float, epsilon: float = 1e-6) -> float:
    if abs(gt) < epsilon:
        return 1.0 if abs(pred) < epsilon else 0.0
    return max(0.0, min(1.0, 1.0 - abs(pred - gt) / max(abs(gt), epsilon)))


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
# MindMap Builder - 复用 DirectQA 的组件
# ============================================================================

class MindMapBuilder:
    """心智地图构建器 - 使用 DA3 + GroundingDINO"""
    
    def __init__(self, device: str = 'cuda', num_frames: int = 32, box_threshold: float = 0.25):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        
        self._labeler = None
        self._depth_estimator = None
        self.focal_length = 500
    
    def load_models(self):
        """加载模型"""
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
        """释放模型"""
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
    
    def build_from_video(self, video_path: str) -> Dict[str, MindMapEntity3D]:
        """从视频构建心智地图"""
        self.load_models()
        
        # 读取视频帧
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not all_frames:
            return {}
        
        # 均匀采样
        num_sample = min(self.num_frames, len(all_frames))
        indices = np.linspace(0, len(all_frames) - 1, num_sample).astype(int)
        frames = [all_frames[i] for i in indices]
        
        # 检测 + 深度估计
        object_stats = defaultdict(lambda: {
            "counts": [],
            "depths": [],
            "sizes": [],
            "confidences": [],
            "first_frame": float('inf'),
            "last_frame": 0,
            "frame_indices": [],
        })
        
        text_prompt = "furniture. chair. table. sofa. bed. lamp. cabinet. tv. refrigerator. door. window. plant. picture. rug. curtain. shelf. desk. stool. washer. dryer. toilet. sink. bathtub. mirror. fan. heater. microwave. oven. dishwasher. clock. vase. book. pillow. blanket. towel."
        
        for frame_idx, (frame, real_idx) in enumerate(zip(frames, indices)):
            try:
                # 检测
                detections = self._labeler.detect(frame, text_prompt)
                
                # 深度估计
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                
                # 校准深度
                median_depth = np.median(depth_map)
                if median_depth > 0:
                    scale = 2.5 / median_depth
                    depth_map = depth_map * scale
                
                h, w = frame.shape[:2]
                
                # 统计
                label_counts = defaultdict(int)
                for det in detections:
                    # DetectionResult 是 dataclass，使用属性访问
                    label = det.label.strip().lower().replace(' ', '_')
                    box = det.bbox_pixels  # 使用像素坐标
                    conf = det.confidence
                    
                    label_counts[label] += 1
                    
                    # 获取深度
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    cx = min(max(cx, 0), depth_map.shape[1] - 1)
                    cy = min(max(cy, 0), depth_map.shape[0] - 1)
                    depth = depth_map[cy, cx]
                    
                    # 估计尺寸
                    box_w = (box[2] - box[0]) / w
                    box_h = (box[3] - box[1]) / h
                    estimated_size = np.array([box_w * depth, box_h * depth, 0.3])
                    
                    object_stats[label]["depths"].append(depth)
                    object_stats[label]["sizes"].append(estimated_size)
                    object_stats[label]["confidences"].append(conf)
                    object_stats[label]["first_frame"] = min(object_stats[label]["first_frame"], real_idx)
                    object_stats[label]["last_frame"] = max(object_stats[label]["last_frame"], real_idx)
                    object_stats[label]["frame_indices"].append(real_idx)
                
                for label, count in label_counts.items():
                    object_stats[label]["counts"].append(count)
                    
            except Exception as e:
                logger.warning(f"处理帧 {frame_idx} 失败: {e}")
                continue
        
        # 构建心智地图
        mind_map = {}
        for label, stats in object_stats.items():
            if not stats["counts"]:
                continue
            
            count = max(stats["counts"])
            avg_depth = np.mean(stats["depths"]) if stats["depths"] else 2.0
            avg_size = np.mean(stats["sizes"], axis=0) if stats["sizes"] else np.array([0.5, 0.5, 0.5])
            avg_conf = np.mean(stats["confidences"]) if stats["confidences"] else 0.5
            
            position_3d = np.array([0, 0, avg_depth])
            
            mind_map[label] = MindMapEntity3D(
                label=label,
                count=count,
                position_3d=position_3d,
                size_3d=avg_size,
                avg_confidence=avg_conf,
                first_seen_frame=int(stats["first_frame"]),
                last_seen_frame=int(stats["last_frame"]),
                frame_indices=sorted(set(stats["frame_indices"])),
            )
        
        return mind_map


# ============================================================================
# VL Reasoner - 使用 Qwen2.5-VL 推理
# ============================================================================

class VLReasoner:
    """VL 推理器 - 使用视觉语言模型回答问题"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
    
    def load_model(self):
        if self.model is not None:
            return
        
        logger.info(f"加载 VL 模型: {self.model_path}")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        logger.info("VL 模型加载完成")
    
    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def mind_map_to_text(self, mind_map: Dict[str, MindMapEntity3D]) -> str:
        if not mind_map:
            return "No objects detected in the scene."
        
        lines = ["Scene Mind Map:"]
        for label, entity in sorted(mind_map.items(), key=lambda x: -x[1].count):
            lines.append(entity.to_text())
        return "\n".join(lines)
    
    def reason(
        self,
        video_path: str,
        mind_map: Dict[str, MindMapEntity3D],
        question: str,
        question_type: str,
        options: List[str] = None,
        num_frames: int = 8
    ) -> Tuple[str, float, str]:
        """使用 VL 模型进行推理"""
        self.load_model()
        
        # 提取关键帧
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        
        if not frames:
            return "unknown", 0.1, "Cannot read video frames"
        
        # 构建心智地图描述
        map_text = self.mind_map_to_text(mind_map)
        
        # 构建提示
        options_text = ""
        if options:
            options_text = "\nOptions:\n" + "\n".join(options)
        
        if question_type in NUMERICAL_TASKS:
            answer_format = "Please answer with a NUMBER only (e.g., 3, 2.5, 25)."
        else:
            answer_format = "Please answer with the OPTION LETTER only (A, B, C, or D)."
        
        prompt = f"""You are a spatial intelligence assistant. Please answer the question based on the video frames and the detected object information.

{map_text}

Question: {question}{options_text}

{answer_format}
After your answer, add [Confidence: X%] to indicate how confident you are.

Answer:"""

        # 构建消息 - 使用 video 输入而不是多图
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "nframes": num_frames,
                },
                {"type": "text", "text": prompt}
            ]
        }]
        
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理视频输入
            from qwen_vl_utils import process_vision_info
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
            
            answer, confidence, reasoning = self._parse_response(response, question_type, options)
            return answer, confidence, reasoning
            
        except Exception as e:
            logger.warning(f"VL 推理失败: {e}")
            # Fallback to rule-based
            return self._fallback_answer(mind_map, question, question_type, options), 0.3, str(e)
    
    def _parse_response(self, response: str, question_type: str, options: List[str] = None) -> Tuple[str, float, str]:
        # 提取置信度
        conf_match = re.search(r'\[Confidence[：:]\s*(\d+)%?\]', response, re.IGNORECASE)
        confidence = float(conf_match.group(1)) / 100 if conf_match else 0.5
        
        # 提取答案
        response_clean = response.split('[')[0].strip()
        
        if question_type in NUMERICAL_TASKS:
            num_match = re.search(r'(\d+\.?\d*)', response_clean)
            answer = num_match.group(1) if num_match else "0"
        else:
            choice_match = re.search(r'^([A-D])', response_clean.upper())
            if choice_match:
                answer = choice_match.group(1)
            else:
                answer = response_clean
        
        return answer, confidence, response_clean
    
    def _fallback_answer(self, mind_map: Dict, question: str, question_type: str, options: List[str]) -> str:
        """规则回退"""
        q_lower = question.lower()
        
        if question_type == 'object_counting':
            match = re.search(r'how many (\w+)', q_lower)
            if match:
                target = match.group(1)
                for label, entity in mind_map.items():
                    if match_object_name(target, label):
                        return str(entity.count)
            return "0"
        
        elif question_type == 'object_size_estimation':
            for label, entity in mind_map.items():
                if label.lower() in q_lower:
                    if entity.size_3d is not None:
                        return str(int(max(entity.size_3d) * 100))
            return "50"
        
        elif question_type == 'room_size_estimation':
            if mind_map:
                positions = [e.position_3d for e in mind_map.values() if e.position_3d is not None]
                if len(positions) >= 2:
                    positions = np.array(positions)
                    area = (positions[:, 2].max() - positions[:, 2].min() + 2) ** 2
                    return str(int(min(80, max(10, area))))
            return "25"
        
        elif question_type == 'object_abs_distance':
            return "2.0"
        
        else:
            return options[0] if options else "A"
    
    def verify_entity(
        self,
        video_path: str,
        entity: MindMapEntity3D,
        question: str,
        question_type: str
    ) -> Tuple[Dict, bool]:
        """验证并可能修正实体信息"""
        self.load_model()
        
        # 定位到实体首次出现的帧
        cap = cv2.VideoCapture(video_path)
        target_frame = entity.first_seen_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {"count": entity.count, "confidence": entity.avg_confidence}, False
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # 构建验证提示
        if question_type == 'object_counting':
            verify_prompt = f"""Look at this image carefully. How many {entity.label}(s) can you see?

Answer with ONLY a number. If you see none, answer 0.
If you're uncertain, add [uncertain] after the number.

Answer:"""
        else:
            verify_prompt = f"""Look at this image. Can you see any {entity.label}?

If yes, describe:
1. How many?
2. Approximate size in centimeters?

Answer briefly:"""
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": verify_prompt}
            ]
        }]
        
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=64)
            
            response = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            # 解析响应
            result = {"count": entity.count, "confidence": entity.avg_confidence}
            was_modified = False
            
            num_match = re.search(r'(\d+)', response)
            if num_match:
                new_count = int(num_match.group(1))
                if new_count != entity.count:
                    result["count"] = new_count
                    result["confidence"] = 0.7
                    was_modified = True
                else:
                    result["confidence"] = min(0.9, entity.avg_confidence + 0.2)
            
            if '[uncertain]' in response.lower() or 'uncertain' in response.lower():
                result["confidence"] = max(0.3, result["confidence"] - 0.1)
            
            return result, was_modified
            
        except Exception as e:
            logger.warning(f"VL 验证失败: {e}")
            return {"count": entity.count, "confidence": entity.avg_confidence}, False


# ============================================================================
# Critic - 一致性检查
# ============================================================================

class Critic:
    """评论家 - 检查心智地图与推理的一致性"""
    
    def evaluate(
        self,
        mind_map: Dict[str, MindMapEntity3D],
        question: str,
        question_type: str,
        answer: str,
        confidence: float,
        target_objects: List[str]
    ) -> CriticFeedback:
        """评估答案的可靠性"""
        issues = []
        entities_to_verify = []
        
        # 1. 检查置信度
        if confidence < 0.4:
            issues.append(f"Low reasoning confidence ({confidence:.0%})")
        
        # 2. 检查目标物体
        for obj in target_objects:
            found = False
            for label in mind_map:
                if match_object_name(obj, label):
                    found = True
                    entities_to_verify.append(label)
                    break
            if not found:
                issues.append(f"Target object '{obj}' not found in mind map")
        
        # 3. 检查计数一致性
        if question_type == 'object_counting':
            answer_num = normalize_number(answer)
            if answer_num == 0:
                issues.append("Count result is 0, possible missed detection")
            elif answer_num is not None:
                for obj in target_objects:
                    for label, entity in mind_map.items():
                        if match_object_name(obj, label):
                            if abs(entity.count - answer_num) > 0:
                                issues.append(f"VL answer ({int(answer_num)}) differs from mind map ({entity.count})")
                            break
        
        # 4. 检查低置信度实体
        for label, entity in mind_map.items():
            if entity.avg_confidence < 0.4 and label not in entities_to_verify:
                entities_to_verify.append(label)
                issues.append(f"Entity '{label}' has low detection confidence ({entity.avg_confidence:.2f})")
        
        # 判断建议动作
        if not issues:
            suggested_action = "accept"
            is_confident = True
        else:
            suggested_action = "verify"
            is_confident = False
        
        return CriticFeedback(
            is_confident=is_confident,
            issues=issues,
            entities_to_verify=entities_to_verify[:3],
            suggested_action=suggested_action,
            reasoning=f"Found {len(issues)} potential issues"
        )


# ============================================================================
# Evolver - VL 回溯演化
# ============================================================================

class Evolver:
    """演化器 - 通过 VL 回溯修正心智地图"""
    
    def __init__(self, vl_reasoner: VLReasoner):
        self.vl_reasoner = vl_reasoner
    
    def evolve(
        self,
        mind_map: Dict[str, MindMapEntity3D],
        video_path: str,
        question: str,
        question_type: str,
        feedback: CriticFeedback,
        target_objects: List[str]
    ) -> Tuple[Dict[str, MindMapEntity3D], List[str], bool]:
        """通过 VL 回溯演化心智地图"""
        corrections = []
        was_evolved = False
        
        # 验证需要检查的实体
        for entity_label in feedback.entities_to_verify:
            if entity_label not in mind_map:
                continue
            
            entity = mind_map[entity_label]
            old_count = entity.count
            old_conf = entity.avg_confidence
            
            # VL 验证
            result, modified = self.vl_reasoner.verify_entity(
                video_path, entity, question, question_type
            )
            
            if modified:
                entity.count = result["count"]
                entity.avg_confidence = result["confidence"]
                corrections.append(f"{entity_label}: count {old_count}->{entity.count}")
                was_evolved = True
            else:
                entity.avg_confidence = result["confidence"]
                if abs(old_conf - entity.avg_confidence) > 0.1:
                    corrections.append(f"{entity_label}: confidence {old_conf:.2f}->{entity.avg_confidence:.2f}")
        
        # 检测漏检物体
        for obj in target_objects:
            found = any(match_object_name(obj, label) for label in mind_map)
            if not found:
                new_entity = self._detect_missing_object(video_path, obj)
                if new_entity:
                    mind_map[obj] = new_entity
                    corrections.append(f"Added: {obj} (count={new_entity.count})")
                    was_evolved = True
        
        return mind_map, corrections, was_evolved
    
    def _detect_missing_object(self, video_path: str, target_obj: str) -> Optional[MindMapEntity3D]:
        """检测漏检物体"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, 4, dtype=int)
        
        max_count = 0
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"How many {target_obj}(s) in this image? Answer with ONLY a number."}
                    ]
                }]
                
                text = self.vl_reasoner.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.vl_reasoner.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.vl_reasoner.device)
                
                with torch.no_grad():
                    generated_ids = self.vl_reasoner.model.generate(**inputs, max_new_tokens=16)
                
                response = self.vl_reasoner.processor.batch_decode(
                    generated_ids[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0]
                
                num_match = re.search(r'(\d+)', response)
                if num_match:
                    max_count = max(max_count, int(num_match.group(1)))
                    
            except Exception as e:
                continue
        
        cap.release()
        
        if max_count > 0:
            return MindMapEntity3D(
                label=target_obj,
                count=max_count,
                avg_confidence=0.6,
            )
        return None


# ============================================================================
# Task Manager
# ============================================================================

class TaskManager:
    """任务管理器"""
    
    @staticmethod
    def analyze_task(question: str, question_type: str, options: List[str] = None) -> Dict:
        result = {
            "question_type": question_type,
            "is_numerical": question_type in NUMERICAL_TASKS,
            "target_objects": [],
        }
        
        q_lower = question.lower()
        
        if question_type == 'object_counting':
            match = re.search(r'how many (\w+)', q_lower)
            if match:
                result["target_objects"] = [match.group(1)]
        
        elif question_type in ['object_size_estimation', 'object_abs_distance']:
            common_objects = ['chair', 'table', 'desk', 'sofa', 'bed', 'tv', 'lamp', 
                            'cabinet', 'door', 'window', 'fridge', 'refrigerator',
                            'stool', 'washer', 'dryer', 'toilet', 'sink']
            for obj in common_objects:
                if obj in q_lower:
                    result["target_objects"].append(obj)
        
        return result


# ============================================================================
# Self-Evolving Agent
# ============================================================================

class SelfEvolvingAgentV3:
    """自进化心智地图智能体 V3"""
    
    def __init__(
        self,
        device: str = 'cuda',
        num_frames: int = 32,
        vl_model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_evolution_rounds: int = 1
    ):
        self.device = device
        self.max_evolution_rounds = max_evolution_rounds
        
        self.manager = TaskManager()
        self.builder = MindMapBuilder(device=device, num_frames=num_frames)
        self.vl_reasoner = VLReasoner(model_path=vl_model_path, device=device)
        self.critic = Critic()
        self.evolver = Evolver(self.vl_reasoner)
        
        self.stats = {"total": 0, "evolved": 0, "improved_after_evolution": 0}
    
    def unload(self):
        self.builder.unload()
        self.vl_reasoner.unload()
    
    def process(
        self,
        video_path: str,
        question: str,
        question_type: str,
        options: List[str] = None,
        ground_truth: str = None
    ) -> Dict[str, Any]:
        self.stats["total"] += 1
        
        # 1. Manager: 分析任务
        task_info = self.manager.analyze_task(question, question_type, options)
        
        # 2. 感知: 构建心智地图
        try:
            mind_map = self.builder.build_from_video(video_path)
        except Exception as e:
            logger.warning(f"心智地图构建失败: {e}")
            mind_map = {}
        
        result = {
            "mind_map_size": len(mind_map),
            "evolved": False,
            "corrections": [],
        }
        
        # 3. Reasoner: VL 推理
        answer_v1, conf_v1, reasoning_v1 = self.vl_reasoner.reason(
            video_path, mind_map, question, question_type, options
        )
        
        result["answer_before_evolution"] = answer_v1
        result["confidence_before_evolution"] = conf_v1
        
        # 4. Critic: 评估
        feedback = self.critic.evaluate(
            mind_map, question, question_type,
            answer_v1, conf_v1, task_info["target_objects"]
        )
        
        result["critic_issues"] = feedback.issues
        
        # 5. Evolver: 演化
        if feedback.suggested_action == "verify" and self.max_evolution_rounds > 0:
            self.stats["evolved"] += 1
            result["evolved"] = True
            
            updated_map, corrections, was_evolved = self.evolver.evolve(
                mind_map, video_path, question, question_type,
                feedback, task_info["target_objects"]
            )
            
            result["corrections"] = corrections
            
            if was_evolved:
                # 重新推理
                answer_v2, conf_v2, _ = self.vl_reasoner.reason(
                    video_path, updated_map, question, question_type, options
                )
                result["answer"] = answer_v2
                result["confidence"] = conf_v2
                
                # 检查改进
                if ground_truth:
                    score_before, _ = evaluate_answer(answer_v1, ground_truth, question_type, options)
                    score_after, _ = evaluate_answer(answer_v2, ground_truth, question_type, options)
                    if score_after > score_before:
                        self.stats["improved_after_evolution"] += 1
            else:
                result["answer"] = answer_v1
                result["confidence"] = conf_v1
        else:
            result["answer"] = answer_v1
            result["confidence"] = conf_v1
        
        return result


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], config: Dict, result_queue: mp.Queue):
    """GPU Worker"""
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    agent = SelfEvolvingAgentV3(
        device='cuda',
        num_frames=config.get('num_frames', 32),
        vl_model_path=config.get('vl_model_path', 'Qwen/Qwen2.5-VL-7B-Instruct'),
        max_evolution_rounds=config.get('max_evolution_rounds', 1)
    )
    
    results = []
    
    # 带进度条
    for sample in tqdm(samples, desc=f"GPU-{gpu_id}", position=gpu_id, leave=False):
        video_path = find_video_path(sample['scene_name'])
        if not video_path:
            results.append({
                'id': sample['id'],
                'scene_name': sample['scene_name'],
                'question_type': sample['question_type'],
                'error': f"Video not found",
                'score': 0.0,
                'score_before': 0.0,
            })
            continue
        
        try:
            result = agent.process(
                video_path=video_path,
                question=sample['question'],
                question_type=sample['question_type'],
                options=sample.get('options'),
                ground_truth=sample.get('ground_truth')
            )
            
            score, correct = evaluate_answer(
                result['answer'], sample['ground_truth'],
                sample['question_type'], sample.get('options')
            )
            score_before, correct_before = evaluate_answer(
                result['answer_before_evolution'], sample['ground_truth'],
                sample['question_type'], sample.get('options')
            )
            
            results.append({
                'id': sample['id'],
                'scene_name': sample['scene_name'],
                'question_type': sample['question_type'],
                'question': sample['question'],
                'ground_truth': sample['ground_truth'],
                'pred': result['answer'],
                'pred_before': result['answer_before_evolution'],
                'score': float(score),
                'score_before': float(score_before),
                'correct': bool(correct),
                'correct_before': bool(correct_before),
                'evolved': bool(result.get('evolved', False)),
                'corrections': result.get('corrections', []),
                'confidence': float(result.get('confidence', 0)),
            })
            
        except Exception as e:
            logger.error(f"处理样本 {sample['id']} 失败: {e}")
            results.append({
                'id': sample['id'],
                'scene_name': sample['scene_name'],
                'question_type': sample['question_type'],
                'error': str(e),
                'score': 0.0,
                'score_before': 0.0,
            })
    
    agent.unload()
    result_queue.put((gpu_id, results, agent.stats))


# ============================================================================
# 数据加载
# ============================================================================

def load_vsibench_dataset(max_samples: int = None) -> List[Dict]:
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
        samples.append({
            'id': i,
            'scene_name': item['scene_name'],
            'question': item['question'],
            'question_type': item['question_type'],
            'options': item.get('options'),
            'ground_truth': item['ground_truth'],
        })
    
    logger.info(f"加载了 {len(samples)} 个样本")
    return samples


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Self-Evolving Agent V3')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--vl-model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--max-evolution-rounds', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'evolving_agent_v3_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Self-Evolving Agent V3 - VL 驱动演化系统")
    print(f"{'='*80}")
    print(f"时间戳: {timestamp}")
    print(f"GPU 数量: {args.num_gpus}")
    print(f"VL 模型: {args.vl_model}")
    print(f"演化轮数: {args.max_evolution_rounds}")
    print(f"{'='*80}\n")
    
    samples = load_vsibench_dataset(args.max_samples)
    
    # 分配样本
    samples_per_gpu = len(samples) // args.num_gpus
    gpu_samples = []
    for i in range(args.num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < args.num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])
    
    config = {
        'num_frames': args.num_frames,
        'vl_model_path': args.vl_model,
        'max_evolution_rounds': args.max_evolution_rounds,
    }
    
    # 启动进程
    result_queue = mp.Queue()
    processes = []
    
    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_samples[gpu_id], config, result_queue)
        )
        p.start()
        processes.append(p)
        logger.info(f"GPU {gpu_id}: {len(gpu_samples[gpu_id])} 样本")
    
    # 收集结果
    all_results = []
    total_stats = {"total": 0, "evolved": 0, "improved_after_evolution": 0}
    
    with tqdm(total=args.num_gpus, desc="Overall", position=args.num_gpus) as pbar:
        completed = 0
        while completed < args.num_gpus:
            try:
                gpu_id, results, stats = result_queue.get(timeout=7200)
                all_results.extend(results)
                for k in total_stats:
                    total_stats[k] += stats.get(k, 0)
                completed += 1
                pbar.update(1)
                pbar.set_postfix({'samples': len(all_results), 'evolved': total_stats['evolved']})
            except Exception as e:
                logger.error(f"等待超时: {e}")
                break
    
    for p in processes:
        p.join(timeout=60)
    
    # 统计
    print(f"\n{'='*100}")
    print("测试结果")
    print(f"{'='*100}")
    
    by_type = defaultdict(lambda: {'total': 0, 'score_sum': 0, 'score_before_sum': 0, 'evolved': 0, 'improved': 0})
    
    for r in all_results:
        qtype = r.get('question_type', 'unknown')
        by_type[qtype]['total'] += 1
        by_type[qtype]['score_sum'] += r.get('score', 0)
        by_type[qtype]['score_before_sum'] += r.get('score_before', 0)
        if r.get('evolved'):
            by_type[qtype]['evolved'] += 1
            if r.get('score', 0) > r.get('score_before', 0):
                by_type[qtype]['improved'] += 1
    
    print(f"\n{'任务类型':<35} {'样本':>6} {'演化前':>10} {'演化后':>10} {'提升':>8} {'演化率':>8} {'改进率':>8}")
    print("-" * 100)
    
    total_score = total_score_before = total_count = 0
    
    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        n = stats['total']
        score_before = stats['score_before_sum'] / n * 100 if n > 0 else 0
        score_after = stats['score_sum'] / n * 100 if n > 0 else 0
        improvement = score_after - score_before
        evolution_rate = stats['evolved'] / n * 100 if n > 0 else 0
        improve_rate = stats['improved'] / stats['evolved'] * 100 if stats['evolved'] > 0 else 0
        
        print(f"{qtype:<35} {n:>6} {score_before:>9.2f}% {score_after:>9.2f}% {improvement:>+7.2f}% {evolution_rate:>7.1f}% {improve_rate:>7.1f}%")
        
        total_score += stats['score_sum']
        total_score_before += stats['score_before_sum']
        total_count += n
    
    print("-" * 100)
    overall_before = total_score_before / total_count * 100 if total_count > 0 else 0
    overall_after = total_score / total_count * 100 if total_count > 0 else 0
    overall_improvement = overall_after - overall_before
    overall_evolution_rate = total_stats['evolved'] / total_count * 100 if total_count > 0 else 0
    overall_improve_rate = total_stats['improved_after_evolution'] / total_stats['evolved'] * 100 if total_stats['evolved'] > 0 else 0
    
    print(f"{'Overall':<35} {total_count:>6} {overall_before:>9.2f}% {overall_after:>9.2f}% {overall_improvement:>+7.2f}% {overall_evolution_rate:>7.1f}% {overall_improve_rate:>7.1f}%")
    print("=" * 100)
    
    # 保存
    summary = {
        'timestamp': timestamp,
        'num_samples': len(all_results),
        'vl_model': args.vl_model,
        'overall_score_before': overall_before,
        'overall_score_after': overall_after,
        'improvement': overall_improvement,
        'evolution_rate': total_stats['evolved'] / total_count if total_count > 0 else 0,
        'by_type': {qtype: {
            'count': stats['total'],
            'score_before': stats['score_before_sum'] / stats['total'] if stats['total'] > 0 else 0,
            'score_after': stats['score_sum'] / stats['total'] if stats['total'] > 0 else 0,
            'evolved': stats['evolved'],
            'improved': stats['improved'],
        } for qtype, stats in by_type.items()}
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
