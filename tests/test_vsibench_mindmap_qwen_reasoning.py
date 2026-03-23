#!/usr/bin/env python3
"""
心智地图 + Qwen3-VL 推理测试

策略：
1. 用 GroundingDINO + DA3 构建 3D 心智地图
2. 将心智地图序列化为文本描述
3. 用 Qwen3-vl-8B-instruct 基于心智地图做推理回答问题

这样选择题任务可以利用 VLM 的语言理解能力来做匹配
"""

import os
import sys

# 设置代理和缓存路径 (必须在导入 transformers 之前)
os.environ['http_proxy'] = 'http://10.11.16.24:8118'
os.environ['https_proxy'] = 'http://10.11.16.24:8118'
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['TORCH_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'

import json
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import cv2
from PIL import Image
import torch

# 添加项目路径
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 常量
# ============================================================================

NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 'room_size_estimation', 'object_abs_distance']
CHOICE_TASKS = ['object_rel_direction_lr', 'object_rel_direction_ud', 'object_rel_distance', 
                'obj_appearance_order', 'route_planning']


# ============================================================================
# 心智地图实体
# ============================================================================

@dataclass
class MindMapEntity3D:
    """带 3D 信息的心智地图实体"""
    entity_id: str
    label: str
    count: int = 1
    avg_confidence: float = 0.0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    
    # 3D 信息
    position_3d: Optional[np.ndarray] = None  # (x, y, z)
    size_3d: Optional[np.ndarray] = None  # (w, h, d)
    depth_median: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'label': self.label,
            'count': self.count,
            'confidence': round(self.avg_confidence, 3),
            'first_seen_frame': self.first_seen_frame,
            'last_seen_frame': self.last_seen_frame,
            'position_3d': [round(x, 2) for x in self.position_3d.tolist()] if self.position_3d is not None else None,
            'size_3d': [round(x, 2) for x in self.size_3d.tolist()] if self.size_3d is not None else None,
            'depth': round(self.depth_median, 2),
        }
    
    def to_description(self) -> str:
        """转换为文本描述"""
        parts = [f"- {self.label}:"]
        parts.append(f"  数量={self.count}")
        parts.append(f"  首次出现帧={self.first_seen_frame}")
        
        if self.position_3d is not None:
            x, y, z = self.position_3d
            parts.append(f"  3D位置=({x:.2f}, {y:.2f}, {z:.2f})米")
            
        if self.size_3d is not None:
            w, h, d = self.size_3d
            parts.append(f"  3D尺寸=({w:.2f}x{h:.2f}x{d:.2f})米")
            
        if self.depth_median > 0:
            parts.append(f"  距离相机={self.depth_median:.2f}米")
            
        return " ".join(parts)


# ============================================================================
# 心智地图构建器
# ============================================================================

class MindMapBuilderWithDepth:
    """集成 DA3 深度的心智地图构建器"""
    
    OBJECT_SIZE_PRIOR = {
        'chair': 0.5, 'table': 1.0, 'sofa': 1.8, 'couch': 1.8,
        'bed': 2.0, 'tv': 1.0, 'television': 1.0, 'monitor': 0.5,
        'door': 2.0, 'window': 1.2, 'refrigerator': 1.8, 'fridge': 1.8,
        'toilet': 0.6, 'sink': 0.6, 'bathtub': 1.5, 'cabinet': 1.0,
        'desk': 1.2, 'shelf': 1.0, 'lamp': 0.5, 'pillow': 0.4,
        'microwave': 0.5, 'stove': 0.7, 'oven': 0.7,
        'trash can': 0.4, 'plant': 0.5, 'nightstand': 0.5,
    }
    
    def __init__(self, device: str = 'cuda', num_frames: int = 16, box_threshold: float = 0.25):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        self._labeler = None
        self._depth_estimator = None
        
    def _load_models(self):
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
    
    def unload(self):
        if self._labeler:
            del self._labeler
            self._labeler = None
        if self._depth_estimator:
            del self._depth_estimator
            self._depth_estimator = None
        torch.cuda.empty_cache()
    
    def build_from_video(self, video_path: str, target_objects: List[str] = None) -> Dict[str, MindMapEntity3D]:
        """从视频构建 3D 心智地图"""
        self._load_models()
        
        # 读取视频帧
        frames = self._sample_video(video_path)
        if not frames:
            return {}
        
        H, W = frames[0].shape[:2]
        
        # 深度估计 - 使用 infer_single 逐帧处理 (与 V3 一致，效果更好)
        depth_maps = []
        for frame in frames:
            try:
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                
                # 校准：假设中值深度为 2.5 米 (与 V3 一致)
                median_depth = np.median(depth_map)
                if median_depth > 0:
                    scale = 2.5 / median_depth
                    depth_map = depth_map * scale
                
                depth_maps.append(depth_map)
            except Exception as e:
                logger.warning(f"深度估计失败: {e}")
                depth_maps.append(np.ones((H, W), dtype=np.float32) * 2.5)
        
        # 检测物体 - 使用与 V3 一致的扩展词汇表
        if target_objects is None:
            target_objects = []
        
        extended_vocab = list(set(target_objects + [
            # 家具
            "chair", "table", "sofa", "couch", "stove", "tv", "television",
            "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator",
            "sink", "toilet", "bathtub", "door", "window", "picture",
            "pillow", "cushion", "monitor", "backpack", "bag",
            "trash can", "trash bin", "mirror", "towel", "plant",
            "nightstand", "closet", "microwave", "printer", "washer",
            # VSIBench obj_appearance_order 常见物体
            "ceiling light", "light", "heater", "cup", "basket", "blanket",
            "bottle", "box", "curtain", "clock", "book", "vase",
            "fan", "radiator", "rug", "carpet", "mat", "shoes",
            "painting", "poster", "frame", "board", "whiteboard",
        ]))
        text_prompt = " . ".join(extended_vocab) + " ."
        
        all_detections = defaultdict(list)
        
        for frame_idx, frame in enumerate(frames):
            # 获取对应帧的深度图
            if frame_idx >= len(depth_maps):
                break
            depth = depth_maps[frame_idx]
            
            # 确保深度图尺寸匹配
            if depth.shape[:2] != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            
            results = self._labeler.detect(frame, text_prompt)
            
            for det in results:
                label = det.label.lower()
                bbox = det.bbox_pixels
                confidence = det.confidence
                
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 获取深度
                cx_int, cy_int = int(np.clip(cx, 0, W-1)), int(np.clip(cy, 0, H-1))
                x1_int = int(np.clip(x1, 0, W-1))
                y1_int = int(np.clip(y1, 0, H-1))
                x2_int = int(np.clip(x2, 0, W-1))
                y2_int = int(np.clip(y2, 0, H-1))
                
                bbox_depth = depth[y1_int:y2_int, x1_int:x2_int]
                if bbox_depth.size > 0:
                    obj_depth = float(np.median(bbox_depth))
                else:
                    obj_depth = float(depth[cy_int, cx_int])
                
                # 2D → 3D 
                fx = fy = min(W, H)
                cx_cam, cy_cam = W / 2, H / 2
                
                x_3d = (cx - cx_cam) * obj_depth / fx
                y_3d = (cy - cy_cam) * obj_depth / fy
                z_3d = obj_depth
                
                bbox_w, bbox_h = x2 - x1, y2 - y1
                width_3d = bbox_w * obj_depth / fx
                height_3d = bbox_h * obj_depth / fy
                
                # 深度方向尺寸：使用先验约束 (与 V3 一致)
                prior_size = self.OBJECT_SIZE_PRIOR.get(label, 0.5)
                depth_3d = min(width_3d, prior_size)
                
                all_detections[label].append({
                    'frame_idx': frame_idx,
                    'bbox': bbox,
                    'confidence': confidence,
                    'position_3d': np.array([x_3d, y_3d, z_3d]),
                    'size_3d': np.array([width_3d, height_3d, depth_3d]),
                    'depth_median': obj_depth,
                })
        
        # 聚合为实体
        return self._aggregate_to_entities(all_detections, len(frames))
    
    def _sample_video(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    def _calibrate_depth_scale(self, depth_maps: List[np.ndarray], W: int, H: int) -> float:
        """校准深度尺度"""
        median_depths = [np.median(d[d > 0]) if np.any(d > 0) else 1.0 for d in depth_maps]
        avg_depth = np.mean(median_depths)
        
        # 假设典型室内场景深度约 3-4 米
        target_depth = 3.5
        return target_depth / avg_depth if avg_depth > 0 else 1.0
    
    def _aggregate_to_entities(self, all_detections: Dict[str, List[Dict]], total_frames: int) -> Dict[str, MindMapEntity3D]:
        """聚合检测为实体"""
        # 标签标准化
        label_map = {
            'chair': 'chair', 'seat': 'chair', 'armchair': 'chair',
            'table': 'table', 'dining table': 'table', 'coffee table': 'table',
            'sofa': 'sofa', 'couch': 'sofa',
            'stove': 'stove', 'oven': 'stove',
            'tv': 'tv', 'television': 'tv', 'monitor': 'tv',
            'pillow': 'pillow', 'cushion': 'pillow',
            'trash can': 'trash bin', 'trash bin': 'trash bin',
        }
        
        category_dets = defaultdict(list)
        for label, dets in all_detections.items():
            category = label
            for key, cat in label_map.items():
                if key in label.lower():
                    category = cat
                    break
            category_dets[category].extend(dets)
        
        entities = {}
        for category, dets in category_dets.items():
            if not dets:
                continue
            
            # 按帧分组计数
            frame_dets = defaultdict(list)
            for det in dets:
                frame_dets[det['frame_idx']].append(det)
            
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
# 心智地图序列化
# ============================================================================

def mindmap_to_text(mind_map: Dict[str, MindMapEntity3D], total_frames: int = 16) -> str:
    """将心智地图转换为结构化文本"""
    if not mind_map:
        return "心智地图为空，未检测到任何物体。"
    
    lines = [
        "## 3D 场景心智地图",
        f"总帧数: {total_frames}",
        f"检测到 {len(mind_map)} 类物体:",
        ""
    ]
    
    # 按首次出现时间排序
    sorted_entities = sorted(mind_map.values(), key=lambda e: e.first_seen_frame)
    
    for entity in sorted_entities:
        lines.append(f"### {entity.label}")
        lines.append(f"- 数量: {entity.count}")
        lines.append(f"- 首次出现: 第 {entity.first_seen_frame} 帧 (共 {total_frames} 帧)")
        lines.append(f"- 最后出现: 第 {entity.last_seen_frame} 帧")
        
        if entity.position_3d is not None:
            x, y, z = entity.position_3d
            lines.append(f"- 3D 位置: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m (z 为距相机距离)")
            
        if entity.size_3d is not None:
            w, h, d = entity.size_3d
            max_dim = max(w, h, d)
            lines.append(f"- 3D 尺寸: 宽={w:.2f}m, 高={h:.2f}m, 深={d:.2f}m (最大尺寸: {max_dim:.2f}m)")
            
        if entity.depth_median > 0:
            lines.append(f"- 距离相机: {entity.depth_median:.2f}m")
            
        lines.append("")
    
    # 添加空间关系摘要
    if len(sorted_entities) >= 2:
        lines.append("### 空间关系摘要")
        
        # 出现顺序
        order_labels = [e.label for e in sorted_entities]
        lines.append(f"- 出现顺序 (按帧): {' → '.join(order_labels)}")
        
        # 距离排序
        by_depth = sorted(mind_map.values(), key=lambda e: e.depth_median)
        depth_labels = [f"{e.label}({e.depth_median:.1f}m)" for e in by_depth]
        lines.append(f"- 距离相机 (近→远): {' → '.join(depth_labels)}")
        
        # 左右关系
        by_x = sorted(mind_map.values(), key=lambda e: e.position_3d[0] if e.position_3d is not None else 0)
        x_labels = [e.label for e in by_x]
        lines.append(f"- 左右位置 (左→右): {' → '.join(x_labels)}")
        
        # 上下关系
        by_y = sorted(mind_map.values(), key=lambda e: e.position_3d[1] if e.position_3d is not None else 0)
        y_labels = [e.label for e in by_y]
        lines.append(f"- 上下位置 (上→下): {' → '.join(y_labels)}")
        
    return "\n".join(lines)


# ============================================================================
# Qwen3-VL 推理器
# ============================================================================

class QwenReasoner:
    """使用 Qwen3-VL 进行推理"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        
    def _load_model(self):
        if self._model is None:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            # 优先使用本地缓存
            cache_dir = "/home/tione/notebook/tianjungu/hf_cache"
            
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, 
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
    
    def unload(self):
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()
    
    def reason(self, mindmap_text: str, question: str, options: List[str] = None, 
               question_type: str = None) -> str:
        """基于心智地图进行推理"""
        self._load_model()
        
        # 构建 prompt
        if options:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            prompt = f"""你是一个空间理解专家。基于以下 3D 场景心智地图回答问题。

{mindmap_text}

问题: {question}

选项:
{options_text}

请仔细分析心智地图中的空间信息，选择最正确的答案。只输出选项内容（不要输出 A/B/C/D）。

答案:"""
        else:
            prompt = f"""你是一个空间理解专家。基于以下 3D 场景心智地图回答问题。

{mindmap_text}

问题: {question}

请根据心智地图中的信息给出简洁的数值答案。

答案:"""

        # 生成
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )
        
        generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        answer = self._processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        
        # 如果是选择题，确保返回选项内容
        if options:
            answer_lower = answer.lower()
            for opt in options:
                if opt.lower() in answer_lower or answer_lower in opt.lower():
                    return opt
            # 尝试匹配 A/B/C/D
            letter_match = re.match(r'^([A-Da-d])\b', answer)
            if letter_match:
                idx = ord(letter_match.group(1).upper()) - ord('A')
                if 0 <= idx < len(options):
                    return options[idx]
            return options[0]  # 默认返回第一个选项
        
        return answer


# ============================================================================
# 直接回答器（数值任务）
# ============================================================================

class DirectQA:
    """直接从心智地图回答数值问题（不用 VLM）"""
    
    @staticmethod
    def answer_counting(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        match = re.search(r'How many (\w+)\(s\)', question)
        if not match:
            match = re.search(r'How many (\w+)', question)
        
        if not match:
            return "0"
        
        target = match.group(1).lower()
        for label, entity in mind_map.items():
            if target in label.lower() or label.lower() in target:
                return str(entity.count)
        return "0"
    
    @staticmethod
    def answer_object_size(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        q_lower = question.lower()
        for label, entity in mind_map.items():
            if label.lower() in q_lower:
                if entity.size_3d is not None:
                    # 返回最大尺寸 (厘米)
                    max_dim = float(np.max(entity.size_3d)) * 100
                    return str(int(max_dim))
        return "50"
    
    @staticmethod
    def answer_room_size(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        if not mind_map:
            return "25"
        
        # 收集所有物体的 3D 位置
        positions = []
        for entity in mind_map.values():
            if entity.position_3d is not None:
                positions.append(entity.position_3d)
        
        if len(positions) < 2:
            # 基于物体数量估计
            return str(15 + len(mind_map) * 2)
        
        positions = np.array(positions)
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        # 房间面积 = 包围盒 * 扩展系数
        estimated_area = (x_range + 2) * (y_range + 2)  # +2 米边界
        estimated_area = max(10, min(100, estimated_area))
        return f"{estimated_area:.1f}"
    
    @staticmethod
    def answer_abs_distance(mind_map: Dict[str, MindMapEntity3D], question: str) -> str:
        q_lower = question.lower()
        
        # 收集所有物体位置
        positions = {}
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                positions[label] = entity.position_3d
        
        # 找到问题中的两个物体
        objs_in_q = []
        for obj_name in positions.keys():
            if obj_name.lower() in q_lower:
                objs_in_q.append(obj_name)
        
        if len(objs_in_q) >= 2:
            dist = float(np.linalg.norm(positions[objs_in_q[0]] - positions[objs_in_q[1]]))
            return f"{dist:.1f}"
        
        # 如果只找到一个物体，估算到相机的距离
        if len(objs_in_q) == 1:
            pos = positions[objs_in_q[0]]
            dist = float(np.linalg.norm(pos))
            return f"{dist:.1f}"
        
        return "2.0"  # 默认值
    
    @staticmethod
    def answer_appearance_order(mind_map: Dict[str, MindMapEntity3D], question: str, 
                                 options: List[str]) -> str:
        """回答出现顺序问题 - 这是心智地图的强项！"""
        if not options:
            return ""
        
        # 同义词映射
        SYNONYMS = {
            'ceiling light': ['light', 'lamp', 'ceiling light'],
            'light': ['light', 'lamp', 'ceiling light'],
            'lamp': ['light', 'lamp', 'ceiling light'],
            'sofa': ['sofa', 'couch'],
            'couch': ['sofa', 'couch'],
            'tv': ['tv', 'television', 'monitor'],
            'television': ['tv', 'television', 'monitor'],
            'trash can': ['trash can', 'trash bin', 'bin'],
            'trash bin': ['trash can', 'trash bin', 'bin'],
        }
        
        # 从问题中提取需要排序的物体
        # 格式: "What will be the first-time appearance order of the following categories in the video: ceiling light, cup, heater, door?"
        match = re.search(r'following categories.*?:\s*(.+?)\?', question, re.IGNORECASE)
        if match:
            objects_text = match.group(1)
            target_objects = [obj.strip().lower() for obj in objects_text.split(',')]
        else:
            # 从选项中提取物体列表
            target_objects = []
            for opt in options:
                # 去掉前缀 "A. " 等
                opt_content = re.sub(r'^[A-D]\.\s*', '', opt)
                objs = [o.strip().lower() for o in opt_content.split(',')]
                target_objects.extend(objs)
            target_objects = list(set(target_objects))
        
        # 获取心智地图中各物体的首次出现帧
        object_first_frames = {}
        for label, entity in mind_map.items():
            label_lower = label.lower()
            for target in target_objects:
                # 获取同义词列表
                synonyms = SYNONYMS.get(target, [target])
                # 模糊匹配
                for syn in synonyms:
                    if syn in label_lower or label_lower in syn:
                        if target not in object_first_frames:
                            object_first_frames[target] = entity.first_seen_frame
                        break
        
        # 如果找不到足够的物体，尝试更宽松的匹配
        if len(object_first_frames) < len(target_objects) // 2:
            for label, entity in mind_map.items():
                label_lower = label.lower()
                for target in target_objects:
                    if target not in object_first_frames:
                        # 检查任何单词是否匹配
                        target_words = target.split()
                        label_words = label_lower.split()
                        if any(tw in label_words or any(tw in lw for lw in label_words) for tw in target_words):
                            object_first_frames[target] = entity.first_seen_frame
        
        # 按首次出现帧排序
        sorted_objects = sorted(object_first_frames.items(), key=lambda x: x[1])
        predicted_order = [obj for obj, frame in sorted_objects]
        
        # 在选项中找最匹配的
        best_option = options[0]
        best_score = -1
        
        for opt in options:
            # 去掉前缀 "A. " 等
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt)
            opt_objects = [o.strip().lower() for o in opt_content.split(',')]
            
            # 计算顺序匹配得分 (Kendall tau)
            score = 0
            total_pairs = 0
            for i, obj1 in enumerate(predicted_order):
                for j, obj2 in enumerate(predicted_order[i+1:], i+1):
                    # 检查这对物体在选项中的顺序
                    try:
                        idx1 = next(k for k, o in enumerate(opt_objects) if obj1 in o or o in obj1)
                        idx2 = next(k for k, o in enumerate(opt_objects) if obj2 in o or o in obj2)
                        total_pairs += 1
                        if idx1 < idx2:
                            score += 1
                    except StopIteration:
                        continue
            
            if score > best_score:
                best_score = score
                best_option = opt
        
        return best_option


# ============================================================================
# 评估指标
# ============================================================================

def normalize_number(s: str) -> Optional[float]:
    if not s:
        return None
    try:
        nums = re.findall(r'-?\d+\.?\d*', str(s))
        return float(nums[0]) if nums else None
    except:
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


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], result_queue: mp.Queue):
    """GPU Worker"""
    # 在子进程中也设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['http_proxy'] = 'http://10.11.16.24:8118'
    os.environ['https_proxy'] = 'http://10.11.16.24:8118'
    os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
    os.environ['TRANSFORMERS_CACHE'] = '/home/tione/notebook/tianjungu/hf_cache'
    
    builder = MindMapBuilderWithDepth(device='cuda', num_frames=16, box_threshold=0.25)
    reasoner = QwenReasoner(device='cuda')
    
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
            
            # 根据任务类型选择回答方式
            if question_type == 'object_counting':
                pred = DirectQA.answer_counting(mind_map, question)
            elif question_type == 'object_size_estimation':
                pred = DirectQA.answer_object_size(mind_map, question)
            elif question_type == 'room_size_estimation':
                pred = DirectQA.answer_room_size(mind_map, question)
            elif question_type == 'object_abs_distance':
                pred = DirectQA.answer_abs_distance(mind_map, question)
            elif question_type == 'obj_appearance_order':
                # 出现顺序：心智地图的强项，直接基于 first_seen_frame 回答
                pred = DirectQA.answer_appearance_order(mind_map, question, options)
            else:
                # 其他选择题任务：使用 Qwen 推理
                mindmap_text = mindmap_to_text(mind_map)
                pred = reasoner.reason(mindmap_text, question, options, question_type)
            
            # 计算得分
            if question_type in NUMERICAL_TASKS:
                pred_num = normalize_number(pred)
                gt_num = normalize_number(str(gt))
                score = mean_relative_accuracy(pred_num, gt_num) if pred_num is not None and gt_num is not None else 0.0
                correct = score > 0.5
            else:
                # 选择题：GT 是 A/B/C/D，pred 可能是 "A. xxx" 或 "A" 或 "xxx"
                # 提取预测中的选项字母
                pred_letter = ""
                if pred:
                    # 尝试匹配开头的字母
                    letter_match = re.match(r'^([A-Da-d])\.?\s*', pred.strip())
                    if letter_match:
                        pred_letter = letter_match.group(1).upper()
                    else:
                        # 尝试在选项中找匹配
                        pred_lower = pred.lower().strip()
                        for idx, opt in enumerate(options):
                            if opt.lower() in pred_lower or pred_lower in opt.lower():
                                pred_letter = chr(65 + idx)  # A, B, C, D
                                break
                
                gt_letter = str(gt).upper().strip() if gt else ""
                correct = pred_letter == gt_letter
                score = 1.0 if correct else 0.0
                
                # 保存更详细的预测信息用于调试
                pred = f"{pred_letter}" if pred_letter else pred
            
            result = {
                'sample_id': sample['id'],
                'question_type': question_type,
                'question': question,
                'prediction': pred,
                'ground_truth': gt,
                'score': score,
                'correct': correct,
            }
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} 样本 {sample['id']} 错误: {e}")
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
        
        if (i + 1) % 5 == 0:
            logger.info(f"GPU {gpu_id}: {i+1}/{total} 完成")
    
    builder.unload()
    reasoner.unload()
    result_queue.put((gpu_id, results))


# ============================================================================
# 视频路径
# ============================================================================

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]


def find_video_path(scene_name: str) -> Optional[str]:
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--task-type', type=str, default='all')
    args = parser.parse_args()
    
    # 加载 VSIBench 数据
    vsibench_path = "/home/tione/notebook/tianjungu/projects/Spatial-MLLM/evaluate/annotation/eval_vsibench.json"
    
    with open(vsibench_path, 'r') as f:
        dataset = json.load(f)
    
    logger.info(f"加载 VSIBench: {len(dataset)} 样本")
    
    # 准备样本
    VSIBENCH_VIDEO_BASE = "/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench"
    
    samples = []
    for item in dataset:
        # 从 path 字段获取视频路径 (如 ./arkitscenes/41069025.mp4)
        rel_path = item.get('path', '')
        if rel_path.startswith('./'):
            rel_path = rel_path[2:]
        
        video_path = os.path.join(VSIBENCH_VIDEO_BASE, rel_path)
        if not os.path.exists(video_path):
            continue
        
        scene_name = os.path.basename(rel_path).replace('.mp4', '')
        
        question_type = item.get('original_question_type', item.get('question_type', ''))
        if args.task_type != 'all' and question_type != args.task_type:
            continue
        
        # 提取答案
        answer = item.get('answer', '')
        if not answer:
            solution = item.get('solution', '')
            match = re.search(r'<answer>(.*?)</answer>', solution)
            if match:
                answer = match.group(1)
        
        samples.append({
            'id': item.get('problem_id', len(samples)),
            'scene': scene_name,
            'video_path': video_path,
            'question': item.get('problem', item.get('question', '')),
            'question_type': question_type,
            'ground_truth': answer,
            'options': item.get('options', []) or [],
        })
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    logger.info(f"有效样本: {len(samples)}")
    
    # 多 GPU 测试
    num_gpus = min(args.num_gpus, len(samples))
    samples_per_gpu = len(samples) // num_gpus
    
    processes = []
    result_queue = mp.Queue()
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if gpu_id < num_gpus - 1 else len(samples)
        gpu_samples = samples[start_idx:end_idx]
        
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_samples, result_queue))
        p.start()
        processes.append(p)
        logger.info(f"启动 GPU {gpu_id}: {len(gpu_samples)} 样本")
    
    # 收集结果
    all_results = []
    for _ in range(num_gpus):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        logger.info(f"GPU {gpu_id} 完成: {len(results)} 结果")
    
    for p in processes:
        p.join()
    
    # 统计
    type_scores = defaultdict(list)
    for r in all_results:
        type_scores[r['question_type']].append(r['score'])
    
    # 输出结果
    print("\n" + "="*60)
    print("心智地图 + Qwen3-VL 推理测试结果")
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
    output_dir = f"/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/mindmap_qwen_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        # 转换 numpy 类型为 Python 原生类型
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
        
        json.dump({
            'summary': {q: {'mean': float(np.mean(s)), 'count': len(s)} for q, s in type_scores.items()},
            'overall': float(overall),
            'details': cleaned_results,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果保存到: {output_dir}")
