"""
语义标签器 - 为 3D 实体添加语义标签

支持两种模式:
1. GroundingDINO 模式: 开放词汇检测 (推荐)
2. CLIP 模式: 从视频帧裁剪实体区域，用 CLIP 分类
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _hf_cache_dir() -> str:
    return os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")


def _from_pretrained_kwargs(model_ref: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"cache_dir": _hf_cache_dir()}
    if os.path.exists(model_ref):
        kwargs["local_files_only"] = True
    elif os.environ.get("MOT_GDINO_LOCAL_ONLY", "0") == "1":
        kwargs["local_files_only"] = True
    return kwargs

# 默认检测提示词 - 常见室内物体
DEFAULT_DETECTION_PROMPT = (
    "chair . table . desk . sofa . couch . bed . cabinet . shelf . drawer . "
    "wardrobe . bookshelf . nightstand . dresser . bench . stool . "
    "television . tv . computer . monitor . laptop . lamp . refrigerator . "
    "microwave . oven . washing machine . air conditioner . fan . "
    "door . window . wall . floor . ceiling . stairs . pillar . column . "
    "picture . painting . mirror . plant . vase . clock . curtain . rug . "
    "box . bag . bottle . cup . book . phone . toy . basket . person"
)


@dataclass
class SemanticLabel:
    """语义标签结果"""
    label: str
    confidence: float
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2] in pixels
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class DetectionResult:
    """检测结果"""
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized 0-1
    bbox_pixels: List[float]  # [x1, y1, x2, y2] in pixels


class GroundingDINOLabeler:
    """
    GroundingDINO 语义标签器 - 开放词汇目标检测
    
    优点:
    - 不需要预定义标签列表
    - 可以检测任意文本描述的物体
    - 同时返回边界框和标签
    """
    
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        """
        Args:
            model_id: HuggingFace 模型 ID
            device: 运行设备 ("cuda" 或 "cpu")
            box_threshold: 边界框置信度阈值 (threshold 参数)
            text_threshold: 文本匹配置信度阈值
        """
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        self.processor = None
        self.model = None
        self._loaded = False
    
    def load_model(self):
        """加载 GroundingDINO 模型"""
        if self._loaded:
            return
        
        try:
            import torch
            from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
            
            logger.info(f"Loading GroundingDINO model: {self.model_id}")

            load_kwargs = _from_pretrained_kwargs(self.model_id)

            self.processor = GroundingDinoProcessor.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
            self.model = GroundingDinoForObjectDetection.from_pretrained(
                self.model_id,
                **load_kwargs,
            )

            if isinstance(self.device, str) and self.device.startswith("cuda"):
                if torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.device = "cpu"
            elif self.device != "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            self._loaded = True
            logger.info("GroundingDINO model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            raise ImportError(
                "transformers and torch are required for GroundingDINOLabeler. "
                "Install with: pip install transformers torch"
            )
    
    def detect(
        self,
        image: np.ndarray,
        text_prompt: str = None,
    ) -> List[DetectionResult]:
        """
        在图像中检测物体（发布版原始实现，无 NaN 检查）
        """
        import torch
        from PIL import Image as PILImage
        
        if not self._loaded:
            self.load_model()
        
        if text_prompt is None:
            text_prompt = DEFAULT_DETECTION_PROMPT
        
        pil_image = PILImage.fromarray(image)
        h, w = image.shape[:2]
        
        inputs = self.processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model = self.model.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)]
        )[0]

        label_key = "text_labels" if "text_labels" in results else "labels"
        detections = []
        for score, label, box in zip(results["scores"], results[label_key], results["boxes"]):
            box = box.cpu().numpy()
            detections.append(DetectionResult(
                label=label,
                confidence=float(score.cpu()),
                bbox=[box[0]/w, box[1]/h, box[2]/w, box[3]/h],
                bbox_pixels=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            ))

        detections.sort(key=lambda x: x.confidence, reverse=True)
        return detections
    
    def detect_and_label_frame(
        self,
        frame: np.ndarray,
        text_prompt: str = None,
        max_detections: int = 20,
    ) -> Dict[str, Any]:
        """
        检测并标注单帧
        
        Returns:
            {
                'detections': List[DetectionResult],
                'frame_shape': (H, W),
                'num_detections': int,
            }
        """
        detections = self.detect(frame, text_prompt)[:max_detections]
        
        return {
            'detections': detections,
            'frame_shape': frame.shape[:2],
            'num_detections': len(detections),
        }
    
    def label_entity_from_detection(
        self,
        entity: Dict,
        detections: List[DetectionResult],
        frame_shape: Tuple[int, int],
        camera_pose: 'CameraPose',
        intrinsic: np.ndarray,
    ) -> Dict:
        """
        根据检测结果为实体匹配语义标签
        
        方法:
        1. 将实体 3D 中心投影到 2D
        2. 找到包含该投影点的检测框
        3. 选择置信度最高的匹配
        """
        center_3d = np.array(entity['center'])
        center_2d = self._project_to_2d(center_3d, camera_pose, intrinsic)
        
        if center_2d is None:
            entity['semantic_label'] = "unknown"
            entity['semantic_confidence'] = 0.0
            return entity
        
        h, w = frame_shape
        
        # 查找包含投影点的检测框
        matches = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_pixels
            if x1 <= center_2d[0] <= x2 and y1 <= center_2d[1] <= y2:
                # 计算投影点到框中心的距离 (归一化)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dist = np.sqrt(((center_2d[0] - cx) / w) ** 2 + ((center_2d[1] - cy) / h) ** 2)
                matches.append((det, dist))
        
        if matches:
            # 按 (置信度, -距离) 排序，选最佳匹配
            matches.sort(key=lambda x: (x[0].confidence, -x[1]), reverse=True)
            best_det = matches[0][0]
            entity['semantic_label'] = best_det.label
            entity['semantic_confidence'] = best_det.confidence
            entity['semantic_bbox'] = best_det.bbox_pixels
        else:
            # 没有直接匹配，查找最近的检测框
            if detections:
                min_dist = float('inf')
                best_det = None
                for det in detections:
                    x1, y1, x2, y2 = det.bbox_pixels
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    dist = np.sqrt((center_2d[0] - cx) ** 2 + (center_2d[1] - cy) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_det = det
                
                # 只有距离足够近才匹配
                max_dist_threshold = max(w, h) * 0.3  # 30% 图像尺寸
                if min_dist < max_dist_threshold:
                    entity['semantic_label'] = best_det.label
                    entity['semantic_confidence'] = best_det.confidence * 0.5  # 降低置信度
                    entity['semantic_bbox'] = best_det.bbox_pixels
                else:
                    entity['semantic_label'] = "object"
                    entity['semantic_confidence'] = 0.3
            else:
                entity['semantic_label'] = "object"
                entity['semantic_confidence'] = 0.1
        
        return entity
    
    def label_entities_from_video(
        self,
        entities: List[Dict],
        video_frames: List[np.ndarray],
        camera_poses: List['CameraPose'],
        intrinsic: np.ndarray,
        text_prompt: str = None,
        sample_frames: int = 5,  # 每个实体采样多少帧
    ) -> List[Dict]:
        """
        使用视频帧为实体添加语义标签
        
        Args:
            entities: 实体列表
            video_frames: 视频帧列表
            camera_poses: 相机位姿列表
            intrinsic: 相机内参
            text_prompt: 检测提示词
            sample_frames: 每个实体采样帧数
        
        Returns:
            带有语义标签的实体列表
        """
        if not self._loaded:
            self.load_model()
        
        logger.info(f"Labeling {len(entities)} entities from {len(video_frames)} frames")
        
        # 预先检测所有采样帧
        frame_indices = np.linspace(0, len(video_frames) - 1, min(20, len(video_frames))).astype(int)
        frame_detections = {}
        
        for idx in frame_indices:
            logger.info(f"  Detecting frame {idx}...")
            frame = video_frames[idx]
            result = self.detect_and_label_frame(frame, text_prompt)
            frame_detections[idx] = result
            logger.info(f"    Found {result['num_detections']} objects")
        
        # 为每个实体匹配标签
        for entity in entities:
            # 获取实体首次出现的帧
            first_frame = entity.get('first_seen_frame', 0)
            
            # 找到最近的已检测帧
            nearest_idx = min(frame_indices, key=lambda x: abs(x - first_frame))
            
            result = frame_detections[nearest_idx]
            pose = camera_poses[nearest_idx] if nearest_idx < len(camera_poses) else camera_poses[0]
            
            entity = self.label_entity_from_detection(
                entity=entity,
                detections=result['detections'],
                frame_shape=result['frame_shape'],
                camera_pose=pose,
                intrinsic=intrinsic,
            )
        
        return entities
    
    def _project_to_2d(
        self,
        point_3d: np.ndarray,
        pose: 'CameraPose',
        K: np.ndarray,
    ) -> Optional[np.ndarray]:
        """将 3D 点投影到 2D 图像坐标"""
        camera_pos = pose.position
        point_cam = point_3d - camera_pos
        
        forward = pose.forward if hasattr(pose, 'forward') else np.array([0, 0, 1])
        if np.dot(point_cam, forward) < 0:
            return None
        
        depth = np.linalg.norm(point_cam)
        if depth < 0.1:
            return None
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x = fx * point_cam[0] / depth + cx
        y = fy * point_cam[1] / depth + cy
        
        return np.array([x, y])


class SemanticLabeler:
    """
    语义标签器 - 使用 CLIP 为实体添加语义标签 (备用方案)
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        candidate_labels: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.candidate_labels = candidate_labels or [
            "chair", "table", "desk", "sofa", "bed", "cabinet", "shelf",
            "television", "computer", "lamp", "door", "window", "plant",
            "picture", "mirror", "box", "bottle", "cup", "book",
        ]
        self.model = None
        self.processor = None
        self._loaded = False
    
    def load_model(self):
        if self._loaded:
            return
        
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            
            cache_dir = HF_CACHE_DIR
            
            logger.info(f"Loading CLIP model: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name, cache_dir=cache_dir)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
            
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                else:
                    self.device = "cpu"
            
            self.model.eval()
            self._loaded = True
            logger.info("CLIP model loaded successfully")
            
        except ImportError as e:
            raise ImportError(f"transformers and torch required: {e}")
    
    def label_entity(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        top_k: int = 5,
    ) -> SemanticLabel:
        if not self._loaded:
            self.load_model()
        
        import torch
        from PIL import Image as PILImage
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]
        
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return SemanticLabel(label="unknown", confidence=0.0, alternatives=[])
        
        pil_image = PILImage.fromarray(image)
        text_inputs = [f"a photo of a {label}" for label in self.candidate_labels]
        
        inputs = self.processor(
            text=text_inputs,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=-1)
        
        probs_np = probs.cpu().numpy()
        top_indices = np.argsort(probs_np)[::-1][:top_k]
        
        best_idx = top_indices[0]
        alternatives = [
            (self.candidate_labels[idx], float(probs_np[idx]))
            for idx in top_indices[1:]
        ]
        
        return SemanticLabel(
            label=self.candidate_labels[best_idx],
            confidence=float(probs_np[best_idx]),
            alternatives=alternatives,
        )


class SimpleSemanticLabeler:
    """简化版语义标签器 - 基于颜色和尺寸启发式"""
    
    def __init__(self):
        pass
    
    def label_entity(self, entity: Dict, colors: Optional[np.ndarray] = None) -> Dict:
        size = np.array(entity.get('size', [1, 1, 1]))
        max_dim = max(size)
        min_dim = min(size)
        volume = np.prod(size)
        
        if size[1] > 2.0:
            label = "tall furniture"
        elif max_dim > 1.5 and min_dim < 0.3:
            label = "table or desk"
        elif volume < 0.1:
            label = "small object"
        elif 0.3 < size[1] < 1.0 and volume < 0.5:
            label = "chair"
        else:
            label = "furniture"
        
        entity['semantic_label'] = label
        entity['semantic_confidence'] = 0.5
        return entity


def create_labeler(mode: str = "grounding_dino", device: str = "cuda"):
    """
    创建语义标签器
    
    Args:
        mode: "grounding_dino" (推荐), "clip", "simple"
        device: 运行设备
    
    Returns:
        标签器实例
    """
    if mode == "simple":
        return SimpleSemanticLabeler()
    
    if mode == "clip":
        return SemanticLabeler(device=device)
    
    # 默认使用 GroundingDINO
    return GroundingDINOLabeler(device=device)
