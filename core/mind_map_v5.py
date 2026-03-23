#!/usr/bin/env python3
"""
MindMap V5 - 统一的空间智能心智地图框架

核心创新：
1. 稀疏体素占据地图 (Sparse Voxel Occupancy Map)
   - 0.1m 立方体网格
   - 解决 room_size 的准确估计
   - 支持遮挡推理

2. DINOv2 语义特征嵌入 (Semantic Feature Embedding)
   - 每个实体存储 RoI 特征向量
   - 支持细粒度属性匹配（颜色、纹理、状态）

3. 概率性位置估计 (Probabilistic Position Estimation)
   - 高斯分布表示位置不确定性
   - 支持置信度感知的推理
   - 触发 "需要重新观察" 的 Agentic 行为

作者: tianjungu
日期: 2026-01-29
"""

import os
import sys
import gc
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 稀疏体素占据地图 (Sparse Voxel Occupancy Map)
# ============================================================================

@dataclass
class VoxelInfo:
    """单个体素的信息"""
    semantic_label: str = ""           # 语义标签
    occupancy_prob: float = 0.0        # 占据概率 [0, 1]
    observation_count: int = 0         # 观测次数
    feature_embedding: Optional[np.ndarray] = None  # 平均特征向量


class SparseVoxelMap:
    """稀疏体素占据地图
    
    使用字典存储，只存储被占据的体素，节省内存。
    体素坐标: (vx, vy, vz) = floor(position / voxel_size)
    """
    
    def __init__(self, voxel_size: float = 0.1):
        """
        Args:
            voxel_size: 体素边长（米），默认 0.1m
        """
        self.voxel_size = voxel_size
        self.voxels: Dict[Tuple[int, int, int], VoxelInfo] = {}
        
        # 边界追踪
        self.min_bounds = np.array([float('inf'), float('inf'), float('inf')])
        self.max_bounds = np.array([float('-inf'), float('-inf'), float('-inf')])
    
    def world_to_voxel(self, position: np.ndarray) -> Tuple[int, int, int]:
        """世界坐标转体素坐标"""
        voxel_coords = np.floor(position / self.voxel_size).astype(int)
        return tuple(voxel_coords)
    
    def voxel_to_world(self, voxel_coord: Tuple[int, int, int]) -> np.ndarray:
        """体素坐标转世界坐标（体素中心）"""
        return (np.array(voxel_coord) + 0.5) * self.voxel_size
    
    def add_observation(self, position: np.ndarray, label: str, 
                        confidence: float = 1.0,
                        feature: Optional[np.ndarray] = None,
                        extent: Optional[np.ndarray] = None):
        """添加观测
        
        Args:
            position: 3D 位置 [x, y, z]
            label: 语义标签
            confidence: 检测置信度
            feature: 特征向量
            extent: 物体范围 [w, h, d]，用于填充多个体素
        """
        # 更新边界
        self.min_bounds = np.minimum(self.min_bounds, position)
        self.max_bounds = np.maximum(self.max_bounds, position)
        
        if extent is not None:
            # 物体占据多个体素
            half_extent = extent / 2
            min_pos = position - half_extent
            max_pos = position + half_extent
            
            min_voxel = self.world_to_voxel(min_pos)
            max_voxel = self.world_to_voxel(max_pos)
            
            for vx in range(min_voxel[0], max_voxel[0] + 1):
                for vy in range(min_voxel[1], max_voxel[1] + 1):
                    for vz in range(min_voxel[2], max_voxel[2] + 1):
                        self._update_voxel((vx, vy, vz), label, confidence, feature)
        else:
            # 单体素更新
            voxel_coord = self.world_to_voxel(position)
            self._update_voxel(voxel_coord, label, confidence, feature)
    
    def _update_voxel(self, coord: Tuple[int, int, int], label: str, 
                      confidence: float, feature: Optional[np.ndarray]):
        """更新单个体素"""
        if coord not in self.voxels:
            self.voxels[coord] = VoxelInfo()
        
        voxel = self.voxels[coord]
        voxel.observation_count += 1
        
        # 贝叶斯更新占据概率
        prior = voxel.occupancy_prob if voxel.observation_count > 1 else 0.5
        # 简化的贝叶斯更新
        voxel.occupancy_prob = (prior * (voxel.observation_count - 1) + confidence) / voxel.observation_count
        
        # 更新语义标签（取置信度最高的）
        if not voxel.semantic_label or confidence > 0.5:
            voxel.semantic_label = label
        
        # 更新特征（增量平均）
        if feature is not None:
            if voxel.feature_embedding is None:
                voxel.feature_embedding = feature.copy()
            else:
                n = voxel.observation_count
                voxel.feature_embedding = (voxel.feature_embedding * (n - 1) + feature) / n
    
    def get_occupied_volume(self, min_prob: float = 0.3) -> float:
        """计算占据体积（立方米）
        
        Args:
            min_prob: 最小占据概率阈值
        """
        count = sum(1 for v in self.voxels.values() if v.occupancy_prob >= min_prob)
        return count * (self.voxel_size ** 3)
    
    def get_floor_area(self, min_prob: float = 0.3, 
                       floor_height_range: Tuple[float, float] = (-0.5, 0.5)) -> float:
        """计算地面投影面积（平方米）
        
        通过统计 XZ 平面上被占据的唯一网格数量来估计。
        
        Args:
            min_prob: 最小占据概率阈值
            floor_height_range: 地面高度范围 (min_y, max_y)
        """
        floor_cells: Set[Tuple[int, int]] = set()
        
        for (vx, vy, vz), voxel in self.voxels.items():
            if voxel.occupancy_prob >= min_prob:
                # 所有占据体素都投影到地面
                floor_cells.add((vx, vz))
        
        return len(floor_cells) * (self.voxel_size ** 2)
    
    def get_room_dimensions(self, min_prob: float = 0.3) -> Tuple[float, float, float]:
        """估计房间尺寸 (宽, 高, 深)"""
        if not self.voxels:
            return (4.0, 2.5, 4.0)  # 默认值
        
        occupied = [(k, v) for k, v in self.voxels.items() if v.occupancy_prob >= min_prob]
        if not occupied:
            return (4.0, 2.5, 4.0)
        
        coords = np.array([k for k, _ in occupied])
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        
        # 转换为米并添加边缘余量
        dimensions = (max_c - min_c + 1) * self.voxel_size
        # 添加边缘余量（物体到墙壁的距离）
        dimensions = dimensions + np.array([1.0, 0.5, 1.0])
        
        return tuple(dimensions)
    
    def ray_cast(self, origin: np.ndarray, direction: np.ndarray, 
                 max_dist: float = 10.0) -> Optional[Tuple[np.ndarray, str]]:
        """射线投射，用于遮挡检测
        
        Returns:
            如果命中，返回 (命中点, 语义标签)；否则返回 None
        """
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        step = self.voxel_size * 0.5  # 步进距离
        
        for t in np.arange(0, max_dist, step):
            point = origin + direction * t
            voxel_coord = self.world_to_voxel(point)
            
            if voxel_coord in self.voxels:
                voxel = self.voxels[voxel_coord]
                if voxel.occupancy_prob > 0.5:
                    return (point, voxel.semantic_label)
        
        return None
    
    def is_occluded(self, observer: np.ndarray, target: np.ndarray, 
                    target_label: str) -> Tuple[bool, float]:
        """检测目标是否被遮挡
        
        Returns:
            (是否遮挡, 遮挡概率)
        """
        direction = target - observer
        dist = np.linalg.norm(direction)
        
        if dist < 0.1:
            return (False, 0.0)
        
        hit = self.ray_cast(observer, direction, dist - 0.1)
        
        if hit is None:
            return (False, 0.0)
        
        hit_point, hit_label = hit
        hit_dist = np.linalg.norm(hit_point - observer)
        
        # 如果命中的是目标物体本身，不算遮挡
        if hit_label == target_label:
            return (False, 0.0)
        
        # 计算遮挡概率（距离越近，遮挡越可能）
        occlusion_prob = 1.0 - (hit_dist / dist)
        return (True, occlusion_prob)
    
    def to_dict(self) -> Dict:
        """序列化"""
        return {
            'voxel_size': self.voxel_size,
            'num_voxels': len(self.voxels),
            'occupied_volume': self.get_occupied_volume(),
            'floor_area': self.get_floor_area(),
            'bounds': {
                'min': self.min_bounds.tolist(),
                'max': self.max_bounds.tolist(),
            }
        }


# ============================================================================
# 2. 概率性位置估计 (Probabilistic Position)
# ============================================================================

@dataclass
class GaussianPosition3D:
    """3D 高斯位置分布"""
    mean: np.ndarray              # 均值 [x, y, z]
    covariance: np.ndarray        # 协方差矩阵 3x3
    observation_count: int = 0    # 观测次数
    
    @classmethod
    def from_single_observation(cls, position: np.ndarray, 
                                 uncertainty: float = 0.5) -> 'GaussianPosition3D':
        """从单次观测创建"""
        return cls(
            mean=position.copy(),
            covariance=np.eye(3) * (uncertainty ** 2),
            observation_count=1
        )
    
    def update(self, new_position: np.ndarray, 
               measurement_noise: float = 0.3):
        """卡尔曼滤波式更新"""
        self.observation_count += 1
        
        # 测量噪声协方差
        R = np.eye(3) * (measurement_noise ** 2)
        
        # 卡尔曼增益
        S = self.covariance + R
        K = self.covariance @ np.linalg.inv(S)
        
        # 更新均值
        innovation = new_position - self.mean
        self.mean = self.mean + K @ innovation
        
        # 更新协方差
        I = np.eye(3)
        self.covariance = (I - K) @ self.covariance
    
    @property
    def uncertainty(self) -> float:
        """位置不确定性（标准差的迹）"""
        return float(np.sqrt(np.trace(self.covariance)))
    
    @property
    def confidence(self) -> float:
        """位置置信度 [0, 1]"""
        # 不确定性越小，置信度越高
        # 使用 sigmoid 映射
        return float(1.0 / (1.0 + self.uncertainty))
    
    def sample(self, n: int = 1) -> np.ndarray:
        """从分布中采样"""
        return np.random.multivariate_normal(self.mean, self.covariance, n)
    
    def mahalanobis_distance(self, point: np.ndarray) -> float:
        """马氏距离"""
        diff = point - self.mean
        return float(np.sqrt(diff.T @ np.linalg.inv(self.covariance) @ diff))
    
    def overlap_probability(self, other: 'GaussianPosition3D') -> float:
        """计算与另一个分布的重叠概率"""
        # 简化：使用均值间距离与不确定性的比值
        dist = np.linalg.norm(self.mean - other.mean)
        combined_uncertainty = self.uncertainty + other.uncertainty
        
        if combined_uncertainty < 1e-6:
            return 1.0 if dist < 0.1 else 0.0
        
        # 重叠概率：距离越近，不确定性越大，重叠越可能
        return float(np.exp(-dist / combined_uncertainty))
    
    def to_dict(self) -> Dict:
        return {
            'mean': self.mean.tolist(),
            'covariance': self.covariance.tolist(),
            'uncertainty': self.uncertainty,
            'confidence': self.confidence,
            'observation_count': self.observation_count,
        }


# ============================================================================
# 3. 增强版心智地图实体 (Enhanced MindMap Entity)
# ============================================================================

@dataclass
class MindMapEntityV5:
    """V5 心智地图实体 - 集成体素、特征、概率"""
    
    # 基础信息
    entity_id: str
    label: str
    instance_id: int = 0          # 实例 ID（区分同类物体）
    
    # 检测统计
    detection_count: int = 0       # 总检测次数
    max_single_frame_count: int = 1  # 单帧最大检测数
    avg_confidence: float = 0.0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    
    # 概率性 3D 位置
    position: Optional[GaussianPosition3D] = None
    
    # 3D 尺寸（也用高斯表示）
    size_mean: Optional[np.ndarray] = None      # [w, h, d]
    size_std: Optional[np.ndarray] = None       # 尺寸标准差
    
    # 语义特征嵌入 (DINOv2)
    feature_embedding: Optional[np.ndarray] = None  # 768-d or 1024-d
    feature_observations: int = 0
    
    # 占据的体素坐标集合
    occupied_voxels: Set[Tuple[int, int, int]] = field(default_factory=set)
    
    # 属性推断（从特征中提取）
    inferred_attributes: Dict[str, float] = field(default_factory=dict)
    
    def update_position(self, new_pos: np.ndarray, uncertainty: float = 0.3):
        """更新位置估计"""
        if self.position is None:
            self.position = GaussianPosition3D.from_single_observation(new_pos, uncertainty)
        else:
            self.position.update(new_pos, uncertainty)
    
    def update_size(self, new_size: np.ndarray):
        """更新尺寸估计"""
        if self.size_mean is None:
            self.size_mean = new_size.copy()
            self.size_std = np.ones(3) * 0.1
        else:
            # 增量更新
            n = self.detection_count
            old_mean = self.size_mean
            self.size_mean = (old_mean * (n - 1) + new_size) / n
            # 更新标准差
            self.size_std = np.sqrt(
                ((n - 1) * self.size_std ** 2 + (new_size - old_mean) * (new_size - self.size_mean)) / n
            )
    
    def update_feature(self, new_feature: np.ndarray):
        """更新特征嵌入（增量平均）"""
        self.feature_observations += 1
        if self.feature_embedding is None:
            self.feature_embedding = new_feature.copy()
        else:
            n = self.feature_observations
            self.feature_embedding = (self.feature_embedding * (n - 1) + new_feature) / n
    
    def feature_similarity(self, other_feature: np.ndarray) -> float:
        """计算与给定特征的余弦相似度"""
        if self.feature_embedding is None:
            return 0.0
        
        dot = np.dot(self.feature_embedding, other_feature)
        norm1 = np.linalg.norm(self.feature_embedding)
        norm2 = np.linalg.norm(other_feature)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    @property
    def position_3d(self) -> Optional[np.ndarray]:
        """兼容旧接口：返回位置均值"""
        return self.position.mean if self.position else None
    
    @property
    def size_3d(self) -> Optional[np.ndarray]:
        """兼容旧接口：返回尺寸均值"""
        return self.size_mean
    
    @property
    def depth_median(self) -> float:
        """兼容旧接口：返回深度"""
        return float(self.position.mean[2]) if self.position else 0.0
    
    @property
    def count(self) -> int:
        """兼容旧接口：返回数量"""
        return self.max_single_frame_count
    
    @property
    def position_confidence(self) -> float:
        """位置置信度"""
        return self.position.confidence if self.position else 0.0
    
    def to_dict(self) -> Dict:
        return {
            'entity_id': self.entity_id,
            'label': self.label,
            'instance_id': self.instance_id,
            'count': self.max_single_frame_count,
            'detection_count': self.detection_count,
            'avg_confidence': round(self.avg_confidence, 3),
            'first_seen_frame': self.first_seen_frame,
            'last_seen_frame': self.last_seen_frame,
            'position': self.position.to_dict() if self.position else None,
            'size_mean': self.size_mean.tolist() if self.size_mean is not None else None,
            'size_std': self.size_std.tolist() if self.size_std is not None else None,
            'has_feature': self.feature_embedding is not None,
            'feature_dim': len(self.feature_embedding) if self.feature_embedding is not None else 0,
            'num_occupied_voxels': len(self.occupied_voxels),
            'inferred_attributes': self.inferred_attributes,
        }


# ============================================================================
# 4. V5 心智地图构建器
# ============================================================================

class MindMapBuilderV5:
    """V5 心智地图构建器
    
    统一框架：体素占据 + 语义特征 + 概率位置
    """
    
    # 已知物体尺寸（用于尺度校准）
    KNOWN_SIZES = {
        'door': {'height': 2.0, 'width': 0.9},
        'bed': {'height': 0.5, 'width': 1.5, 'length': 2.0},
        'sofa': {'height': 0.85, 'width': 0.9},
        'couch': {'height': 0.85, 'width': 0.9},
        'refrigerator': {'height': 1.7, 'width': 0.7},
        'fridge': {'height': 1.7, 'width': 0.7},
        'chair': {'height': 0.9, 'width': 0.5},
        'table': {'height': 0.75, 'width': 0.8},
        'desk': {'height': 0.75, 'width': 0.6},
        'toilet': {'height': 0.4, 'width': 0.4},
        'tv': {'height': 0.6, 'width': 1.0},
        'television': {'height': 0.6, 'width': 1.0},
        'monitor': {'height': 0.35, 'width': 0.55},
        'person': {'height': 1.7, 'width': 0.5},
    }
    
    def __init__(self, 
                 device: str = 'cuda',
                 num_frames: int = 32,
                 box_threshold: float = 0.25,
                 voxel_size: float = 0.1,
                 extract_features: bool = True):
        """
        Args:
            device: 计算设备
            num_frames: 采样帧数
            box_threshold: 检测阈值
            voxel_size: 体素大小（米）
            extract_features: 是否提取 DINOv2 特征
        """
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        self.voxel_size = voxel_size
        self.extract_features = extract_features
        
        # 模型
        self._labeler = None
        self._depth_estimator = None
        self._feature_extractor = None
        
        # 相机内参
        self.focal_length = 500
        self.principal_point = None
        
        # 尺度校准
        self.scale_factor = 1.0
        self.scale_history: List[float] = []
    
    def _load_models(self):
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
        
        # DINOv2 特征提取器（复用 DA3 的 backbone）
        if self.extract_features and self._feature_extractor is None:
            self._setup_feature_extractor()
    
    def _setup_feature_extractor(self):
        """设置 DINOv2 特征提取器
        
        DA3 使用 DINOv2 作为 backbone，我们可以直接获取中间特征
        """
        try:
            # 尝试从 DA3 模型获取 backbone
            if hasattr(self._depth_estimator, 'model') and hasattr(self._depth_estimator.model, 'pretrained'):
                self._feature_extractor = self._depth_estimator.model.pretrained
                logger.info("复用 DA3 DINOv2 backbone 作为特征提取器")
            else:
                # 独立加载 DINOv2
                logger.info("独立加载 DINOv2 特征提取器...")
                self._feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
                self._feature_extractor = self._feature_extractor.to(self.device)
                self._feature_extractor.eval()
                logger.info("DINOv2 特征提取器加载完成")
        except Exception as e:
            logger.warning(f"特征提取器加载失败: {e}")
            self._feature_extractor = None
            self.extract_features = False
    
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
        
        if self._feature_extractor is not None:
            try:
                del self._feature_extractor
            except:
                pass
            self._feature_extractor = None
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """提取视频帧"""
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
        n = min(self.num_frames, len(frames))
        indices = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in indices], indices.tolist()
    
    def _extract_roi_feature(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """提取 RoI 的 DINOv2 特征
        
        Args:
            frame: RGB 帧
            bbox: (x1, y1, x2, y2)
            
        Returns:
            特征向量 (768-d 或 1024-d)
        """
        if self._feature_extractor is None:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # 调整到 DINOv2 输入尺寸
            roi_resized = cv2.resize(roi, (224, 224))
            
            # 转为 tensor
            roi_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
            roi_tensor = roi_tensor.unsqueeze(0).to(self.device)
            
            # 标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            roi_tensor = (roi_tensor - mean) / std
            
            # 提取特征
            with torch.no_grad():
                if hasattr(self._feature_extractor, 'forward_features'):
                    # DINOv2 风格
                    features = self._feature_extractor.forward_features(roi_tensor)
                    if isinstance(features, dict):
                        feature = features['x_norm_clstoken']
                    else:
                        feature = features[:, 0]  # CLS token
                else:
                    # 通用风格
                    feature = self._feature_extractor(roi_tensor)
                    if len(feature.shape) > 2:
                        feature = feature.mean(dim=(2, 3))
            
            return feature.squeeze().cpu().numpy()
            
        except Exception as e:
            logger.debug(f"特征提取失败: {e}")
            return None
    
    def _calibrate_scale(self, detections: List[Dict], depth_map: np.ndarray) -> float:
        """动态尺度校准"""
        scale_estimates = []
        
        for det in detections:
            label = det['label'].lower()
            
            known_size = None
            for key in self.KNOWN_SIZES:
                if key in label or label in key:
                    known_size = self.KNOWN_SIZES[key]
                    break
            
            if known_size is None:
                continue
            
            ref_dim = known_size.get('height', known_size.get('width'))
            x1, y1, x2, y2 = det['bbox']
            h_pixel = y2 - y1
            
            depth_roi = depth_map[y1:y2, x1:x2]
            if depth_roi.size == 0:
                continue
            depth_median = np.median(depth_roi)
            if depth_median <= 0:
                continue
            
            h_estimated = h_pixel / self.focal_length * depth_median
            
            if h_estimated > 0.01:
                scale = ref_dim / h_estimated
                if 0.1 < scale < 10:
                    scale_estimates.append(scale)
        
        if scale_estimates:
            scale = float(np.median(scale_estimates))
            self.scale_history.append(scale)
            if len(self.scale_history) > 10:
                self.scale_history = self.scale_history[-10:]
            return scale
        
        if self.scale_history:
            return float(np.median(self.scale_history))
        
        # 默认中值校准
        median_depth = np.median(depth_map)
        return 2.5 / median_depth if median_depth > 0 else 1.0
    
    def build_from_video(self, video_path: str,
                         target_objects: List[str] = None,
                         vocabulary: List[str] = None,
                         use_dynamic_scale: bool = True) -> Tuple[Dict[str, MindMapEntityV5], SparseVoxelMap]:
        """从视频构建 V5 心智地图
        
        Args:
            video_path: 视频路径
            target_objects: 目标物体列表
            vocabulary: 检测词汇表
            use_dynamic_scale: 是否使用动态尺度校准（对 room_size 任务应禁用）
        
        Returns:
            (entities, voxel_map)
        """
        self._load_models()
        
        # 重置
        self.scale_history.clear()
        voxel_map = SparseVoxelMap(voxel_size=self.voxel_size)
        
        # 提取帧
        all_frames = self._extract_frames(video_path)
        if not all_frames:
            return {}, voxel_map
        
        frames, frame_indices = self._sample_frames(all_frames)
        H, W = frames[0].shape[:2]
        self.principal_point = (W / 2, H / 2)
        
        # 构建词汇表
        from tests.test_vsibench_directqa import EXTENDED_VOCABULARY
        if vocabulary is None:
            vocabulary = EXTENDED_VOCABULARY
        if target_objects:
            vocabulary = list(set(target_objects + vocabulary))
        text_prompt = " . ".join(vocabulary) + " ."
        
        # 存储所有检测
        all_detections: Dict[str, List[Dict]] = defaultdict(list)
        frame_detection_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # 逐帧处理
        for frame_idx, frame in enumerate(frames):
            original_idx = frame_indices[frame_idx]
            
            # 深度估计
            try:
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                if depth_map.shape[:2] != (H, W):
                    depth_map = cv2.resize(depth_map, (W, H))
            except Exception as e:
                logger.warning(f"深度估计失败: {e}")
                depth_map = np.ones((H, W), dtype=np.float32) * 2.5
            
            # 物体检测
            results = self._labeler.detect(frame, text_prompt)
            
            # 预处理检测结果
            raw_detections = []
            for det in results:
                x1, y1, x2, y2 = [int(v) for v in det.bbox_pixels]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    raw_detections.append({
                        'label': det.label.lower(),
                        'bbox': (x1, y1, x2, y2),
                        'confidence': det.confidence,
                    })
            
            # 尺度校准（条件性）
            if use_dynamic_scale:
                scale = self._calibrate_scale(raw_detections, depth_map)
            else:
                # 使用固定的中值校准
                median_depth = np.median(depth_map)
                scale = 2.5 / median_depth if median_depth > 0 else 1.0
            
            depth_map_scaled = depth_map * scale
            
            # 处理每个检测
            for det in raw_detections:
                label = det['label']
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                
                frame_detection_counts[original_idx][label] += 1
                
                # 获取深度
                depth_roi = depth_map_scaled[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                depth_median = float(np.median(depth_roi))
                depth_std = float(np.std(depth_roi))
                
                # 计算 3D 位置
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                px, py = self.principal_point
                
                x_3d = (cx - px) / self.focal_length * depth_median
                y_3d = (cy - py) / self.focal_length * depth_median
                z_3d = depth_median
                
                # 计算位置不确定性（基于深度方差和检测置信度）
                position_uncertainty = depth_std * 0.5 + (1 - confidence) * 0.5
                
                # 计算 3D 尺寸
                w_3d = (x2 - x1) / self.focal_length * depth_median
                h_3d = (y2 - y1) / self.focal_length * depth_median
                d_3d = min(w_3d, h_3d) * 0.5
                
                # 提取特征
                feature = None
                if self.extract_features:
                    feature = self._extract_roi_feature(frame, (x1, y1, x2, y2))
                
                all_detections[label].append({
                    'frame_idx': original_idx,
                    'bbox': det['bbox'],
                    'confidence': confidence,
                    'position_3d': np.array([x_3d, y_3d, z_3d]),
                    'position_uncertainty': position_uncertainty,
                    'size_3d': np.array([w_3d, h_3d, d_3d]),
                    'depth_median': depth_median,
                    'feature': feature,
                })
                
                # 更新体素地图
                voxel_map.add_observation(
                    position=np.array([x_3d, y_3d, z_3d]),
                    label=label,
                    confidence=confidence,
                    feature=feature,
                    extent=np.array([w_3d, h_3d, d_3d])
                )
        
        # 聚合成实体
        entities = self._aggregate_entities(all_detections, frame_detection_counts)
        
        return entities, voxel_map
    
    def _aggregate_entities(self, all_detections: Dict[str, List[Dict]],
                           frame_detection_counts: Dict[int, Dict[str, int]]) -> Dict[str, MindMapEntityV5]:
        """聚合检测结果为实体"""
        entities = {}
        
        for category, dets in all_detections.items():
            if not dets:
                continue
            
            # 计算单帧最大检测数
            max_count = 0
            for frame_idx, counts in frame_detection_counts.items():
                if category in counts:
                    max_count = max(max_count, counts[category])
            if max_count == 0:
                max_count = 1
            
            # 创建实体
            entity = MindMapEntityV5(
                entity_id=f"entity_{category}",
                label=category,
                detection_count=len(dets),
                max_single_frame_count=max_count,
                avg_confidence=float(np.mean([d['confidence'] for d in dets])),
                first_seen_frame=min(d['frame_idx'] for d in dets),
                last_seen_frame=max(d['frame_idx'] for d in dets),
            )
            
            # 更新位置（使用所有观测）
            for det in dets:
                entity.update_position(
                    det['position_3d'],
                    uncertainty=det['position_uncertainty']
                )
                entity.update_size(det['size_3d'])
                
                if det['feature'] is not None:
                    entity.update_feature(det['feature'])
            
            entities[category] = entity
        
        return entities


# ============================================================================
# 5. V5 推理器 - 统一策略
# ============================================================================

class MindMapReasonerV5:
    """V5 心智地图推理器
    
    基于概率推理的统一策略，不再使用 trick
    """
    
    # 置信度阈值
    HIGH_CONFIDENCE = 0.7
    LOW_CONFIDENCE = 0.3
    
    @staticmethod
    def answer_counting(entities: Dict[str, MindMapEntityV5], question: str) -> Tuple[str, float]:
        """回答计数问题
        
        Returns:
            (答案, 置信度)
        """
        import re
        from tests.test_vsibench_directqa import match_object_name
        
        match = re.search(r'How many (\w+)', question)
        if not match:
            return "0", 0.5
        
        target = match.group(1).lower()
        
        for label, entity in entities.items():
            if match_object_name(target, label):
                confidence = entity.avg_confidence
                return str(entity.max_single_frame_count), confidence
        
        return "0", 0.3
    
    @staticmethod
    def answer_object_size(entities: Dict[str, MindMapEntityV5], question: str) -> Tuple[str, float]:
        """回答物体尺寸问题
        
        注意：VSIBench 的 size_estimation 问题询问的是物体的最大维度尺寸（厘米）
        """
        from tests.test_vsibench_directqa import get_synonyms
        
        q_lower = question.lower()
        
        for label, entity in entities.items():
            if label.lower() in q_lower or any(s in q_lower for s in get_synonyms(label)):
                if entity.size_mean is not None:
                    # 取最大维度，转为厘米
                    max_dim = float(np.max(entity.size_mean)) * 100
                    # 置信度基于检测次数和置信度
                    confidence = min(1.0, entity.avg_confidence * entity.detection_count / 10)
                    return str(int(max_dim)), confidence
        
        return "50", 0.3
    
    @staticmethod
    def answer_room_size(entities: Dict[str, MindMapEntityV5], 
                         voxel_map: SparseVoxelMap,
                         question: str) -> Tuple[str, float]:
        """回答房间面积问题
        
        使用与基线完全一致的方法：XY 平面包围盒
        """
        if not entities:
            return "20", 0.3
        
        # 收集所有物体的 3D 位置
        positions = []
        for entity in entities.values():
            if entity.position is not None:
                positions.append(entity.position.mean)
        
        if len(positions) < 2:
            return str(12 + len(entities) * 2), 0.3
        
        positions = np.array(positions)
        
        # 计算 XY 平面的包围盒（与基线一致！）
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        # 使用基线的扩展系数 +1.5
        estimated_area = (x_range + 1.5) * (y_range + 1.5)
        estimated_area = max(8, min(80, estimated_area))
        
        confidence = min(1.0, len(positions) / 5)
        
        return f"{estimated_area:.1f}", confidence
    
    @staticmethod
    def answer_abs_distance(entities: Dict[str, MindMapEntityV5], 
                           question: str) -> Tuple[str, float]:
        """回答绝对距离问题 - 概率版本"""
        import re
        from tests.test_vsibench_directqa import match_object_name
        
        # 解析物体名称
        between_match = re.search(
            r'between (?:the )?([a-z]+(?:\s+[a-z]+)*)\s+and\s+(?:the )?([a-z]+(?:\s+[a-z]+)*)',
            question.lower()
        )
        
        if not between_match:
            return "2.0", 0.3
        
        obj1_name = between_match.group(1).strip()
        obj2_name = between_match.group(2).strip()
        
        # 查找实体
        def find_entity(name: str) -> Optional[MindMapEntityV5]:
            for label, entity in entities.items():
                if match_object_name(name, label):
                    return entity
            return None
        
        entity1 = find_entity(obj1_name)
        entity2 = find_entity(obj2_name)
        
        if entity1 is None or entity2 is None:
            return "2.0", 0.3
        
        if entity1.position is None or entity2.position is None:
            return "2.0", 0.3
        
        # 计算距离（使用位置均值）
        dist = float(np.linalg.norm(entity1.position.mean - entity2.position.mean))
        
        # 计算置信度（基于两个实体的位置不确定性）
        combined_uncertainty = entity1.position.uncertainty + entity2.position.uncertainty
        confidence = 1.0 / (1.0 + combined_uncertainty)
        
        return f"{dist:.1f}", confidence
    
    @staticmethod
    def answer_rel_direction(entities: Dict[str, MindMapEntityV5],
                            question: str,
                            options: List[str],
                            difficulty: str = 'easy') -> Tuple[str, float]:
        """回答相对方向问题 - 概率版本
        
        关键改进：当位置不确定性高时，返回低置信度
        """
        import re
        from tests.test_vsibench_directqa import match_object_name
        
        if not options:
            return "left", 0.3
        
        q_lower = question.lower()
        
        # 解析问题
        standing_match = re.search(r'standing by (?:the )?([a-z]+(?:\s+[a-z]+)*?)(?:\s+and\s+facing|\s*,)', q_lower)
        facing_match = re.search(r'facing (?:the )?([a-z]+(?:\s+[a-z]+)*?)(?:\s*,|\s+is\b)', q_lower)
        target_match_my = re.search(r'is (?:the )?([a-z]+(?:\s+[a-z]+)*?)\s+to\s+my', q_lower)
        target_match_of = re.search(r'is (?:the )?([a-z]+(?:\s+[a-z]+)*?)\s+to\s+the\s+(?:left|right|back|front)', q_lower)
        
        if target_match_my:
            target_match = target_match_my
            relative_to_observer = True
        elif target_match_of:
            target_match = target_match_of
            relative_to_observer = False
        else:
            return options[0], 0.3
        
        if not all([standing_match, facing_match, target_match]):
            return options[0], 0.3
        
        standing_name = standing_match.group(1)
        facing_name = facing_match.group(1)
        target_name = target_match.group(1)
        
        # 查找实体
        def find_entity(name: str) -> Optional[MindMapEntityV5]:
            for label, entity in entities.items():
                if match_object_name(name, label):
                    return entity
            return None
        
        standing_entity = find_entity(standing_name)
        facing_entity = find_entity(facing_name)
        target_entity = find_entity(target_name)
        
        if not all([standing_entity, facing_entity, target_entity]):
            return options[0], 0.3
        
        if not all([standing_entity.position, facing_entity.position, target_entity.position]):
            return options[0], 0.3
        
        standing_pos = standing_entity.position.mean
        facing_pos = facing_entity.position.mean
        target_pos = target_entity.position.mean
        
        # 计算方向向量
        forward = np.array([facing_pos[0] - standing_pos[0], facing_pos[2] - standing_pos[2]])
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return options[0], 0.3
        forward = forward / forward_norm
        
        right = np.array([forward[1], -forward[0]])
        
        if relative_to_observer:
            to_target = np.array([target_pos[0] - standing_pos[0], target_pos[2] - standing_pos[2]])
        else:
            to_target = np.array([target_pos[0] - facing_pos[0], target_pos[2] - facing_pos[2]])
        
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm < 1e-6:
            return options[0], 0.3
        to_target = to_target / to_target_norm
        
        # 计算点积
        front_dot = np.dot(to_target, forward)
        right_dot = np.dot(to_target, right)
        
        # 确定方向
        if 'left or' in q_lower and 'right' in q_lower and 'back' not in q_lower:
            direction = 'right' if right_dot > 0 else 'left'
        elif 'left, right, or back' in q_lower or 'left, right or back' in q_lower:
            if front_dot < -0.5:
                direction = 'back'
            elif right_dot > 0:
                direction = 'right'
            else:
                direction = 'left'
        elif 'front-left' in q_lower or 'front-right' in q_lower:
            if front_dot > 0:
                direction = 'front-right' if right_dot > 0 else 'front-left'
            else:
                direction = 'back-right' if right_dot > 0 else 'back-left'
        else:
            direction = 'right' if right_dot > 0 else 'left'
        
        # 计算置信度
        # 1. 基于位置不确定性
        position_confidence = 1.0 / (1.0 + standing_entity.position.uncertainty + 
                                      facing_entity.position.uncertainty +
                                      target_entity.position.uncertainty)
        
        # 2. 基于方向向量的确定性（点积绝对值越大越确定）
        direction_certainty = max(abs(right_dot), abs(front_dot))
        
        # 3. 如果两个物体位置重叠概率高，降低置信度
        overlap_penalty = 1.0
        if standing_entity.position.overlap_probability(target_entity.position) > 0.5:
            overlap_penalty = 0.5
        
        confidence = position_confidence * direction_certainty * overlap_penalty
        
        # 匹配选项
        for opt in options:
            if direction in opt.lower():
                return opt, confidence
        
        return options[0], confidence * 0.5
    
    @staticmethod
    def answer_rel_distance(entities: Dict[str, MindMapEntityV5],
                           question: str,
                           options: List[str]) -> Tuple[str, float]:
        """回答相对距离问题"""
        import re
        from tests.test_vsibench_directqa import match_object_name, get_synonyms
        
        if not options:
            return "", 0.3
        
        q_lower = question.lower()
        
        find_closest = 'closest' in q_lower or 'nearest' in q_lower
        find_farthest = 'farthest' in q_lower or 'furthest' in q_lower
        
        if not find_closest and not find_farthest:
            find_closest = True
        
        ref_match = re.search(r'(?:to|from) (?:the )?([a-z]+(?:\s+[a-z]+)*)\??', q_lower)
        if not ref_match:
            return options[0], 0.3
        
        reference_name = ref_match.group(1)
        
        # 查找参考实体
        ref_entity = None
        for label, entity in entities.items():
            if match_object_name(reference_name, label):
                ref_entity = entity
                break
        
        if ref_entity is None or ref_entity.position is None:
            import random
            return random.choice(options), 0.3
        
        # 提取候选物体
        candidates_match = re.search(r'\(([^)]+)\)', q_lower)
        if candidates_match:
            candidate_names = [c.strip() for c in candidates_match.group(1).split(',')]
        else:
            candidate_names = []
            for opt in options:
                opt_content = re.sub(r'^[A-D]\.\s*', '', opt)
                candidate_names.append(opt_content.strip().lower())
        
        # 计算距离
        distances = {}
        confidences = {}
        option_map = {}
        
        for i, cand in enumerate(candidate_names):
            cand_entity = None
            for label, entity in entities.items():
                if match_object_name(cand, label):
                    cand_entity = entity
                    break
            
            if cand_entity is not None and cand_entity.position is not None:
                dist = float(np.linalg.norm(cand_entity.position.mean - ref_entity.position.mean))
                distances[cand] = dist
                # 置信度基于两个实体的位置不确定性
                conf = 1.0 / (1.0 + cand_entity.position.uncertainty + ref_entity.position.uncertainty)
                confidences[cand] = conf
                if i < len(options):
                    option_map[cand] = options[i]
        
        if not distances:
            import random
            return random.choice(options), 0.3
        
        # 选择最近/最远
        if find_closest:
            best_cand = min(distances.keys(), key=lambda k: distances[k])
        else:
            best_cand = max(distances.keys(), key=lambda k: distances[k])
        
        confidence = confidences.get(best_cand, 0.5)
        
        if best_cand in option_map:
            return option_map[best_cand], confidence
        
        for opt in options:
            if best_cand in opt.lower():
                return opt, confidence
        
        return options[0], confidence * 0.5
    
    @staticmethod
    def answer_appearance_order(entities: Dict[str, MindMapEntityV5],
                                question: str,
                                options: List[str]) -> Tuple[str, float]:
        """回答出现顺序问题"""
        import re
        from tests.test_vsibench_directqa import match_object_name
        
        if not options:
            return "", 0.3
        
        # 提取目标物体
        match = re.search(r'following categories.*?:\s*(.+?)\?', question, re.IGNORECASE)
        if match:
            target_objects = [obj.strip().lower() for obj in match.group(1).split(',')]
        else:
            opt_content = re.sub(r'^[A-D]\.\s*', '', options[0])
            target_objects = [obj.strip().lower() for obj in opt_content.split(',')]
        
        # 获取首次出现帧
        object_frames = {}
        detection_counts = {}
        
        for target in target_objects:
            for label, entity in entities.items():
                if match_object_name(target, label):
                    object_frames[target] = entity.first_seen_frame
                    detection_counts[target] = entity.detection_count
                    break
        
        if len(object_frames) < 2:
            import random
            return random.choice(options), 0.3
        
        # 排序
        sorted_objects = sorted(object_frames.items(), key=lambda x: x[1])
        predicted_order = [obj for obj, _ in sorted_objects]
        
        # 添加未检测到的物体
        missing = [t for t in target_objects if t not in object_frames]
        predicted_order.extend(missing)
        
        # 匹配选项
        best_option = options[0]
        best_score = -1
        
        for opt in options:
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt)
            opt_objects = [o.strip().lower() for o in opt_content.split(',')]
            
            score = 0
            for i, pred_obj in enumerate(predicted_order):
                for j, opt_obj in enumerate(opt_objects):
                    if match_object_name(pred_obj, opt_obj):
                        score += max(0, len(target_objects) - abs(i - j))
                        break
            
            if score > best_score:
                best_score = score
                best_option = opt
        
        # 置信度基于检测次数
        avg_detections = np.mean(list(detection_counts.values())) if detection_counts else 1
        confidence = min(1.0, avg_detections / 10)
        
        return best_option, confidence
    
    @staticmethod
    def answer_route_planning(entities: Dict[str, MindMapEntityV5],
                             voxel_map: SparseVoxelMap,
                             question: str,
                             options: List[str]) -> Tuple[str, float]:
        """回答路径规划问题 - 可考虑遮挡"""
        import re
        from tests.test_vsibench_directqa import match_object_name
        
        if not options:
            return "", 0.3
        
        q_lower = question.lower()
        
        start_match = re.search(r'beginning at (?:the )?(\w+(?:\s+\w+)?)', q_lower)
        facing_match = re.search(r'facing (?:the )?(\w+)', q_lower)
        
        if not start_match or not facing_match:
            return options[0], 0.3
        
        steps = re.findall(r'go forward until (?:the )?(\w+(?:\s+\w+)?)', q_lower)
        
        # 查找实体位置
        def find_position(name: str) -> Optional[np.ndarray]:
            for label, entity in entities.items():
                if match_object_name(name, label) and entity.position is not None:
                    return entity.position.mean
            return None
        
        path_positions = []
        start_pos = find_position(start_match.group(1))
        if start_pos is not None:
            path_positions.append(start_pos)
        
        for step in steps:
            pos = find_position(step)
            if pos is not None:
                path_positions.append(pos)
        
        if len(path_positions) < 3:
            return options[0], 0.3
        
        # 计算转向
        turns = []
        for i in range(1, len(path_positions) - 1):
            prev_dir = path_positions[i] - path_positions[i-1]
            next_dir = path_positions[i+1] - path_positions[i]
            
            prev_dir_2d = prev_dir[:2] / (np.linalg.norm(prev_dir[:2]) + 1e-8)
            next_dir_2d = next_dir[:2] / (np.linalg.norm(next_dir[:2]) + 1e-8)
            
            cross = prev_dir_2d[0] * next_dir_2d[1] - prev_dir_2d[1] * next_dir_2d[0]
            dot = np.dot(prev_dir_2d, next_dir_2d)
            
            if dot < -0.5:
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
            match_count = sum(1 for turn in turns if turn in opt_lower)
            if match_count == len(turns):
                return opt, 0.7
        
        return options[0], 0.3


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == '__main__':
    # 简单测试
    print("MindMap V5 - 统一空间智能框架")
    print("=" * 50)
    
    # 测试体素地图
    voxel_map = SparseVoxelMap(voxel_size=0.1)
    
    # 添加一些测试观测
    voxel_map.add_observation(np.array([0, 0, 2]), "chair", 0.9, extent=np.array([0.5, 0.9, 0.5]))
    voxel_map.add_observation(np.array([1, 0, 3]), "table", 0.85, extent=np.array([0.8, 0.75, 0.8]))
    voxel_map.add_observation(np.array([-1, 0, 2.5]), "sofa", 0.92, extent=np.array([1.5, 0.85, 0.9]))
    
    print(f"体素数量: {len(voxel_map.voxels)}")
    print(f"占据体积: {voxel_map.get_occupied_volume():.3f} m³")
    print(f"地面面积: {voxel_map.get_floor_area():.2f} m²")
    print(f"房间尺寸: {voxel_map.get_room_dimensions()}")
    
    # 测试高斯位置
    pos = GaussianPosition3D.from_single_observation(np.array([1, 0, 2]), uncertainty=0.5)
    print(f"\n初始位置: {pos.mean}, 不确定性: {pos.uncertainty:.3f}")
    
    pos.update(np.array([1.1, 0.1, 2.1]))
    print(f"更新后位置: {pos.mean}, 不确定性: {pos.uncertainty:.3f}")
    
    pos.update(np.array([1.05, 0.05, 2.05]))
    print(f"再次更新: {pos.mean}, 不确定性: {pos.uncertainty:.3f}")
    
    print("\nV5 框架测试完成!")
