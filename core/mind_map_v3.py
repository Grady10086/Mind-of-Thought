"""
心智地图 V3 - 改进版本

核心改进：
1. 动态尺度校准 - 利用已知物体尺寸反推深度尺度因子
2. 实例级追踪 - 区分多个同类物体 (chair_001, chair_002)
3. 概率性表示 - 位置/尺寸用高斯分布，支持不确定性推理
4. 相机位姿估计 - 统一到世界坐标系（简化版，基于帧间光流）

设计原则：
- 增量式改进，兼容原有接口
- 每个模块可独立测试
- 保持原有指标不变差

作者: tianjungu
日期: 2026-01-28
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

logger = logging.getLogger(__name__)


# ============================================================================
# 已知物体尺寸数据库 (用于动态尺度校准)
# ============================================================================

KNOWN_OBJECT_SIZES = {
    # 门窗类 - 尺寸较标准，适合校准
    'door': {'height': 2.0, 'width': 0.9},
    'window': {'height': 1.2, 'width': 1.0},
    
    # 大型家具 - 尺寸相对稳定
    'bed': {'height': 0.5, 'width': 1.5, 'length': 2.0},
    'sofa': {'height': 0.85, 'width': 0.9, 'length': 2.0},
    'couch': {'height': 0.85, 'width': 0.9, 'length': 2.0},
    'refrigerator': {'height': 1.7, 'width': 0.7, 'depth': 0.7},
    'fridge': {'height': 1.7, 'width': 0.7, 'depth': 0.7},
    'wardrobe': {'height': 2.0, 'width': 1.2, 'depth': 0.6},
    'closet': {'height': 2.0, 'width': 1.2, 'depth': 0.6},
    
    # 中型家具
    'chair': {'height': 0.9, 'seat_height': 0.45, 'width': 0.5},
    'table': {'height': 0.75, 'width': 0.8, 'length': 1.2},
    'desk': {'height': 0.75, 'width': 0.6, 'length': 1.2},
    'toilet': {'height': 0.4, 'width': 0.4, 'depth': 0.65},
    'sink': {'height': 0.85, 'width': 0.5},
    'bathtub': {'height': 0.6, 'width': 0.75, 'length': 1.5},
    
    # 电器
    'tv': {'height': 0.6, 'width': 1.0},  # 约 43 英寸
    'television': {'height': 0.6, 'width': 1.0},
    'monitor': {'height': 0.35, 'width': 0.55},  # 约 24 英寸
    'washer': {'height': 0.85, 'width': 0.6, 'depth': 0.6},
    'dryer': {'height': 0.85, 'width': 0.6, 'depth': 0.6},
    
    # 小型物品 (用于辅助验证，不作为主要校准依据)
    'pillow': {'height': 0.15, 'width': 0.5},
    'bottle': {'height': 0.25, 'width': 0.08},
}


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class Detection:
    """单次检测结果"""
    frame_idx: int
    label: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    
    # 相机坐标系下的位置
    position_camera: Optional[np.ndarray] = None  # (3,)
    size_estimated: Optional[np.ndarray] = None   # (3,)
    depth_median: float = 0.0


@dataclass
class CameraPose:
    """相机位姿 (简化版)"""
    frame_idx: int
    R: np.ndarray = field(default_factory=lambda: np.eye(3))  # 旋转矩阵
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 平移向量
    
    def camera_to_world(self, point_camera: np.ndarray) -> np.ndarray:
        """相机坐标 → 世界坐标"""
        # world = R^T @ (camera - t) = R^T @ camera - R^T @ t
        # 简化：假设相机原点在世界坐标系原点移动
        return self.R.T @ point_camera + self.t
    
    def world_to_camera(self, point_world: np.ndarray) -> np.ndarray:
        """世界坐标 → 相机坐标"""
        return self.R @ (point_world - self.t)


@dataclass 
class TrackedInstance:
    """追踪的物体实例"""
    instance_id: str              # "chair_001"
    label: str                    # "chair"
    
    # 观测历史
    detections: List[Detection] = field(default_factory=list)
    
    # 世界坐标系下的聚合位置
    world_positions: List[np.ndarray] = field(default_factory=list)
    
    # 统计量
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_detections: int = 0
    avg_confidence: float = 0.0
    
    def get_position_mean(self) -> Optional[np.ndarray]:
        """获取位置均值"""
        if not self.world_positions:
            return None
        return np.mean(self.world_positions, axis=0)
    
    def get_position_cov(self) -> Optional[np.ndarray]:
        """获取位置协方差矩阵"""
        if len(self.world_positions) < 2:
            # 观测太少，返回较大的协方差
            return np.eye(3) * 0.5
        positions = np.array(self.world_positions)
        return np.cov(positions.T) + np.eye(3) * 0.01  # 加小量防止奇异
    
    def get_size_mean(self) -> Optional[np.ndarray]:
        """获取尺寸均值"""
        sizes = [d.size_estimated for d in self.detections if d.size_estimated is not None]
        if not sizes:
            return None
        return np.median(sizes, axis=0)


@dataclass
class ProbabilisticEntity:
    """概率性实体表示 (心智地图中的物体)"""
    instance_id: str
    label: str
    
    # 位置的高斯分布
    position_mean: np.ndarray         # (3,) 均值
    position_cov: np.ndarray          # (3,3) 协方差矩阵
    
    # 尺寸估计
    size_mean: np.ndarray             # (3,)
    size_std: np.ndarray              # (3,)
    
    # 置信度
    existence_probability: float      # 物体存在的概率 [0, 1]
    observation_count: int            # 观测次数
    
    # 时序信息
    first_seen_frame: int
    last_seen_frame: int
    
    def get_confidence_radius(self, sigma: float = 2.0) -> float:
        """获取位置置信区间半径 (2-sigma 椭球)"""
        eigenvalues = np.linalg.eigvalsh(self.position_cov)
        return sigma * np.sqrt(np.max(eigenvalues))
    
    def position_uncertainty(self) -> float:
        """位置不确定性 (协方差矩阵的迹)"""
        return np.trace(self.position_cov)


# ============================================================================
# 动态尺度校准器
# ============================================================================

class DynamicScaleCalibrator:
    """
    动态尺度校准器
    
    原理：
    - DA3 输出的是相对深度，需要乘以尺度因子得到绝对深度
    - 利用检测到的已知尺寸物体（门、床、冰箱等）反推尺度因子
    - 多物体取中值，增强鲁棒性
    """
    
    def __init__(self, focal_length: float = 500.0):
        self.focal_length = focal_length
        self.scale_history: List[float] = []
        self.default_scale = 1.0
        
    def calibrate_from_detections(
        self, 
        detections: List[Detection],
        depth_map: np.ndarray,
        image_height: int,
    ) -> float:
        """
        从检测结果校准尺度因子
        
        Args:
            detections: 当前帧的检测结果
            depth_map: 深度图 (H, W)
            image_height: 图像高度
            
        Returns:
            尺度因子
        """
        scale_estimates = []
        
        for det in detections:
            label_lower = det.label.lower()
            
            # 检查是否有已知尺寸
            known_size = None
            for key in KNOWN_OBJECT_SIZES:
                if key in label_lower or label_lower in key:
                    known_size = KNOWN_OBJECT_SIZES[key]
                    break
            
            if known_size is None:
                continue
            
            # 获取物体的主要参考尺寸（优先用高度）
            ref_dimension = known_size.get('height')
            if ref_dimension is None:
                ref_dimension = known_size.get('width')
            if ref_dimension is None:
                continue
            
            # 计算当前估计的 3D 高度
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
            
            if y2 <= y1 or x2 <= x1:
                continue
            
            h_pixel = y2 - y1
            
            # 获取物体区域的中值深度
            depth_roi = depth_map[y1:y2, x1:x2]
            if depth_roi.size == 0:
                continue
            depth_median = np.median(depth_roi)
            if depth_median <= 0:
                continue
            
            # 计算估计的 3D 高度
            h_estimated = h_pixel / self.focal_length * depth_median
            
            # 计算尺度因子
            if h_estimated > 0.01:  # 避免除以太小的数
                scale = ref_dimension / h_estimated
                
                # 合理性检查：尺度因子应该在 [0.1, 10] 范围内
                if 0.1 < scale < 10:
                    scale_estimates.append(scale)
                    logger.debug(f"尺度校准: {label_lower}, 真实={ref_dimension:.2f}m, "
                               f"估计={h_estimated:.2f}m, scale={scale:.2f}")
        
        if scale_estimates:
            # 使用中值，对异常值更鲁棒
            scale = float(np.median(scale_estimates))
            self.scale_history.append(scale)
            
            # 保持最近 10 帧的历史
            if len(self.scale_history) > 10:
                self.scale_history = self.scale_history[-10:]
            
            return scale
        
        # 如果当前帧没有可用的参考物体，使用历史中值
        if self.scale_history:
            return float(np.median(self.scale_history))
        
        return self.default_scale
    
    def get_smoothed_scale(self) -> float:
        """获取平滑后的尺度因子"""
        if self.scale_history:
            return float(np.median(self.scale_history))
        return self.default_scale


# ============================================================================
# 实例级追踪器
# ============================================================================

class InstanceTracker:
    """
    实例级追踪器
    
    解决问题：
    - 原来按类别聚合 "chair"，场景中 3 把椅子会变成 1 把"幽灵椅"
    - 现在追踪每个实例：chair_001, chair_002, chair_003
    
    算法：
    - 基于 3D 位置的匈牙利匹配
    - 新检测与现有 track 匹配，距离超过阈值则创建新 track
    """
    
    def __init__(
        self, 
        distance_threshold: float = 1.5,  # 匹配距离阈值 (米)
        max_lost_frames: int = 10,        # 允许丢失的最大帧数
    ):
        self.distance_threshold = distance_threshold
        self.max_lost_frames = max_lost_frames
        
        # 活跃的追踪实例
        self.active_tracks: Dict[str, TrackedInstance] = {}
        
        # 每个类别的下一个 ID
        self.next_id: Dict[str, int] = defaultdict(int)
        
        # 当前帧
        self.current_frame = 0
    
    def _generate_instance_id(self, label: str) -> str:
        """生成实例 ID"""
        self.next_id[label] += 1
        return f"{label}_{self.next_id[label]:03d}"
    
    def update(
        self, 
        frame_idx: int,
        detections: List[Detection],
        camera_pose: Optional[CameraPose] = None,
    ) -> Dict[str, TrackedInstance]:
        """
        更新追踪
        
        Args:
            frame_idx: 当前帧索引
            detections: 当前帧的检测结果（已包含 position_camera）
            camera_pose: 相机位姿（用于转换到世界坐标）
            
        Returns:
            更新后的所有追踪实例
        """
        self.current_frame = frame_idx
        
        # 如果没有相机位姿，使用单位变换
        if camera_pose is None:
            camera_pose = CameraPose(frame_idx=frame_idx)
        
        # 按类别分组检测
        detections_by_label: Dict[str, List[Detection]] = defaultdict(list)
        for det in detections:
            if det.position_camera is not None:
                detections_by_label[det.label.lower()].append(det)
        
        # 对每个类别分别进行匹配
        matched_track_ids = set()
        
        for label, label_detections in detections_by_label.items():
            # 获取该类别的现有 tracks
            existing_tracks = [
                (tid, track) for tid, track in self.active_tracks.items()
                if track.label.lower() == label
            ]
            
            if not existing_tracks and not label_detections:
                continue
            
            # 转换检测到世界坐标
            world_positions = []
            for det in label_detections:
                world_pos = camera_pose.camera_to_world(det.position_camera)
                world_positions.append(world_pos)
            
            if not existing_tracks:
                # 没有现有 track，全部创建新的
                for det, world_pos in zip(label_detections, world_positions):
                    self._create_track(det, world_pos, frame_idx)
            elif not label_detections:
                # 没有新检测，现有 track 保持
                pass
            else:
                # 构建代价矩阵并匹配
                cost_matrix = np.zeros((len(existing_tracks), len(label_detections)))
                
                for i, (tid, track) in enumerate(existing_tracks):
                    track_pos = track.get_position_mean()
                    if track_pos is None:
                        cost_matrix[i, :] = self.distance_threshold * 2
                    else:
                        for j, world_pos in enumerate(world_positions):
                            cost_matrix[i, j] = np.linalg.norm(track_pos - world_pos)
                
                # 匈牙利匹配
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                matched_det_indices = set()
                for track_idx, det_idx in zip(row_ind, col_ind):
                    if cost_matrix[track_idx, det_idx] < self.distance_threshold:
                        # 匹配成功，更新 track
                        tid, track = existing_tracks[track_idx]
                        det = label_detections[det_idx]
                        world_pos = world_positions[det_idx]
                        
                        self._update_track(track, det, world_pos, frame_idx)
                        matched_track_ids.add(tid)
                        matched_det_indices.add(det_idx)
                
                # 未匹配的检测创建新 track
                for det_idx, (det, world_pos) in enumerate(zip(label_detections, world_positions)):
                    if det_idx not in matched_det_indices:
                        self._create_track(det, world_pos, frame_idx)
        
        # 清理长期丢失的 tracks
        self._cleanup_lost_tracks(frame_idx)
        
        return self.active_tracks
    
    def _create_track(self, det: Detection, world_pos: np.ndarray, frame_idx: int):
        """创建新的追踪实例"""
        instance_id = self._generate_instance_id(det.label.lower())
        
        track = TrackedInstance(
            instance_id=instance_id,
            label=det.label.lower(),
            detections=[det],
            world_positions=[world_pos],
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
            total_detections=1,
            avg_confidence=det.confidence,
        )
        
        self.active_tracks[instance_id] = track
        logger.debug(f"创建新 track: {instance_id} at {world_pos}")
    
    def _update_track(self, track: TrackedInstance, det: Detection, 
                      world_pos: np.ndarray, frame_idx: int):
        """更新现有追踪实例"""
        track.detections.append(det)
        track.world_positions.append(world_pos)
        track.last_seen_frame = frame_idx
        track.total_detections += 1
        
        # 更新平均置信度
        track.avg_confidence = (
            track.avg_confidence * (track.total_detections - 1) + det.confidence
        ) / track.total_detections
    
    def _cleanup_lost_tracks(self, frame_idx: int):
        """清理长期丢失的 tracks"""
        to_remove = []
        for tid, track in self.active_tracks.items():
            if frame_idx - track.last_seen_frame > self.max_lost_frames:
                # 只有观测次数少于 2 才删除
                if track.total_detections < 2:
                    to_remove.append(tid)
        
        for tid in to_remove:
            del self.active_tracks[tid]
            logger.debug(f"删除丢失的 track: {tid}")
    
    def get_instances_by_label(self, label: str) -> List[TrackedInstance]:
        """获取指定类别的所有实例"""
        label_lower = label.lower()
        return [
            track for track in self.active_tracks.values()
            if track.label == label_lower or label_lower in track.label or track.label in label_lower
        ]
    
    def get_instance_count(self, label: str) -> int:
        """获取指定类别的实例数量"""
        return len(self.get_instances_by_label(label))


# ============================================================================
# 概率性心智地图
# ============================================================================

class ProbabilisticMindMap:
    """
    概率性心智地图
    
    特点：
    - 每个物体用高斯分布表示位置
    - 支持不确定性推理
    - 可以判断是否需要重新观察
    """
    
    def __init__(self):
        self.entities: Dict[str, ProbabilisticEntity] = {}
        
    def build_from_tracker(self, tracker: InstanceTracker, 
                           min_observations: int = 1):
        """从追踪器构建概率性心智地图"""
        self.entities.clear()
        
        for instance_id, track in tracker.active_tracks.items():
            if track.total_detections < min_observations:
                continue
            
            pos_mean = track.get_position_mean()
            pos_cov = track.get_position_cov()
            size_mean = track.get_size_mean()
            
            if pos_mean is None:
                continue
            
            if size_mean is None:
                size_mean = np.array([0.5, 0.5, 0.5])
            
            entity = ProbabilisticEntity(
                instance_id=instance_id,
                label=track.label,
                position_mean=pos_mean,
                position_cov=pos_cov,
                size_mean=size_mean,
                size_std=size_mean * 0.2,  # 假设 20% 标准差
                existence_probability=min(1.0, 0.5 + track.total_detections * 0.1),
                observation_count=track.total_detections,
                first_seen_frame=track.first_seen_frame,
                last_seen_frame=track.last_seen_frame,
            )
            
            self.entities[instance_id] = entity
    
    def get_entity_by_label(self, label: str) -> Optional[ProbabilisticEntity]:
        """获取指定标签的实体（返回置信度最高的）"""
        label_lower = label.lower()
        candidates = []
        
        for entity in self.entities.values():
            if (entity.label == label_lower or 
                label_lower in entity.label or 
                entity.label in label_lower):
                candidates.append(entity)
        
        if not candidates:
            return None
        
        # 返回观测次数最多的
        return max(candidates, key=lambda e: e.observation_count)
    
    def get_all_entities_by_label(self, label: str) -> List[ProbabilisticEntity]:
        """获取指定标签的所有实体"""
        label_lower = label.lower()
        return [
            e for e in self.entities.values()
            if e.label == label_lower or label_lower in e.label or e.label in label_lower
        ]
    
    def query_direction_probabilistic(
        self,
        observer_pos: np.ndarray,
        facing_pos: np.ndarray,
        target_entity: ProbabilisticEntity,
        n_samples: int = 500,
    ) -> Dict[str, float]:
        """
        概率性方向查询 - Monte Carlo 采样
        
        Returns:
            {"left": 0.7, "right": 0.2, "back": 0.1}
        """
        direction_counts = {"left": 0, "right": 0, "front": 0, "back": 0}
        
        for _ in range(n_samples):
            # 从目标位置分布采样
            try:
                target_pos = np.random.multivariate_normal(
                    target_entity.position_mean, 
                    target_entity.position_cov
                )
            except:
                target_pos = target_entity.position_mean
            
            # 计算方向
            direction = self._compute_direction_single(
                observer_pos, facing_pos, target_pos
            )
            direction_counts[direction] += 1
        
        # 转换为概率
        total = sum(direction_counts.values())
        return {k: v / total for k, v in direction_counts.items()}
    
    def _compute_direction_single(
        self,
        observer_pos: np.ndarray,
        facing_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> str:
        """计算单次方向判断"""
        # 使用 X-Z 平面（俯视图）
        forward = np.array([
            facing_pos[0] - observer_pos[0],
            facing_pos[2] - observer_pos[2]
        ])
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return "front"
        forward = forward / forward_norm
        
        # 右方向向量
        right = np.array([forward[1], -forward[0]])
        
        # 目标相对向量
        to_target = np.array([
            target_pos[0] - observer_pos[0],
            target_pos[2] - observer_pos[2]
        ])
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm < 1e-6:
            return "front"
        to_target = to_target / to_target_norm
        
        # 计算点积
        front_dot = np.dot(to_target, forward)
        right_dot = np.dot(to_target, right)
        
        # 判断方向
        if front_dot < -0.5:
            return "back"
        elif right_dot > 0:
            return "right"
        else:
            return "left"
    
    def needs_reobservation(self, entity: ProbabilisticEntity, 
                           task_type: str) -> bool:
        """
        判断是否需要重新观察
        
        触发条件：
        - 存在概率太低
        - 位置不确定性太高（相对于任务要求）
        """
        if entity.existence_probability < 0.6:
            return True
        
        # 任务相关的不确定性阈值
        uncertainty_thresholds = {
            "rel_direction": 1.0,   # 方向任务允许较大不确定性
            "abs_distance": 0.5,    # 距离任务需要较高精度
            "rel_distance": 0.8,    # 相对距离允许中等不确定性
            "counting": float('inf'),  # 计数任务不关心位置
        }
        
        threshold = uncertainty_thresholds.get(task_type, 1.0)
        if entity.get_confidence_radius() > threshold:
            return True
        
        return False


# ============================================================================
# 兼容层：转换为旧格式的 MindMapEntity3D
# ============================================================================

def convert_to_legacy_mindmap(
    prob_mind_map: ProbabilisticMindMap,
    tracker: InstanceTracker,
) -> Dict[str, Any]:
    """
    将概率性心智地图转换为旧格式
    
    旧格式：Dict[label, MindMapEntity3D] (按类别聚合)
    新格式：Dict[instance_id, ProbabilisticEntity] (按实例)
    
    为了兼容原有的 DirectQA 函数，需要转换回按类别聚合的格式
    """
    from dataclasses import dataclass
    
    @dataclass
    class LegacyEntity:
        entity_id: str
        label: str
        count: int
        avg_confidence: float
        first_seen_frame: int
        last_seen_frame: int
        position_3d: np.ndarray
        size_3d: np.ndarray
        depth_median: float
    
    legacy_map = {}
    
    # 按类别分组
    entities_by_label: Dict[str, List[ProbabilisticEntity]] = defaultdict(list)
    for entity in prob_mind_map.entities.values():
        entities_by_label[entity.label].append(entity)
    
    for label, entities in entities_by_label.items():
        # 计算类别级别的聚合统计
        count = len(entities)  # 实例数量
        
        # 位置取最置信的实例
        best_entity = max(entities, key=lambda e: e.observation_count)
        position_3d = best_entity.position_mean
        size_3d = best_entity.size_mean
        
        # 置信度取平均
        avg_conf = np.mean([e.existence_probability for e in entities])
        
        # 首次/最后出现帧
        first_frame = min(e.first_seen_frame for e in entities)
        last_frame = max(e.last_seen_frame for e in entities)
        
        legacy_entity = LegacyEntity(
            entity_id=f"entity_{label}",
            label=label,
            count=count,
            avg_confidence=avg_conf,
            first_seen_frame=first_frame,
            last_seen_frame=last_frame,
            position_3d=position_3d,
            size_3d=size_3d,
            depth_median=float(position_3d[2]) if position_3d is not None else 2.5,
        )
        
        legacy_map[label] = legacy_entity
    
    return legacy_map


# ============================================================================
# 测试函数
# ============================================================================

def test_scale_calibrator():
    """测试尺度校准器"""
    calibrator = DynamicScaleCalibrator(focal_length=500)
    
    # 模拟一个检测：门，像素高度 400，深度 2.0m
    det = Detection(
        frame_idx=0,
        label="door",
        bbox=(100, 50, 200, 450),  # 高度 400 像素
        confidence=0.9,
    )
    
    # 模拟深度图：门区域深度约 2.0m
    depth_map = np.ones((480, 640)) * 2.0
    
    scale = calibrator.calibrate_from_detections([det], depth_map, 480)
    
    # 估计高度 = 400 / 500 * 2.0 = 1.6m
    # 真实高度 = 2.0m
    # 尺度 = 2.0 / 1.6 = 1.25
    print(f"尺度因子: {scale:.3f}")
    assert 1.0 < scale < 1.5, f"尺度因子应该在 1.0-1.5 之间，实际: {scale}"
    print("✓ 尺度校准器测试通过")


def test_instance_tracker():
    """测试实例追踪器"""
    tracker = InstanceTracker(distance_threshold=1.0)
    
    # 帧 0: 检测到 2 把椅子
    det1 = Detection(0, "chair", (100, 100, 200, 300), 0.9, 
                     position_camera=np.array([1.0, 0.0, 2.0]))
    det2 = Detection(0, "chair", (300, 100, 400, 300), 0.85,
                     position_camera=np.array([2.0, 0.0, 2.5]))
    
    tracker.update(0, [det1, det2])
    assert len(tracker.active_tracks) == 2, "应该有 2 个追踪实例"
    
    # 帧 1: 两把椅子位置略有变化
    det1_new = Detection(1, "chair", (105, 100, 205, 300), 0.88,
                         position_camera=np.array([1.05, 0.0, 2.1]))
    det2_new = Detection(1, "chair", (295, 100, 395, 300), 0.87,
                         position_camera=np.array([1.95, 0.0, 2.4]))
    
    tracker.update(1, [det1_new, det2_new])
    assert len(tracker.active_tracks) == 2, "仍然应该有 2 个追踪实例"
    
    # 验证每个实例有 2 次检测
    for track in tracker.active_tracks.values():
        assert track.total_detections == 2, f"每个实例应有 2 次检测"
    
    print("✓ 实例追踪器测试通过")


def test_probabilistic_mindmap():
    """测试概率性心智地图"""
    tracker = InstanceTracker()
    
    # 添加一些检测
    for i in range(5):
        det = Detection(i, "chair", (100, 100, 200, 300), 0.9,
                       position_camera=np.array([1.0 + np.random.randn()*0.1, 
                                                 0.0 + np.random.randn()*0.1, 
                                                 2.0 + np.random.randn()*0.1]),
                       size_estimated=np.array([0.5, 0.9, 0.5]))
        tracker.update(i, [det])
    
    # 构建概率性心智地图
    mind_map = ProbabilisticMindMap()
    mind_map.build_from_tracker(tracker)
    
    assert len(mind_map.entities) == 1, "应该有 1 个实体"
    
    entity = list(mind_map.entities.values())[0]
    print(f"实体: {entity.instance_id}")
    print(f"位置均值: {entity.position_mean}")
    print(f"位置协方差:\n{entity.position_cov}")
    print(f"置信区间半径: {entity.get_confidence_radius():.3f}m")
    
    # 测试方向查询
    observer_pos = np.array([0.0, 0.0, 0.0])
    facing_pos = np.array([0.0, 0.0, 3.0])
    
    direction_probs = mind_map.query_direction_probabilistic(
        observer_pos, facing_pos, entity
    )
    print(f"方向概率: {direction_probs}")
    
    print("✓ 概率性心智地图测试通过")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_scale_calibrator()
    test_instance_tracker()
    test_probabilistic_mindmap()
    print("\n所有测试通过!")
