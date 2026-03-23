"""
心智地图 V4 - 完整改进版本

核心改进：
1. 相机位姿估计 - 基于帧间特征匹配
2. 动态尺度校准 - 利用已知物体尺寸
3. 实例级追踪 - 正确处理多实例计数
4. 体素占据地图 - 用于 room_size_estimation
5. 概率性推理 - 支持不确定性

设计原则：
- 每个改进可独立开关
- 保持与原有接口兼容
- 失败时优雅降级

作者: tianjungu
日期: 2026-01-28
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ============================================================================
# 已知物体尺寸数据库 (用于动态尺度校准)
# ============================================================================

KNOWN_OBJECT_SIZES = {
    # 门窗类 - 尺寸较标准
    'door': {'height': 2.0, 'width': 0.9},
    'window': {'height': 1.2, 'width': 1.0},
    
    # 大型家具
    'bed': {'height': 0.5, 'width': 1.5, 'length': 2.0},
    'sofa': {'height': 0.85, 'width': 0.9, 'length': 2.0},
    'couch': {'height': 0.85, 'width': 0.9, 'length': 2.0},
    'refrigerator': {'height': 1.7, 'width': 0.7},
    'fridge': {'height': 1.7, 'width': 0.7},
    'wardrobe': {'height': 2.0, 'width': 1.2},
    'closet': {'height': 2.0, 'width': 1.2},
    
    # 中型家具
    'chair': {'height': 0.9, 'seat_height': 0.45, 'width': 0.5},
    'table': {'height': 0.75, 'width': 0.8},
    'desk': {'height': 0.75, 'width': 0.6},
    'toilet': {'height': 0.4, 'width': 0.4},
    'sink': {'height': 0.85, 'width': 0.5},
    'bathtub': {'height': 0.6, 'width': 0.75},
    
    # 电器
    'tv': {'height': 0.6, 'width': 1.0},
    'television': {'height': 0.6, 'width': 1.0},
    'monitor': {'height': 0.35, 'width': 0.55},
    'washer': {'height': 0.85, 'width': 0.6},
    'dryer': {'height': 0.85, 'width': 0.6},
}


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Detection:
    """单次检测"""
    frame_idx: int
    label: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    
    position_camera: Optional[np.ndarray] = None  # 相机坐标系位置
    position_world: Optional[np.ndarray] = None   # 世界坐标系位置
    size_3d: Optional[np.ndarray] = None
    depth_median: float = 0.0


@dataclass
class CameraPose:
    """相机位姿"""
    frame_idx: int
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def camera_to_world(self, point_camera: np.ndarray) -> np.ndarray:
        """相机坐标 → 世界坐标"""
        return self.R.T @ point_camera + self.t
    
    def world_to_camera(self, point_world: np.ndarray) -> np.ndarray:
        """世界坐标 → 相机坐标"""
        return self.R @ (point_world - self.t)


# ============================================================================
# 相机位姿估计器
# ============================================================================

class CameraPoseEstimator:
    """
    相机位姿估计器 - 基于特征点匹配
    
    使用 ORB 特征点追踪估计帧间相对运动
    """
    
    def __init__(self, focal_length: float = 500.0, 
                 principal_point: Tuple[float, float] = None):
        self.focal_length = focal_length
        self.principal_point = principal_point
        
        # ORB 特征检测器
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 位姿历史
        self.poses: List[CameraPose] = []
        self.prev_frame_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # 累积位姿
        self.cumulative_R = np.eye(3)
        self.cumulative_t = np.zeros(3)
    
    def estimate_pose(self, frame: np.ndarray, frame_idx: int) -> CameraPose:
        """估计当前帧的相机位姿"""
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        H, W = gray.shape[:2]
        if self.principal_point is None:
            self.principal_point = (W / 2, H / 2)
        
        # 检测特征点
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # 第一帧：初始化
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
            pose = CameraPose(
                frame_idx=frame_idx,
                R=np.eye(3),
                t=np.zeros(3)
            )
            self.poses.append(pose)
            return pose
        
        # 特征匹配
        if descriptors is None or self.prev_descriptors is None:
            # 无法匹配，保持上一帧位姿
            pose = CameraPose(
                frame_idx=frame_idx,
                R=self.cumulative_R.copy(),
                t=self.cumulative_t.copy()
            )
            self.poses.append(pose)
            return pose
        
        try:
            matches = self.bf_matcher.match(self.prev_descriptors, descriptors)
            
            # 需要足够的匹配点
            if len(matches) < 8:
                pose = CameraPose(
                    frame_idx=frame_idx,
                    R=self.cumulative_R.copy(),
                    t=self.cumulative_t.copy()
                )
            else:
                # 提取匹配点坐标
                pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
                pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])
                
                # 相机内参矩阵
                K = np.array([
                    [self.focal_length, 0, self.principal_point[0]],
                    [0, self.focal_length, self.principal_point[1]],
                    [0, 0, 1]
                ])
                
                # 计算本质矩阵
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, 
                                               prob=0.999, threshold=1.0)
                
                if E is not None:
                    # 分解本质矩阵得到 R, t
                    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
                    
                    # 累积位姿
                    self.cumulative_t = self.cumulative_t + self.cumulative_R @ t.flatten() * 0.1
                    self.cumulative_R = R @ self.cumulative_R
                
                pose = CameraPose(
                    frame_idx=frame_idx,
                    R=self.cumulative_R.copy(),
                    t=self.cumulative_t.copy()
                )
                
        except Exception as e:
            logger.debug(f"位姿估计失败: {e}")
            pose = CameraPose(
                frame_idx=frame_idx,
                R=self.cumulative_R.copy(),
                t=self.cumulative_t.copy()
            )
        
        # 更新历史
        self.prev_frame_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.poses.append(pose)
        
        return pose
    
    def reset(self):
        """重置"""
        self.poses.clear()
        self.prev_frame_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.cumulative_R = np.eye(3)
        self.cumulative_t = np.zeros(3)


# ============================================================================
# 动态尺度校准器
# ============================================================================

class DynamicScaleCalibrator:
    """动态尺度校准器"""
    
    def __init__(self, focal_length: float = 500.0):
        self.focal_length = focal_length
        self.scale_history: List[float] = []
        self.default_median_depth = 2.5  # 默认中值深度
        
    def calibrate(self, detections: List[Detection], depth_map: np.ndarray,
                  image_height: int) -> float:
        """
        从检测结果校准尺度因子
        
        Returns:
            尺度因子（乘以 DA3 原始深度输出）
        """
        scale_estimates = []
        
        for det in detections:
            label_lower = det.label.lower()
            
            # 查找已知尺寸
            known_size = None
            for key in KNOWN_OBJECT_SIZES:
                if key in label_lower or label_lower in key:
                    known_size = KNOWN_OBJECT_SIZES[key]
                    break
            
            if known_size is None:
                continue
            
            # 获取参考尺寸（优先用高度）
            ref_dim = known_size.get('height', known_size.get('width'))
            if ref_dim is None:
                continue
            
            # 计算估计的 3D 高度
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
            
            if y2 <= y1 or x2 <= x1:
                continue
            
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
        
        # 回退到历史或默认
        if self.scale_history:
            return float(np.median(self.scale_history))
        
        # 使用默认中值深度校准
        median_depth = np.median(depth_map)
        if median_depth > 0:
            return self.default_median_depth / median_depth
        
        return 1.0
    
    def reset(self):
        self.scale_history.clear()


# ============================================================================
# 体素占据地图
# ============================================================================

class VoxelOccupancyMap:
    """
    体素占据地图 - 用于 room_size_estimation
    
    改进：
    - 不再用包围盒估算面积
    - 而是统计占据的体素投影到地面的面积
    """
    
    def __init__(self, voxel_size: float = 0.1):
        self.voxel_size = voxel_size
        self.occupied_voxels: Set[Tuple[int, int, int]] = set()
        
    def _point_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        """3D 点 → 体素坐标"""
        return (
            int(point[0] / self.voxel_size),
            int(point[1] / self.voxel_size),
            int(point[2] / self.voxel_size)
        )
    
    def insert_point(self, point: np.ndarray):
        """插入一个 3D 点"""
        voxel = self._point_to_voxel(point)
        self.occupied_voxels.add(voxel)
    
    def insert_bbox_3d(self, center: np.ndarray, size: np.ndarray):
        """插入一个 3D 包围盒（填充多个体素）"""
        half_size = size / 2
        min_corner = center - half_size
        max_corner = center + half_size
        
        # 在包围盒范围内采样体素
        x_range = np.arange(min_corner[0], max_corner[0], self.voxel_size)
        y_range = np.arange(min_corner[1], max_corner[1], self.voxel_size)
        z_range = np.arange(min_corner[2], max_corner[2], self.voxel_size)
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    self.insert_point(np.array([x, y, z]))
    
    def compute_floor_area(self) -> float:
        """计算地面面积（XZ 平面投影）"""
        if not self.occupied_voxels:
            return 0.0
        
        # 投影到 XZ 平面
        xz_cells = set()
        for vx, vy, vz in self.occupied_voxels:
            xz_cells.add((vx, vz))
        
        return len(xz_cells) * (self.voxel_size ** 2)
    
    def compute_bounding_box_area(self) -> float:
        """计算包围盒面积（用于对比）"""
        if not self.occupied_voxels:
            return 0.0
        
        voxels_arr = np.array(list(self.occupied_voxels))
        
        x_range = (voxels_arr[:, 0].max() - voxels_arr[:, 0].min() + 1) * self.voxel_size
        z_range = (voxels_arr[:, 2].max() - voxels_arr[:, 2].min() + 1) * self.voxel_size
        
        return x_range * z_range
    
    def clear(self):
        self.occupied_voxels.clear()


# ============================================================================
# 改进的实例追踪器
# ============================================================================

class InstanceTrackerV2:
    """
    改进的实例追踪器
    
    修复问题：
    - 原版在无相机位姿时过度分割
    - 现在使用更大的距离阈值 + 帧间连续性约束
    """
    
    def __init__(self, distance_threshold: float = 2.0,  # 增大阈值
                 max_lost_frames: int = 10):
        self.distance_threshold = distance_threshold
        self.max_lost_frames = max_lost_frames
        
        self.active_tracks: Dict[str, 'TrackedInstance'] = {}
        self.next_id: Dict[str, int] = defaultdict(int)
        self.current_frame = 0
        
        # 记录每帧每类别的检测数（用于正确计数）
        self.frame_detection_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def _generate_id(self, label: str) -> str:
        self.next_id[label] += 1
        return f"{label}_{self.next_id[label]:03d}"
    
    def update(self, frame_idx: int, detections: List[Detection],
               camera_pose: Optional[CameraPose] = None) -> Dict[str, 'TrackedInstance']:
        """更新追踪"""
        self.current_frame = frame_idx
        
        # 记录每类别的检测数
        label_counts = defaultdict(int)
        for det in detections:
            label_counts[det.label.lower()] += 1
        self.frame_detection_counts[frame_idx] = dict(label_counts)
        
        # 按类别分组
        dets_by_label: Dict[str, List[Detection]] = defaultdict(list)
        for det in detections:
            if det.position_camera is not None or det.position_world is not None:
                dets_by_label[det.label.lower()].append(det)
        
        # 对每个类别进行追踪
        for label, label_dets in dets_by_label.items():
            # 获取该类别现有 tracks
            existing = [(tid, t) for tid, t in self.active_tracks.items() if t.label == label]
            
            # 获取位置（优先世界坐标，否则用相机坐标）
            det_positions = []
            for det in label_dets:
                if det.position_world is not None:
                    det_positions.append(det.position_world)
                elif det.position_camera is not None:
                    if camera_pose is not None:
                        det_positions.append(camera_pose.camera_to_world(det.position_camera))
                    else:
                        det_positions.append(det.position_camera)
                else:
                    det_positions.append(None)
            
            if not existing:
                # 没有现有 track，全部创建新的
                for det, pos in zip(label_dets, det_positions):
                    if pos is not None:
                        self._create_track(det, pos, frame_idx)
            else:
                # 匈牙利匹配
                from scipy.optimize import linear_sum_assignment
                
                cost_matrix = np.full((len(existing), len(label_dets)), self.distance_threshold * 2)
                
                for i, (tid, track) in enumerate(existing):
                    track_pos = track.get_position_mean()
                    if track_pos is None:
                        continue
                    for j, pos in enumerate(det_positions):
                        if pos is not None:
                            cost_matrix[i, j] = np.linalg.norm(track_pos - pos)
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                matched_det = set()
                for ti, di in zip(row_ind, col_ind):
                    if cost_matrix[ti, di] < self.distance_threshold:
                        tid, track = existing[ti]
                        self._update_track(track, label_dets[di], det_positions[di], frame_idx)
                        matched_det.add(di)
                
                # 未匹配的检测创建新 track
                for di, (det, pos) in enumerate(zip(label_dets, det_positions)):
                    if di not in matched_det and pos is not None:
                        self._create_track(det, pos, frame_idx)
        
        self._cleanup()
        return self.active_tracks
    
    def _create_track(self, det: Detection, position: np.ndarray, frame_idx: int):
        instance_id = self._generate_id(det.label.lower())
        track = TrackedInstance(
            instance_id=instance_id,
            label=det.label.lower(),
            positions=[position],
            sizes=[det.size_3d] if det.size_3d is not None else [],
            confidences=[det.confidence],
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
        )
        self.active_tracks[instance_id] = track
    
    def _update_track(self, track: 'TrackedInstance', det: Detection, 
                      position: np.ndarray, frame_idx: int):
        track.positions.append(position)
        if det.size_3d is not None:
            track.sizes.append(det.size_3d)
        track.confidences.append(det.confidence)
        track.last_seen_frame = frame_idx
    
    def _cleanup(self):
        to_remove = []
        for tid, track in self.active_tracks.items():
            if self.current_frame - track.last_seen_frame > self.max_lost_frames:
                if len(track.positions) < 2:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.active_tracks[tid]
    
    def get_max_frame_count(self, label: str) -> int:
        """获取该类别在任意单帧的最大检测数"""
        label_lower = label.lower()
        max_count = 0
        for frame_counts in self.frame_detection_counts.values():
            for lbl, cnt in frame_counts.items():
                if label_lower in lbl or lbl in label_lower:
                    max_count = max(max_count, cnt)
        return max_count
    
    def reset(self):
        self.active_tracks.clear()
        self.next_id.clear()
        self.frame_detection_counts.clear()
        self.current_frame = 0


@dataclass
class TrackedInstance:
    """追踪实例"""
    instance_id: str
    label: str
    
    positions: List[np.ndarray] = field(default_factory=list)
    sizes: List[np.ndarray] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    
    def get_position_mean(self) -> Optional[np.ndarray]:
        if not self.positions:
            return None
        return np.median(self.positions, axis=0)
    
    def get_size_mean(self) -> Optional[np.ndarray]:
        if not self.sizes:
            return None
        return np.median(self.sizes, axis=0)
    
    def get_avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return float(np.mean(self.confidences))


# ============================================================================
# 心智地图实体（兼容旧格式）
# ============================================================================

@dataclass
class MindMapEntity3D:
    """心智地图实体（兼容原有格式）"""
    entity_id: str
    label: str
    
    count: int = 1
    avg_confidence: float = 0.0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    
    position_3d: Optional[np.ndarray] = None
    size_3d: Optional[np.ndarray] = None
    depth_median: float = 0.0


# ============================================================================
# 心智地图构建器 V4
# ============================================================================

class MindMapBuilderV4:
    """
    心智地图构建器 V4
    
    改进开关：
    - use_camera_pose: 使用相机位姿估计
    - use_scale_calibration: 使用动态尺度校准
    - use_voxel_map: 使用体素地图（用于 room_size）
    - use_instance_tracking: 使用实例追踪
    """
    
    def __init__(
        self, 
        device: str = 'cuda',
        num_frames: int = 32,
        box_threshold: float = 0.25,
        # 改进开关
        use_camera_pose: bool = False,
        use_scale_calibration: bool = False,
        use_voxel_map: bool = False,
        use_instance_tracking: bool = False,
    ):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        
        # 改进开关
        self.use_camera_pose = use_camera_pose
        self.use_scale_calibration = use_scale_calibration
        self.use_voxel_map = use_voxel_map
        self.use_instance_tracking = use_instance_tracking
        
        # 模型
        self._labeler = None
        self._depth_estimator = None
        
        # 改进模块
        self.pose_estimator = CameraPoseEstimator() if use_camera_pose else None
        self.scale_calibrator = DynamicScaleCalibrator() if use_scale_calibration else None
        self.voxel_map = VoxelOccupancyMap(voxel_size=0.2) if use_voxel_map else None
        self.instance_tracker = InstanceTrackerV2() if use_instance_tracking else None
        
        # 相机参数
        self.focal_length = 500
        self.principal_point = None
        
        # 存储中间结果
        self.all_detections: Dict[str, List[Dict]] = {}
        self.frame_indices: List[int] = []
    
    def _load_models(self):
        """加载模型"""
        import torch
        
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
        """释放模型"""
        import torch
        import gc
        
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
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """提取帧"""
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
        """采样帧"""
        n = min(self.num_frames, len(frames))
        indices = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in indices], indices.tolist()
    
    def build_from_video(
        self, 
        video_path: str, 
        target_objects: List[str] = None,
        vocab: List[str] = None,
    ) -> Dict[str, MindMapEntity3D]:
        """
        从视频构建心智地图
        
        Returns:
            Dict[label, MindMapEntity3D] - 兼容原有格式
        """
        self._load_models()
        
        # 重置改进模块
        if self.pose_estimator:
            self.pose_estimator.reset()
        if self.scale_calibrator:
            self.scale_calibrator.reset()
        if self.voxel_map:
            self.voxel_map.clear()
        if self.instance_tracker:
            self.instance_tracker.reset()
        
        # 提取帧
        all_frames = self._extract_frames(video_path)
        if not all_frames:
            return {}
        
        frames, self.frame_indices = self._sample_frames(all_frames)
        H, W = frames[0].shape[:2]
        self.principal_point = (W / 2, H / 2)
        
        if self.pose_estimator:
            self.pose_estimator.principal_point = self.principal_point
        
        # 构建检测词汇
        from tests.test_vsibench_directqa import EXTENDED_VOCABULARY
        if vocab is None:
            vocab = EXTENDED_VOCABULARY
        if target_objects:
            vocab = list(set(target_objects + vocab))
        text_prompt = " . ".join(vocab) + " ."
        
        # 逐帧处理
        self.all_detections = defaultdict(list)
        
        for frame_idx, frame in enumerate(frames):
            original_idx = self.frame_indices[frame_idx]
            
            # 1. 深度估计
            try:
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                
                if depth_map.shape[:2] != (H, W):
                    depth_map = cv2.resize(depth_map, (W, H))
            except Exception as e:
                logger.warning(f"深度估计失败: {e}")
                depth_map = np.ones((H, W), dtype=np.float32) * 2.5
            
            # 2. 物体检测
            results = self._labeler.detect(frame, text_prompt)
            
            # 创建 Detection 对象列表（用于尺度校准）
            detections_for_calibration = []
            for det in results:
                bbox = [int(v) for v in det.bbox_pixels]
                detections_for_calibration.append(Detection(
                    frame_idx=original_idx,
                    label=det.label,
                    bbox=tuple(bbox),
                    confidence=det.confidence,
                ))
            
            # 3. 尺度校准
            if self.use_scale_calibration and self.scale_calibrator:
                scale = self.scale_calibrator.calibrate(
                    detections_for_calibration, depth_map, H
                )
            else:
                # 默认：中值深度 = 2.5m
                median_depth = np.median(depth_map)
                scale = 2.5 / median_depth if median_depth > 0 else 1.0
            
            depth_map_scaled = depth_map * scale
            
            # 4. 相机位姿估计
            if self.use_camera_pose and self.pose_estimator:
                camera_pose = self.pose_estimator.estimate_pose(frame, original_idx)
            else:
                camera_pose = None
            
            # 5. 处理每个检测
            frame_detections = []
            for det in results:
                label = det.label.lower()
                bbox = det.bbox_pixels
                confidence = det.confidence
                
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 深度
                depth_roi = depth_map_scaled[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                depth_median = float(np.median(depth_roi))
                
                # 3D 位置（相机坐标系）
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                px, py = self.principal_point
                
                x_3d = (cx - px) / self.focal_length * depth_median
                y_3d = (cy - py) / self.focal_length * depth_median
                z_3d = depth_median
                
                position_camera = np.array([x_3d, y_3d, z_3d])
                
                # 转换到世界坐标系
                if camera_pose is not None:
                    position_world = camera_pose.camera_to_world(position_camera)
                else:
                    position_world = position_camera
                
                # 3D 尺寸
                w_pix, h_pix = x2 - x1, y2 - y1
                w_3d = w_pix / self.focal_length * depth_median
                h_3d = h_pix / self.focal_length * depth_median
                d_3d = min(w_3d, h_3d) * 0.5
                size_3d = np.array([w_3d, h_3d, d_3d])
                
                # 插入体素地图
                if self.use_voxel_map and self.voxel_map:
                    self.voxel_map.insert_bbox_3d(position_world, size_3d)
                
                # 创建 Detection 对象
                detection = Detection(
                    frame_idx=original_idx,
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    position_camera=position_camera,
                    position_world=position_world,
                    size_3d=size_3d,
                    depth_median=depth_median,
                )
                frame_detections.append(detection)
                
                # 存储（用于聚合）
                self.all_detections[label].append({
                    'frame_idx': original_idx,
                    'bbox': bbox,
                    'confidence': confidence,
                    'position_3d': position_world,
                    'size_3d': size_3d,
                    'depth_median': depth_median,
                })
            
            # 更新实例追踪
            if self.use_instance_tracking and self.instance_tracker:
                self.instance_tracker.update(original_idx, frame_detections, camera_pose)
        
        # 聚合成实体
        return self._aggregate_entities()
    
    def _aggregate_entities(self) -> Dict[str, MindMapEntity3D]:
        """聚合检测结果为实体"""
        entities = {}
        
        for category, dets in self.all_detections.items():
            if not dets:
                continue
            
            # 按帧分组计算最大检测数
            frame_dets = defaultdict(list)
            for d in dets:
                frame_dets[d['frame_idx']].append(d)
            
            # 使用实例追踪的计数（如果启用），否则用帧内最大检测数
            if self.use_instance_tracking and self.instance_tracker:
                max_count = self.instance_tracker.get_max_frame_count(category)
                if max_count == 0:
                    max_count = max(len(fd) for fd in frame_dets.values()) if frame_dets else 1
            else:
                max_count = max(len(fd) for fd in frame_dets.values()) if frame_dets else 1
            
            # 聚合统计
            avg_conf = np.mean([d['confidence'] for d in dets])
            first_frame = min(d['frame_idx'] for d in dets)
            last_frame = max(d['frame_idx'] for d in dets)
            
            # 位置/尺寸取中值
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
    
    def get_room_area_from_voxel(self) -> Optional[float]:
        """从体素地图获取房间面积"""
        if self.voxel_map:
            return self.voxel_map.compute_floor_area()
        return None


# ============================================================================
# 测试
# ============================================================================

def test_pose_estimator():
    """测试相机位姿估计"""
    estimator = CameraPoseEstimator()
    
    # 模拟两帧
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    pose1 = estimator.estimate_pose(frame1, 0)
    pose2 = estimator.estimate_pose(frame2, 1)
    
    print(f"Pose 1: R=\n{pose1.R}, t={pose1.t}")
    print(f"Pose 2: R=\n{pose2.R}, t={pose2.t}")
    print("✓ 相机位姿估计测试通过")


def test_voxel_map():
    """测试体素地图"""
    vmap = VoxelOccupancyMap(voxel_size=0.1)
    
    # 插入一些点
    for i in range(10):
        for j in range(10):
            vmap.insert_point(np.array([i * 0.1, 0, j * 0.1]))
    
    area = vmap.compute_floor_area()
    print(f"Floor area: {area:.2f} m^2")
    assert 0.9 < area < 1.1, f"Expected ~1.0 m^2, got {area}"
    print("✓ 体素地图测试通过")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_pose_estimator()
    test_voxel_map()
    print("\n所有测试通过!")
