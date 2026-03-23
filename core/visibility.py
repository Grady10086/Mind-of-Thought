"""
可见性计算模块 - 基于射线投射

提供两种实现:
1. RaycastVisibility: 基于体素的射线投射
2. BVHVisibility: 使用 BVH 加速的射线投射 (适用于大规模场景)
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisibilityResult:
    """可见性计算结果"""
    visible: bool
    occlusion_ratio: float      # 0-1, 被遮挡射线比例
    distance: float             # 到实体的距离
    reason: Optional[str] = None  # 不可见原因
    
    def to_dict(self) -> dict:
        return {
            'visible': self.visible,
            'occlusion_ratio': round(self.occlusion_ratio, 3),
            'distance': round(self.distance, 3),
            'reason': self.reason,
        }


class RaycastVisibility:
    """
    基于体素的射线投射可见性计算
    
    方法:
    1. 构建占据体素集合 (加速结构)
    2. 从相机向实体发射多条射线
    3. 使用 3D DDA 算法沿射线步进
    4. 检测射线是否被遮挡
    """
    
    def __init__(self, voxel_map: 'VoxelMap'):
        """
        Args:
            voxel_map: 体素地图
        """
        self.voxel_map = voxel_map
        self.resolution = voxel_map.resolution
        self.occupied_voxels: Set[Tuple[int, int, int]] = set()
        self._build_acceleration_structure()
    
    def _build_acceleration_structure(self):
        """构建加速结构 - 占据体素集合"""
        for key, voxel in self.voxel_map.voxels.items():
            if voxel.is_occupied():
                self.occupied_voxels.add(key)
        logger.info(f"Built acceleration structure with {len(self.occupied_voxels)} occupied voxels")
    
    def compute_visibility(
        self,
        camera_position: np.ndarray,
        entity: Dict,
        max_distance: float = 15.0,
        num_rays: int = 9,
        occlusion_threshold: float = 0.5,
    ) -> VisibilityResult:
        """
        计算从相机到单个实体的可见性
        
        Args:
            camera_position: 相机位置 [x, y, z]
            entity: 实体字典，包含 'center', 'size'
            max_distance: 最大可见距离 (米)
            num_rays: 发射的射线数量
            occlusion_threshold: 遮挡比例阈值
        
        Returns:
            VisibilityResult
        """
        center = np.array(entity['center'])
        size = np.array(entity.get('size', [0.5, 0.5, 0.5]))
        
        # 检查距离
        distance = np.linalg.norm(center - camera_position)
        if distance > max_distance:
            return VisibilityResult(
                visible=False,
                occlusion_ratio=1.0,
                distance=distance,
                reason='too_far',
            )
        
        # 生成射线目标点
        target_points = self._sample_entity_points(center, size, num_rays)
        
        # 射线投射
        occluded_count = 0
        for target in target_points:
            if self._is_ray_occluded(camera_position, target, center):
                occluded_count += 1
        
        occlusion_ratio = occluded_count / len(target_points)
        visible = occlusion_ratio < occlusion_threshold
        
        return VisibilityResult(
            visible=visible,
            occlusion_ratio=occlusion_ratio,
            distance=distance,
            reason='occluded' if not visible else None,
        )
    
    def compute_visibility_batch(
        self,
        camera_position: np.ndarray,
        entities: List[Dict],
        **kwargs,
    ) -> List[Dict]:
        """
        批量计算多个实体的可见性
        
        Args:
            camera_position: 相机位置
            entities: 实体列表
            **kwargs: 传递给 compute_visibility 的参数
        
        Returns:
            带有 'visibility' 字段的实体列表
        """
        results = []
        for entity in entities:
            vis = self.compute_visibility(camera_position, entity, **kwargs)
            entity_copy = entity.copy()
            entity_copy['visibility'] = vis.to_dict()
            results.append(entity_copy)
        return results
    
    def _sample_entity_points(
        self,
        center: np.ndarray,
        size: np.ndarray,
        num_points: int,
    ) -> List[np.ndarray]:
        """
        在实体表面采样点
        
        采样策略:
        - 中心点 (必须)
        - 8 个角点
        - 6 个面中心
        """
        points = [center]  # 始终包含中心点
        
        half_size = size / 2 * 0.8  # 稍微收缩以避免边界问题
        
        if num_points >= 9:
            # 添加 8 个角点
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        corner = center + np.array([dx, dy, dz]) * half_size
                        points.append(corner)
        
        if num_points >= 15:
            # 添加 6 个面中心
            for axis in range(3):
                for sign in [-1, 1]:
                    face_center = center.copy()
                    face_center[axis] += sign * half_size[axis]
                    points.append(face_center)
        
        return points[:num_points]
    
    def _is_ray_occluded(
        self,
        origin: np.ndarray,
        target: np.ndarray,
        entity_center: np.ndarray,
    ) -> bool:
        """
        检查射线是否被遮挡
        
        使用 3D DDA (Digital Differential Analyzer) 算法
        沿射线步进，检查是否穿过占据的体素
        
        Args:
            origin: 射线起点 (相机位置)
            target: 射线终点 (实体采样点)
            entity_center: 实体中心 (用于避免自遮挡)
        
        Returns:
            True 如果射线被遮挡
        """
        direction = target - origin
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return False
        
        direction = direction / distance
        
        # 步进参数
        step_size = self.resolution * 0.5  # 半体素步长
        num_steps = int(distance / step_size)
        
        # 到实体中心的距离 (用于提前终止)
        dist_to_entity = np.linalg.norm(entity_center - origin)
        
        for i in range(1, num_steps):  # 从 1 开始，跳过起点
            t = i * step_size
            
            # 接近目标实体时停止检查 (避免自遮挡)
            if t > dist_to_entity * 0.85:
                break
            
            point = origin + direction * t
            
            # 转换为体素坐标
            voxel_key = self._point_to_key(point)
            
            # 检查是否占据
            if voxel_key in self.occupied_voxels:
                # 检查这个体素是否属于目标实体 (简化判断)
                voxel_center = self._key_to_center(voxel_key)
                dist_to_entity_center = np.linalg.norm(voxel_center - entity_center)
                
                # 如果遮挡体素离目标实体较远，则认为是被其他物体遮挡
                if dist_to_entity_center > self.resolution * 3:
                    return True
        
        return False
    
    def _point_to_key(self, point: np.ndarray) -> Tuple[int, int, int]:
        """将点转换为体素 key"""
        key = tuple((point / self.resolution).astype(int))
        return key
    
    def _key_to_center(self, key: Tuple[int, int, int]) -> np.ndarray:
        """将体素 key 转换为中心点"""
        return (np.array(key) + 0.5) * self.resolution


class SimpleVisibility:
    """
    简化版可见性计算 - 基于距离和角度阈值
    
    用于快速测试或作为回退方案
    """
    
    def __init__(
        self,
        max_distance: float = 10.0,
        max_angle: float = 75.0,  # 度
    ):
        self.max_distance = max_distance
        self.max_angle_cos = np.cos(np.radians(max_angle))
    
    def compute_visibility(
        self,
        camera_position: np.ndarray,
        camera_forward: np.ndarray,
        entity: Dict,
    ) -> VisibilityResult:
        """
        基于距离和角度的简单可见性判断
        
        Args:
            camera_position: 相机位置
            camera_forward: 相机朝向 (单位向量)
            entity: 实体字典
        
        Returns:
            VisibilityResult
        """
        center = np.array(entity['center'])
        
        # 计算方向和距离
        to_entity = center - camera_position
        distance = np.linalg.norm(to_entity)
        
        if distance > self.max_distance:
            return VisibilityResult(
                visible=False,
                occlusion_ratio=1.0,
                distance=distance,
                reason='too_far',
            )
        
        # 计算角度
        if distance > 1e-6:
            to_entity_norm = to_entity / distance
            cos_angle = np.dot(to_entity_norm, camera_forward)
            
            if cos_angle < self.max_angle_cos:
                return VisibilityResult(
                    visible=False,
                    occlusion_ratio=1.0,
                    distance=distance,
                    reason='out_of_fov',
                )
        
        return VisibilityResult(
            visible=True,
            occlusion_ratio=0.0,
            distance=distance,
        )


class BVHVisibility:
    """
    使用 BVH (Bounding Volume Hierarchy) 加速的可见性计算
    
    适用于大规模场景，利用 trimesh 的射线投射功能
    """
    
    def __init__(self, voxel_map: 'VoxelMap'):
        self.voxel_map = voxel_map
        self.resolution = voxel_map.resolution
        self.mesh = None
        self._built = False
    
    def build_bvh(self, max_voxels: int = 50000):
        """
        构建 BVH 加速结构
        
        将占据体素转换为网格，利用 trimesh 的 BVH
        
        Args:
            max_voxels: 最大体素数量 (防止内存溢出)
        """
        if self._built:
            return
        
        try:
            import trimesh
            
            # 收集占据体素中心
            occupied_centers = []
            for key, voxel in self.voxel_map.voxels.items():
                if voxel.is_occupied():
                    center = (np.array(key) + 0.5) * self.resolution
                    occupied_centers.append(center)
                    if len(occupied_centers) >= max_voxels:
                        break
            
            if not occupied_centers:
                logger.warning("No occupied voxels for BVH")
                return
            
            logger.info(f"Building BVH with {len(occupied_centers)} voxels")
            
            # 创建体素盒子
            box = trimesh.primitives.Box(extents=[self.resolution] * 3)
            
            # 合并所有体素
            meshes = []
            for center in occupied_centers:
                mesh = box.copy()
                mesh.apply_translation(center)
                meshes.append(mesh)
            
            self.mesh = trimesh.util.concatenate(meshes)
            self._built = True
            logger.info(f"BVH built successfully")
            
        except ImportError:
            logger.error("trimesh not available for BVH")
        except Exception as e:
            logger.error(f"Failed to build BVH: {e}")
    
    def raycast(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        """
        批量射线投射
        
        Args:
            origins: (N, 3) 射线起点
            directions: (N, 3) 射线方向 (单位向量)
        
        Returns:
            hit_distances: (N,) 命中距离，无命中返回 inf
        """
        if not self._built:
            self.build_bvh()
        
        if self.mesh is None:
            return np.full(len(origins), np.inf)
        
        # 使用 trimesh 射线投射
        locations, index_ray, _ = self.mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
        )
        
        # 计算每条射线的最近命中
        hit_distances = np.full(len(origins), np.inf)
        for loc, ray_idx in zip(locations, index_ray):
            dist = np.linalg.norm(loc - origins[ray_idx])
            hit_distances[ray_idx] = min(hit_distances[ray_idx], dist)
        
        return hit_distances
    
    def compute_visibility(
        self,
        camera_position: np.ndarray,
        entity: Dict,
        num_rays: int = 9,
        occlusion_threshold: float = 0.5,
    ) -> VisibilityResult:
        """
        计算可见性
        
        Args:
            camera_position: 相机位置
            entity: 实体字典
            num_rays: 射线数量
            occlusion_threshold: 遮挡阈值
        
        Returns:
            VisibilityResult
        """
        if not self._built:
            self.build_bvh()
        
        center = np.array(entity['center'])
        size = np.array(entity.get('size', [0.5, 0.5, 0.5]))
        distance = np.linalg.norm(center - camera_position)
        
        # 生成射线
        targets = self._sample_entity_points(center, size, num_rays)
        origins = np.tile(camera_position, (len(targets), 1))
        directions = targets - origins
        
        # 归一化方向
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norms, 1e-8)
        target_distances = norms.flatten()
        
        # 射线投射
        hit_distances = self.raycast(origins, directions)
        
        # 判断遮挡 (命中距离 < 目标距离 * 0.95)
        occluded = hit_distances < target_distances * 0.95
        occlusion_ratio = np.mean(occluded)
        
        return VisibilityResult(
            visible=occlusion_ratio < occlusion_threshold,
            occlusion_ratio=float(occlusion_ratio),
            distance=distance,
            reason='occluded' if occlusion_ratio >= occlusion_threshold else None,
        )
    
    def _sample_entity_points(
        self,
        center: np.ndarray,
        size: np.ndarray,
        num_points: int,
    ) -> np.ndarray:
        """采样实体表面点"""
        points = [center]
        half_size = size / 2 * 0.8
        
        if num_points >= 9:
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        corner = center + np.array([dx, dy, dz]) * half_size
                        points.append(corner)
        
        return np.array(points[:num_points])


def create_visibility_calculator(
    voxel_map: 'VoxelMap',
    mode: str = "auto",
) -> 'RaycastVisibility':
    """
    创建可见性计算器
    
    Args:
        voxel_map: 体素地图
        mode: "raycast", "bvh", "simple", 或 "auto"
    
    Returns:
        可见性计算器实例
    """
    if mode == "simple":
        return SimpleVisibility()
    
    if mode == "bvh":
        return BVHVisibility(voxel_map)
    
    if mode == "auto":
        # 根据体素数量选择
        num_voxels = len(voxel_map.voxels)
        if num_voxels > 100000:
            logger.info(f"Using BVH visibility for {num_voxels} voxels")
            return BVHVisibility(voxel_map)
    
    return RaycastVisibility(voxel_map)
