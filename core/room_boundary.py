"""
房间边界检测器 - 基于点云平面分割

用于 VSIBench room_size_estimation 任务
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Plane:
    """平面数据结构"""
    normal: np.ndarray      # 法向量 (单位向量)
    point: np.ndarray       # 平面上一点
    inliers: int            # 内点数量
    distance: float         # 原点到平面距离
    
    @property
    def d(self) -> float:
        """平面方程 ax + by + cz + d = 0 中的 d"""
        return -np.dot(self.normal, self.point)


@dataclass
class RoomBounds:
    """房间边界数据"""
    floor_height: float
    ceiling_height: float
    room_height: float
    bbox: List[float]           # [xmin, ymin, zmin, xmax, ymax, zmax]
    dimensions: List[float]     # [width, height, depth] in meters
    walls: List[Plane]
    floor_plane: Optional[Plane]
    ceiling_plane: Optional[Plane]
    
    def to_dict(self) -> dict:
        return {
            'floor_height': round(self.floor_height, 3),
            'ceiling_height': round(self.ceiling_height, 3),
            'room_height': round(self.room_height, 3),
            'bbox': [round(v, 3) for v in self.bbox],
            'dimensions': [round(v, 3) for v in self.dimensions],
            'num_walls': len(self.walls),
            'has_floor': self.floor_plane is not None,
            'has_ceiling': self.ceiling_plane is not None,
        }


class RoomBoundaryDetector:
    """
    房间边界检测器 - 基于 RANSAC 平面分割
    
    检测流程:
    1. RANSAC 多平面分割 - 检测主要平面
    2. 平面分类 - 根据法向量分为水平面(地板/天花板)和垂直面(墙壁)
    3. 边界计算 - 从平面位置推断房间尺寸
    """
    
    def __init__(
        self,
        ransac_threshold: float = 0.05,     # RANSAC 距离阈值 (米)
        ransac_iterations: int = 1000,       # RANSAC 迭代次数
        min_plane_points: int = 100,         # 最小平面点数
        horizontal_threshold: float = 0.85,  # 水平面法向量 Y 分量阈值
        vertical_threshold: float = 0.15,    # 垂直面法向量 Y 分量阈值
    ):
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        self.min_plane_points = min_plane_points
        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold
    
    def detect(self, points: np.ndarray, max_planes: int = 8) -> RoomBounds:
        """
        检测房间边界
        
        Args:
            points: (N, 3) 点云
            max_planes: 最多检测的平面数量
        
        Returns:
            RoomBounds 对象
        """
        logger.info(f"Detecting room boundaries from {len(points)} points")
        
        # 1. RANSAC 多平面分割
        planes = self._ransac_multi_plane(points, max_planes)
        logger.info(f"Detected {len(planes)} planes")
        
        # 2. 分类平面
        floor, ceiling, walls = self._classify_planes(planes)
        logger.info(f"Classified: floor={floor is not None}, ceiling={ceiling is not None}, walls={len(walls)}")
        
        # 3. 计算房间边界
        bounds = self._compute_room_bounds(floor, ceiling, walls, points)
        
        return bounds
    
    def _ransac_multi_plane(self, points: np.ndarray, max_planes: int) -> List[Plane]:
        """
        RANSAC 多平面分割
        
        迭代检测平面，每次移除已检测平面的内点
        """
        remaining_points = points.copy()
        planes = []
        
        for _ in range(max_planes):
            if len(remaining_points) < self.min_plane_points:
                break
            
            plane = self._ransac_single_plane(remaining_points)
            
            if plane is None or plane.inliers < self.min_plane_points:
                break
            
            planes.append(plane)
            
            # 移除内点
            distances = self._point_to_plane_distance(remaining_points, plane)
            mask = distances > self.ransac_threshold
            remaining_points = remaining_points[mask]
        
        return planes
    
    def _ransac_single_plane(self, points: np.ndarray) -> Optional[Plane]:
        """
        RANSAC 单平面检测
        
        算法:
        1. 随机采样 3 点
        2. 拟合平面
        3. 计算内点
        4. 迭代找最佳平面
        """
        n_points = len(points)
        if n_points < 3:
            return None
        
        best_plane = None
        best_inliers = 0
        
        for _ in range(self.ransac_iterations):
            # 随机采样 3 点
            indices = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[indices]
            
            # 计算平面法向量
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal = normal / norm
            
            # 计算内点
            distances = np.abs(np.dot(points - p1, normal))
            inliers = np.sum(distances < self.ransac_threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = Plane(
                    normal=normal,
                    point=p1,
                    inliers=inliers,
                    distance=np.abs(np.dot(normal, p1)),
                )
        
        return best_plane
    
    def _point_to_plane_distance(self, points: np.ndarray, plane: Plane) -> np.ndarray:
        """计算点到平面的距离"""
        return np.abs(np.dot(points - plane.point, plane.normal))
    
    def _classify_planes(self, planes: List[Plane]) -> Tuple[Optional[Plane], Optional[Plane], List[Plane]]:
        """
        根据法向量分类平面
        
        - 水平面 (|normal·Y| > threshold): 地板(最低)/天花板(最高)
        - 垂直面 (|normal·Y| < threshold): 墙壁
        
        Returns:
            (floor, ceiling, walls)
        """
        horizontal = []
        vertical = []
        
        for plane in planes:
            y_component = abs(plane.normal[1])  # Y 轴分量
            
            if y_component > self.horizontal_threshold:
                horizontal.append(plane)
            elif y_component < self.vertical_threshold:
                vertical.append(plane)
        
        # 按高度排序水平平面 (使用平面上的点的 Y 坐标)
        horizontal.sort(key=lambda p: p.point[1])
        
        floor = horizontal[0] if horizontal else None
        ceiling = horizontal[-1] if len(horizontal) > 1 else None
        
        # 如果只检测到一个水平面，判断是地板还是天花板
        if floor is not None and ceiling is None and len(horizontal) == 1:
            # 检查法向量方向: 地板法向量朝上 (Y > 0)
            if floor.normal[1] < 0:
                ceiling = floor
                floor = None
        
        return floor, ceiling, vertical
    
    def _compute_room_bounds(
        self,
        floor: Optional[Plane],
        ceiling: Optional[Plane],
        walls: List[Plane],
        points: np.ndarray,
    ) -> RoomBounds:
        """从平面和点云计算房间边界"""
        
        # 计算点云边界 (作为回退)
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        
        # 地板高度
        if floor is not None:
            floor_height = floor.point[1]
        else:
            floor_height = pmin[1]
        
        # 天花板高度
        if ceiling is not None:
            ceiling_height = ceiling.point[1]
        else:
            ceiling_height = pmax[1]
        
        # 房间高度
        room_height = ceiling_height - floor_height
        
        # X, Z 边界 - 优先从墙壁推断
        xmin, xmax = pmin[0], pmax[0]
        zmin, zmax = pmin[2], pmax[2]
        
        for wall in walls:
            # 分析墙壁法向量
            nx, ny, nz = wall.normal
            
            # X 方向墙壁 (法向量主要在 X 方向)
            if abs(nx) > 0.7:
                wall_x = wall.point[0]
                if nx > 0:
                    xmin = max(xmin, wall_x - 0.1)
                else:
                    xmax = min(xmax, wall_x + 0.1)
            
            # Z 方向墙壁 (法向量主要在 Z 方向)
            if abs(nz) > 0.7:
                wall_z = wall.point[2]
                if nz > 0:
                    zmin = max(zmin, wall_z - 0.1)
                else:
                    zmax = min(zmax, wall_z + 0.1)
        
        bbox = [xmin, floor_height, zmin, xmax, ceiling_height, zmax]
        dimensions = [xmax - xmin, room_height, zmax - zmin]
        
        return RoomBounds(
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            room_height=room_height,
            bbox=bbox,
            dimensions=dimensions,
            walls=walls,
            floor_plane=floor,
            ceiling_plane=ceiling,
        )


def detect_room_from_voxel_map(voxel_map: 'VoxelMap', **kwargs) -> RoomBounds:
    """
    从体素地图检测房间边界
    
    便捷函数
    """
    # 获取点云
    points = voxel_map.get_point_cloud()
    
    if points is None or len(points) == 0:
        raise ValueError("VoxelMap has no points")
    
    detector = RoomBoundaryDetector(**kwargs)
    return detector.detect(points)
