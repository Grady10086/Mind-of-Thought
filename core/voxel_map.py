"""
稀疏体素地图模块 (Sparse Voxel Map)

实现心智地图构建的第一步：几何"积木化"

核心数据结构：
- SparseVoxelMap: 稀疏体素哈希表，存储空间占据和语义特征

参考：
- 体素大小: 0.1m (DA3 的物理尺度精度)
- 存储方式: 哈希表 {(i,j,k): VoxelData}
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoxelData:
    """单个体素的数据"""
    # 占据信息
    occupancy: float = 0.0           # 占据概率 [0, 1]
    hit_count: int = 0               # 被观测到的次数
    
    # 语义特征 (来自 DA3 的 DINOv2)
    feature_sum: Optional[np.ndarray] = None   # 特征累加
    feature_count: int = 0                      # 特征计数
    
    # 颜色
    color_sum: Optional[np.ndarray] = None     # RGB 累加
    color_count: int = 0
    
    # 时间戳
    first_seen: int = -1             # 首次观测帧
    last_seen: int = -1              # 最后观测帧
    
    # 所属实体 ID (聚类后填充)
    entity_id: Optional[str] = None
    
    @property
    def feature(self) -> Optional[np.ndarray]:
        """平均特征向量"""
        if self.feature_sum is not None and self.feature_count > 0:
            return self.feature_sum / self.feature_count
        return None
    
    @property
    def color(self) -> Optional[np.ndarray]:
        """平均颜色"""
        if self.color_sum is not None and self.color_count > 0:
            return self.color_sum / self.color_count
        return None
    
    def update(
        self,
        frame_id: int,
        feature: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
    ):
        """更新体素数据"""
        self.hit_count += 1
        # 首次观测立即设为占据，后续更新增加置信度
        self.occupancy = min(1.0, max(0.6, self.occupancy) + 0.1)
        
        if self.first_seen < 0:
            self.first_seen = frame_id
        self.last_seen = frame_id
        
        if feature is not None:
            if self.feature_sum is None:
                self.feature_sum = feature.copy()
            else:
                self.feature_sum += feature
            self.feature_count += 1
        
        if color is not None:
            if self.color_sum is None:
                self.color_sum = color.copy()
            else:
                self.color_sum += color
            self.color_count += 1
    
    def is_occupied(self, threshold: float = 0.5) -> bool:
        """检查体素是否被占据"""
        return self.occupancy >= threshold


class SparseVoxelMap:
    """
    稀疏体素哈希表
    
    核心功能：
    1. 将连续坐标量化为体素索引
    2. 高效存储和查询占据信息
    3. 融合多帧观测的语义特征
    
    使用方法：
        voxel_map = SparseVoxelMap(voxel_size=0.1)
        
        # 从点云填充
        voxel_map.integrate_points(points, colors, frame_id=0)
        
        # 查询占据
        is_occupied = voxel_map.is_occupied((1, 2, 3))
        
        # 获取所有占据体素
        occupied = voxel_map.get_occupied_voxels()
    """
    
    def __init__(
        self,
        voxel_size: float = 0.1,
        occupancy_threshold: float = 0.5,
    ):
        """
        Args:
            voxel_size: 体素大小 (米)，默认 0.1m
            occupancy_threshold: 占据判定阈值
        """
        self.voxel_size = voxel_size
        self.occupancy_threshold = occupancy_threshold
        
        # 稀疏哈希表: {(i, j, k): VoxelData}
        self._voxels: Dict[Tuple[int, int, int], VoxelData] = {}
        
        # 统计信息
        self._bounds_min: Optional[np.ndarray] = None
        self._bounds_max: Optional[np.ndarray] = None
        self._total_points_integrated: int = 0
    
    def _world_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        """世界坐标转体素索引"""
        idx = np.floor(point / self.voxel_size).astype(int)
        return tuple(idx)
    
    def _voxel_to_world(self, voxel_idx: Tuple[int, int, int]) -> np.ndarray:
        """体素索引转世界坐标 (体素中心)"""
        return (np.array(voxel_idx) + 0.5) * self.voxel_size
    
    def integrate_points(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        frame_id: int = 0,
    ):
        """
        将点云积分到体素地图
        
        Args:
            points: (N, 3) 点云坐标
            colors: (N, 3) 点云颜色 [0, 1]
            features: (N, D) 语义特征向量
            frame_id: 帧 ID
        """
        if len(points) == 0:
            return
        
        # 更新边界
        if self._bounds_min is None:
            self._bounds_min = points.min(axis=0)
            self._bounds_max = points.max(axis=0)
        else:
            self._bounds_min = np.minimum(self._bounds_min, points.min(axis=0))
            self._bounds_max = np.maximum(self._bounds_max, points.max(axis=0))
        
        # 逐点积分
        for i in range(len(points)):
            voxel_idx = self._world_to_voxel(points[i])
            
            if voxel_idx not in self._voxels:
                self._voxels[voxel_idx] = VoxelData()
            
            color = colors[i] if colors is not None else None
            feature = features[i] if features is not None else None
            
            self._voxels[voxel_idx].update(frame_id, feature, color)
        
        self._total_points_integrated += len(points)
        logger.debug(f"Integrated {len(points)} points, total voxels: {len(self._voxels)}")
    
    def integrate_scene(self, scene: 'Scene3D', frame_id: int = 0):
        """从 Scene3D 对象积分"""
        from .scene import Scene3D
        if scene.point_cloud is not None:
            self.integrate_points(
                scene.point_cloud,
                scene.colors,
                frame_id=frame_id,
            )
    
    def is_occupied(self, voxel_idx: Tuple[int, int, int]) -> bool:
        """检查体素是否被占据"""
        if voxel_idx not in self._voxels:
            return False
        return self._voxels[voxel_idx].occupancy >= self.occupancy_threshold
    
    def get_voxel(self, voxel_idx: Tuple[int, int, int]) -> Optional[VoxelData]:
        """获取体素数据"""
        return self._voxels.get(voxel_idx)
    
    def get_occupied_voxels(self) -> List[Tuple[int, int, int]]:
        """获取所有被占据的体素索引"""
        return [
            idx for idx, voxel in self._voxels.items()
            if voxel.occupancy >= self.occupancy_threshold
        ]
    
    def get_occupied_centers(self) -> np.ndarray:
        """获取所有被占据体素的中心坐标"""
        occupied = self.get_occupied_voxels()
        if not occupied:
            return np.array([]).reshape(0, 3)
        return np.array([self._voxel_to_world(idx) for idx in occupied])
    
    def get_point_cloud(self) -> np.ndarray:
        """
        获取点云 (所有占据体素的中心)
        
        Returns:
            (N, 3) 点云数组
        """
        return self.get_occupied_centers()
    
    def point_to_key(self, point: np.ndarray) -> Tuple[int, int, int]:
        """将世界坐标点转换为体素索引"""
        return self._world_to_voxel(point)
    
    def key_to_center(self, key: Tuple[int, int, int]) -> np.ndarray:
        """将体素索引转换为中心坐标"""
        return self._voxel_to_world(key)
    
    def query_nearby_voxels(
        self,
        center: np.ndarray,
        radius: float,
    ) -> List[Tuple[Tuple[int, int, int], VoxelData]]:
        """查询指定范围内的体素"""
        center_idx = self._world_to_voxel(center)
        voxel_radius = int(np.ceil(radius / self.voxel_size))
        
        results = []
        for di in range(-voxel_radius, voxel_radius + 1):
            for dj in range(-voxel_radius, voxel_radius + 1):
                for dk in range(-voxel_radius, voxel_radius + 1):
                    idx = (center_idx[0] + di, center_idx[1] + dj, center_idx[2] + dk)
                    if idx in self._voxels:
                        voxel_center = self._voxel_to_world(idx)
                        if np.linalg.norm(voxel_center - center) <= radius:
                            results.append((idx, self._voxels[idx]))
        
        return results
    
    def raycast(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float = 10.0,
    ) -> Optional[Tuple[Tuple[int, int, int], float]]:
        """
        射线投射，找到第一个被占据的体素
        
        Args:
            origin: 射线起点
            direction: 射线方向 (需归一化)
            max_distance: 最大射线距离
            
        Returns:
            (体素索引, 距离) 或 None
        """
        direction = direction / np.linalg.norm(direction)
        step_size = self.voxel_size * 0.5
        
        current_pos = origin.copy()
        distance = 0.0
        
        while distance < max_distance:
            voxel_idx = self._world_to_voxel(current_pos)
            if self.is_occupied(voxel_idx):
                return voxel_idx, distance
            
            current_pos += direction * step_size
            distance += step_size
        
        return None
    
    @property
    def num_voxels(self) -> int:
        """体素总数"""
        return len(self._voxels)
    
    @property
    def voxel_count(self) -> int:
        """体素总数 (别名)"""
        return self.num_voxels
    
    @property
    def num_occupied(self) -> int:
        """被占据体素数"""
        return len(self.get_occupied_voxels())
    
    @property
    def occupied_count(self) -> int:
        """被占据体素数 (别名)"""
        return self.num_occupied
    
    @property
    def voxels(self) -> Dict[Tuple[int, int, int], VoxelData]:
        """体素字典"""
        return self._voxels
    
    @property
    def resolution(self) -> float:
        """体素分辨率 (别名)"""
        return self.voxel_size
    
    @property
    def bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """空间边界"""
        if self._bounds_min is None:
            return None
        return self._bounds_min.copy(), self._bounds_max.copy()
    
    @property
    def dimensions(self) -> Optional[np.ndarray]:
        """空间尺寸 (米)"""
        if self._bounds_min is None:
            return None
        return self._bounds_max - self._bounds_min
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            'voxel_size': self.voxel_size,
            'num_voxels': self.num_voxels,
            'num_occupied': self.num_occupied,
            'bounds': [b.tolist() for b in self.bounds] if self.bounds else None,
            'dimensions': self.dimensions.tolist() if self.dimensions is not None else None,
            'total_points_integrated': self._total_points_integrated,
        }
    
    def __repr__(self) -> str:
        return (
            f"SparseVoxelMap(voxel_size={self.voxel_size}, "
            f"num_voxels={self.num_voxels}, "
            f"num_occupied={self.num_occupied})"
        )


def connected_components_3d(
    voxel_map: SparseVoxelMap,
    feature_threshold: float = 0.5,
) -> Dict[int, Set[Tuple[int, int, int]]]:
    """
    3D 连通域分析
    
    将相邻且语义特征相似的体素聚类为实体
    
    Args:
        voxel_map: 体素地图
        feature_threshold: 特征相似度阈值
        
    Returns:
        {entity_id: {voxel_indices}}
    """
    occupied = set(voxel_map.get_occupied_voxels())
    visited = set()
    entities: Dict[int, Set[Tuple[int, int, int]]] = {}
    entity_id = 0
    
    # 6-邻域偏移
    neighbors = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    
    def feature_similar(v1: VoxelData, v2: VoxelData) -> bool:
        """检查两个体素的特征是否相似"""
        f1, f2 = v1.feature, v2.feature
        if f1 is None or f2 is None:
            return True  # 无特征时默认相似
        
        # 余弦相似度
        sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
        return sim >= feature_threshold
    
    def bfs(start: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
        """广度优先搜索连通域"""
        component = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            current = queue.pop(0)
            component.add(current)
            current_voxel = voxel_map.get_voxel(current)
            
            for di, dj, dk in neighbors:
                neighbor = (current[0] + di, current[1] + dj, current[2] + dk)
                
                if neighbor in occupied and neighbor not in visited:
                    neighbor_voxel = voxel_map.get_voxel(neighbor)
                    
                    if feature_similar(current_voxel, neighbor_voxel):
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return component
    
    # 遍历所有占据体素
    for voxel_idx in occupied:
        if voxel_idx not in visited:
            component = bfs(voxel_idx)
            if len(component) > 0:
                entities[entity_id] = component
                
                # 更新体素的 entity_id
                for idx in component:
                    voxel = voxel_map.get_voxel(idx)
                    if voxel:
                        voxel.entity_id = f"entity_{entity_id:03d}"
                
                entity_id += 1
    
    logger.info(f"Found {len(entities)} connected components")
    return entities
