"""
空间记忆与心智地图模块

功能:
- 构建空间心智地图 (Spatial Mind Map)
- 管理场景中的物体、区域、空间关系
- 支持空间推理查询

核心概念:
- SpatialObject: 场景中的单个物体
- SpatialRegion: 语义区域 (如 "厨房", "客厅")
- SpatialRelation: 物体间的空间关系 (如 "在...上方", "在...旁边")
- MindMap: 整合以上信息的心智地图

使用方法:
    from core.memory import MindMap, SpatialObject
    from core.scene import SceneLoader
    
    # 加载 3D 场景
    scene = SceneLoader.load_glb("path/to/scene.glb")
    
    # 构建心智地图
    mind_map = MindMap.from_scene(scene)
    
    # 添加检测到的物体
    mind_map.add_object(SpatialObject(
        name="chair",
        position=np.array([1.0, 0.5, 2.0]),
        category="furniture"
    ))
    
    # 空间查询
    nearby = mind_map.query_nearby("chair", radius=1.5)
    relations = mind_map.get_relations("chair")
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from enum import Enum

import numpy as np

from .scene import Scene3D, CameraPose

logger = logging.getLogger(__name__)


class SpatialRelationType(Enum):
    """空间关系类型"""
    # 垂直关系
    ABOVE = "above"           # 在...上方
    BELOW = "below"           # 在...下方
    ON_TOP_OF = "on_top_of"   # 在...顶部
    
    # 水平关系
    LEFT_OF = "left_of"       # 在...左边
    RIGHT_OF = "right_of"     # 在...右边
    IN_FRONT_OF = "in_front_of"  # 在...前面
    BEHIND = "behind"         # 在...后面
    
    # 距离关系
    NEAR = "near"             # 靠近
    FAR = "far"               # 远离
    NEXT_TO = "next_to"       # 紧邻
    
    # 包含关系
    INSIDE = "inside"         # 在...内部
    OUTSIDE = "outside"       # 在...外部
    CONTAINS = "contains"     # 包含
    
    # 对齐关系
    ALIGNED_WITH = "aligned_with"  # 对齐
    FACING = "facing"         # 面向


@dataclass
class BoundingBox3D:
    """3D 包围盒"""
    center: np.ndarray        # (3,) 中心点
    size: np.ndarray          # (3,) 尺寸 (长宽高)
    rotation: Optional[np.ndarray] = None  # (3, 3) 旋转矩阵
    
    @property
    def min_point(self) -> np.ndarray:
        """最小角点"""
        return self.center - self.size / 2
    
    @property
    def max_point(self) -> np.ndarray:
        """最大角点"""
        return self.center + self.size / 2
    
    @property
    def volume(self) -> float:
        """体积"""
        return float(np.prod(self.size))
    
    def contains_point(self, point: np.ndarray) -> bool:
        """检查点是否在包围盒内"""
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)
    
    def intersects(self, other: 'BoundingBox3D') -> bool:
        """检查是否与另一个包围盒相交"""
        return (
            np.all(self.min_point <= other.max_point) and
            np.all(self.max_point >= other.min_point)
        )
    
    def distance_to(self, other: 'BoundingBox3D') -> float:
        """到另一个包围盒的距离"""
        return float(np.linalg.norm(self.center - other.center))


@dataclass
class SpatialObject:
    """
    场景中的空间物体
    
    表示检测到的物体及其空间属性
    """
    name: str                           # 物体名称/标签
    position: np.ndarray                # (3,) 位置坐标
    category: Optional[str] = None      # 类别 (如 "furniture", "appliance")
    
    # 几何属性
    bbox: Optional[BoundingBox3D] = None  # 3D 包围盒
    size: Optional[np.ndarray] = None     # (3,) 尺寸估计
    
    # 外观属性
    color: Optional[str] = None           # 主色调
    material: Optional[str] = None        # 材质
    
    # 状态
    confidence: float = 1.0               # 检测置信度
    frame_id: Optional[int] = None        # 检测帧 ID
    
    # 唯一标识
    object_id: Optional[str] = None
    
    # 语义属性
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.object_id is None:
            self.object_id = f"{self.name}_{id(self)}"
        if isinstance(self.position, list):
            self.position = np.array(self.position)
    
    def distance_to(self, other: Union['SpatialObject', np.ndarray]) -> float:
        """计算到另一个物体/点的距离"""
        if isinstance(other, SpatialObject):
            other_pos = other.position
        else:
            other_pos = np.array(other)
        return float(np.linalg.norm(self.position - other_pos))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'object_id': self.object_id,
            'position': self.position.tolist(),
            'category': self.category,
            'confidence': self.confidence,
            'attributes': self.attributes,
        }


@dataclass
class SpatialRelation:
    """
    空间关系
    
    描述两个物体之间的空间关系
    """
    subject: str              # 主体物体 ID
    relation: SpatialRelationType  # 关系类型
    object: str               # 客体物体 ID
    confidence: float = 1.0   # 关系置信度
    distance: Optional[float] = None  # 距离 (米)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'subject': self.subject,
            'relation': self.relation.value,
            'object': self.object,
            'confidence': self.confidence,
            'distance': self.distance,
        }
    
    def __str__(self) -> str:
        return f"{self.subject} {self.relation.value} {self.object}"


@dataclass
class SpatialRegion:
    """
    语义区域
    
    表示场景中的功能区域 (如厨房、客厅)
    """
    name: str                           # 区域名称
    center: np.ndarray                  # (3,) 中心点
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (min, max)
    objects: List[str] = field(default_factory=list)  # 区域内物体 ID 列表
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """检查点是否在区域内"""
        if self.bounds is None:
            return False
        min_pt, max_pt = self.bounds
        return np.all(point >= min_pt) and np.all(point <= max_pt)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'center': self.center.tolist(),
            'bounds': [b.tolist() for b in self.bounds] if self.bounds else None,
            'objects': self.objects,
            'attributes': self.attributes,
        }


class MindMap:
    """
    空间心智地图
    
    整合 3D 场景信息，构建结构化的空间理解:
    - 物体位置与属性
    - 物体间空间关系
    - 语义区域划分
    - 相机轨迹与视角
    """
    
    def __init__(self, scene: Optional[Scene3D] = None):
        """
        Args:
            scene: 3D 场景数据 (可选)
        """
        self.scene = scene
        
        # 物体存储
        self._objects: Dict[str, SpatialObject] = {}
        
        # 空间关系
        self._relations: List[SpatialRelation] = []
        
        # 语义区域
        self._regions: Dict[str, SpatialRegion] = {}
        
        # 场景元信息
        self.scene_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.scene_center: Optional[np.ndarray] = None
        
        # 从场景初始化
        if scene is not None:
            self._init_from_scene(scene)
    
    def _init_from_scene(self, scene: Scene3D):
        """从 3D 场景初始化"""
        self.scene_bounds = scene.bounds
        self.scene_center = scene.center
    
    @classmethod
    def from_scene(cls, scene: Scene3D) -> 'MindMap':
        """从 Scene3D 创建心智地图"""
        return cls(scene=scene)
    
    @classmethod
    def from_glb(cls, glb_path: str) -> 'MindMap':
        """从 GLB 文件创建心智地图"""
        from .scene import SceneLoader
        scene = SceneLoader.load_glb(glb_path)
        return cls(scene=scene)
    
    # ==================== 物体管理 ====================
    
    def add_object(self, obj: SpatialObject) -> str:
        """
        添加物体
        
        Args:
            obj: 空间物体
            
        Returns:
            物体 ID
        """
        self._objects[obj.object_id] = obj
        logger.debug(f"Added object: {obj.name} at {obj.position}")
        return obj.object_id
    
    def add_objects(self, objects: List[SpatialObject]):
        """批量添加物体"""
        for obj in objects:
            self.add_object(obj)
    
    def get_object(self, object_id: str) -> Optional[SpatialObject]:
        """获取物体"""
        return self._objects.get(object_id)
    
    def get_objects_by_name(self, name: str) -> List[SpatialObject]:
        """按名称获取物体"""
        return [obj for obj in self._objects.values() if obj.name == name]
    
    def get_objects_by_category(self, category: str) -> List[SpatialObject]:
        """按类别获取物体"""
        return [obj for obj in self._objects.values() if obj.category == category]
    
    def remove_object(self, object_id: str) -> bool:
        """移除物体"""
        if object_id in self._objects:
            del self._objects[object_id]
            # 移除相关关系
            self._relations = [
                r for r in self._relations 
                if r.subject != object_id and r.object != object_id
            ]
            return True
        return False
    
    @property
    def objects(self) -> List[SpatialObject]:
        """所有物体列表"""
        return list(self._objects.values())
    
    @property
    def object_count(self) -> int:
        """物体数量"""
        return len(self._objects)
    
    # ==================== 空间关系 ====================
    
    def add_relation(self, relation: SpatialRelation):
        """添加空间关系"""
        self._relations.append(relation)
    
    def compute_relation(
        self, 
        obj1: SpatialObject, 
        obj2: SpatialObject,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[SpatialRelation]:
        """
        计算两个物体之间的空间关系
        
        Args:
            obj1: 物体 1
            obj2: 物体 2
            thresholds: 距离阈值
            
        Returns:
            空间关系列表
        """
        if thresholds is None:
            thresholds = {
                'near': 1.0,       # 近距离阈值 (米)
                'next_to': 0.5,    # 紧邻阈值
                'vertical': 0.3,   # 垂直关系阈值
            }
        
        relations = []
        
        pos1 = obj1.position
        pos2 = obj2.position
        diff = pos2 - pos1
        distance = np.linalg.norm(diff)
        
        # 距离关系
        if distance < thresholds['next_to']:
            relations.append(SpatialRelation(
                subject=obj1.object_id,
                relation=SpatialRelationType.NEXT_TO,
                object=obj2.object_id,
                distance=distance,
            ))
        elif distance < thresholds['near']:
            relations.append(SpatialRelation(
                subject=obj1.object_id,
                relation=SpatialRelationType.NEAR,
                object=obj2.object_id,
                distance=distance,
            ))
        
        # 垂直关系 (假设 Y 轴向上)
        if abs(diff[1]) > thresholds['vertical']:
            if diff[1] > 0:
                relations.append(SpatialRelation(
                    subject=obj1.object_id,
                    relation=SpatialRelationType.BELOW,
                    object=obj2.object_id,
                    distance=abs(diff[1]),
                ))
            else:
                relations.append(SpatialRelation(
                    subject=obj1.object_id,
                    relation=SpatialRelationType.ABOVE,
                    object=obj2.object_id,
                    distance=abs(diff[1]),
                ))
        
        return relations
    
    def compute_all_relations(self, max_distance: float = 3.0):
        """
        计算所有物体对之间的空间关系
        
        Args:
            max_distance: 最大考虑距离
        """
        objects = self.objects
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                if obj1.distance_to(obj2) < max_distance:
                    relations = self.compute_relation(obj1, obj2)
                    for rel in relations:
                        self.add_relation(rel)
    
    def get_relations(self, object_id: str) -> List[SpatialRelation]:
        """获取物体的所有空间关系"""
        return [
            r for r in self._relations 
            if r.subject == object_id or r.object == object_id
        ]
    
    @property
    def relations(self) -> List[SpatialRelation]:
        """所有空间关系"""
        return self._relations.copy()
    
    # ==================== 空间查询 ====================
    
    def query_nearby(
        self, 
        target: Union[str, np.ndarray], 
        radius: float = 1.0,
        category: Optional[str] = None,
    ) -> List[SpatialObject]:
        """
        查询目标附近的物体
        
        Args:
            target: 物体 ID 或位置坐标
            radius: 搜索半径 (米)
            category: 筛选类别
            
        Returns:
            附近物体列表
        """
        # 获取目标位置
        if isinstance(target, str):
            obj = self.get_object(target)
            if obj is None:
                # 尝试按名称查找
                objs = self.get_objects_by_name(target)
                if not objs:
                    return []
                center = objs[0].position
            else:
                center = obj.position
        else:
            center = np.array(target)
        
        # 搜索附近物体
        nearby = []
        for obj in self.objects:
            if isinstance(target, str) and obj.object_id == target:
                continue
            if obj.distance_to(center) <= radius:
                if category is None or obj.category == category:
                    nearby.append(obj)
        
        # 按距离排序
        nearby.sort(key=lambda x: x.distance_to(center))
        return nearby
    
    def query_in_region(self, region_name: str) -> List[SpatialObject]:
        """查询区域内的物体"""
        region = self._regions.get(region_name)
        if region is None:
            return []
        return [self.get_object(oid) for oid in region.objects if self.get_object(oid)]
    
    def count_objects(self, name: Optional[str] = None, category: Optional[str] = None) -> int:
        """
        统计物体数量
        
        Args:
            name: 物体名称 (可选)
            category: 物体类别 (可选)
            
        Returns:
            物体数量
        """
        if name is None and category is None:
            return self.object_count
        
        count = 0
        for obj in self.objects:
            if name is not None and obj.name != name:
                continue
            if category is not None and obj.category != category:
                continue
            count += 1
        return count
    
    # ==================== 区域管理 ====================
    
    def add_region(self, region: SpatialRegion):
        """添加语义区域"""
        self._regions[region.name] = region
    
    def get_region(self, name: str) -> Optional[SpatialRegion]:
        """获取区域"""
        return self._regions.get(name)
    
    def assign_objects_to_regions(self):
        """将物体分配到区域"""
        for obj in self.objects:
            for region in self._regions.values():
                if region.contains_point(obj.position):
                    if obj.object_id not in region.objects:
                        region.objects.append(obj.object_id)
    
    # ==================== 相机轨迹 ====================
    
    def get_camera_trajectory(self) -> Optional[np.ndarray]:
        """获取相机轨迹"""
        if self.scene is None:
            return None
        return self.scene.get_camera_trajectory()
    
    def get_camera_at_frame(self, frame_id: int) -> Optional[CameraPose]:
        """获取指定帧的相机位姿"""
        if self.scene is None or not self.scene.camera_poses:
            return None
        
        sorted_names = sorted(self.scene.camera_poses.keys())
        if frame_id < len(sorted_names):
            return self.scene.camera_poses[sorted_names[frame_id]]
        return None
    
    # ==================== 序列化 ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'scene_bounds': [b.tolist() for b in self.scene_bounds] if self.scene_bounds else None,
            'scene_center': self.scene_center.tolist() if self.scene_center is not None else None,
            'objects': [obj.to_dict() for obj in self.objects],
            'relations': [rel.to_dict() for rel in self.relations],
            'regions': {name: reg.to_dict() for name, reg in self._regions.items()},
            'object_count': self.object_count,
        }
    
    def save(self, path: str):
        """保存到 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved mind map to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MindMap':
        """从 JSON 文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        mind_map = cls()
        
        # 恢复场景元信息
        if data.get('scene_bounds'):
            mind_map.scene_bounds = tuple(np.array(b) for b in data['scene_bounds'])
        if data.get('scene_center'):
            mind_map.scene_center = np.array(data['scene_center'])
        
        # 恢复物体
        for obj_data in data.get('objects', []):
            obj = SpatialObject(
                name=obj_data['name'],
                position=np.array(obj_data['position']),
                object_id=obj_data.get('object_id'),
                category=obj_data.get('category'),
                confidence=obj_data.get('confidence', 1.0),
                attributes=obj_data.get('attributes', {}),
            )
            mind_map.add_object(obj)
        
        # 恢复关系
        for rel_data in data.get('relations', []):
            rel = SpatialRelation(
                subject=rel_data['subject'],
                relation=SpatialRelationType(rel_data['relation']),
                object=rel_data['object'],
                confidence=rel_data.get('confidence', 1.0),
                distance=rel_data.get('distance'),
            )
            mind_map.add_relation(rel)
        
        logger.info(f"Loaded mind map from {path}")
        return mind_map
    
    def __repr__(self) -> str:
        return (
            f"MindMap(objects={self.object_count}, "
            f"relations={len(self._relations)}, "
            f"regions={len(self._regions)})"
        )


# 便捷函数
def create_mind_map(scene_path: str) -> MindMap:
    """从场景文件创建心智地图"""
    from .scene import SceneLoader
    scene = SceneLoader.load(scene_path)
    return MindMap.from_scene(scene)
