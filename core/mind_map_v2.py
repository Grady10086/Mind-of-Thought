"""
心智地图 V2 - 四步构建法

实现基于新方法论的空间心智地图：
1. 几何"积木化" (Voxel Construction)
2. 实体"符号化" (Object Instantiation)
3. 轨迹"线索化" (Trajectory & POV)
4. 结构"输出化" (Promptable Structure)

核心改进：
- 稀疏体素表示替代原始点云
- 自动聚类发现物体（不依赖外部检测器）
- 可见性关联（记录物体在哪些帧可见）
- LLM 友好的结构化输出

使用方法：
    from core.mind_map_v2 import MindMapBuilder
    from core.scene import SceneLoader
    
    # 加载场景
    scene = SceneLoader.load_glb("scene.glb")
    
    # 四步构建
    builder = MindMapBuilder(voxel_size=0.1)
    mind_map = builder.build(scene)
    
    # 生成 LLM Prompt
    prompt = mind_map.to_prompt()
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from pathlib import Path

import numpy as np

from .scene import Scene3D, CameraPose
from .voxel_map import SparseVoxelMap, VoxelData, connected_components_3d

logger = logging.getLogger(__name__)


@dataclass
class EntityV2:
    """
    实体 (物体) 数据结构 V2
    
    由连通域分析自动生成，包含完整的时空信息
    """
    # 基本信息
    entity_id: str                    # 实体 ID (如 "entity_001")
    
    # 几何信息 (由体素聚类计算)
    centroid: np.ndarray              # (3,) 中心点坐标
    bbox_min: np.ndarray              # (3,) 包围盒最小点
    bbox_max: np.ndarray              # (3,) 包围盒最大点
    voxel_count: int = 0              # 体素数量
    
    # 外观信息
    mean_color: Optional[np.ndarray] = None    # 平均颜色
    mean_feature: Optional[np.ndarray] = None  # 平均语义特征
    
    # 时序信息
    first_seen_frame: int = -1        # 首次出现的帧
    last_seen_frame: int = -1         # 最后出现的帧
    visible_frames: List[int] = field(default_factory=list)  # 可见帧列表
    
    # 语义标签 (可选，由外部检测器或 CLIP 提供)
    semantic_label: Optional[str] = None
    confidence: float = 1.0
    
    @property
    def size(self) -> np.ndarray:
        """物体尺寸 (米)"""
        return self.bbox_max - self.bbox_min
    
    @property
    def volume(self) -> float:
        """体积估计 (立方米)"""
        return float(np.prod(self.size))
    
    def distance_to(self, other: Union['EntityV2', np.ndarray]) -> float:
        """到另一个实体/点的距离"""
        if isinstance(other, EntityV2):
            return float(np.linalg.norm(self.centroid - other.centroid))
        return float(np.linalg.norm(self.centroid - np.array(other)))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (LLM 友好)"""
        return {
            'id': self.entity_id,
            'label': self.semantic_label or 'unknown',
            'center': [round(x, 2) for x in self.centroid.tolist()],
            'size': [round(x, 2) for x in self.size.tolist()],
            'first_seen': self.first_seen_frame,
            'visible_in': self.visible_frames[:10],  # 限制长度
        }


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    frame_id: int
    position: np.ndarray      # (3,) 相机位置
    rotation: np.ndarray      # (3, 3) 旋转矩阵
    visible_entities: List[str] = field(default_factory=list)  # 当前可见的实体 ID
    
    @property
    def forward(self) -> np.ndarray:
        """相机朝向"""
        return self.rotation[2, :]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame': self.frame_id,
            'position': [round(x, 2) for x in self.position.tolist()],
            'visible': self.visible_entities,
        }


class MindMapV2:
    """
    心智地图 V2
    
    三层结构：
    1. 底层 (体素层): 几何约束 - "挡不挡得住"、"能不能走"
    2. 中层 (对象层): 逻辑符号 - "这是什么"、"它在哪"
    3. 顶层 (时空层): 时空关联 - "先看到什么"、"离我多远"
    """
    
    def __init__(self, voxel_size: float = 0.1):
        """
        Args:
            voxel_size: 体素大小 (米)
        """
        self.voxel_size = voxel_size
        
        # 底层: 体素地图
        self.voxel_map: Optional[SparseVoxelMap] = None
        
        # 中层: 实体列表
        self._entities: Dict[str, EntityV2] = {}
        
        # 顶层: 轨迹
        self._trajectory: List[TrajectoryPoint] = []
        
        # 空间关系图 (邻接表)
        self._spatial_graph: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # 元信息
        self.scene_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.world_scale: str = f"1 unit = {voxel_size}m"
    
    # ==================== 实体管理 ====================
    
    def add_entity(self, entity: EntityV2):
        """添加实体"""
        self._entities[entity.entity_id] = entity
    
    def get_entity(self, entity_id: str) -> Optional[EntityV2]:
        """获取实体"""
        return self._entities.get(entity_id)
    
    def get_entities_by_label(self, label: str) -> List[EntityV2]:
        """按语义标签获取实体"""
        return [e for e in self._entities.values() if e.semantic_label == label]
    
    @property
    def entities(self) -> List[EntityV2]:
        """所有实体"""
        return list(self._entities.values())
    
    @property
    def entity_count(self) -> int:
        """实体数量"""
        return len(self._entities)
    
    # ==================== 轨迹管理 ====================
    
    def add_trajectory_point(self, point: TrajectoryPoint):
        """添加轨迹点"""
        self._trajectory.append(point)
    
    @property
    def trajectory(self) -> List[TrajectoryPoint]:
        """轨迹"""
        return self._trajectory
    
    def get_camera_path(self) -> np.ndarray:
        """获取相机路径 (T, 3)"""
        if not self._trajectory:
            return np.array([]).reshape(0, 3)
        return np.array([p.position for p in self._trajectory])
    
    # ==================== 空间关系 ====================
    
    def compute_spatial_relations(self, max_distance: float = 5.0):
        """
        计算实体间的空间关系图
        
        Args:
            max_distance: 最大关联距离
        """
        entities = self.entities
        self._spatial_graph = {e.entity_id: {} for e in entities}
        
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                dist = e1.distance_to(e2)
                if dist <= max_distance:
                    # 计算相对方位
                    direction = e2.centroid - e1.centroid
                    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
                    
                    # 简化方位描述
                    relation = self._direction_to_relation(direction_norm)
                    
                    self._spatial_graph[e1.entity_id][e2.entity_id] = {
                        'distance': round(dist, 2),
                        'relation': relation,
                    }
                    
                    # 反向关系
                    reverse_relation = self._reverse_relation(relation)
                    self._spatial_graph[e2.entity_id][e1.entity_id] = {
                        'distance': round(dist, 2),
                        'relation': reverse_relation,
                    }
    
    def _direction_to_relation(self, direction: np.ndarray) -> str:
        """将方向向量转换为关系描述"""
        # 假设 Y 轴向上
        x, y, z = direction
        
        relations = []
        
        # 垂直关系
        if y > 0.5:
            relations.append("above")
        elif y < -0.5:
            relations.append("below")
        
        # 水平关系 (简化为前后左右)
        horizontal = np.array([x, 0, z])
        horizontal = horizontal / (np.linalg.norm(horizontal) + 1e-8)
        
        if abs(horizontal[2]) > abs(horizontal[0]):
            if horizontal[2] > 0:
                relations.append("in_front_of")
            else:
                relations.append("behind")
        else:
            if horizontal[0] > 0:
                relations.append("right_of")
            else:
                relations.append("left_of")
        
        return "_".join(relations) if relations else "near"
    
    def _reverse_relation(self, relation: str) -> str:
        """获取反向关系"""
        reverses = {
            "above": "below",
            "below": "above",
            "left_of": "right_of",
            "right_of": "left_of",
            "in_front_of": "behind",
            "behind": "in_front_of",
            "near": "near",
        }
        parts = relation.split("_")
        reversed_parts = []
        for part in parts:
            if part in reverses:
                reversed_parts.append(reverses[part])
            else:
                reversed_parts.append(part)
        return "_".join(reversed_parts)
    
    def get_relations(self, entity_id: str) -> Dict[str, Dict[str, Any]]:
        """获取实体的空间关系"""
        return self._spatial_graph.get(entity_id, {})
    
    # ==================== 空间查询 ====================
    
    def query_nearby(
        self,
        target: Union[str, np.ndarray],
        radius: float = 2.0,
    ) -> List[EntityV2]:
        """查询附近的实体"""
        if isinstance(target, str):
            entity = self.get_entity(target)
            if entity is None:
                return []
            center = entity.centroid
        else:
            center = np.array(target)
        
        nearby = []
        for e in self.entities:
            if isinstance(target, str) and e.entity_id == target:
                continue
            if e.distance_to(center) <= radius:
                nearby.append(e)
        
        nearby.sort(key=lambda x: x.distance_to(center))
        return nearby
    
    def query_visible_at_frame(self, frame_id: int) -> List[EntityV2]:
        """查询某帧可见的实体"""
        if frame_id < len(self._trajectory):
            visible_ids = self._trajectory[frame_id].visible_entities
            return [self.get_entity(eid) for eid in visible_ids if self.get_entity(eid)]
        return []
    
    def query_first_seen(self, entity_id: str) -> Optional[int]:
        """查询实体首次出现的帧"""
        entity = self.get_entity(entity_id)
        return entity.first_seen_frame if entity else None
    
    # ==================== LLM 输出 ====================
    
    def to_prompt(self, format: str = 'yaml') -> str:
        """
        生成 LLM 可读的结构化 Prompt
        
        Args:
            format: 输出格式 ('yaml' 或 'markdown')
            
        Returns:
            结构化文本
        """
        if format == 'yaml':
            return self._to_yaml_prompt()
        else:
            return self._to_markdown_prompt()
    
    def _to_yaml_prompt(self) -> str:
        """生成 YAML 格式的 Prompt"""
        lines = [
            "Spatial_Mind_Map:",
            f"  World_Scale: \"{self.world_scale}\"",
            f"  Total_Entities: {self.entity_count}",
            "",
            "  Entities:",
        ]
        
        for entity in self.entities:
            label = entity.semantic_label or 'object'
            lines.append(f"    - id: \"{entity.entity_id}\"")
            lines.append(f"      label: \"{label}\"")
            lines.append(f"      center: {[round(x, 2) for x in entity.centroid.tolist()]}")
            lines.append(f"      size: {[round(x, 2) for x in entity.size.tolist()]}")
            if entity.visible_frames:
                lines.append(f"      first_seen: frame_{entity.first_seen_frame}")
        
        if self._trajectory:
            lines.extend([
                "",
                "  Robot_Path:",
            ])
            # 简化轨迹描述
            key_frames = [0, len(self._trajectory) // 2, len(self._trajectory) - 1]
            for i in key_frames:
                if i < len(self._trajectory):
                    tp = self._trajectory[i]
                    pos = [round(x, 2) for x in tp.position.tolist()]
                    lines.append(f"    - frame_{i}: position={pos}, visible={len(tp.visible_entities)} objects")
        
        return "\n".join(lines)
    
    def _to_markdown_prompt(self) -> str:
        """生成 Markdown 格式的 Prompt"""
        lines = [
            "## Spatial Mind Map",
            "",
            f"**World Scale**: {self.world_scale}",
            f"**Total Entities**: {self.entity_count}",
            "",
            "### Entities",
            "",
            "| ID | Label | Center (x,y,z) | Size (m) | First Seen |",
            "|---|---|---|---|---|",
        ]
        
        for entity in self.entities:
            label = entity.semantic_label or 'object'
            center = ", ".join([f"{x:.2f}" for x in entity.centroid])
            size = ", ".join([f"{x:.2f}" for x in entity.size])
            first_seen = f"frame_{entity.first_seen_frame}" if entity.first_seen_frame >= 0 else "N/A"
            lines.append(f"| {entity.entity_id} | {label} | ({center}) | ({size}) | {first_seen} |")
        
        if self._spatial_graph:
            lines.extend([
                "",
                "### Spatial Relations",
                "",
            ])
            for eid, relations in self._spatial_graph.items():
                if relations:
                    rel_strs = [f"{eid} is {r['relation']} {other} ({r['distance']}m)" 
                               for other, r in list(relations.items())[:3]]
                    lines.append(f"- {'; '.join(rel_strs)}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为完整字典"""
        return {
            'world_scale': self.world_scale,
            'voxel_size': self.voxel_size,
            'entity_count': self.entity_count,
            'voxel_stats': self.voxel_map.to_dict() if self.voxel_map else None,
            'entities': [e.to_dict() for e in self.entities],
            'trajectory': [t.to_dict() for t in self._trajectory],
            'spatial_graph': self._spatial_graph,
        }
    
    def save(self, path: str):
        """保存到 JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved mind map to {path}")
    
    def __repr__(self) -> str:
        return (
            f"MindMapV2(entities={self.entity_count}, "
            f"trajectory_points={len(self._trajectory)}, "
            f"voxel_size={self.voxel_size}m)"
        )


class MindMapBuilder:
    """
    心智地图构建器
    
    实现四步构建法：
    1. 几何"积木化" - 构建稀疏体素地图
    2. 实体"符号化" - 连通域聚类得到物体
    3. 轨迹"线索化" - 关联相机位姿与可见性
    4. 结构"输出化" - 生成 LLM 友好的输出
    """
    
    def __init__(
        self,
        voxel_size: float = 0.1,
        min_entity_voxels: int = 10,
        feature_threshold: float = 0.5,
    ):
        """
        Args:
            voxel_size: 体素大小 (米)
            min_entity_voxels: 实体最小体素数 (过滤噪声)
            feature_threshold: 特征相似度阈值 (用于聚类)
        """
        self.voxel_size = voxel_size
        self.min_entity_voxels = min_entity_voxels
        self.feature_threshold = feature_threshold
    
    def build(
        self,
        scene: Scene3D,
        semantic_labels: Optional[Dict[str, str]] = None,
    ) -> MindMapV2:
        """
        四步构建心智地图
        
        Args:
            scene: Scene3D 对象
            semantic_labels: 可选的语义标签 {entity_id: label}
            
        Returns:
            MindMapV2 对象
        """
        logger.info("=" * 50)
        logger.info("开始构建心智地图 (四步构建法)")
        logger.info("=" * 50)
        
        mind_map = MindMapV2(voxel_size=self.voxel_size)
        
        # ========== 第一步：几何"积木化" ==========
        logger.info("[Step 1/4] 几何积木化 - 构建稀疏体素地图")
        mind_map.voxel_map = self._step1_voxelize(scene)
        logger.info(f"  -> {mind_map.voxel_map}")
        
        # ========== 第二步：实体"符号化" ==========
        logger.info("[Step 2/4] 实体符号化 - 连通域聚类")
        entities = self._step2_instantiate(mind_map.voxel_map, scene)
        for entity in entities:
            if semantic_labels and entity.entity_id in semantic_labels:
                entity.semantic_label = semantic_labels[entity.entity_id]
            mind_map.add_entity(entity)
        logger.info(f"  -> 发现 {len(entities)} 个实体")
        
        # ========== 第三步：轨迹"线索化" ==========
        logger.info("[Step 3/4] 轨迹线索化 - 关联相机与可见性")
        trajectory = self._step3_trajectory(scene, mind_map)
        for point in trajectory:
            mind_map.add_trajectory_point(point)
        logger.info(f"  -> {len(trajectory)} 个轨迹点")
        
        # ========== 第四步：结构"输出化" ==========
        logger.info("[Step 4/4] 结构输出化 - 计算空间关系")
        mind_map.compute_spatial_relations()
        mind_map.scene_bounds = scene.bounds
        logger.info(f"  -> 空间关系图构建完成")
        
        logger.info("=" * 50)
        logger.info(f"心智地图构建完成: {mind_map}")
        logger.info("=" * 50)
        
        return mind_map
    
    def _step1_voxelize(self, scene: Scene3D) -> SparseVoxelMap:
        """第一步：体素化"""
        voxel_map = SparseVoxelMap(voxel_size=self.voxel_size)
        
        if scene.point_cloud is not None:
            voxel_map.integrate_points(
                scene.point_cloud,
                colors=scene.colors,
                frame_id=0,
            )
        
        return voxel_map
    
    def _step2_instantiate(
        self,
        voxel_map: SparseVoxelMap,
        scene: Scene3D,
    ) -> List[EntityV2]:
        """第二步：实体实例化"""
        # 连通域分析
        components = connected_components_3d(voxel_map, self.feature_threshold)
        
        entities = []
        for entity_id_num, voxel_indices in components.items():
            # 过滤小噪声
            if len(voxel_indices) < self.min_entity_voxels:
                continue
            
            entity_id = f"entity_{entity_id_num:03d}"
            
            # 计算几何属性
            centers = []
            colors = []
            features = []
            first_seen = float('inf')
            last_seen = -1
            
            for idx in voxel_indices:
                voxel = voxel_map.get_voxel(idx)
                if voxel:
                    center = (np.array(idx) + 0.5) * voxel_map.voxel_size
                    centers.append(center)
                    
                    if voxel.color is not None:
                        colors.append(voxel.color)
                    if voxel.feature is not None:
                        features.append(voxel.feature)
                    
                    if voxel.first_seen >= 0:
                        first_seen = min(first_seen, voxel.first_seen)
                    if voxel.last_seen >= 0:
                        last_seen = max(last_seen, voxel.last_seen)
            
            if not centers:
                continue
            
            centers = np.array(centers)
            
            entity = EntityV2(
                entity_id=entity_id,
                centroid=centers.mean(axis=0),
                bbox_min=centers.min(axis=0),
                bbox_max=centers.max(axis=0),
                voxel_count=len(voxel_indices),
                mean_color=np.mean(colors, axis=0) if colors else None,
                mean_feature=np.mean(features, axis=0) if features else None,
                first_seen_frame=int(first_seen) if first_seen != float('inf') else -1,
                last_seen_frame=int(last_seen),
            )
            
            entities.append(entity)
        
        return entities
    
    def _step3_trajectory(
        self,
        scene: Scene3D,
        mind_map: MindMapV2,
    ) -> List[TrajectoryPoint]:
        """第三步：轨迹与可见性"""
        trajectory = []
        
        if not scene.camera_poses:
            return trajectory
        
        sorted_names = sorted(scene.camera_poses.keys())
        
        for frame_id, name in enumerate(sorted_names):
            camera = scene.camera_poses[name]
            
            # 计算相机位置
            position = camera.position
            rotation = camera.rotation
            
            # 计算可见性 (简化版：基于距离和角度)
            visible_entities = []
            for entity in mind_map.entities:
                # 检查是否在视野内
                to_entity = entity.centroid - position
                distance = np.linalg.norm(to_entity)
                
                if distance < 10.0:  # 10m 内
                    # 检查是否在相机前方
                    forward = rotation[2, :] if rotation is not None else np.array([0, 0, 1])
                    dot = np.dot(to_entity / (distance + 1e-8), forward)
                    
                    if dot > 0.3:  # 大致在视野内
                        visible_entities.append(entity.entity_id)
                        
                        # 更新实体的可见帧
                        if frame_id not in entity.visible_frames:
                            entity.visible_frames.append(frame_id)
            
            trajectory.append(TrajectoryPoint(
                frame_id=frame_id,
                position=position,
                rotation=rotation if rotation is not None else np.eye(3),
                visible_entities=visible_entities,
            ))
        
        return trajectory


# 便捷函数
def build_mind_map(
    scene_path: str,
    voxel_size: float = 0.1,
) -> MindMapV2:
    """
    从场景文件构建心智地图
    
    Args:
        scene_path: GLB/PLY 文件路径
        voxel_size: 体素大小
        
    Returns:
        MindMapV2 对象
    """
    from .scene import SceneLoader
    scene = SceneLoader.load(scene_path)
    
    builder = MindMapBuilder(voxel_size=voxel_size)
    return builder.build(scene)
