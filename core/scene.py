"""
3D 场景加载与表示模块

功能:
- 加载 DA3 生成的 GLB/PLY/NPZ 文件
- 提取点云、网格、相机位姿
- 提供统一的 3D 场景数据接口

使用方法:
    from core.scene import SceneLoader, Scene3D
    
    # 加载 GLB 文件
    scene = SceneLoader.load_glb("path/to/scene.glb")
    
    # 访问数据
    points = scene.point_cloud       # (N, 3) 点云坐标
    colors = scene.colors            # (N, 3) 点云颜色
    cameras = scene.camera_poses     # {name: 4x4 变换矩阵}
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    logger.warning("trimesh not installed. GLB loading will be limited.")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning("open3d not installed. PLY loading will be limited.")


@dataclass
class CameraPose:
    """相机位姿数据结构"""
    name: str                           # 相机名称 (如 "frame_001")
    extrinsic: np.ndarray              # 4x4 外参矩阵 (world-to-camera)
    intrinsic: Optional[np.ndarray] = None  # 3x3 内参矩阵
    width: Optional[int] = None
    height: Optional[int] = None
    _position_override: Optional[np.ndarray] = None  # 直接指定的位置 (用于 DA3 GLB)
    
    @property
    def position(self) -> np.ndarray:
        """相机在世界坐标系中的位置"""
        # 如果有直接指定的位置，使用它
        if self._position_override is not None:
            return self._position_override
        # 否则从外参矩阵提取: C = -R^T @ t
        R = self.extrinsic[:3, :3]
        t = self.extrinsic[:3, 3]
        return -R.T @ t
    
    @property
    def rotation(self) -> np.ndarray:
        """相机旋转矩阵"""
        return self.extrinsic[:3, :3]
    
    @property
    def forward(self) -> np.ndarray:
        """相机朝向 (Z 轴方向)"""
        return self.rotation[2, :]


@dataclass
class Scene3D:
    """
    3D 场景数据结构
    
    统一表示 DA3 输出的各种格式
    """
    # 基础几何
    point_cloud: Optional[np.ndarray] = None   # (N, 3) 点云坐标
    colors: Optional[np.ndarray] = None        # (N, 3) 或 (N, 4) 点云颜色 [0, 1]
    normals: Optional[np.ndarray] = None       # (N, 3) 法向量
    
    # 网格数据
    mesh_vertices: Optional[np.ndarray] = None    # (V, 3) 顶点坐标
    mesh_faces: Optional[np.ndarray] = None       # (F, 3) 面片索引
    mesh_vertex_colors: Optional[np.ndarray] = None  # (V, 3) 顶点颜色
    
    # 相机数据
    camera_poses: Dict[str, CameraPose] = field(default_factory=dict)
    
    # 深度数据
    depth_maps: Optional[np.ndarray] = None    # (T, H, W) 深度图序列
    
    # 元信息
    source_path: Optional[str] = None
    source_format: Optional[str] = None
    num_frames: int = 0
    
    # 原始 trimesh/open3d 对象 (可选)
    _raw_scene: Any = None
    
    @property
    def num_points(self) -> int:
        """点云点数"""
        if self.point_cloud is not None:
            return len(self.point_cloud)
        return 0
    
    @property
    def num_cameras(self) -> int:
        """相机数量"""
        return len(self.camera_poses)
    
    @property
    def has_mesh(self) -> bool:
        """是否包含网格数据"""
        return self.mesh_vertices is not None and self.mesh_faces is not None
    
    @property
    def bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """场景边界 (min, max)"""
        if self.point_cloud is not None:
            return self.point_cloud.min(axis=0), self.point_cloud.max(axis=0)
        return None
    
    @property
    def center(self) -> Optional[np.ndarray]:
        """场景中心点"""
        if self.point_cloud is not None:
            return self.point_cloud.mean(axis=0)
        return None
    
    def get_camera_trajectory(self) -> Optional[np.ndarray]:
        """
        获取相机轨迹
        
        Returns:
            (T, 3) 相机位置序列，按名称排序
        """
        if not self.camera_poses:
            return None
        
        # 按名称排序
        sorted_names = sorted(self.camera_poses.keys())
        positions = [self.camera_poses[name].position for name in sorted_names]
        return np.array(positions)
    
    def downsample(self, voxel_size: float = 0.01) -> 'Scene3D':
        """
        体素下采样
        
        Args:
            voxel_size: 体素大小
            
        Returns:
            下采样后的新 Scene3D 对象
        """
        if not HAS_OPEN3D or self.point_cloud is None:
            return self
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors[:, :3])
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
        
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        return Scene3D(
            point_cloud=np.asarray(pcd_down.points),
            colors=np.asarray(pcd_down.colors) if pcd_down.has_colors() else None,
            normals=np.asarray(pcd_down.normals) if pcd_down.has_normals() else None,
            camera_poses=self.camera_poses,
            source_path=self.source_path,
            source_format=self.source_format,
            num_frames=self.num_frames,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        result = {
            'num_points': self.num_points,
            'num_cameras': self.num_cameras,
            'has_mesh': self.has_mesh,
            'source_path': self.source_path,
            'source_format': self.source_format,
            'num_frames': self.num_frames,
        }
        
        if self.bounds:
            result['bounds'] = {
                'min': self.bounds[0].tolist(),
                'max': self.bounds[1].tolist(),
            }
        
        if self.center is not None:
            result['center'] = self.center.tolist()
        
        if self.camera_poses:
            result['cameras'] = {
                name: {
                    'position': pose.position.tolist(),
                    'rotation': pose.rotation.tolist(),
                }
                for name, pose in self.camera_poses.items()
            }
        
        return result


class SceneLoader:
    """
    3D 场景加载器
    
    支持格式:
    - GLB: DA3 输出的 3D 网格 + 相机棱锥
    - PLY: 点云格式
    - NPZ: DA3 原始数据 (深度、位姿、内参)
    """
    
    @staticmethod
    def load(path: Union[str, Path]) -> Scene3D:
        """
        自动检测格式并加载
        
        Args:
            path: 文件路径
            
        Returns:
            Scene3D 对象
        """
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix == '.glb' or suffix == '.gltf':
            return SceneLoader.load_glb(str(path))
        elif suffix == '.ply':
            return SceneLoader.load_ply(str(path))
        elif suffix == '.npz':
            return SceneLoader.load_npz(str(path))
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    @staticmethod
    def load_glb(glb_path: str) -> Scene3D:
        """
        加载 GLB 文件
        
        GLB 是 DA3 的主要输出格式，包含:
        - 3D 点云/网格 (重建的场景)
        - 相机棱锥 (每帧的相机位姿可视化，Path3D 类型)
        
        Args:
            glb_path: GLB 文件路径
            
        Returns:
            Scene3D 对象
        """
        if not HAS_TRIMESH:
            raise ImportError("trimesh is required for GLB loading. Install with: pip install trimesh")
        
        logger.info(f"Loading GLB: {glb_path}")
        scene = trimesh.load(glb_path)
        
        # 提取点云和网格
        point_cloud = None
        colors = None
        mesh_vertices = None
        mesh_faces = None
        mesh_vertex_colors = None
        
        if isinstance(scene, trimesh.Scene):
            # DA3 输出的 GLB 包含多种几何体类型:
            # - PointCloud: 场景点云 (通常是 geometry_0)
            # - Path3D: 相机棱锥线框
            # - Trimesh: 可能的网格数据
            
            point_clouds = []
            point_colors = []
            meshes = []
            
            for name, geom in scene.geometry.items():
                geom_type = type(geom).__name__
                
                if geom_type == 'PointCloud':
                    # 提取点云
                    if hasattr(geom, 'vertices'):
                        point_clouds.append(np.array(geom.vertices))
                        # 提取点云颜色
                        if hasattr(geom, 'colors') and geom.colors is not None:
                            c = np.array(geom.colors)
                            if c.max() > 1:
                                c = c / 255.0
                            point_colors.append(c[:, :3] if c.shape[1] >= 3 else c)
                        elif hasattr(geom, 'visual') and hasattr(geom.visual, 'vertex_colors'):
                            c = np.array(geom.visual.vertex_colors)
                            if c.max() > 1:
                                c = c / 255.0
                            point_colors.append(c[:, :3])
                        else:
                            # 无颜色，使用默认灰色
                            point_colors.append(np.ones((len(geom.vertices), 3)) * 0.5)
                    logger.info(f"  Found PointCloud: {name} with {len(geom.vertices)} points")
                
                elif geom_type == 'Trimesh':
                    # 提取网格
                    meshes.append(geom)
                    logger.info(f"  Found Trimesh: {name} with {len(geom.vertices)} vertices")
                
                # Path3D 类型跳过 (相机棱锥线框)
            
            # 合并点云
            if point_clouds:
                point_cloud = np.vstack(point_clouds)
                if point_colors:
                    colors = np.vstack(point_colors)
                logger.info(f"  Total point cloud: {len(point_cloud)} points")
            
            # 如果没有点云但有网格，从网格提取顶点
            if point_cloud is None and meshes:
                mesh = trimesh.util.concatenate(meshes)
                mesh_vertices = np.array(mesh.vertices)
                point_cloud = mesh_vertices.copy()
                if hasattr(mesh, 'faces'):
                    mesh_faces = np.array(mesh.faces)
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                    vertex_colors = np.array(mesh.visual.vertex_colors)
                    if vertex_colors.max() > 1:
                        vertex_colors = vertex_colors / 255.0
                    colors = vertex_colors[:, :3]
                    mesh_vertex_colors = colors.copy()
        else:
            # 直接是网格或点云
            if hasattr(scene, 'vertices'):
                point_cloud = np.array(scene.vertices)
                if hasattr(scene, 'visual') and hasattr(scene.visual, 'vertex_colors'):
                    vertex_colors = np.array(scene.visual.vertex_colors)
                    if vertex_colors.max() > 1:
                        vertex_colors = vertex_colors / 255.0
                    colors = vertex_colors[:, :3]
        
        # 提取相机位姿 (从 Path3D 几何体的变换)
        camera_poses = {}
        if isinstance(scene, trimesh.Scene):
            # 方法 1: 从场景图提取
            if hasattr(scene, 'graph'):
                for node_name in scene.graph.nodes:
                    node_lower = node_name.lower()
                    if 'camera' in node_lower or 'frustum' in node_lower or 'pyramid' in node_lower:
                        try:
                            transform, _ = scene.graph.get(node_name)
                            if transform is not None:
                                camera_poses[node_name] = CameraPose(
                                    name=node_name,
                                    extrinsic=np.array(transform),
                                )
                        except Exception as e:
                            logger.warning(f"Failed to extract camera {node_name}: {e}")
            
            # 方法 2: 从 Path3D 几何体提取 (DA3 的相机棱锥)
            if not camera_poses:
                frame_idx = 0
                for geom_name, geom in scene.geometry.items():
                    geom_type = type(geom).__name__
                    if geom_type == 'Path3D':
                        try:
                            # DA3 的棱锥顶点直接包含相机位置
                            if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
                                # 第一个顶点是相机中心
                                camera_pos = np.array(geom.vertices[0])
                                
                                # 计算相机朝向 (从相机中心指向棱锥底面中心)
                                if len(geom.vertices) >= 5:
                                    # 棱锥底面顶点
                                    bottom_verts = geom.vertices[1:5]
                                    bottom_center = np.mean(bottom_verts, axis=0)
                                    forward = bottom_center - camera_pos
                                    forward = forward / (np.linalg.norm(forward) + 1e-8)
                                else:
                                    forward = np.array([0, 0, 1])
                                
                                # 构建简化的外参矩阵 (用于存储朝向信息)
                                # 注意：这只是近似，真实外参需要完整的位姿估计
                                extrinsic = np.eye(4)
                                extrinsic[2, :3] = forward  # Z 轴为朝向
                                
                                camera_name = f"frame_{frame_idx:05d}"
                                camera_poses[camera_name] = CameraPose(
                                    name=camera_name,
                                    extrinsic=extrinsic,
                                    _position_override=camera_pos,  # 直接存储位置
                                )
                                frame_idx += 1
                        except Exception:
                            pass
            
            logger.info(f"  Extracted {len(camera_poses)} camera poses")
        
        return Scene3D(
            point_cloud=point_cloud,
            colors=colors,
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            mesh_vertex_colors=mesh_vertex_colors,
            camera_poses=camera_poses,
            source_path=glb_path,
            source_format='glb',
            num_frames=len(camera_poses),
            _raw_scene=scene,
        )
    
    @staticmethod
    def load_ply(ply_path: str) -> Scene3D:
        """
        加载 PLY 点云文件
        
        Args:
            ply_path: PLY 文件路径
            
        Returns:
            Scene3D 对象
        """
        if not HAS_OPEN3D:
            raise ImportError("open3d is required for PLY loading. Install with: pip install open3d")
        
        logger.info(f"Loading PLY: {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        
        point_cloud = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        
        return Scene3D(
            point_cloud=point_cloud,
            colors=colors,
            normals=normals,
            source_path=ply_path,
            source_format='ply',
        )
    
    @staticmethod
    def load_npz(npz_path: str) -> Scene3D:
        """
        加载 NPZ 原始数据
        
        NPZ 包含 DA3 的原始输出:
        - points: 点云坐标
        - colors: 点云颜色
        - depths: 深度图序列
        - camera_poses: 相机外参
        - intrinsics: 相机内参
        
        Args:
            npz_path: NPZ 文件路径
            
        Returns:
            Scene3D 对象
        """
        logger.info(f"Loading NPZ: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        
        # 提取点云
        point_cloud = data.get('points')
        colors = data.get('colors')
        
        # 提取深度图
        depth_maps = data.get('depths') or data.get('depth')
        
        # 提取相机位姿
        camera_poses = {}
        raw_poses = data.get('camera_poses') or data.get('extrinsics')
        raw_intrinsics = data.get('intrinsics')
        
        if raw_poses is not None:
            for i, pose in enumerate(raw_poses):
                name = f"frame_{i:05d}"
                intrinsic = raw_intrinsics[i] if raw_intrinsics is not None and i < len(raw_intrinsics) else None
                camera_poses[name] = CameraPose(
                    name=name,
                    extrinsic=np.array(pose),
                    intrinsic=np.array(intrinsic) if intrinsic is not None else None,
                )
        
        return Scene3D(
            point_cloud=point_cloud,
            colors=colors,
            depth_maps=depth_maps,
            camera_poses=camera_poses,
            source_path=npz_path,
            source_format='npz',
            num_frames=len(camera_poses),
        )
    
    @staticmethod
    def load_from_da3_output(output_dir: str) -> Scene3D:
        """
        从 DA3 输出目录加载场景
        
        优先级: GLB > PLY > NPZ
        
        Args:
            output_dir: DA3 输出目录
            
        Returns:
            Scene3D 对象
        """
        output_dir = Path(output_dir)
        
        # 查找可用文件
        glb_file = output_dir / "scene.glb"
        ply_file = output_dir / "scene.ply"
        npz_file = output_dir / "scene.npz"
        
        if glb_file.exists():
            return SceneLoader.load_glb(str(glb_file))
        elif ply_file.exists():
            return SceneLoader.load_ply(str(ply_file))
        elif npz_file.exists():
            return SceneLoader.load_npz(str(npz_file))
        else:
            raise FileNotFoundError(f"No scene file found in {output_dir}")


def merge_point_clouds(
    scenes: List[Scene3D],
    transforms: Optional[List[np.ndarray]] = None,
    voxel_size: float = 0.01,
) -> Scene3D:
    """
    合并多个场景的点云
    
    Args:
        scenes: 场景列表
        transforms: 每个场景的变换矩阵
        voxel_size: 下采样体素大小
        
    Returns:
        合并后的 Scene3D 对象
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for point cloud merging")
    
    merged_pcd = o3d.geometry.PointCloud()
    
    for i, scene in enumerate(scenes):
        if scene.point_cloud is None:
            continue
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene.point_cloud)
        if scene.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(scene.colors[:, :3])
        
        if transforms and i < len(transforms):
            pcd.transform(transforms[i])
        
        merged_pcd += pcd
    
    # 下采样
    if voxel_size > 0:
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    return Scene3D(
        point_cloud=np.asarray(merged_pcd.points),
        colors=np.asarray(merged_pcd.colors) if merged_pcd.has_colors() else None,
        source_format='merged',
    )


# 便捷函数
def load_scene(path: Union[str, Path]) -> Scene3D:
    """加载 3D 场景的便捷函数"""
    return SceneLoader.load(path)
