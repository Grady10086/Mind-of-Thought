"""
深度感知模块 - DA3 完整能力版本

使用 DA3 的完整能力:
- Ray Map 输出 (7通道): 相机射线方向
- 统一 3D 坐标系: 通过 ray 预测相机位姿
- 多帧融合: Cross-view self-attention
- 相机内外参估计: 从 ray 预测 extrinsics/intrinsics

关键改进:
1. 使用 da3nested-giant-large 模型 (多视图融合)
2. 启用 use_ray_pose=True 获取相机位姿
3. 多帧输入获得统一的世界坐标系
4. 使用预测的相机参数进行精确的 3D 投影

参考: https://github.com/ByteDance-Seed/depth-anything-3
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_config import resolve_da3_src_path

DA3_SRC_PATH = resolve_da3_src_path()
if DA3_SRC_PATH is not None and str(DA3_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(DA3_SRC_PATH))


@dataclass
class DA3FullPrediction:
    """DA3 完整预测结果"""
    depth_maps: np.ndarray          # (N, H, W) 深度图
    extrinsics: np.ndarray          # (N, 3, 4) 相机外参 (世界坐标系)
    intrinsics: np.ndarray          # (N, 3, 3) 相机内参
    processed_images: np.ndarray    # (N, H, W, 3) 处理后的图像
    confidence: Optional[np.ndarray] = None  # (N, H, W) 置信度
    
    @property
    def num_frames(self) -> int:
        return self.depth_maps.shape[0]
    
    def get_camera_center(self, frame_idx: int = 0) -> np.ndarray:
        """
        获取相机中心位置 (世界坐标系)
        
        DA3 返回的 extrinsics 是 world-to-camera (w2c) 变换:
        cam_point = R @ world_point + t
        
        相机中心在相机坐标系中是原点 (0,0,0)，所以:
        0 = R @ camera_center + t
        camera_center = -R^T @ t
        """
        R = self.extrinsics[frame_idx, :3, :3]
        t = self.extrinsics[frame_idx, :3, 3]
        return -R.T @ t
    
    def pixel_to_world(self, u: float, v: float, depth: float, frame_idx: int = 0) -> np.ndarray:
        """
        将像素坐标转换为世界坐标
        
        流程:
        1. 像素坐标 (u, v) + 深度 d -> 相机坐标系中的 3D 点
        2. 相机坐标 -> 世界坐标 (使用 w2c 的逆变换)
        
        Args:
            u, v: 像素坐标
            depth: 深度值
            frame_idx: 帧索引
            
        Returns:
            world_point: (3,) 世界坐标
        """
        K = self.intrinsics[frame_idx]  # (3, 3)
        E = self.extrinsics[frame_idx]  # (3, 4) = [R|t], world-to-camera
        
        # 1. 像素坐标 -> 相机坐标
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # 相机坐标系: X 右, Y 下, Z 前 (标准 OpenCV 约定)
        x_cam = (u - cx) / fx * depth
        y_cam = (v - cy) / fy * depth
        z_cam = depth
        
        cam_point = np.array([x_cam, y_cam, z_cam])
        
        # 2. 相机坐标 -> 世界坐标
        # w2c: cam = R @ world + t
        # 逆变换: world = R^T @ (cam - t) = R^T @ cam - R^T @ t
        R = E[:3, :3]
        t = E[:3, 3]
        
        world_point = R.T @ cam_point - R.T @ t
        
        return world_point
    
    def get_c2w(self, frame_idx: int = 0) -> np.ndarray:
        """获取 camera-to-world 变换矩阵 (4x4)"""
        E = self.extrinsics[frame_idx]  # (3, 4) w2c
        R = E[:3, :3]
        t = E[:3, 3]
        
        # c2w = inverse of w2c
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        return c2w


class DA3FullEstimator:
    """
    DA3 完整能力深度估计器
    
    使用 DA3 的全部能力:
    - da3nested-giant-large 模型 (多视图融合)
    - Ray Map 预测相机位姿
    - 统一的世界坐标系
    
    Example:
        >>> estimator = DA3FullEstimator(device='cuda')
        >>> prediction = estimator.estimate_multiview(frames)
        >>> world_point = prediction.pixel_to_world(320, 240, 2.5, frame_idx=0)
    """
    
    # 可用模型
    AVAILABLE_MODELS = [
        "da3-small",
        "da3-base",
        "da3-large",
        "da3-giant",
        "da3nested-giant-large",  # 推荐: 多视图融合
    ]
    
    def __init__(
        self,
        model_name: str = "da3nested-giant-large",
        device: Union[str, torch.device] = 'cuda',
        process_res: int = 504,
        use_ray_pose: bool = True,  # 关键: 启用 ray-based pose estimation
    ):
        """
        Args:
            model_name: 模型名称，推荐 "da3nested-giant-large"
            device: 计算设备
            process_res: 处理分辨率
            use_ray_pose: 是否使用 ray-based 位姿估计
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.process_res = process_res
        self.use_ray_pose = use_ray_pose
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载 DA3 模型"""
        try:
            from depth_anything_3.api import DepthAnything3
            
            logger.info(f"加载 DA3 完整模型: {self.model_name}")
            
            # 检查是否是本地路径
            if os.path.isdir(self.model_name):
                # 从本地路径加载 (如 DA3 1.1)
                logger.info(f"从本地路径加载模型: {self.model_name}")
                self.model = DepthAnything3.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                )
            else:
                # 使用预设模型名称
                self.model = DepthAnything3(model_name=self.model_name)
            
            self.model = self.model.to(self.device)
            self.model.device = self.device
            self.model.eval()
            
            logger.info(f"DA3 完整模型加载完成，设备: {self.device}")
            logger.info(f"  - 模型: {self.model_name}")
            logger.info(f"  - use_ray_pose: {self.use_ray_pose}")
            
        except Exception as e:
            logger.error(f"DA3 模型加载失败: {e}")
            raise RuntimeError(
                "Failed to load Depth-Anything-3. Set MOT_DA3_ROOT or MOT_DA3_SRC in config/local.env "
                "or your shell environment, and ensure depth_anything_3 dependencies are installed."
            ) from e
    
    @torch.no_grad()
    def estimate_multiview(
        self,
        images: List[np.ndarray],
        ref_view_strategy: str = "saddle_balanced",
    ) -> DA3FullPrediction:
        """
        多视图深度估计 - 使用 DA3 完整 API
        
        Args:
            images: RGB 图像列表，每个 (H, W, 3) uint8
            ref_view_strategy: 参考视图选择策略
                - "first": 第一帧
                - "middle": 中间帧
                - "saddle_balanced": 鞍点平衡 (推荐)
                
        Returns:
            DA3FullPrediction 包含:
                - depth_maps: 深度图
                - extrinsics: 相机外参 (统一世界坐标系)
                - intrinsics: 相机内参
        """
        # 转换图像格式
        pil_images = []
        for img in images:
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        
        logger.info(f"DA3 多视图推理: {len(pil_images)} 帧, use_ray_pose={self.use_ray_pose}")
        
        # 调用 DA3 完整 API
        prediction = self.model.inference(
            image=pil_images,
            extrinsics=None,  # 让 DA3 自己估计
            intrinsics=None,  # 让 DA3 自己估计
            use_ray_pose=self.use_ray_pose,  # 关键: 使用 ray-based 位姿估计
            ref_view_strategy=ref_view_strategy,
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
        )
        
        # 提取结果
        depth_maps = prediction.depth  # (N, H, W)
        extrinsics = prediction.extrinsics  # (N, 3, 4) 或 (N, 4, 4)
        intrinsics = prediction.intrinsics  # (N, 3, 3)
        processed_images = prediction.processed_images  # (N, H, W, 3)
        
        # 确保外参是 (N, 3, 4) 格式
        if extrinsics is not None and extrinsics.shape[-2] == 4:
            extrinsics = extrinsics[:, :3, :]
        
        # 获取置信度
        confidence = getattr(prediction, 'conf', None)
        
        logger.info(f"DA3 推理完成:")
        logger.info(f"  - depth_maps: {depth_maps.shape}")
        logger.info(f"  - extrinsics: {extrinsics.shape if extrinsics is not None else 'None'}")
        logger.info(f"  - intrinsics: {intrinsics.shape if intrinsics is not None else 'None'}")
        
        return DA3FullPrediction(
            depth_maps=depth_maps,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            processed_images=processed_images,
            confidence=confidence,
        )
    
    @torch.no_grad()
    def estimate_single(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        单帧深度估计 (兼容旧接口)
        
        Args:
            image: RGB 图像 (H, W, 3) uint8
            
        Returns:
            (depth_map, intrinsics, None): depth_map (H, W), intrinsics (3, 3)
        """
        prediction = self.estimate_multiview([image])
        
        depth = prediction.depth_maps[0]
        intrinsics = prediction.intrinsics[0] if prediction.intrinsics is not None else None
        
        return depth, intrinsics, None
    
    def compute_3d_point(
        self,
        prediction: DA3FullPrediction,
        frame_idx: int,
        u: float,
        v: float,
    ) -> np.ndarray:
        """
        计算像素的 3D 世界坐标
        
        Args:
            prediction: DA3 预测结果
            frame_idx: 帧索引
            u, v: 像素坐标
            
        Returns:
            world_point: (3,) 世界坐标
        """
        # 获取深度
        H, W = prediction.depth_maps.shape[1:3]
        u_int = int(np.clip(u, 0, W - 1))
        v_int = int(np.clip(v, 0, H - 1))
        depth = prediction.depth_maps[frame_idx, v_int, u_int]
        
        # 计算世界坐标
        return prediction.pixel_to_world(u, v, depth, frame_idx)
    
    def compute_object_center_3d(
        self,
        prediction: DA3FullPrediction,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],
        use_camera_coords: bool = False,  # 新增参数
    ) -> np.ndarray:
        """
        计算物体边界框的 3D 中心点
        
        Args:
            prediction: DA3 预测结果
            frame_idx: 帧索引
            bbox: (x1, y1, x2, y2) 边界框
            use_camera_coords: 如果 True，返回相机坐标系; 否则返回世界坐标系
            
        Returns:
            center_3d: (3,) 坐标
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # 使用中心点深度
        H, W = prediction.depth_maps.shape[1:3]
        cx_int = int(np.clip(cx, 0, W - 1))
        cy_int = int(np.clip(cy, 0, H - 1))
        
        # 使用边界框区域的中值深度 (更鲁棒)
        x1_int = int(np.clip(x1, 0, W - 1))
        x2_int = int(np.clip(x2, 0, W - 1))
        y1_int = int(np.clip(y1, 0, H - 1))
        y2_int = int(np.clip(y2, 0, H - 1))
        
        depth_region = prediction.depth_maps[frame_idx, y1_int:y2_int+1, x1_int:x2_int+1]
        depth = np.median(depth_region) if depth_region.size > 0 else prediction.depth_maps[frame_idx, cy_int, cx_int]
        
        if use_camera_coords:
            # 直接返回相机坐标系 (像 V7 原始版本那样)
            K = prediction.intrinsics[frame_idx]
            fx, fy = K[0, 0], K[1, 1]
            cx_cam, cy_cam = K[0, 2], K[1, 2]
            
            x_cam = (cx - cx_cam) / fx * depth
            y_cam = (cy - cy_cam) / fy * depth
            z_cam = depth
            
            return np.array([x_cam, y_cam, z_cam])
        else:
            return prediction.pixel_to_world(cx, cy, depth, frame_idx)
    
    def compute_object_size_3d(
        self,
        prediction: DA3FullPrediction,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, float]:
        """
        计算物体的 3D 尺寸 (米)
        
        Args:
            prediction: DA3 预测结果
            frame_idx: 帧索引
            bbox: (x1, y1, x2, y2) 边界框
            
        Returns:
            (width_3d, height_3d): 物体的 3D 宽度和高度 (米)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # 获取深度
        H, W = prediction.depth_maps.shape[1:3]
        cx_int = int(np.clip(cx, 0, W - 1))
        cy_int = int(np.clip(cy, 0, H - 1))
        depth = prediction.depth_maps[frame_idx, cy_int, cx_int]
        
        # 获取相机内参
        K = prediction.intrinsics[frame_idx]
        fx, fy = K[0, 0], K[1, 1]
        
        # 计算 3D 尺寸
        width_px = x2 - x1
        height_px = y2 - y1
        
        width_3d = width_px * depth / fx
        height_3d = height_px * depth / fy
        
        return width_3d, height_3d
    
    def to(self, device: Union[str, torch.device]) -> 'DA3FullEstimator':
        """移动模型到指定设备"""
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.device = self.device
        return self


class MindMapBuilder3D:
    """
    基于 DA3 完整能力的心智地图构建器
    
    核心特性:
    - 使用 DA3 的 Ray Map 获取统一世界坐标系
    - 多帧融合获得一致的 3D 坐标
    - 物体位置在世界坐标系中表示
    - 可选使用相机坐标系 (用于方向判断)
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        num_frames: int = 16,  # 多视图需要的帧数
        box_threshold: float = 0.25,
        model_name: str = "da3nested-giant-large",
        use_ray_pose: bool = True,
        use_camera_coords: bool = False,  # 是否使用相机坐标系
    ):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        self.model_name = model_name
        self.use_ray_pose = use_ray_pose
        self.use_camera_coords = use_camera_coords
        
        self._labeler = None
        self._da3_estimator = None
    
    def load_models(self):
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
        
        if self._da3_estimator is None:
            self._da3_estimator = DA3FullEstimator(
                model_name=self.model_name,
                device=self.device,
                use_ray_pose=self.use_ray_pose,
            )
            logger.info(f"DA3 完整估计器加载完成: {self.model_name}")
    
    def unload(self):
        """卸载模型"""
        if self._labeler is not None:
            try:
                del self._labeler.model
                del self._labeler.processor
            except:
                pass
            self._labeler = None
        
        if self._da3_estimator is not None:
            del self._da3_estimator.model
            self._da3_estimator = None
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    def build_from_video(
        self,
        video_path: str,
        target_objects: List[str] = None,
        extended_vocabulary: List[str] = None,
    ) -> Tuple[Dict[str, Any], DA3FullPrediction, List[np.ndarray]]:
        """
        从视频构建 3D 心智地图
        
        Args:
            video_path: 视频路径
            target_objects: 目标物体列表
            extended_vocabulary: 扩展词汇表
            
        Returns:
            (mind_map_3d, da3_prediction, sampled_frames)
            - mind_map_3d: 3D 心智地图 {label: entity}
            - da3_prediction: DA3 完整预测结果
            - sampled_frames: 采样的帧
        """
        self.load_models()
        
        # 1. 采样视频帧
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {}, None, []
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        sampled_frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(frame_rgb)
        
        cap.release()
        
        if len(sampled_frames) < 2:
            logger.warning(f"采样帧数不足: {len(sampled_frames)}")
            return {}, None, sampled_frames
        
        logger.info(f"采样 {len(sampled_frames)} 帧用于 DA3 多视图估计")
        
        # 2. DA3 多视图深度估计 (获取统一世界坐标系)
        da3_prediction = self._da3_estimator.estimate_multiview(
            sampled_frames,
            ref_view_strategy="saddle_balanced",
        )
        
        # 3. 在每帧上检测物体
        vocab = list(set((target_objects or []) + (extended_vocabulary or [])))
        if not vocab:
            vocab = ["chair", "table", "sofa", "bed", "tv", "door", "window"]
        prompt = " . ".join(vocab) + " ."
        
        # 收集所有检测
        from collections import defaultdict
        all_detections = defaultdict(list)
        
        # 处理后的图像尺寸可能与原始不同
        proc_H, proc_W = da3_prediction.depth_maps.shape[1:3]
        
        for i, (frame_rgb, frame_idx) in enumerate(zip(sampled_frames, frame_indices)):
            orig_H, orig_W = frame_rgb.shape[:2]
            
            # 物体检测
            detections = self._labeler.detect(frame_rgb, prompt)
            
            for det in detections:
                raw_label = det.label.strip().lower()
                if raw_label.startswith('##'):
                    continue
                
                bbox = det.bbox_pixels
                conf = det.confidence
                
                # 将原始坐标映射到处理后的尺寸
                scale_x = proc_W / orig_W
                scale_y = proc_H / orig_H
                
                bbox_scaled = (
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y,
                )
                
                # 计算 3D 坐标 (可选使用相机坐标系)
                center_3d = self._da3_estimator.compute_object_center_3d(
                    da3_prediction, i, bbox_scaled,
                    use_camera_coords=self.use_camera_coords,
                )
                
                # 计算 3D 尺寸
                width_3d, height_3d = self._da3_estimator.compute_object_size_3d(
                    da3_prediction, i, bbox_scaled
                )
                
                all_detections[raw_label].append({
                    'frame_idx': int(frame_idx),
                    'frame_order': i,
                    'bbox': bbox,
                    'bbox_scaled': bbox_scaled,
                    'confidence': float(conf),
                    'position_3d': center_3d,
                    'width_3d': width_3d,
                    'height_3d': height_3d,
                })
        
        # 4. 聚合成 3D 心智地图
        mind_map_3d = {}
        for label, dets in all_detections.items():
            entity = self._aggregate_detections_3d(label, dets)
            mind_map_3d[label] = entity
        
        return mind_map_3d, da3_prediction, sampled_frames
    
    def _aggregate_detections_3d(self, label: str, detections: List[Dict]) -> Dict:
        """聚合检测结果为 3D 实体"""
        if not detections:
            return {
                'label': label,
                'count': 0,
                'position_3d': None,
                'size_3d': None,
                'avg_confidence': 0,
                'first_seen_frame': -1,
                'detections': [],
            }
        
        # 按帧统计数量
        from collections import defaultdict
        frame_dets = defaultdict(list)
        for det in detections:
            frame_dets[det['frame_idx']].append(det)
        
        max_count = max(len(fd) for fd in frame_dets.values())
        avg_conf = np.mean([d['confidence'] for d in detections])
        
        # 聚合 3D 位置 (使用中位数)
        positions = np.array([d['position_3d'] for d in detections])
        avg_pos = np.median(positions, axis=0)
        
        # 聚合 3D 尺寸
        best_det = max(detections, key=lambda x: x['confidence'])
        size_3d = np.array([best_det['width_3d'], best_det['height_3d'], 0.3])
        
        first_frame = min(d['frame_idx'] for d in detections)
        
        return {
            'label': label,
            'count': max_count,
            'position_3d': avg_pos,
            'size_3d': size_3d,
            'avg_confidence': float(avg_conf),
            'first_seen_frame': first_frame,
            'detections': detections,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("测试 DA3 完整能力估计器...")
    
    try:
        # 创建估计器
        estimator = DA3FullEstimator(
            model_name="da3nested-giant-large",
            device='cuda',
            use_ray_pose=True,
        )
        
        # 创建测试数据
        dummy_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        
        # 多视图估计
        prediction = estimator.estimate_multiview(dummy_frames)
        
        print(f"深度图形状: {prediction.depth_maps.shape}")
        print(f"外参形状: {prediction.extrinsics.shape if prediction.extrinsics is not None else 'None'}")
        print(f"内参形状: {prediction.intrinsics.shape if prediction.intrinsics is not None else 'None'}")
        
        # 测试坐标转换
        if prediction.extrinsics is not None:
            world_point = prediction.pixel_to_world(320, 240, 2.5, frame_idx=0)
            print(f"像素 (320, 240) 深度 2.5m 的世界坐标: {world_point}")
            
            camera_center = prediction.get_camera_center(0)
            print(f"相机中心 (帧0): {camera_center}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
