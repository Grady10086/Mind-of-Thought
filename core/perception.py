"""
深度感知模块 - 集成 Depth Anything V3 (DA3)

特性:
- 基于 AMD ROCm 优化
- 支持 FP16 半精度推理
- 时序一致性平滑滤波
- 内存高效的 Buffer 机制
- 相对深度归一化处理

参考: https://github.com/ByteDance-Seed/depth-anything-3
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Generator, Dict, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 0.001) -> np.ndarray:
    """
    引导滤波 - 使用 RGB 图像引导深度图平滑，有效减少 ViT 的网格纹理
    
    Args:
        guide: 引导图像 (H, W) float32 [0, 1]
        src: 待滤波图像 (H, W) float32
        radius: 滤波窗口半径
        eps: 正则化参数，越大平滑效果越强
        
    Returns:
        滤波后的图像 (H, W) float32
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    
    box_size = (radius, radius)
    
    mean_guide = cv2.boxFilter(guide, -1, box_size)
    mean_src = cv2.boxFilter(src, -1, box_size)
    corr_guide = cv2.boxFilter(guide * guide, -1, box_size)
    corr_gs = cv2.boxFilter(guide * src, -1, box_size)
    
    var_guide = corr_guide - mean_guide * mean_guide
    cov_gs = corr_gs - mean_guide * mean_src
    
    a = cov_gs / (var_guide + eps)
    b = mean_src - a * mean_guide
    
    mean_a = cv2.boxFilter(a, -1, box_size)
    mean_b = cv2.boxFilter(b, -1, box_size)
    
    return mean_a * guide + mean_b


class TemporalSmoothingMethod(Enum):
    """时序平滑方法"""
    NONE = "none"
    EMA = "exponential_moving_average"  # 指数移动平均
    GAUSSIAN = "gaussian"  # 高斯平滑
    BILATERAL = "bilateral"  # 双边滤波（保边平滑）


@dataclass
class DepthPrediction:
    """深度预测结果"""
    depth_maps: torch.Tensor  # (T, 1, H, W) 归一化深度图 [0, 1]
    depth_raw: torch.Tensor  # (T, H, W) 原始深度值
    confidence: Optional[torch.Tensor] = None  # (T, H, W) 置信度
    extrinsics: Optional[torch.Tensor] = None  # (T, 3, 4) 外参
    intrinsics: Optional[torch.Tensor] = None  # (T, 3, 3) 内参
    timestamps: Optional[List[float]] = None


class DepthEstimator:
    """
    深度估计器 - 基于 Depth Anything V3
    
    专为 AMD ROCm (308X) 优化:
    - 使用 aiter flash attention
    - 支持 FP16 半精度推理
    - 时序一致性平滑
    - 内存高效的分批处理
    
    Example:
        >>> estimator = DepthEstimator(device='cuda', half_precision=True)
        >>> depth_pred = estimator.infer_video(video_frames)
        >>> depth_maps = depth_pred.depth_maps  # (T, 1, H, W)
    """
    
    # 预训练模型列表
    AVAILABLE_MODELS = [
        "depth-anything/DA3-Small",
        "depth-anything/DA3-Base",
        "depth-anything/DA3-Large",
    ]
    
    def __init__(
        self,
        model_name: str = "depth-anything/DA3-Large",
        device: Union[str, torch.device] = 'cuda',
        half_precision: bool = True,
        process_res: int = 504,
        temporal_smoothing: TemporalSmoothingMethod = TemporalSmoothingMethod.EMA,
        ema_alpha: float = 0.3,
        batch_size: int = 4,  # 内存管理：每批处理帧数
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: 预训练模型名称
            device: 计算设备
            half_precision: 是否使用 FP16
            process_res: 处理分辨率
            temporal_smoothing: 时序平滑方法
            ema_alpha: EMA 平滑系数 (0-1)，越小越平滑
            batch_size: 分批处理大小（内存管理）
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.half_precision = half_precision
        self.process_res = process_res
        self.temporal_smoothing = temporal_smoothing
        self.ema_alpha = ema_alpha
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        self.model = None
        self.transform = None  # MiDaS transform
        self.dtype = torch.float16 if half_precision else torch.float32
        self._use_fallback = False
        
        self._load_model()
    
    def _load_model(self):
        """加载 DA3 模型"""
        try:
            from depth_anything_3.api import DepthAnything3
            
            # 模型名称映射
            model_name_map = {
                "depth-anything/DA3-Small": "da3-small",
                "depth-anything/DA3-Base": "da3-base", 
                "depth-anything/DA3-Large": "da3-large",
            }
            
            da3_model_name = model_name_map.get(self.model_name, "da3-large")
            logger.info(f"加载 DA3 模型: {da3_model_name}")
            
            # 使用本地配置创建模型
            self.model = DepthAnything3(model_name=da3_model_name)
            
            # 移动到设备
            self.model = self.model.to(self.device)
            self.model.device = self.device
            
            # 设置为评估模式
            self.model.eval()
            
            self._use_fallback = False
            logger.info(f"DA3 模型加载完成，设备: {self.device}")
            
        except ImportError as e:
            logger.error(f"无法导入 DA3: {e}")
            logger.info("尝试使用备用深度估计...")
            self._load_fallback_model()
        except Exception as e:
            logger.error(f"DA3 模型加载失败: {e}")
            logger.info("尝试使用备用深度估计...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """
        备用深度估计模型 (MiDaS)
        当 DA3 不可用时使用
        
        注意: 在 spawn 子进程中 torch.hub.load 可能失败，
        所以优先使用缓存路径直接加载
        """
        try:
            # 方法1: 尝试从缓存直接加载（适用于 spawn 子进程）
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "intel-isl_MiDaS_master")
            if os.path.exists(cache_dir):
                sys.path.insert(0, cache_dir)
                try:
                    from midas.midas_net_custom import MidasNet_small
                    from midas.transforms import Resize, NormalizeImage, PrepareForNet
                    
                    # 查找权重文件
                    weights_path = None
                    for p in [
                        os.path.join(cache_dir, "weights", "midas_v21_small_256.pt"),
                        os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "midas_v21_small_256.pt"),
                    ]:
                        if os.path.exists(p):
                            weights_path = p
                            break
                    
                    self.model = MidasNet_small(
                        weights_path,
                        features=64,
                        backbone="efficientnet_lite3",
                        exportable=True,
                        non_negative=True,
                        blocks={'expand': True}
                    )
                    self.model = self.model.to(self.device)
                    if self.half_precision:
                        self.model = self.model.half()
                    self.model.eval()
                    
                    # 手动构建 small_transform
                    from torchvision.transforms import Compose
                    self.transform = Compose([
                        lambda img: {"image": img / 255.0},
                        Resize(
                            256, 256,
                            resize_target=None,
                            keep_aspect_ratio=True,
                            ensure_multiple_of=32,
                            resize_method="upper_bound",
                            image_interpolation_method=cv2.INTER_CUBIC,
                        ),
                        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        PrepareForNet(),
                        lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
                    ])
                    
                    self._use_fallback = True
                    logger.info("使用备用 MiDaS 模型 (从缓存加载)")
                    return
                except Exception as e_cache:
                    logger.warning(f"从缓存加载 MiDaS 失败: {e_cache}")
                finally:
                    if cache_dir in sys.path:
                        sys.path.remove(cache_dir)
            
            # 方法2: 标准 torch.hub.load
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small",
                trust_repo=True,
            )
            self.model = self.model.to(self.device)
            if self.half_precision:
                self.model = self.model.half()
            self.model.eval()
            
            # 加载 transforms
            try:
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self.transform = midas_transforms.small_transform
            except:
                self.transform = None
            
            self._use_fallback = True
            logger.info("使用备用 MiDaS 模型 (torch.hub)")
        except Exception as e:
            logger.error(f"备用模型也加载失败: {e}")
            self._use_fallback = None
            self.transform = None
    
    @torch.no_grad()
    def infer_single(
        self,
        image: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        单帧深度推理
        
        Args:
            image: RGB 图像 (H, W, 3) np.ndarray 或 (3, H, W) Tensor
            normalize: 是否归一化到 [0, 1]
            
        Returns:
            (depth_map, confidence): depth_map (1, H, W), confidence (H, W) 或 None
        """
        # 预处理 - 转换为 PIL Image 或 numpy array for DA3
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                # (3, H, W) -> (H, W, 3)
                image = image.permute(1, 2, 0).cpu().numpy()
            elif image.dim() == 3 and image.shape[-1] == 3:
                image = image.cpu().numpy()
            else:
                image = image.cpu().numpy()
        
        # 确保是 uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        H_orig, W_orig = image.shape[:2]
        
        # 推理
        if hasattr(self, '_use_fallback') and self._use_fallback is None:
            # 所有模型都加载失败，返回默认深度
            logger.warning("无可用深度模型，返回默认深度值")
            default_depth = torch.ones(1, H_orig, W_orig, device=self.device) * 5.0
            return default_depth, None
        elif hasattr(self, '_use_fallback') and self._use_fallback:
            # MiDaS 备用路径
            return self._infer_midas(image, normalize)
        else:
            # DA3 路径 - 使用 inference API
            try:
                # DA3 接受 list of images
                prediction = self.model.inference(
                    image=[image],
                    process_res=self.process_res,
                    process_res_method="upper_bound_resize",
                )
                
                # 获取深度图 (N, H, W)
                depth_raw = prediction.depth[0]  # (H, W)
                
                # 应用引导滤波减少 ViT 网格纹理
                # 先上采样到原始分辨率
                depth_up = cv2.resize(depth_raw, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
                
                # 使用 RGB 图像的灰度作为引导
                guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                
                # 应用引导滤波
                depth_filtered = guided_filter(guide, depth_up, radius=16, eps=0.001)
                
                depth = torch.from_numpy(depth_filtered).to(self.device)
                depth = depth.unsqueeze(0)  # (1, H, W)
                
                # 获取置信度
                confidence = None
                if prediction.conf is not None:
                    conf = prediction.conf[0]
                    confidence = torch.from_numpy(conf).to(self.device)
                    confidence = F.interpolate(
                        confidence.unsqueeze(0).unsqueeze(0),
                        size=(H_orig, W_orig),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
            except Exception as e:
                logger.warning(f"DA3 推理失败: {e}，回退到 MiDaS")
                return self._infer_midas(image, normalize)
        
        # 归一化深度图到 [0, 1]
        if normalize:
            depth = self._normalize_depth(depth)
        
        return depth, confidence
    
    def _infer_midas(self, image: np.ndarray, normalize: bool = True) -> Tuple[torch.Tensor, None]:
        """MiDaS 备用推理"""
        H_orig, W_orig = image.shape[:2]
        
        if self.transform is not None:
            input_tensor = self.transform(image).to(self.device)
        else:
            # 基本预处理
            img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            img = img / 255.0
            img = F.interpolate(img, size=(384, 384), mode='bilinear', align_corners=False)
            input_tensor = img.to(self.device)
        
        # 确保输入与模型权重的dtype一致
        if hasattr(self.model, 'parameters'):
            model_dtype = next(self.model.parameters()).dtype
            input_tensor = input_tensor.to(model_dtype)
        
        depth = self.model(input_tensor)
        
        # 调整回原始尺寸
        depth = F.interpolate(
            depth.unsqueeze(1) if depth.dim() == 3 else depth,
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        )
        
        # 确保形状 (1, H, W)
        if depth.dim() == 4:
            depth = depth.squeeze(0)
        
        if normalize:
            depth = self._normalize_depth(depth)
        
        return depth, None
    
    @torch.no_grad()
    def infer_video(
        self,
        video_frames: Union[torch.Tensor, List[np.ndarray]],
        normalize: bool = True,
        apply_temporal_smoothing: bool = True,
    ) -> DepthPrediction:
        """
        视频深度推理 - 内存高效的分批处理
        
        Args:
            video_frames: 视频帧
                - Tensor: (T, C, H, W) 或 (T, H, W, C)
                - List[np.ndarray]: 每个 (H, W, 3)
            normalize: 是否归一化到 [0, 1]
            apply_temporal_smoothing: 是否应用时序平滑
            
        Returns:
            DepthPrediction 包含深度图和相关信息
        """
        # 转换输入格式
        if isinstance(video_frames, list):
            # List[np.ndarray] -> (T, C, H, W)
            frames = np.stack(video_frames, axis=0)  # (T, H, W, 3)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
        else:
            frames = video_frames
            if frames.shape[-1] == 3:  # (T, H, W, C) -> (T, C, H, W)
                frames = frames.permute(0, 3, 1, 2)
        
        frames = frames.to(self.device, dtype=self.dtype)
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        T, C, H, W = frames.shape
        
        # 分批处理以管理内存
        depth_maps = []
        confidences = []
        
        for batch_start in range(0, T, self.batch_size):
            batch_end = min(batch_start + self.batch_size, T)
            batch_frames = frames[batch_start:batch_end]
            
            batch_depths = []
            batch_confs = []
            
            for i in range(batch_frames.shape[0]):
                depth, conf = self.infer_single(batch_frames[i], normalize=False)
                batch_depths.append(depth)
                if conf is not None:
                    batch_confs.append(conf)
            
            depth_maps.extend(batch_depths)
            confidences.extend(batch_confs)
            
            # 清理 GPU 内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 堆叠结果
        depth_raw = torch.stack(depth_maps, dim=0).squeeze(1)  # (T, H, W)
        
        # 归一化
        if normalize:
            depth_normalized = self._normalize_depth(depth_raw.unsqueeze(1))  # (T, 1, H, W)
        else:
            depth_normalized = depth_raw.unsqueeze(1)
        
        # 时序平滑
        if apply_temporal_smoothing and self.temporal_smoothing != TemporalSmoothingMethod.NONE:
            depth_normalized = self._apply_temporal_smoothing(depth_normalized)
        
        # 构建结果
        confidence = torch.stack(confidences, dim=0) if confidences else None
        
        return DepthPrediction(
            depth_maps=depth_normalized,
            depth_raw=depth_raw,
            confidence=confidence,
        )
    
    def infer_video_generator(
        self,
        video_frames: Union[torch.Tensor, List[np.ndarray]],
        buffer_size: int = 8,
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        """
        内存高效的生成器模式推理 - 适用于超长视频
        
        Args:
            video_frames: 视频帧
            buffer_size: 缓冲区大小（用于时序平滑）
            
        Yields:
            (frame_idx, depth_map): 帧索引和对应的深度图
        """
        # 转换格式
        if isinstance(video_frames, list):
            T = len(video_frames)
        else:
            T = video_frames.shape[0]
        
        # 深度缓冲区（用于时序平滑）
        depth_buffer = []
        
        for i in range(T):
            # 获取当前帧
            if isinstance(video_frames, list):
                frame = video_frames[i]
            else:
                frame = video_frames[i]
            
            # 推理
            depth, _ = self.infer_single(frame, normalize=True)
            
            # 添加到缓冲区
            depth_buffer.append(depth)
            if len(depth_buffer) > buffer_size:
                depth_buffer.pop(0)
            
            # 应用时序平滑（使用缓冲区）
            if self.temporal_smoothing == TemporalSmoothingMethod.EMA and len(depth_buffer) > 1:
                smoothed = self._ema_smooth_buffer(depth_buffer)
            else:
                smoothed = depth
            
            yield i, smoothed
            
            # 定期清理内存
            if i % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        归一化深度图到 [0, 1]
        
        注意: DA3 输出的是相对深度，我们保持相对关系而非绝对数值
        """
        # 按帧归一化（保持每帧的相对深度关系）
        if depth.dim() == 4:  # (T, 1, H, W)
            depth_min = depth.view(depth.shape[0], -1).min(dim=1, keepdim=True)[0]
            depth_max = depth.view(depth.shape[0], -1).max(dim=1, keepdim=True)[0]
            depth_min = depth_min.view(-1, 1, 1, 1)
            depth_max = depth_max.view(-1, 1, 1, 1)
        else:  # (1, H, W) 或 (H, W)
            depth_min = depth.min()
            depth_max = depth.max()
        
        # 避免除零
        depth_range = depth_max - depth_min
        depth_range = torch.clamp(depth_range, min=1e-6)
        
        normalized = (depth - depth_min) / depth_range
        return normalized
    
    def _apply_temporal_smoothing(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """
        应用时序平滑 - 减少帧间闪烁
        
        Args:
            depth_maps: (T, 1, H, W)
            
        Returns:
            平滑后的深度图 (T, 1, H, W)
        """
        if self.temporal_smoothing == TemporalSmoothingMethod.NONE:
            return depth_maps
        
        T = depth_maps.shape[0]
        if T <= 1:
            return depth_maps
        
        if self.temporal_smoothing == TemporalSmoothingMethod.EMA:
            return self._ema_smooth(depth_maps)
        elif self.temporal_smoothing == TemporalSmoothingMethod.GAUSSIAN:
            return self._gaussian_smooth(depth_maps)
        elif self.temporal_smoothing == TemporalSmoothingMethod.BILATERAL:
            return self._bilateral_smooth(depth_maps)
        
        return depth_maps
    
    def _ema_smooth(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """指数移动平均平滑"""
        T = depth_maps.shape[0]
        alpha = self.ema_alpha
        
        smoothed = torch.zeros_like(depth_maps)
        smoothed[0] = depth_maps[0]
        
        for t in range(1, T):
            smoothed[t] = alpha * depth_maps[t] + (1 - alpha) * smoothed[t - 1]
        
        return smoothed
    
    def _ema_smooth_buffer(self, buffer: List[torch.Tensor]) -> torch.Tensor:
        """基于缓冲区的 EMA 平滑"""
        alpha = self.ema_alpha
        result = buffer[0]
        for i in range(1, len(buffer)):
            result = alpha * buffer[i] + (1 - alpha) * result
        return result
    
    def _gaussian_smooth(self, depth_maps: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """高斯时序平滑"""
        T, C, H, W = depth_maps.shape
        
        # 创建1D高斯核
        sigma = kernel_size / 6.0
        x = torch.arange(kernel_size, device=depth_maps.device, dtype=depth_maps.dtype) - kernel_size // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        # 重塑用于卷积
        depth_flat = depth_maps.permute(2, 3, 1, 0).reshape(-1, 1, T)  # (H*W, 1, T)
        kernel = kernel.view(1, 1, -1)
        
        # 时序卷积
        padding = kernel_size // 2
        smoothed = F.conv1d(depth_flat, kernel, padding=padding)
        
        # 恢复形状
        smoothed = smoothed.reshape(H, W, C, T).permute(3, 2, 0, 1)
        
        return smoothed
    
    def _bilateral_smooth(self, depth_maps: torch.Tensor, sigma_space: float = 2.0, sigma_depth: float = 0.1) -> torch.Tensor:
        """双边时序平滑 - 保边平滑"""
        T = depth_maps.shape[0]
        window_size = 5
        half_window = window_size // 2
        
        smoothed = torch.zeros_like(depth_maps)
        
        for t in range(T):
            weights_sum = torch.zeros_like(depth_maps[t])
            result = torch.zeros_like(depth_maps[t])
            
            for dt in range(-half_window, half_window + 1):
                t_neighbor = max(0, min(T - 1, t + dt))
                
                # 空间权重
                space_weight = np.exp(-(dt ** 2) / (2 * sigma_space ** 2))
                
                # 深度差异权重
                depth_diff = torch.abs(depth_maps[t] - depth_maps[t_neighbor])
                depth_weight = torch.exp(-(depth_diff ** 2) / (2 * sigma_depth ** 2))
                
                weight = space_weight * depth_weight
                weights_sum += weight
                result += weight * depth_maps[t_neighbor]
            
            smoothed[t] = result / (weights_sum + 1e-6)
        
        return smoothed
    
    def get_depth_gradient(self, depth_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算深度梯度 - 用于理解相对深度关系
        
        Scale Invariance 提示：VLM 应该关注梯度和相对关系，而非绝对数值
        
        Args:
            depth_map: (1, H, W) 或 (H, W)
            
        Returns:
            (grad_x, grad_y): 水平和垂直方向的深度梯度
        """
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        elif depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(0)
        
        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(depth_map, sobel_x, padding=1)
        grad_y = F.conv2d(depth_map, sobel_y, padding=1)
        
        return grad_x.squeeze(), grad_y.squeeze()
    
    def to(self, device: Union[str, torch.device]) -> 'DepthEstimator':
        """移动模型到指定设备"""
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self


class SimpleDepthEstimator:
    """
    简化版深度估计器 - 当 DA3 不可用时的备选方案
    使用 MiDaS 或其他轻量级模型
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """加载 MiDaS 模型"""
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.model = self.model.to(self.device).eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
            
            logger.info("SimpleDepthEstimator: MiDaS 加载成功")
        except Exception as e:
            logger.error(f"MiDaS 加载失败: {e}")
            self.model = None
    
    @torch.no_grad()
    def infer_video(self, video_frames: List[np.ndarray]) -> torch.Tensor:
        """简化的视频深度推理"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        depth_maps = []
        for frame in video_frames:
            input_tensor = self.transform(frame).to(self.device)
            depth = self.model(input_tensor)
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            # 归一化
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            depth_maps.append(depth.squeeze())
        
        return torch.stack(depth_maps, dim=0).unsqueeze(1)  # (T, 1, H, W)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试
    print("测试 DepthEstimator...")
    
    try:
        estimator = DepthEstimator(
            device='cuda',
            half_precision=True,
            temporal_smoothing=TemporalSmoothingMethod.EMA,
        )
        
        # 创建测试数据
        dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
        
        # 推理
        result = estimator.infer_video(dummy_frames)
        
        print(f"深度图形状: {result.depth_maps.shape}")
        print(f"深度范围: [{result.depth_maps.min():.3f}, {result.depth_maps.max():.3f}]")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("尝试 SimpleDepthEstimator...")
        
        try:
            simple_estimator = SimpleDepthEstimator(device='cuda')
            dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
            depth = simple_estimator.infer_video(dummy_frames)
            print(f"Simple 深度图形状: {depth.shape}")
        except Exception as e2:
            print(f"SimpleDepthEstimator 也失败: {e2}")
