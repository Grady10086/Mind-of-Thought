"""
空间感知可视化模块 - Vibe 调试工具

特性:
- RGB + Depth 并排对比
- 深度图伪彩色增强 (MAGMA colormap)
- 视频预览和导出
- 问题文本叠加显示
- 时序一致性检测可视化

用于快速验证深度估计质量，建立科研直觉 (Vibe)
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import cv2

logger = logging.getLogger(__name__)

# 尝试导入 matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib 不可用，部分可视化功能受限")


@dataclass
class VisualizationConfig:
    """可视化配置"""
    colormap: int = cv2.COLORMAP_MAGMA  # 深度图颜色映射
    font_scale: float = 0.7
    font_thickness: int = 2
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_bg_color: Tuple[int, int, int] = (0, 0, 0)
    border_color: Tuple[int, int, int] = (128, 128, 128)
    output_fps: int = 10
    output_codec: str = 'mp4v'


class SpatialVisualizer:
    """
    空间感知可视化器
    
    功能:
    1. Side-by-side RGB/Depth 对比
    2. 深度图伪彩色增强
    3. 问题文本叠加
    4. 时序一致性热力图
    5. 视频导出
    
    Example:
        >>> viz = SpatialVisualizer()
        >>> viz.create_comparison_video(
        ...     rgb_frames, depth_maps, 
        ...     question="How many chairs are in the room?",
        ...     output_path="demo.mp4"
        ... )
    """
    
    # 支持的 colormaps
    COLORMAPS = {
        'magma': cv2.COLORMAP_MAGMA,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'jet': cv2.COLORMAP_JET,
        'turbo': cv2.COLORMAP_TURBO,
        'bone': cv2.COLORMAP_BONE,
    }
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def depth_to_colormap(
        self,
        depth: Union[np.ndarray, torch.Tensor],
        colormap: Optional[str] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        将深度图转换为伪彩色图像
        
        Args:
            depth: 深度图 (H, W) 或 (1, H, W)
            colormap: 颜色映射名称
            normalize: 是否归一化
            
        Returns:
            RGB 伪彩色图像 (H, W, 3)
        """
        # 转换为 numpy
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        
        # 确保是 2D
        if depth.ndim == 3:
            depth = depth.squeeze()
        
        # 归一化到 [0, 255]
        if normalize:
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max - depth_min > 1e-6:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth)
        
        depth_uint8 = (depth * 255).astype(np.uint8)
        
        # 应用 colormap
        cmap = self.COLORMAPS.get(colormap, self.config.colormap) if colormap else self.config.colormap
        colored = cv2.applyColorMap(depth_uint8, cmap)
        
        # BGR -> RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored
    
    def create_side_by_side(
        self,
        rgb: np.ndarray,
        depth: Union[np.ndarray, torch.Tensor],
        colormap: str = 'magma',
        add_labels: bool = True,
    ) -> np.ndarray:
        """
        创建 RGB 和 Depth 并排对比图
        
        Args:
            rgb: RGB 图像 (H, W, 3)
            depth: 深度图 (H, W) 或 (1, H, W)
            colormap: 深度图颜色映射
            add_labels: 是否添加标签
            
        Returns:
            并排图像 (H, W*2, 3)
        """
        # 确保 RGB 是正确格式
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        # 转换深度图为彩色
        depth_colored = self.depth_to_colormap(depth, colormap)
        
        # 调整大小以匹配
        h, w = rgb.shape[:2]
        if depth_colored.shape[:2] != (h, w):
            depth_colored = cv2.resize(depth_colored, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 并排拼接
        combined = np.concatenate([rgb, depth_colored], axis=1)
        
        # 添加标签
        if add_labels:
            combined = self._add_label(combined, "RGB", (10, 30))
            combined = self._add_label(combined, "Depth (Relative)", (w + 10, 30))
        
        return combined
    
    def _add_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: Optional[float] = None,
    ) -> np.ndarray:
        """添加文本标签"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = font_scale or self.config.font_scale
        thickness = self.config.font_thickness
        
        # 获取文本大小
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # 绘制背景
        x, y = position
        cv2.rectangle(
            image,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + baseline + 5),
            self.config.text_bg_color,
            -1
        )
        
        # 绘制文本
        cv2.putText(
            image,
            text,
            position,
            font,
            scale,
            self.config.text_color,
            thickness,
            cv2.LINE_AA
        )
        
        return image
    
    def add_question_overlay(
        self,
        image: np.ndarray,
        question: str,
        task_type: Optional[str] = None,
        max_width: int = 80,
    ) -> np.ndarray:
        """
        在图像上方添加问题文本
        
        Args:
            image: 输入图像
            question: 问题文本
            task_type: 任务类型
            max_width: 每行最大字符数
            
        Returns:
            添加问题后的图像
        """
        # 文本换行
        words = question.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_width:
                current_line = current_line + " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # 添加任务类型
        if task_type:
            lines.insert(0, f"[Task: {task_type}]")
        
        # 计算文本区域高度
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.config.font_scale * 0.8
        thickness = 1
        line_height = 25
        header_height = len(lines) * line_height + 20
        
        # 创建带头部的新图像
        h, w = image.shape[:2]
        new_image = np.zeros((h + header_height, w, 3), dtype=np.uint8)
        new_image[header_height:, :] = image
        
        # 填充头部背景
        new_image[:header_height, :] = (40, 40, 40)  # 深灰色背景
        
        # 绘制文本
        for i, line in enumerate(lines):
            y = 20 + i * line_height
            cv2.putText(
                new_image,
                line,
                (10, y),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
        
        return new_image
    
    def visualize_temporal_consistency(
        self,
        depth_maps: Union[torch.Tensor, List[np.ndarray]],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        可视化时序一致性 - 检测帧间闪烁
        
        通过计算相邻帧的深度差异来检测闪烁
        
        Args:
            depth_maps: 深度图序列 (T, H, W) 或 (T, 1, H, W)
            output_path: 保存路径
            
        Returns:
            时序一致性热力图 (T-1, H, W, 3)
        """
        # 转换格式
        if isinstance(depth_maps, torch.Tensor):
            depth_maps = depth_maps.detach().cpu().numpy()
        
        if depth_maps.ndim == 4:
            depth_maps = depth_maps.squeeze(1)
        
        T = depth_maps.shape[0]
        
        # 计算帧间差异
        diff_maps = []
        for t in range(1, T):
            diff = np.abs(depth_maps[t] - depth_maps[t-1])
            # 归一化差异
            diff = diff / (diff.max() + 1e-6)
            diff_maps.append(diff)
        
        diff_array = np.stack(diff_maps, axis=0)
        
        # 转换为热力图
        heatmaps = []
        for diff in diff_array:
            heatmap = (diff * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmaps.append(heatmap)
        
        result = np.stack(heatmaps, axis=0)
        
        # 保存
        if output_path:
            self._save_image_sequence(result, output_path)
        
        # 打印统计
        mean_diff = diff_array.mean()
        max_diff = diff_array.max()
        logger.info(f"时序一致性分析: 平均差异={mean_diff:.4f}, 最大差异={max_diff:.4f}")
        
        if mean_diff > 0.1:
            logger.warning("检测到较大的帧间闪烁，建议启用时序平滑")
        
        return result
    
    def create_comparison_video(
        self,
        rgb_frames: Union[List[np.ndarray], torch.Tensor],
        depth_maps: Union[List[np.ndarray], torch.Tensor],
        output_path: str,
        question: Optional[str] = None,
        task_type: Optional[str] = None,
        colormap: str = 'magma',
        fps: Optional[int] = None,
        show_frame_number: bool = True,
    ) -> str:
        """
        创建 RGB/Depth 对比视频
        
        Args:
            rgb_frames: RGB 帧序列
            depth_maps: 深度图序列
            output_path: 输出视频路径
            question: 问题文本（显示在视频上方）
            task_type: 任务类型
            colormap: 深度图颜色映射
            fps: 输出帧率
            show_frame_number: 是否显示帧号
            
        Returns:
            输出文件路径
        """
        # 转换格式
        if isinstance(rgb_frames, torch.Tensor):
            rgb_frames = rgb_frames.detach().cpu().numpy()
            if rgb_frames.shape[1] == 3:  # (T, C, H, W) -> (T, H, W, C)
                rgb_frames = rgb_frames.transpose(0, 2, 3, 1)
        
        if isinstance(depth_maps, torch.Tensor):
            depth_maps = depth_maps.detach().cpu().numpy()
            if depth_maps.ndim == 4:
                depth_maps = depth_maps.squeeze(1)
        
        # 转换为列表
        if isinstance(rgb_frames, np.ndarray):
            rgb_frames = [rgb_frames[i] for i in range(rgb_frames.shape[0])]
        if isinstance(depth_maps, np.ndarray):
            depth_maps = [depth_maps[i] for i in range(depth_maps.shape[0])]
        
        T = min(len(rgb_frames), len(depth_maps))
        
        # 创建第一帧以获取尺寸
        first_frame = self.create_side_by_side(rgb_frames[0], depth_maps[0], colormap)
        if question:
            first_frame = self.add_question_overlay(first_frame, question, task_type)
        
        h, w = first_frame.shape[:2]
        
        # 初始化视频写入器
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*self.config.output_codec)
        out_fps = fps or self.config.output_fps
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
        
        # 处理每一帧
        for i in range(T):
            frame = self.create_side_by_side(rgb_frames[i], depth_maps[i], colormap)
            
            if question:
                frame = self.add_question_overlay(frame, question, task_type)
            
            if show_frame_number:
                frame = self._add_label(frame, f"Frame: {i+1}/{T}", (w - 150, h - 20), 0.5)
            
            # RGB -> BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        logger.info(f"对比视频已保存: {output_path}")
        
        return output_path
    
    def create_depth_video(
        self,
        depth_maps: Union[List[np.ndarray], torch.Tensor],
        output_path: str,
        colormap: str = 'magma',
        fps: Optional[int] = None,
    ) -> str:
        """
        创建纯深度图视频
        
        Args:
            depth_maps: 深度图序列
            output_path: 输出路径
            colormap: 颜色映射
            fps: 帧率
            
        Returns:
            输出文件路径
        """
        # 转换格式
        if isinstance(depth_maps, torch.Tensor):
            depth_maps = depth_maps.detach().cpu().numpy()
            if depth_maps.ndim == 4:
                depth_maps = depth_maps.squeeze(1)
        
        if isinstance(depth_maps, np.ndarray) and depth_maps.ndim == 3:
            depth_maps = [depth_maps[i] for i in range(depth_maps.shape[0])]
        
        # 转换为彩色
        colored_frames = [self.depth_to_colormap(d, colormap) for d in depth_maps]
        
        h, w = colored_frames[0].shape[:2]
        
        # 写入视频
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*self.config.output_codec)
        out_fps = fps or self.config.output_fps
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
        
        for frame in colored_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        logger.info(f"深度视频已保存: {output_path}")
        
        return output_path
    
    def save_comparison_image(
        self,
        rgb: np.ndarray,
        depth: Union[np.ndarray, torch.Tensor],
        output_path: str,
        question: Optional[str] = None,
        task_type: Optional[str] = None,
        colormap: str = 'magma',
    ) -> str:
        """保存单帧对比图"""
        frame = self.create_side_by_side(rgb, depth, colormap)
        
        if question:
            frame = self.add_question_overlay(frame, question, task_type)
        
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, frame_bgr)
        
        logger.info(f"对比图已保存: {output_path}")
        return output_path
    
    def _save_image_sequence(
        self,
        images: np.ndarray,
        output_dir: str,
        prefix: str = "frame",
    ):
        """保存图像序列"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img_bgr)
    
    @staticmethod
    def create_grid_visualization(
        images: List[np.ndarray],
        grid_size: Tuple[int, int],
        titles: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        创建图像网格
        
        Args:
            images: 图像列表
            grid_size: (rows, cols)
            titles: 每个图像的标题
            
        Returns:
            网格图像
        """
        rows, cols = grid_size
        
        # 调整所有图像到相同大小
        h, w = images[0].shape[:2]
        resized = [cv2.resize(img, (w, h)) for img in images]
        
        # 创建网格
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        
        for i, img in enumerate(resized[:rows * cols]):
            r, c = i // cols, i % cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
            
            # 添加标题
            if titles and i < len(titles):
                cv2.putText(
                    grid,
                    titles[i],
                    (c * w + 10, r * h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        return grid


def quick_visualize(
    rgb_frames: List[np.ndarray],
    depth_maps: Union[List[np.ndarray], torch.Tensor],
    output_path: str = "output_comparison.mp4",
    question: Optional[str] = None,
):
    """
    快速可视化函数 - 一行代码生成对比视频
    
    Example:
        >>> quick_visualize(rgb_frames, depth_maps, "demo.mp4", "How many chairs?")
    """
    viz = SpatialVisualizer()
    return viz.create_comparison_video(
        rgb_frames,
        depth_maps,
        output_path,
        question=question,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("测试 SpatialVisualizer...")
    
    # 创建测试数据
    T, H, W = 8, 480, 640
    rgb_frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(T)]
    depth_maps = [np.random.rand(H, W).astype(np.float32) for _ in range(T)]
    
    viz = SpatialVisualizer()
    
    # 测试并排显示
    combined = viz.create_side_by_side(rgb_frames[0], depth_maps[0])
    print(f"并排图像形状: {combined.shape}")
    
    # 测试添加问题
    with_question = viz.add_question_overlay(
        combined,
        "How many chairs are visible in this room? Please count carefully.",
        "object_counting"
    )
    print(f"带问题图像形状: {with_question.shape}")
    
    # 测试视频生成
    try:
        output_path = viz.create_comparison_video(
            rgb_frames,
            depth_maps,
            "/tmp/test_comparison.mp4",
            question="Test question for spatial intelligence research",
            task_type="object_counting",
        )
        print(f"视频已生成: {output_path}")
    except Exception as e:
        print(f"视频生成失败: {e}")
    
    # 测试时序一致性
    depth_tensor = torch.tensor(np.stack(depth_maps))
    consistency = viz.visualize_temporal_consistency(depth_tensor)
    print(f"时序一致性热力图形状: {consistency.shape}")
    
    print("测试完成!")
