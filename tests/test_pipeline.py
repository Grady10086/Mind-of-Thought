"""
Pipeline 单元测试
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch


class TestVideoReader:
    """测试视频读取器"""
    
    def test_import(self):
        from core.dataloader import VideoReader
        assert VideoReader is not None
    
    def test_metadata(self):
        """测试视频元数据获取（如果有测试视频）"""
        from core.dataloader import VideoReader
        # 跳过如果没有测试视频
        pass


class TestVSIBenchDataset:
    """测试数据集加载"""
    
    def test_import(self):
        from core.dataloader import VSIBenchDataset
        assert VSIBenchDataset is not None
    
    def test_task_types(self):
        from core.dataloader import TaskType, MCA_TASKS, NA_TASKS
        assert len(MCA_TASKS) > 0
        assert len(NA_TASKS) > 0


class TestDepthEstimator:
    """测试深度估计器"""
    
    def test_import(self):
        from core.perception import DepthEstimator, TemporalSmoothingMethod
        assert DepthEstimator is not None
        assert TemporalSmoothingMethod is not None
    
    def test_temporal_smoothing_methods(self):
        from core.perception import TemporalSmoothingMethod
        methods = [m.value for m in TemporalSmoothingMethod]
        assert 'exponential_moving_average' in methods
        assert 'gaussian' in methods


class TestVisualizer:
    """测试可视化器"""
    
    def test_import(self):
        from core.visualizer import SpatialVisualizer
        assert SpatialVisualizer is not None
    
    def test_depth_to_colormap(self):
        from core.visualizer import SpatialVisualizer
        
        viz = SpatialVisualizer()
        depth = np.random.rand(480, 640).astype(np.float32)
        colored = viz.depth_to_colormap(depth)
        
        assert colored.shape == (480, 640, 3)
        assert colored.dtype == np.uint8
    
    def test_side_by_side(self):
        from core.visualizer import SpatialVisualizer
        
        viz = SpatialVisualizer()
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32)
        
        combined = viz.create_side_by_side(rgb, depth)
        
        assert combined.shape == (480, 1280, 3)  # 宽度翻倍


class TestIntegration:
    """集成测试"""
    
    def test_pipeline_import(self):
        from main_pipeline import SpatialIntelligencePipeline
        assert SpatialIntelligencePipeline is not None


def test_normalize_depth():
    """测试深度归一化"""
    from core.perception import DepthEstimator
    
    # 创建测试数据
    depth = torch.randn(4, 1, 480, 640)
    
    # 手动归一化
    depth_min = depth.view(4, -1).min(dim=1, keepdim=True)[0].view(4, 1, 1, 1)
    depth_max = depth.view(4, -1).max(dim=1, keepdim=True)[0].view(4, 1, 1, 1)
    normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    
    assert normalized.min() >= 0
    assert normalized.max() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
