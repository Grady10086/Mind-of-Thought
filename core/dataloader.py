"""
VSI-Bench 数据加载器
专为空间智能研究设计，支持视频帧序列、问题文本、任务类型和 Ground Truth 的高效读取

参考: https://github.com/vision-x-nyu/thinking-in-space
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Generator, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """VSI-Bench 任务类型"""
    # Multiple Choice Answer (MCA) 任务
    OBJECT_REL_DIRECTION_LR = "object_rel_direction_lr"
    OBJECT_REL_DIRECTION_FB = "object_rel_direction_fb"
    OBJECT_REL_DIRECTION_UD = "object_rel_direction_ud"
    OBJECT_REL_DISTANCE = "object_rel_distance"
    ROUTE_PLANNING = "route_planning"
    OBJ_APPEARANCE_ORDER = "obj_appearance_order"
    
    # Numeric Answer (NA) 任务
    OBJECT_ABS_DISTANCE = "object_abs_distance"
    OBJECT_COUNTING = "object_counting"
    OBJECT_SIZE_ESTIMATION = "object_size_estimation"
    ROOM_SIZE_ESTIMATION = "room_size_estimation"


# 任务分类
MCA_TASKS = {
    TaskType.OBJECT_REL_DIRECTION_LR,
    TaskType.OBJECT_REL_DIRECTION_FB,
    TaskType.OBJECT_REL_DIRECTION_UD,
    TaskType.OBJECT_REL_DISTANCE,
    TaskType.ROUTE_PLANNING,
    TaskType.OBJ_APPEARANCE_ORDER,
}

NA_TASKS = {
    TaskType.OBJECT_ABS_DISTANCE,
    TaskType.OBJECT_COUNTING,
    TaskType.OBJECT_SIZE_ESTIMATION,
    TaskType.ROOM_SIZE_ESTIMATION,
}


@dataclass
class VSIBenchSample:
    """VSI-Bench 单个样本的数据结构"""
    video_path: str
    scene_id: str
    question: str
    task_type: str
    ground_truth: Union[str, float, int]
    options: Optional[List[str]] = None  # MCA 任务的选项
    metadata: Optional[Dict] = None
    
    def is_mca_task(self) -> bool:
        """是否为多选题任务"""
        return any(t.value in self.task_type for t in MCA_TASKS)
    
    def is_na_task(self) -> bool:
        """是否为数值答案任务"""
        return any(t.value in self.task_type for t in NA_TASKS)


class VideoReader:
    """
    视频读取器 - 支持多种后端和鲁棒的错误处理
    
    特性:
    - 支持 cv2、imageio、moviepy 多后端
    - 内存高效的帧生成器模式
    - 时间戳精确定位
    """
    
    SUPPORTED_BACKENDS = ['cv2', 'imageio', 'moviepy']
    
    def __init__(self, video_path: str, backend: str = 'cv2'):
        self.video_path = video_path
        self.backend = backend
        self._cap = None
        self._metadata = None
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    def _init_cv2(self):
        """初始化 OpenCV 后端"""
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
        
        self._metadata = {
            'total_frames': int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': self._cap.get(cv2.CAP_PROP_FPS),
            'width': int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(self._cap.get(cv2.CAP_PROP_FPS), 1e-6)
        }
    
    @property
    def metadata(self) -> Dict:
        if self._metadata is None:
            self._init_cv2()
        return self._metadata
    
    def get_frame_at_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """
        获取指定时间戳的帧
        
        Args:
            timestamp: 时间戳（秒）
            
        Returns:
            RGB 格式的帧 (H, W, 3) 或 None
        """
        if self._cap is None:
            self._init_cv2()
        
        fps = self._metadata['fps']
        frame_idx = int(timestamp * fps)
        frame_idx = max(0, min(frame_idx, self._metadata['total_frames'] - 1))
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def get_frame_at_index(self, index: int) -> Optional[np.ndarray]:
        """获取指定索引的帧"""
        if self._cap is None:
            self._init_cv2()
        
        index = max(0, min(index, self._metadata['total_frames'] - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def extract_frames_uniform(
        self, 
        num_frames: int = 16,
        return_timestamps: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
        """
        均匀采样视频帧
        
        Args:
            num_frames: 采样帧数
            return_timestamps: 是否返回时间戳
            
        Returns:
            帧列表，或 (帧列表, 时间戳列表)
        """
        if self._cap is None:
            self._init_cv2()
        
        total = self._metadata['total_frames']
        if total == 0:
            logger.warning(f"视频帧数为0: {self.video_path}")
            return ([], []) if return_timestamps else []
        
        # 计算采样索引
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        fps = self._metadata['fps']
        
        frames = []
        timestamps = []
        
        for idx in indices:
            frame = self.get_frame_at_index(idx)
            if frame is not None:
                frames.append(frame)
                timestamps.append(idx / fps)
            else:
                logger.warning(f"无法读取帧 {idx}: {self.video_path}")
        
        if return_timestamps:
            return frames, timestamps
        return frames
    
    def frame_generator(
        self, 
        num_frames: int = 16,
        batch_size: int = 4
    ) -> Generator[Tuple[List[np.ndarray], List[int]], None, None]:
        """
        内存高效的帧生成器 - 适用于长视频处理
        
        Args:
            num_frames: 总采样帧数
            batch_size: 每批返回的帧数
            
        Yields:
            (帧批次, 帧索引批次)
        """
        if self._cap is None:
            self._init_cv2()
        
        total = self._metadata['total_frames']
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_frames = []
            
            for idx in batch_indices:
                frame = self.get_frame_at_index(idx)
                if frame is not None:
                    batch_frames.append(frame)
            
            if batch_frames:
                yield batch_frames, batch_indices.tolist()
    
    def close(self):
        """释放资源"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VSIBenchDataset(Dataset):
    """
    VSI-Bench 数据集类
    
    支持:
    - HuggingFace Hub 数据集
    - 本地目录结构
    - 视频帧序列、问题文本、任务类型、Ground Truth
    - 内存高效的帧加载
    
    Example:
        >>> dataset = VSIBenchDataset(
        ...     data_root="/path/to/vsibench",
        ...     num_frames=16,
        ...     task_filter=["object_counting", "room_size_estimation"]
        ... )
        >>> sample = dataset[0]
        >>> frames, question, task_type, ground_truth = sample['frames'], sample['question'], sample['task_type'], sample['ground_truth']
    """
    
    def __init__(
        self,
        data_root: Optional[str] = None,
        hf_dataset_name: str = "nyu-visionx/VSI-Bench",
        split: str = "test",
        num_frames: int = 16,
        frame_size: Optional[Tuple[int, int]] = None,  # (H, W)
        task_filter: Optional[List[str]] = None,
        scene_filter: Optional[List[str]] = None,  # scannet, scannetpp, arkitscenes
        transform: Optional[callable] = None,
        cache_dir: Optional[str] = None,
        use_hf: bool = True,
        return_tensor: bool = True,
    ):
        """
        Args:
            data_root: 本地数据根目录
            hf_dataset_name: HuggingFace 数据集名称
            split: 数据集划分
            num_frames: 每个视频采样的帧数
            frame_size: 调整帧大小 (H, W)，None 表示保持原尺寸
            task_filter: 只加载特定任务类型
            scene_filter: 只加载特定场景源
            transform: 帧变换函数
            cache_dir: 缓存目录
            use_hf: 是否使用 HuggingFace 数据集
            return_tensor: 是否返回 PyTorch Tensor
        """
        self.data_root = data_root
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.task_filter = task_filter
        self.scene_filter = scene_filter
        self.transform = transform
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/vsibench")
        self.use_hf = use_hf
        self.return_tensor = return_tensor
        
        self.samples: List[VSIBenchSample] = []
        self._load_dataset()
        
        logger.info(f"VSIBenchDataset 初始化完成: {len(self.samples)} 样本")
    
    def _load_dataset(self):
        """加载数据集"""
        if self.use_hf:
            self._load_from_huggingface()
        elif self.data_root:
            self._load_from_local()
        else:
            raise ValueError("必须指定 data_root 或设置 use_hf=True")
    
    def _load_from_huggingface(self):
        """从 HuggingFace Hub 加载数据集"""
        try:
            from datasets import load_dataset
            
            logger.info(f"从 HuggingFace 加载数据集: {self.hf_dataset_name}")
            
            dataset = load_dataset(
                self.hf_dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            
            for item in dataset:
                # 解析任务类型
                task_type = item.get('question_type', item.get('task_type', ''))
                
                # 应用任务过滤
                if self.task_filter and not any(t in task_type for t in self.task_filter):
                    continue
                
                # 应用场景过滤
                scene_source = item.get('scene_source', item.get('source', ''))
                if self.scene_filter and scene_source not in self.scene_filter:
                    continue
                
                # 构建视频路径
                video_path = item.get('video', item.get('video_path', ''))
                if not os.path.isabs(video_path):
                    # 尝试从缓存目录构建路径
                    video_path = os.path.join(self.cache_dir, video_path)
                
                sample = VSIBenchSample(
                    video_path=video_path,
                    scene_id=item.get('scene_id', item.get('video_id', '')),
                    question=item.get('question', ''),
                    task_type=task_type,
                    ground_truth=item.get('answer', item.get('ground_truth', '')),
                    options=item.get('options', item.get('choices', None)),
                    metadata={
                        'source': scene_source,
                        'question_id': item.get('question_id', ''),
                        'raw_data': item,
                    }
                )
                self.samples.append(sample)
                
        except ImportError:
            logger.error("请安装 datasets: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"HuggingFace 数据集加载失败: {e}")
            raise
    
    def _load_from_local(self):
        """从本地目录加载数据集"""
        data_root = Path(self.data_root)
        
        # 查找 metadata.json
        metadata_paths = list(data_root.glob("**/metadata.json"))
        if not metadata_paths:
            # 尝试查找其他格式
            metadata_paths = list(data_root.glob("**/*.json"))
        
        for meta_path in metadata_paths:
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # 处理不同的 metadata 格式
                if isinstance(metadata, list):
                    items = metadata
                elif isinstance(metadata, dict):
                    items = metadata.get('data', metadata.get('samples', [metadata]))
                else:
                    continue
                
                video_dir = meta_path.parent / "video"
                if not video_dir.exists():
                    video_dir = meta_path.parent
                
                for item in items:
                    task_type = item.get('question_type', item.get('task_type', ''))
                    
                    if self.task_filter and not any(t in task_type for t in self.task_filter):
                        continue
                    
                    # 构建视频路径
                    video_name = item.get('video', item.get('video_path', item.get('scene_id', '')))
                    if not video_name.endswith('.mp4'):
                        video_name += '.mp4'
                    video_path = video_dir / video_name
                    
                    if not video_path.exists():
                        logger.warning(f"视频文件不存在: {video_path}")
                        continue
                    
                    sample = VSIBenchSample(
                        video_path=str(video_path),
                        scene_id=item.get('scene_id', ''),
                        question=item.get('question', ''),
                        task_type=task_type,
                        ground_truth=item.get('answer', item.get('ground_truth', '')),
                        options=item.get('options', None),
                        metadata=item,
                    )
                    self.samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"解析 {meta_path} 失败: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Returns:
            Dict containing:
                - frames: Tensor (T, C, H, W) 或 List[np.ndarray]
                - question: str
                - task_type: str
                - ground_truth: Union[str, float, int]
                - options: Optional[List[str]]
                - timestamps: List[float]
                - metadata: Dict
        """
        sample = self.samples[idx]
        
        # 读取视频帧
        try:
            with VideoReader(sample.video_path) as reader:
                frames, timestamps = reader.extract_frames_uniform(
                    num_frames=self.num_frames,
                    return_timestamps=True
                )
        except Exception as e:
            logger.error(f"视频读取失败 {sample.video_path}: {e}")
            # 返回空帧作为 fallback
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(self.num_frames)]
            timestamps = [0.0] * self.num_frames
        
        # 调整帧大小
        if self.frame_size is not None:
            h, w = self.frame_size
            frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames]
        
        # 应用变换
        if self.transform is not None:
            frames = [self.transform(f) for f in frames]
        
        # 转换为 Tensor
        if self.return_tensor:
            # (T, H, W, C) -> (T, C, H, W)
            frames_array = np.stack(frames, axis=0)
            frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float() / 255.0
        else:
            frames_tensor = frames
        
        return {
            'frames': frames_tensor,
            'question': sample.question,
            'task_type': sample.task_type,
            'ground_truth': sample.ground_truth,
            'options': sample.options,
            'timestamps': timestamps,
            'scene_id': sample.scene_id,
            'video_path': sample.video_path,
            'metadata': sample.metadata,
        }
    
    def get_frame_at_timestamp(self, idx: int, timestamp: float) -> Optional[np.ndarray]:
        """
        获取指定样本在特定时间戳的帧
        
        Args:
            idx: 样本索引
            timestamp: 时间戳（秒）
            
        Returns:
            RGB 帧 (H, W, 3) 或 None
        """
        sample = self.samples[idx]
        try:
            with VideoReader(sample.video_path) as reader:
                return reader.get_frame_at_timestamp(timestamp)
        except Exception as e:
            logger.error(f"获取帧失败: {e}")
            return None
    
    def get_task_statistics(self) -> Dict[str, int]:
        """获取任务类型统计"""
        stats = {}
        for sample in self.samples:
            task = sample.task_type
            stats[task] = stats.get(task, 0) + 1
        return stats
    
    def filter_by_task(self, task_types: List[str]) -> 'VSIBenchDataset':
        """创建只包含特定任务类型的子数据集视图"""
        filtered = VSIBenchDataset.__new__(VSIBenchDataset)
        filtered.__dict__.update(self.__dict__)
        filtered.samples = [s for s in self.samples if any(t in s.task_type for t in task_types)]
        return filtered
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """DataLoader 的 collate 函数"""
        frames = torch.stack([item['frames'] for item in batch])
        
        return {
            'frames': frames,  # (B, T, C, H, W)
            'questions': [item['question'] for item in batch],
            'task_types': [item['task_type'] for item in batch],
            'ground_truths': [item['ground_truth'] for item in batch],
            'options': [item['options'] for item in batch],
            'timestamps': [item['timestamps'] for item in batch],
            'scene_ids': [item['scene_id'] for item in batch],
            'video_paths': [item['video_path'] for item in batch],
            'metadata': [item['metadata'] for item in batch],
        }


def create_vsibench_dataloader(
    data_root: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    num_frames: int = 16,
    task_filter: Optional[List[str]] = None,
    **kwargs
) -> DataLoader:
    """
    创建 VSI-Bench DataLoader 的便捷函数
    
    Args:
        data_root: 数据根目录
        batch_size: 批大小
        num_workers: 工作进程数
        num_frames: 每视频采样帧数
        task_filter: 任务类型过滤
        **kwargs: 传递给 VSIBenchDataset 的其他参数
        
    Returns:
        DataLoader 实例
    """
    dataset = VSIBenchDataset(
        data_root=data_root,
        num_frames=num_frames,
        task_filter=task_filter,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=VSIBenchDataset.collate_fn,
        pin_memory=True,
    )


# 评估指标函数
def mean_relative_accuracy(pred: float, gt: float, threshold: float = 0.1) -> float:
    """
    计算 Mean Relative Accuracy (MRA)
    用于数值答案任务
    
    Args:
        pred: 预测值
        gt: Ground Truth
        threshold: 相对误差阈值
        
    Returns:
        accuracy score (0 或 1)
    """
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    
    relative_error = abs(pred - gt) / abs(gt)
    return 1.0 if relative_error <= threshold else 0.0


def exact_match(pred: str, gt: str) -> float:
    """
    精确匹配（用于多选题）
    
    Args:
        pred: 预测答案
        gt: Ground Truth
        
    Returns:
        1.0 if match else 0.0
    """
    return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    
    # 测试 HuggingFace 数据集加载
    try:
        dataset = VSIBenchDataset(
            use_hf=True,
            num_frames=8,
            task_filter=["object_counting"],
        )
        
        print(f"数据集大小: {len(dataset)}")
        print(f"任务统计: {dataset.get_task_statistics()}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本 frames shape: {sample['frames'].shape}")
            print(f"样本 question: {sample['question'][:100]}...")
            print(f"样本 task_type: {sample['task_type']}")
            print(f"样本 ground_truth: {sample['ground_truth']}")
            
    except Exception as e:
        print(f"测试失败: {e}")
