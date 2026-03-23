#!/usr/bin/env python3
"""
VSIBench 消融实验测试 - 逐一测试各项改进

改进项：
1. 相机位姿估计 (use_camera_pose)
2. 动态尺度校准 (use_scale_calibration)  
3. 体素占据地图 (use_voxel_map)
4. 实例级追踪 (use_instance_tracking)

用法：
  # 基线（无改进）
  python test_vsibench_ablation.py --num-gpus 8

  # 测试单个改进
  python test_vsibench_ablation.py --num-gpus 8 --use-camera-pose
  python test_vsibench_ablation.py --num-gpus 8 --use-scale-calibration
  python test_vsibench_ablation.py --num-gpus 8 --use-voxel-map
  python test_vsibench_ablation.py --num-gpus 8 --use-instance-tracking

  # 组合测试
  python test_vsibench_ablation.py --num-gpus 8 --use-scale-calibration --use-voxel-map

作者: tianjungu
日期: 2026-01-28
"""

import os
import sys
import json
import gc
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import re
import logging

import numpy as np
import cv2

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 导入原有模块
# ============================================================================

from tests.test_vsibench_directqa import (
    normalize_number, mean_relative_accuracy,
    NUMERICAL_TASKS, CHOICE_TASKS,
    EXTENDED_VOCABULARY, SYNONYM_MAP,
    get_synonyms, match_object_name,
    DirectQA,
    find_video_path, get_scene_source,
    VIDEO_DIRS,
)


# ============================================================================
# 改进版心智地图构建器
# ============================================================================

class MindMapBuilderAblation:
    """心智地图构建器 - 消融实验版"""
    
    def __init__(
        self, 
        device: str = 'cuda',
        num_frames: int = 32,
        box_threshold: float = 0.25,
        # 改进开关
        use_camera_pose: bool = False,
        use_scale_calibration: bool = False,
        use_voxel_map: bool = False,
        use_instance_tracking: bool = False,
    ):
        self.device = device
        self.num_frames = num_frames
        self.box_threshold = box_threshold
        
        # 改进开关
        self.use_camera_pose = use_camera_pose
        self.use_scale_calibration = use_scale_calibration
        self.use_voxel_map = use_voxel_map
        self.use_instance_tracking = use_instance_tracking
        
        # 模型
        self._labeler = None
        self._depth_estimator = None
        
        # 改进模块（按需加载）
        self.pose_estimator = None
        self.scale_calibrator = None
        self.voxel_map = None
        self.instance_tracker = None
        
        # 相机参数
        self.focal_length = 500
        self.principal_point = None
        
        # 中间结果
        self.all_detections: Dict[str, List[Dict]] = {}
        self.frame_indices: List[int] = []
        
        # 已知物体尺寸（用于尺度校准）
        self.known_sizes = {
            'door': {'height': 2.0, 'width': 0.9},
            'bed': {'height': 0.5, 'width': 1.5, 'length': 2.0},
            'sofa': {'height': 0.85, 'width': 0.9},
            'couch': {'height': 0.85, 'width': 0.9},
            'refrigerator': {'height': 1.7, 'width': 0.7},
            'fridge': {'height': 1.7, 'width': 0.7},
            'chair': {'height': 0.9, 'width': 0.5},
            'table': {'height': 0.75, 'width': 0.8},
            'desk': {'height': 0.75, 'width': 0.6},
            'toilet': {'height': 0.4, 'width': 0.4},
            'tv': {'height': 0.6, 'width': 1.0},
            'television': {'height': 0.6, 'width': 1.0},
            'monitor': {'height': 0.35, 'width': 0.55},
        }
    
    def _load_models(self):
        """加载模型"""
        import torch
        
        if self._labeler is None:
            from core.semantic_labeler import GroundingDINOLabeler
            self._labeler = GroundingDINOLabeler(
                model_id="IDEA-Research/grounding-dino-base",
                device=self.device,
                box_threshold=self.box_threshold,
                text_threshold=0.25,
            )
            self._labeler.load_model()
        
        if self._depth_estimator is None:
            from core.perception import DepthEstimator
            self._depth_estimator = DepthEstimator(
                model_name="depth-anything/DA3-Large",
                device=self.device,
                half_precision=True,
            )
        
        # 按需初始化改进模块
        if self.use_camera_pose and self.pose_estimator is None:
            self.pose_estimator = SimplePoseEstimator(self.focal_length)
        
        if self.use_voxel_map and self.voxel_map is None:
            self.voxel_map = SimpleVoxelMap(voxel_size=0.2)
        
        if self.use_instance_tracking and self.instance_tracker is None:
            self.instance_tracker = SimpleInstanceTracker()
    
    def unload(self):
        """释放模型"""
        import torch
        
        if self._labeler is not None:
            try:
                del self._labeler.model
                del self._labeler.processor
            except:
                pass
            self._labeler = None
        
        if self._depth_estimator is not None:
            try:
                del self._depth_estimator.model
            except:
                pass
            self._depth_estimator = None
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    
    def _sample_frames(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:
        """采样帧"""
        n = min(self.num_frames, len(frames))
        indices = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in indices], indices.tolist()
    
    def _calibrate_scale(self, detections: List[Dict], depth_map: np.ndarray) -> float:
        """动态尺度校准"""
        if not self.use_scale_calibration:
            # 默认中值深度校准
            median_depth = np.median(depth_map)
            return 2.5 / median_depth if median_depth > 0 else 1.0
        
        scale_estimates = []
        
        for det in detections:
            label = det['label'].lower()
            
            # 查找已知尺寸
            known_size = None
            for key in self.known_sizes:
                if key in label or label in key:
                    known_size = self.known_sizes[key]
                    break
            
            if known_size is None:
                continue
            
            ref_dim = known_size.get('height', known_size.get('width'))
            if ref_dim is None:
                continue
            
            # 计算估计高度
            x1, y1, x2, y2 = det['bbox']
            h_pixel = y2 - y1
            
            depth_roi = depth_map[y1:y2, x1:x2]
            if depth_roi.size == 0:
                continue
            depth_median = np.median(depth_roi)
            if depth_median <= 0:
                continue
            
            h_estimated = h_pixel / self.focal_length * depth_median
            
            if h_estimated > 0.01:
                scale = ref_dim / h_estimated
                if 0.1 < scale < 10:
                    scale_estimates.append(scale)
        
        if scale_estimates:
            return float(np.median(scale_estimates))
        
        # 回退到默认
        median_depth = np.median(depth_map)
        return 2.5 / median_depth if median_depth > 0 else 1.0
    
    def build_from_video(self, video_path: str, 
                         target_objects: List[str] = None) -> Dict[str, Any]:
        """从视频构建心智地图"""
        self._load_models()
        
        # 重置模块
        if self.pose_estimator:
            self.pose_estimator.reset()
        if self.voxel_map:
            self.voxel_map.clear()
        if self.instance_tracker:
            self.instance_tracker.reset()
        
        # 提取帧
        all_frames = self._extract_frames(video_path)
        if not all_frames:
            return {}
        
        frames, self.frame_indices = self._sample_frames(all_frames)
        H, W = frames[0].shape[:2]
        self.principal_point = (W / 2, H / 2)
        
        # 构建词汇表
        if target_objects is None:
            target_objects = []
        vocab = list(set(target_objects + EXTENDED_VOCABULARY))
        text_prompt = " . ".join(vocab) + " ."
        
        # 存储所有检测
        self.all_detections = defaultdict(list)
        frame_detection_counts = defaultdict(lambda: defaultdict(int))
        
        for frame_idx, frame in enumerate(frames):
            original_idx = self.frame_indices[frame_idx]
            
            # 1. 深度估计
            try:
                depth_tensor, _ = self._depth_estimator.infer_single(frame, normalize=False)
                depth_map = depth_tensor.squeeze().cpu().numpy()
                if depth_map.shape[:2] != (H, W):
                    depth_map = cv2.resize(depth_map, (W, H))
            except Exception as e:
                logger.warning(f"深度估计失败: {e}")
                depth_map = np.ones((H, W), dtype=np.float32) * 2.5
            
            # 2. 物体检测
            results = self._labeler.detect(frame, text_prompt)
            
            # 预处理检测结果
            raw_detections = []
            for det in results:
                x1, y1, x2, y2 = [int(v) for v in det.bbox_pixels]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    raw_detections.append({
                        'label': det.label.lower(),
                        'bbox': (x1, y1, x2, y2),
                        'confidence': det.confidence,
                    })
            
            # 3. 尺度校准
            scale = self._calibrate_scale(raw_detections, depth_map)
            depth_map_scaled = depth_map * scale
            
            # 4. 相机位姿估计
            if self.use_camera_pose and self.pose_estimator:
                camera_pose = self.pose_estimator.estimate(frame, original_idx)
            else:
                camera_pose = None
            
            # 5. 处理每个检测
            for det in raw_detections:
                label = det['label']
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                
                # 记录帧内检测数
                frame_detection_counts[original_idx][label] += 1
                
                # 深度
                depth_roi = depth_map_scaled[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                depth_median = float(np.median(depth_roi))
                
                # 3D 位置
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                px, py = self.principal_point
                
                x_3d = (cx - px) / self.focal_length * depth_median
                y_3d = (cy - py) / self.focal_length * depth_median
                z_3d = depth_median
                
                position_camera = np.array([x_3d, y_3d, z_3d])
                
                # 转换到世界坐标
                if camera_pose is not None:
                    position_world = camera_pose.camera_to_world(position_camera)
                else:
                    position_world = position_camera
                
                # 3D 尺寸
                w_3d = (x2 - x1) / self.focal_length * depth_median
                h_3d = (y2 - y1) / self.focal_length * depth_median
                d_3d = min(w_3d, h_3d) * 0.5
                size_3d = np.array([w_3d, h_3d, d_3d])
                
                # 插入体素地图
                if self.use_voxel_map and self.voxel_map:
                    self.voxel_map.insert_bbox(position_world, size_3d)
                
                # 存储检测
                self.all_detections[label].append({
                    'frame_idx': original_idx,
                    'bbox': det['bbox'],
                    'confidence': confidence,
                    'position_3d': position_world,
                    'size_3d': size_3d,
                    'depth_median': depth_median,
                })
        
        # 聚合实体
        return self._aggregate(frame_detection_counts)
    
    def _aggregate(self, frame_detection_counts) -> Dict[str, Any]:
        """聚合检测结果"""
        
        @dataclass
        class Entity:
            entity_id: str
            label: str
            count: int
            avg_confidence: float
            first_seen_frame: int
            last_seen_frame: int
            position_3d: np.ndarray
            size_3d: np.ndarray
            depth_median: float
        
        entities = {}
        
        for category, dets in self.all_detections.items():
            if not dets:
                continue
            
            # 计算最大单帧检测数
            max_count = 0
            for frame_idx, counts in frame_detection_counts.items():
                if category in counts:
                    max_count = max(max_count, counts[category])
            
            if max_count == 0:
                max_count = 1
            
            # 聚合
            avg_conf = np.mean([d['confidence'] for d in dets])
            first_frame = min(d['frame_idx'] for d in dets)
            last_frame = max(d['frame_idx'] for d in dets)
            
            positions = np.array([d['position_3d'] for d in dets])
            sizes = np.array([d['size_3d'] for d in dets])
            depths = [d['depth_median'] for d in dets]
            
            entity = Entity(
                entity_id=f"entity_{category}",
                label=category,
                count=max_count,
                avg_confidence=float(avg_conf),
                first_seen_frame=first_frame,
                last_seen_frame=last_frame,
                position_3d=np.median(positions, axis=0),
                size_3d=np.median(sizes, axis=0),
                depth_median=float(np.median(depths)),
            )
            entities[category] = entity
        
        return entities
    
    def get_voxel_floor_area(self) -> Optional[float]:
        """获取体素地图计算的地面面积"""
        if self.voxel_map:
            return self.voxel_map.compute_floor_area()
        return None


# ============================================================================
# 简化的改进模块
# ============================================================================

class SimplePoseEstimator:
    """简化的相机位姿估计"""
    
    def __init__(self, focal_length: float = 500):
        self.focal_length = focal_length
        self.cumulative_t = np.zeros(3)
        self.prev_frame = None
        
    def estimate(self, frame: np.ndarray, frame_idx: int) -> 'SimpleCameraPose':
        """估计位姿"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # 简单的光流估计位移
        if self.prev_frame is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_frame, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # 估计平均位移
                mean_flow = np.mean(flow, axis=(0, 1))
                
                # 转换为 3D 位移（简化：假设深度为 2m）
                dx = mean_flow[0] / self.focal_length * 2.0
                dz = mean_flow[1] / self.focal_length * 2.0
                
                self.cumulative_t[0] += dx * 0.1
                self.cumulative_t[2] += dz * 0.1
                
            except Exception as e:
                pass
        
        self.prev_frame = gray
        
        return SimpleCameraPose(
            frame_idx=frame_idx,
            t=self.cumulative_t.copy()
        )
    
    def reset(self):
        self.cumulative_t = np.zeros(3)
        self.prev_frame = None


@dataclass
class SimpleCameraPose:
    """简化的相机位姿"""
    frame_idx: int
    t: np.ndarray = None
    
    def __post_init__(self):
        if self.t is None:
            self.t = np.zeros(3)
    
    def camera_to_world(self, point: np.ndarray) -> np.ndarray:
        return point + self.t


class SimpleVoxelMap:
    """简化的体素地图"""
    
    def __init__(self, voxel_size: float = 0.2):
        self.voxel_size = voxel_size
        self.occupied: set = set()
    
    def insert_bbox(self, center: np.ndarray, size: np.ndarray):
        """插入包围盒"""
        half = size / 2
        min_corner = center - half
        max_corner = center + half
        
        # 简化：只插入几个代表性点
        for x in np.linspace(min_corner[0], max_corner[0], 3):
            for z in np.linspace(min_corner[2], max_corner[2], 3):
                vx = int(x / self.voxel_size)
                vz = int(z / self.voxel_size)
                self.occupied.add((vx, vz))
    
    def compute_floor_area(self) -> float:
        """计算地面面积"""
        return len(self.occupied) * (self.voxel_size ** 2)
    
    def clear(self):
        self.occupied.clear()


class SimpleInstanceTracker:
    """简化的实例追踪"""
    
    def __init__(self):
        self.frame_counts: Dict[int, Dict[str, int]] = {}
    
    def record_frame(self, frame_idx: int, label: str):
        if frame_idx not in self.frame_counts:
            self.frame_counts[frame_idx] = defaultdict(int)
        self.frame_counts[frame_idx][label] += 1
    
    def get_max_count(self, label: str) -> int:
        max_count = 0
        for counts in self.frame_counts.values():
            if label in counts:
                max_count = max(max_count, counts[label])
        return max_count
    
    def reset(self):
        self.frame_counts.clear()


# ============================================================================
# 改进版 DirectQA
# ============================================================================

class DirectQAAblation:
    """DirectQA 消融版 - 支持使用改进模块"""
    
    @staticmethod
    def answer_room_size(mind_map: Dict, question: str, 
                         voxel_area: Optional[float] = None) -> str:
        """
        回答房间面积
        
        如果启用了体素地图，使用体素计算的面积
        否则使用原来的包围盒方法
        """
        # 优先使用体素面积
        if voxel_area is not None and voxel_area > 0:
            # 体素面积需要一些扩展（因为只覆盖了物体区域）
            estimated_area = voxel_area * 2.5  # 扩展系数
            estimated_area = max(8, min(80, estimated_area))
            return f"{estimated_area:.1f}"
        
        # 回退到原来的包围盒方法
        if not mind_map:
            return "20"
        
        positions = []
        for entity in mind_map.values():
            if entity.position_3d is not None:
                positions.append(entity.position_3d)
        
        if len(positions) < 2:
            return str(12 + len(mind_map) * 2)
        
        positions = np.array(positions)
        
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        estimated_area = (x_range + 1.5) * (y_range + 1.5)
        estimated_area = max(8, min(80, estimated_area))
        
        return f"{estimated_area:.1f}"


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], config: Dict, result_queue: mp.Queue):
    """GPU Worker"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    builder = MindMapBuilderAblation(
        device='cuda',
        num_frames=32,
        box_threshold=0.25,
        use_camera_pose=config.get('use_camera_pose', False),
        use_scale_calibration=config.get('use_scale_calibration', False),
        use_voxel_map=config.get('use_voxel_map', False),
        use_instance_tracking=config.get('use_instance_tracking', False),
    )
    
    results = []
    total = len(samples)
    
    for i, sample in enumerate(samples):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 构建心智地图
            target_objects = []
            if question_type == 'object_counting':
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map = builder.build_from_video(video_path, target_objects)
            voxel_area = builder.get_voxel_floor_area()
            
            # 回答问题
            if question_type == 'object_counting':
                pred = DirectQA.answer_counting(mind_map, question)
            elif question_type == 'object_size_estimation':
                pred = DirectQA.answer_object_size(mind_map, question)
            elif question_type == 'room_size_estimation':
                pred = DirectQAAblation.answer_room_size(mind_map, question, voxel_area)
            elif question_type == 'object_abs_distance':
                pred = DirectQA.answer_abs_distance(mind_map, question)
            elif question_type.startswith('object_rel_direction'):
                difficulty = question_type.split('_')[-1] if '_' in question_type else 'easy'
                pred = DirectQA.answer_rel_direction(mind_map, question, options, difficulty)
            elif question_type == 'object_rel_distance':
                pred = DirectQA.answer_rel_distance(mind_map, question, options)
            elif question_type == 'obj_appearance_order':
                pred = DirectQA.answer_appearance_order(mind_map, question, options)
            elif question_type == 'route_planning':
                pred = DirectQA.answer_route_planning(mind_map, question, options)
            else:
                pred = str(options[0]) if options else "0"
            
            # 计算指标
            if question_type in NUMERICAL_TASKS:
                pred_num = normalize_number(pred)
                gt_num = normalize_number(str(gt))
                score = mean_relative_accuracy(pred_num, gt_num) if pred_num and gt_num else 0.0
                correct = score > 0.5
            else:
                # 选择题评估
                pred_letter = None
                gt_letter = str(gt).strip().upper()
                
                if len(gt_letter) > 1:
                    for idx, opt in enumerate(options):
                        opt_content = re.sub(r'^[A-D]\.\s*', '', opt).strip()
                        if gt_letter.lower() == opt_content.lower():
                            gt_letter = chr(65 + idx)
                            break
                
                if pred:
                    letter_match = re.match(r'^([A-D])[\.\s]', pred.strip().upper())
                    if letter_match:
                        pred_letter = letter_match.group(1)
                    else:
                        pred_clean = re.sub(r'^[A-D]\.\s*', '', pred).lower().strip()
                        for idx, opt in enumerate(options):
                            opt_content = re.sub(r'^[A-D]\.\s*', '', opt).lower().strip()
                            if pred_clean == opt_content or pred_clean in opt_content or opt_content in pred_clean:
                                pred_letter = chr(65 + idx)
                                break
                
                correct = pred_letter == gt_letter if pred_letter else False
                score = 1.0 if correct else 0.0
            
            result = {
                'sample_id': sample['id'],
                'question_type': question_type,
                'question': question,
                'prediction': pred,
                'ground_truth': gt,
                'score': score,
                'correct': bool(correct),
            }
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} 样本 {sample['id']} 错误: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'sample_id': sample['id'],
                'question_type': question_type,
                'question': question,
                'prediction': '',
                'ground_truth': gt,
                'score': 0.0,
                'correct': False,
                'error': str(e),
            }
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"GPU {gpu_id}: {i+1}/{total} 完成")
    
    builder.unload()
    result_queue.put((gpu_id, results))


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--task-type', type=str, default='all')
    
    # 改进开关
    parser.add_argument('--use-camera-pose', action='store_true', help='启用相机位姿估计')
    parser.add_argument('--use-scale-calibration', action='store_true', help='启用动态尺度校准')
    parser.add_argument('--use-voxel-map', action='store_true', help='启用体素地图')
    parser.add_argument('--use-instance-tracking', action='store_true', help='启用实例追踪')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'use_camera_pose': args.use_camera_pose,
        'use_scale_calibration': args.use_scale_calibration,
        'use_voxel_map': args.use_voxel_map,
        'use_instance_tracking': args.use_instance_tracking,
    }
    
    # 生成实验名称
    enabled_features = [k.replace('use_', '') for k, v in config.items() if v]
    exp_name = '_'.join(enabled_features) if enabled_features else 'baseline'
    
    print("=" * 70)
    print(f"VSIBench 消融实验 - {exp_name}")
    print("=" * 70)
    print(f"配置: {config}")
    print(f"GPU数量: {args.num_gpus}")
    
    from datasets import load_dataset
    dataset = load_dataset(
        "nyu-visionx/VSI-Bench",
        split="test",
        cache_dir="/home/tione/notebook/tianjungu/hf_cache/vsibench"
    )
    
    # 准备样本
    samples = []
    for idx, item in enumerate(dataset):
        scene_name = item['scene_name']
        question_type = item['question_type']
        
        if args.task_type != 'all' and question_type != args.task_type:
            continue
        
        video_path = find_video_path(scene_name)
        if not video_path:
            continue
        
        samples.append({
            'id': idx,
            'scene_name': scene_name,
            'source': get_scene_source(scene_name),
            'video_path': video_path,
            'question': item['question'],
            'question_type': question_type,
            'ground_truth': item['ground_truth'],
            'options': item.get('options', []),
        })
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"总样本数: {len(samples)}")
    
    # 任务统计
    type_counts = defaultdict(int)
    for s in samples:
        type_counts[s['question_type']] += 1
    print("\n任务类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    
    # 分配到各 GPU
    num_gpus = min(args.num_gpus, len(samples))
    samples_per_gpu = len(samples) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])
    
    # 启动多进程
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    
    start_time = datetime.now()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_samples[gpu_id], config, result_queue))
        p.start()
        processes.append(p)
        logger.info(f"启动 GPU {gpu_id}: {len(gpu_samples[gpu_id])} 样本")
    
    # 收集结果
    all_results = []
    for _ in range(num_gpus):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        logger.info(f"GPU {gpu_id} 完成: {len(results)} 结果")
    
    for p in processes:
        p.join()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n总耗时: {duration:.1f}秒")
    
    # 统计
    type_scores = defaultdict(list)
    for r in all_results:
        type_scores[r['question_type']].append(r['score'])
    
    # 输出结果
    print("\n" + "="*60)
    print(f"消融实验结果 - {exp_name}")
    print("="*60)
    
    total_score = 0
    total_count = 0
    
    for q_type in sorted(type_scores.keys()):
        scores = type_scores[q_type]
        if q_type in NUMERICAL_TASKS:
            avg = np.mean(scores) * 100
            metric = "MRA"
        else:
            avg = np.mean(scores) * 100
            metric = "Acc"
        
        print(f"{q_type:30s}: {avg:6.2f}% {metric} ({len(scores)} 样本)")
        total_score += sum(scores)
        total_count += len(scores)
    
    overall = total_score / total_count * 100 if total_count > 0 else 0
    print("-"*60)
    print(f"{'Overall':30s}: {overall:6.2f}% ({total_count} 样本)")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/ablation_{exp_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换 numpy 类型
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    cleaned_results = []
    for r in all_results:
        cleaned = {k: convert_numpy(v) for k, v in r.items()}
        cleaned_results.append(cleaned)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'config': config,
            'experiment_name': exp_name,
            'summary': {q: {'mean': float(np.mean(s)), 'count': len(s)} for q, s in type_scores.items()},
            'overall': float(overall),
            'details': cleaned_results,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果保存到: {output_dir}")
