#!/usr/bin/env python3
"""
64³ Grid Mind Map 真实小样本测试

核心流程:
1. 从V7基准结果(VL=63.61%)中选取样本(每种任务类型各选一些)
2. 对每个样本视频: DA3多视图推理 → 获取绝对深度+内外参
3. GroundingDINO检测 → 反投影到世界坐标 → 构建64³ Grid
4. Grid工具确定性回答所有问题
5. 对比 Grid工具 vs V7 VL(63.61%)

对比基准: outputs/evolving_agent_v7_20260203_134612/ (VL Overall = 63.61%)
"""

import os
import sys
import json
import re
import gc
import time
import logging
import numpy as np
import cv2
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime

# ============================================================================
# 环境配置
# ============================================================================
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 添加 DA3 路径
DA3_ROOT = PROJECT_ROOT / "projects" / "Depth-Anything-3"
sys.path.insert(0, str(DA3_ROOT))
sys.path.insert(0, str(DA3_ROOT / "src"))

# 固定随机种子以确保DA3输出可重复
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 视频目录
VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

# 扩展词汇表 (v6: 覆盖VSIBench中常见物体 + 缺失实体补充)
# GroundingDINO prompt有长度限制，分批检测（每批≤30词）
EXTENDED_VOCABULARY_BATCH1 = [
    "chair", "table", "sofa", "couch", "bed", "desk", "lamp", "door", "window",
    "cabinet", "shelf", "stove", "tv", "television", "refrigerator", "fridge",
    "sink", "toilet", "bathtub", "mirror", "curtain", "rug", "carpet",
    "pillow", "cushion", "monitor", "nightstand", "closet", "bookshelf",
]
EXTENDED_VOCABULARY_BATCH2 = [
    "microwave", "oven", "washer", "dryer", "printer", "counter", "drawer",
    "clock", "fan", "armchair", "stool", "fireplace", "blanket", "towel",
    "plant", "cup", "heater", "picture", "painting", "backpack", "bag",
    "trash can", "trash bin", "air conditioner",
]
EXTENDED_VOCABULARY_BATCH3 = [
    "telephone", "phone", "light", "laptop", "keyboard", "computer",
    "vase", "bottle", "box", "basket", "wardrobe", "dresser", "bench",
    "shower", "tub", "faucet", "pot", "pan", "kettle", "toaster",
    "book", "shoe", "speaker", "screen", "remote control",
    "whiteboard", "heater", "radiator", "chandelier", "ceiling light",
]
# BATCH4: VSIBench高频缺失实体 (从全量结果统计得到)
EXTENDED_VOCABULARY_BATCH4 = [
    "suitcase", "power strip", "crate", "guitar", "bowl", "coat rack",
    "piano", "washing machine", "bucket", "exhaust fan", "cutting board",
    "paper", "column", "doorframe", "wall", "floor",
    "mouse", "mat", "iron", "hanger", "rack", "cart", "tray",
    "mop", "broom", "scale", "tube", "rope", "wire", "cord",
]
# 全部词汇（用于名称匹配）
ALL_VOCABULARY_BATCHES = [EXTENDED_VOCABULARY_BATCH1, EXTENDED_VOCABULARY_BATCH2,
                          EXTENDED_VOCABULARY_BATCH3, EXTENDED_VOCABULARY_BATCH4]
EXTENDED_VOCABULARY = EXTENDED_VOCABULARY_BATCH1 + EXTENDED_VOCABULARY_BATCH2 + EXTENDED_VOCABULARY_BATCH3 + EXTENDED_VOCABULARY_BATCH4

# 标定物参考尺寸 (m) - v6扩展
CALIBRATION_OBJECTS = {
    'door': 2.0, 'chair': 0.80, 'dining chair': 0.80, 'office chair': 1.0,
    'bed': 2.0, 'toilet': 0.40, 'refrigerator': 1.75, 'fridge': 1.75,
    'sofa': 0.85, 'couch': 0.85, 'table': 0.75, 'dining table': 0.75,
    'desk': 0.75, 'window': 1.2, 'sink': 0.85, 'stove': 0.9,
    'bathtub': 0.6, 'tv': 0.6, 'television': 0.6, 'monitor': 0.5,
    'bookshelf': 1.8, 'cabinet': 1.0, 'nightstand': 0.6, 'lamp': 0.5,
    'washer': 0.85, 'dryer': 0.85, 'microwave': 0.35, 'oven': 0.6,
    'fireplace': 1.0, 'shower': 1.8, 'mirror': 0.8,
    'washing machine': 0.85, 'piano': 1.0, 'guitar': 1.0, 'suitcase': 0.65,
    'bucket': 0.35, 'crate': 0.5, 'coat rack': 1.7, 'column': 2.5,
}


def find_video_path(scene_name: str) -> Optional[str]:
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


# ============================================================================
# 64³ Grid 数据结构
# ============================================================================

@dataclass
class GridEntity:
    entity_id: str
    category: str
    grid_position: Tuple[int, int, int]  # (x, y, z) in [0, 63]
    position_3d: np.ndarray              # 世界坐标 (m)
    size_3d: np.ndarray = None           # (width, height, depth) in meters
    confidence: float = 0.5
    first_seen_frame: int = 0
    count_in_frame: int = 1
    detections: List[Dict] = field(default_factory=list)
    
    # === Mind-of-Thought: Belief Probing + Uncertainty Fields ===
    position_cov: np.ndarray = None      # 3x3 协方差矩阵
    position_uncertainty: float = 1.0    # 标量 uncertainty
    size_uncertainty: float = 1.0
    depth_std_mean: float = 0.0
    obs_count: int = 0
    support_frames: List[int] = field(default_factory=list)
    frame_confidences: Dict[int, float] = field(default_factory=dict)
    
    def summarize_detections(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Mind-of-Thought: 带权聚合多帧检测结果
        权重 = confidence / uncertainty^2
        """
        if not detections:
            return None
        
        # 计算权重
        weights = []
        positions = []
        sizes = []
        confidences = []
        
        for det in detections:
            u = max(det.get('position_uncertainty', 1.0), 1e-4)
            conf = det.get('confidence', 0.5)
            w = conf / (u ** 2)
            weights.append(w)
            positions.append(det['position_3d'])
            sizes.append(det.get('size_3d', np.array([0.1, 0.1, 0.1])))
            confidences.append(conf)
        
        weights = np.array(weights)
        positions = np.array(positions)
        
        # 加权平均位置
        total_weight = np.sum(weights)
        if total_weight < 1e-6:
            return None
        
        mean_pos = np.sum(positions * weights[:, None], axis=0) / total_weight
        
        # 加权协方差
        diff = positions - mean_pos
        cov = np.dot(diff.T * weights, diff) / total_weight
        
        # 标量 uncertainty (trace of cov)
        scalar_unc = float(np.trace(cov)) + 1e-4
        
        return {
            'position_3d': mean_pos,
            'position_cov': cov,
            'position_uncertainty': scalar_unc,
            'confidence': float(np.mean(confidences)),
        }


class Grid64:
    """64×64×64 场景Grid - 从DA3多视图构建
    
    核心概念: meters_per_grid
    - 每个grid格子代表的物理尺寸(米)
    - 所有距离/尺寸计算统一基于此值
    - 通过标定物+常识物体校准此值
    - SET_SCALE Evolution指令本质上就是修改此值
    
    Grid物理尺寸 = GRID_SIZE × meters_per_grid
    两点物理距离 = grid距离 × meters_per_grid
    """
    
    GRID_SIZE = 64
    
    def __init__(self):
        self.entities: Dict[str, GridEntity] = {}
        self.scene_min = None      # DA3原始坐标边界
        self.scene_max = None
        self.camera_positions: List[Dict] = []
        # 统一尺度: 每格代表多少米 (校准前=DA3原始尺度/64, 校准后=真实物理尺度/64)
        self._meters_per_grid: float = None  # None表示未校准，用raw计算
        self._raw_meters_per_grid: float = None  # DA3原始尺度
        self.calibration_log: List[str] = []  # 校准过程日志
    
    def set_scene_bounds(self, all_world_points: np.ndarray):
        """从点云设置场景边界 (用百分位数避免异常值)"""
        valid = all_world_points[~np.any(np.isnan(all_world_points) | np.isinf(all_world_points), axis=1)]
        self.scene_min = np.percentile(valid, 2, axis=0)
        self.scene_max = np.percentile(valid, 98, axis=0)
        for i in range(3):
            if self.scene_max[i] - self.scene_min[i] < 0.5:
                center = (self.scene_max[i] + self.scene_min[i]) / 2
                self.scene_min[i] = center - 0.5
                self.scene_max[i] = center + 0.5
        # 计算原始的meters_per_grid (DA3尺度)
        raw_range = float(np.max(self.scene_max - self.scene_min))
        self._raw_meters_per_grid = raw_range / self.GRID_SIZE
    
    @property
    def meters_per_grid(self) -> float:
        """每格代表的物理尺寸(米) - 校准后的值"""
        if self._meters_per_grid is not None:
            return self._meters_per_grid
        if self._raw_meters_per_grid is not None:
            return self._raw_meters_per_grid
        return 0.08  # 默认~5m房间
    
    @meters_per_grid.setter
    def meters_per_grid(self, value: float):
        """SET_SCALE: 直接设置每格的物理尺寸"""
        self._meters_per_grid = value
        self.calibration_log.append(f"SET_SCALE: meters_per_grid={value:.4f}m")
    
    @property
    def scale_correction_factor(self) -> float:
        """DA3原始尺度到校准尺度的修正系数"""
        if self._raw_meters_per_grid and self._raw_meters_per_grid > 1e-8:
            return self.meters_per_grid / self._raw_meters_per_grid
        return 1.0
    
    @property
    def scene_physical_size(self) -> np.ndarray:
        """场景在三个维度的物理尺寸(米), 校准后"""
        return np.array([self.GRID_SIZE * self.meters_per_grid] * 3)
    
    def world_to_grid(self, pos_3d: np.ndarray) -> Tuple[int, int, int]:
        if self.scene_min is None:
            return (32, 32, 32)
        normalized = (pos_3d - self.scene_min) / (self.scene_max - self.scene_min + 1e-6)
        normalized = np.clip(normalized, 0, 1)
        grid_coord = (normalized * (self.GRID_SIZE - 1)).astype(int)
        grid_coord = np.clip(grid_coord, 0, self.GRID_SIZE - 1)
        return tuple(grid_coord)
    
    def grid_to_physical(self, grid_pos: Tuple[int, int, int]) -> np.ndarray:
        """Grid坐标→物理坐标(米), 基于meters_per_grid"""
        g = np.array(grid_pos, dtype=float)
        return g * self.meters_per_grid
    
    def physical_distance(self, id1: str, id2: str) -> Optional[float]:
        """两Entity间的物理距离(m) - 用原始position_3d × scale_correction_factor"""
        if id1 not in self.entities or id2 not in self.entities:
            return None
        e1, e2 = self.entities[id1], self.entities[id2]
        # 用DA3原始世界坐标的距离 × 统一修正系数
        raw_dist = float(np.linalg.norm(e1.position_3d - e2.position_3d))
        return raw_dist * self.scale_correction_factor
    
    def physical_size(self, entity_id: str) -> Optional[float]:
        """Entity的物理尺寸(m) - 基于meters_per_grid统一校准"""
        if entity_id not in self.entities:
            return None
        entity = self.entities[entity_id]
        if entity.size_3d is None:
            return None
        # size_3d是DA3原始尺度，乘以校准修正系数
        raw_size = max(float(entity.size_3d[0]), float(entity.size_3d[1]))
        return raw_size * self.scale_correction_factor
    
    def get_by_category(self, target: str) -> List[GridEntity]:
        return [e for e in self.entities.values() if _match_name(target, e.category)]
    
    def calibrate_scale(self):
        """统一Grid级别尺度校准 (v6: 信任标定物，不再scene_clamp覆盖)
        
        核心思路: 标定物factor = known_size / da3_estimated_size
        这个factor直接代表DA3尺度到真实物理尺度的全局修正系数。
        
        calibrated_mpg = raw_mpg * correction_factor
        所有物理量(距离/尺寸/面积)都通过 scale_correction_factor 统一修正。
        
        多层校准策略:
        1. 标定物校准 (最可靠): 用中位数factor
        2. 常识物体约束: 作为fallback或validation
        3. 场景级约束: 仅在没有标定物时使用
        """
        self.calibration_log = []
        
        if self._raw_meters_per_grid is None:
            self.calibration_log.append("NO_BOUNDS: scene bounds not set")
            return
        
        raw_mpg = self._raw_meters_per_grid
        self.calibration_log.append(f"RAW: meters_per_grid={raw_mpg:.4f} (scene_range={raw_mpg*64:.2f}m)")
        
        # =====================================================================
        # 第1层: 标定物校准 (最可靠)
        # factor = known_physical_size / da3_estimated_size
        # 这是全局统一修正系数
        # =====================================================================
        calibration_factors = []
        calibration_details = []
        
        for eid, entity in self.entities.items():
            for cal_name, cal_size in CALIBRATION_OBJECTS.items():
                if _match_name(cal_name, entity.category):
                    if entity.size_3d is not None:
                        est_size_raw = max(float(entity.size_3d[0]), float(entity.size_3d[1]))
                        if est_size_raw > 0.001:
                            factor = cal_size / est_size_raw
                            if 0.01 < factor < 1000:
                                calibration_factors.append(factor)
                                calibration_details.append(
                                    f"  {eid}({cal_name}): est={est_size_raw:.3f} known={cal_size:.2f} → factor={factor:.3f}"
                                )
                    break
        
        correction_factor = None
        
        if len(calibration_factors) >= 3:
            # 有足够标定物: 用中位数，直接信任，不做scene_clamp
            correction_factor = float(np.median(calibration_factors))
            self.calibration_log.append(
                f"CALIBRATION_OBJECTS: {len(calibration_factors)} objects (TRUSTED), "
                f"median_factor={correction_factor:.3f}, "
                f"range=[{min(calibration_factors):.3f}, {max(calibration_factors):.3f}]"
            )
            for detail in calibration_details[:5]:
                self.calibration_log.append(detail)
        elif calibration_factors:
            correction_factor = float(np.median(calibration_factors))
            self.calibration_log.append(
                f"CALIBRATION_OBJECTS: {len(calibration_factors)} objects (few), "
                f"median_factor={correction_factor:.3f}"
            )
        else:
            self.calibration_log.append("CALIBRATION_OBJECTS: none found")
        
        # =====================================================================
        # 第2层: 常识物体约束
        # =====================================================================
        COMMONSENSE_SIZES = {
            'door': (1.5, 2.5), 'chair': (0.4, 1.2), 'bed': (1.5, 2.5),
            'sofa': (1.2, 3.0), 'couch': (1.2, 3.0), 'table': (0.5, 2.5),
            'desk': (0.5, 2.0), 'toilet': (0.3, 0.7), 'refrigerator': (1.2, 2.0),
            'fridge': (1.2, 2.0), 'bathtub': (1.0, 2.0), 'window': (0.6, 2.0),
            'sink': (0.3, 1.0), 'stove': (0.5, 1.0), 'tv': (0.3, 1.5),
            'bookshelf': (0.8, 2.5), 'cabinet': (0.5, 2.0), 'nightstand': (0.3, 0.8),
            'washer': (0.6, 1.0), 'dryer': (0.6, 1.0), 'microwave': (0.2, 0.5),
            'oven': (0.4, 0.8), 'mirror': (0.3, 1.5), 'lamp': (0.2, 1.8),
            'monitor': (0.3, 0.8), 'pillow': (0.2, 0.7), 'rug': (0.5, 4.0),
            'curtain': (1.0, 3.0), 'shower': (0.8, 1.2),
        }
        
        commonsense_factors = []
        for eid, entity in self.entities.items():
            for cs_name, (min_s, max_s) in COMMONSENSE_SIZES.items():
                if _match_name(cs_name, entity.category):
                    if entity.size_3d is not None:
                        est_raw = max(float(entity.size_3d[0]), float(entity.size_3d[1]))
                        if est_raw > 0.001:
                            typical = (min_s + max_s) / 2
                            cs_factor = typical / est_raw
                            if 0.005 < cs_factor < 5000:
                                commonsense_factors.append(cs_factor)
                    break
        
        if commonsense_factors:
            cs_median = float(np.median(commonsense_factors))
            self.calibration_log.append(
                f"COMMONSENSE: {len(commonsense_factors)} objects, "
                f"median_factor={cs_median:.3f}"
            )
            if correction_factor is None:
                correction_factor = cs_median
                self.calibration_log.append("USING commonsense as primary calibration")
        
        # =====================================================================
        # 第3层: 场景级约束
        # - 无标定物时: 直接用场景约束
        # - 有标定物时: soft clamp (加权平均，不hard override)
        # =====================================================================
        if correction_factor is None:
            raw_span = raw_mpg * self.GRID_SIZE
            if raw_span > 15.0:
                correction_factor = 7.0 / raw_span
            elif raw_span < 2.0:
                correction_factor = 5.0 / raw_span
            else:
                correction_factor = 1.0
            self.calibration_log.append(
                f"SCENE_FALLBACK: raw_span={raw_span:.2f}m, "
                f"correction_factor={correction_factor:.3f}"
            )
        else:
            # Soft clamp: 标定物校准后，如果scene_span极端不合理，做加权修正
            calibrated_span = raw_mpg * correction_factor * self.GRID_SIZE
            if calibrated_span > 25.0:
                # 极端偏大: 用加权平均拉向10m目标
                target_factor = 10.0 / (raw_mpg * self.GRID_SIZE)
                # weight: span越大，越倾向target (sigmoid-like)
                excess = (calibrated_span - 25.0) / 25.0  # 0~几
                w = min(0.8, excess)  # 最多80%权重给target
                old_factor = correction_factor
                correction_factor = correction_factor * (1 - w) + target_factor * w
                self.calibration_log.append(
                    f"SOFT_CLAMP_MAX: span={calibrated_span:.1f}m>25m, "
                    f"w={w:.2f}, factor: {old_factor:.3f}→{correction_factor:.3f}"
                )
            elif calibrated_span < 3.0:
                target_factor = 5.0 / (raw_mpg * self.GRID_SIZE)
                deficit = (3.0 - calibrated_span) / 3.0
                w = min(0.7, deficit)
                old_factor = correction_factor
                correction_factor = correction_factor * (1 - w) + target_factor * w
                self.calibration_log.append(
                    f"SOFT_CLAMP_MIN: span={calibrated_span:.1f}m<3m, "
                    f"w={w:.2f}, factor: {old_factor:.3f}→{correction_factor:.3f}"
                )
        
        # 应用校准
        self._meters_per_grid = raw_mpg * correction_factor
        final_span = self._meters_per_grid * self.GRID_SIZE
        self.calibration_log.append(
            f"FINAL: meters_per_grid={self._meters_per_grid:.4f}m, "
            f"scene_span={final_span:.2f}m, "
            f"correction={correction_factor:.3f}"
        )
        
        logger.info(
            f"Scale calibrated: mpg={self._meters_per_grid:.4f}m "
            f"(scene={final_span:.1f}m, raw={raw_mpg*64:.1f}m, "
            f"correction={correction_factor:.3f}, "
            f"cal_objects={len(calibration_factors)}, cs_objects={len(commonsense_factors)})"
        )
    
    def get_relative_direction(self, observer_id: str, facing_id: str, target_id: str) -> str:
        """站在observer面向facing，target在哪个方位
        注意: 方向判断只需要grid坐标的相对关系，不受meters_per_grid影响
        """
        if observer_id not in self.entities or facing_id not in self.entities or target_id not in self.entities:
            return "unknown"
        
        obs = np.array(self.entities[observer_id].grid_position, dtype=float)
        fac = np.array(self.entities[facing_id].grid_position, dtype=float)
        tgt = np.array(self.entities[target_id].grid_position, dtype=float)
        
        # 面向向量 (xz平面, grid空间)
        facing_dir = np.array([fac[0] - obs[0], fac[2] - obs[2]])
        if np.linalg.norm(facing_dir) < 1e-6:
            facing_dir = np.array([0.0, 1.0])
        facing_dir = facing_dir / np.linalg.norm(facing_dir)
        
        target_dir = np.array([tgt[0] - obs[0], tgt[2] - obs[2]])
        if np.linalg.norm(target_dir) < 1e-6:
            return "same_position"
        
        forward = np.dot(target_dir, facing_dir)
        right = facing_dir[0] * target_dir[1] - facing_dir[1] * target_dir[0]
        
        parts = []
        if forward >= 0:
            parts.append("front")
        else:
            parts.append("behind")
        if right >= 0:
            parts.append("right")
        else:
            parts.append("left")
        return "-".join(parts)
    
    def to_text(self) -> str:
        """Grid文本表示 (给VL看)"""
        lines = [f"=== 64³ Grid Mind Map ==="]
        lines.append(f"meters_per_grid: {self.meters_per_grid:.4f}m (scene_span≈{self.meters_per_grid*64:.1f}m)")
        lines.append(f"Entities ({len(self.entities)}):")
        for eid, e in sorted(self.entities.items()):
            gp = e.grid_position
            phys = self.grid_to_physical(gp)
            size_str = ""
            if e.size_3d is not None:
                ps = self.physical_size(eid)
                if ps:
                    size_str = f", phys_size={ps:.2f}m"
            lines.append(f"  - {eid}: category={e.category}, grid=({gp[0]},{gp[1]},{gp[2]}), "
                         f"pos=({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m{size_str}, "
                         f"count={e.count_in_frame}, conf={e.confidence:.2f}, "
                         f"first_frame={e.first_seen_frame}")
        if self.calibration_log:
            lines.append(f"Calibration: {self.calibration_log[-1]}")
        return "\n".join(lines)


# ============================================================================
# 名称匹配
# ============================================================================

SYNONYMS = {
    'sofa': ['couch'], 'couch': ['sofa'],
    'tv': ['television', 'monitor', 'screen'], 'television': ['tv', 'screen'],
    'fridge': ['refrigerator'], 'refrigerator': ['fridge'],
    'desk': ['table'], 'stool': ['chair'],
    'trash': ['bin', 'garbage', 'trashcan', 'trash can'], 'bin': ['trash', 'garbage'],
    'rug': ['carpet', 'mat'], 'carpet': ['rug', 'mat'],
    'wardrobe': ['closet', 'cabinet'], 'closet': ['wardrobe'],
    'backpack': ['bag'], 'bag': ['backpack'],
    'telephone': ['phone'], 'phone': ['telephone'],
    'light': ['lamp', 'ceiling light', 'chandelier'], 'lamp': ['light', 'ceiling light'],
    'ceiling light': ['light', 'lamp', 'chandelier'],
    'fan': ['exhaust fan'], 'exhaust fan': ['fan'],
    'picture': ['painting', 'art'], 'painting': ['picture'],
    'cushion': ['pillow'], 'pillow': ['cushion'],
    'nightstand': ['bedside table', 'night stand'], 'bedside table': ['nightstand'],
    'bathtub': ['tub'], 'tub': ['bathtub'],
    'monitor': ['screen', 'tv', 'display'], 'screen': ['monitor', 'tv'],
    'laptop': ['computer', 'notebook'], 'computer': ['laptop', 'pc'],
    'cup': ['mug', 'glass'], 'mug': ['cup'],
    'shelf': ['bookshelf', 'shelving'], 'bookshelf': ['shelf'],
    'dresser': ['drawer', 'chest'], 'drawer': ['dresser'],
    'whiteboard': ['board'], 'board': ['whiteboard'],
    'heater': ['radiator'], 'radiator': ['heater'],
    'chandelier': ['ceiling light', 'light'],
    'trash can': ['bin', 'trash bin', 'garbage'], 'trash bin': ['trash can', 'bin'],
    'washer': ['washing machine'], 'washing machine': ['washer'],
    'mouse': ['computer mouse'], 'computer mouse': ['mouse'],
    'coat rack': ['rack', 'hanger'], 'doorframe': ['door frame', 'door'],
    'power strip': ['power'], 'mat': ['rug', 'carpet'],
}

def _match_name(target: str, label: str) -> bool:
    target = target.lower().strip()
    label = label.lower().strip()
    if target == label or target in label or label in target:
        return True
    tw = set(target.replace('_', ' ').replace('-', ' ').split())
    lw = set(label.replace('_', ' ').replace('-', ' ').split())
    if tw & lw:
        return True
    for t in tw:
        if t in SYNONYMS and any(s in lw for s in SYNONYMS[t]):
            return True
    return False


# ============================================================================
# Grid 构建器 (DA3 + GroundingDINO)
# ============================================================================

class Grid64Builder:
    """用DA3多视图推理 + GroundingDINO构建64³ Grid"""
    GRID_CLASS = Grid64
    
    def __init__(self, device='cuda', num_frames=16):
        self.device = device
        self.num_frames = num_frames
        self._da3 = None
        self._labeler = None
    
    def load_models(self):
        if self._da3 is None:
            from core.perception_da3_full import DA3FullEstimator
            self._da3 = DA3FullEstimator(
                model_name="da3nested-giant-large",
                device=self.device,
                use_ray_pose=True,
            )
            logger.info("DA3 loaded")
        
        if self._labeler is None:
            from core.semantic_labeler import GroundingDINOLabeler
            self._labeler = GroundingDINOLabeler(
                model_id="IDEA-Research/grounding-dino-base",
                device=self.device,
                box_threshold=0.25,
                text_threshold=0.25,
            )
            self._labeler.load_model()
            logger.info("GroundingDINO loaded")
    
    def unload(self):
        if self._da3 is not None:
            del self._da3.model
            self._da3 = None
        if self._labeler is not None:
            try:
                del self._labeler.model
                del self._labeler.processor
            except:
                pass
            self._labeler = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def build_grid(self, video_path: str, target_objects: List[str] = None) -> Grid64:
        """从视频构建64³ Grid
        
        构建完成后，DA3结果和采样帧缓存在:
          self._cached_da3_pred   — DA3FullPrediction (depth_maps, intrinsics, extrinsics)
          self._cached_frames     — List[np.ndarray] RGB采样帧
          self._cached_proc_shape — (proc_H, proc_W) DA3处理分辨率
          self._cached_orig_shape — (orig_H, orig_W) 原始帧分辨率
        供ADD Evolution复用。
        """
        self.load_models()
        grid = self.GRID_CLASS()
        
        # 清空缓存
        self._cached_da3_pred = None
        self._cached_frames = None
        self._cached_proc_shape = None
        self._cached_orig_shape = None
        
        # 1. 采样视频帧
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return grid
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        sampled_frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if len(sampled_frames) < 2:
            return grid
        
        # 2. DA3 多视图推理
        logger.info(f"DA3 inference: {len(sampled_frames)} frames")
        da3_pred = self._da3.estimate_multiview(sampled_frames, ref_view_strategy="saddle_balanced")
        
        depth_maps = da3_pred.depth_maps
        intrinsics = da3_pred.intrinsics
        extrinsics = da3_pred.extrinsics
        proc_H, proc_W = depth_maps.shape[1:3]
        
        # 缓存DA3结果和帧供ADD Evolution复用
        self._cached_da3_pred = da3_pred
        self._cached_frames = sampled_frames
        self._cached_proc_shape = (proc_H, proc_W)
        self._cached_orig_shape = (sampled_frames[0].shape[0], sampled_frames[0].shape[1])
        
        # 3. 相机位置提取 (w2c约定: cam = R @ world + t)
        #    相机中心: R @ cam_center + t = 0 → cam_center = -R^T @ t
        for i in range(len(extrinsics)):
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            cam_pos = -R.T @ t  # w2c → 相机世界坐标
            view_dir = R.T @ np.array([0, 0, 1.0])  # 相机Z轴在世界坐标系中的方向
            grid.camera_positions.append({
                'frame_idx': int(frame_indices[i]) if i < len(frame_indices) else i,
                'world_pos': cam_pos,
                'view_dir': view_dir / (np.linalg.norm(view_dir) + 1e-6),
            })
        
        # 5. 物体检测 + 3D定位 (分批检测避免prompt过长)
        extra = target_objects or []
        vocab_batches = [list(set(extra + ALL_VOCABULARY_BATCHES[0]))] + ALL_VOCABULARY_BATCHES[1:]
        
        all_detections = defaultdict(list)
        
        for i, frame_rgb in enumerate(sampled_frames):
            orig_H, orig_W = frame_rgb.shape[:2]
            
            # 分批检测
            frame_detections = []
            for batch in vocab_batches:
                prompt = " . ".join(batch) + " ."
                dets = self._labeler.detect(frame_rgb, prompt)
                frame_detections.extend(dets)
            
            for det in frame_detections:
                raw_label = det.label.strip().lower()
                if raw_label.startswith('##'):
                    continue
                
                bbox = det.bbox_pixels
                conf = det.confidence
                
                # 将原始坐标映射到DA3处理后的尺寸
                scale_x = proc_W / orig_W
                scale_y = proc_H / orig_H
                bbox_scaled = (
                    bbox[0] * scale_x, bbox[1] * scale_y,
                    bbox[2] * scale_x, bbox[3] * scale_y,
                )
                
                # 计算3D世界坐标 (使用bbox区域中值深度)
                x1_int = int(np.clip(bbox_scaled[0], 0, proc_W - 1))
                y1_int = int(np.clip(bbox_scaled[1], 0, proc_H - 1))
                x2_int = int(np.clip(bbox_scaled[2], 0, proc_W - 1))
                y2_int = int(np.clip(bbox_scaled[3], 0, proc_H - 1))
                
                depth_region = depth_maps[i, y1_int:y2_int+1, x1_int:x2_int+1]
                if depth_region.size > 0:
                    depth = np.median(depth_region)
                    # Mind-of-Thought: 计算深度区域的 std 作为 uncertainty 指标
                    valid_depths = depth_region[depth_region > 0.01]
                    depth_std = float(np.std(valid_depths)) if len(valid_depths) > 1 else 1.0
                else:
                    cx_int = int((x1_int + x2_int) / 2)
                    cy_int = int((y1_int + y2_int) / 2)
                    depth = depth_maps[i, cy_int, cx_int]
                    depth_std = 1.0  # 回退到高 uncertainty
                
                if depth < 0.01:
                    continue
                
                # Mind-of-Thought: 计算 position_uncertainty
                # u = 0.5 * depth_std + 0.5 * (1.0 - confidence)
                position_uncertainty = 0.5 * depth_std + 0.5 * (1.0 - float(conf))
                
                # 像素 → 相机坐标
                center_u = (bbox_scaled[0] + bbox_scaled[2]) / 2
                center_v = (bbox_scaled[1] + bbox_scaled[3]) / 2
                K = intrinsics[i]
                fx, fy = K[0, 0], K[1, 1]
                cx_k, cy_k = K[0, 2], K[1, 2]
                
                x_cam = (center_u - cx_k) / fx * depth
                y_cam = (center_v - cy_k) / fy * depth
                z_cam = depth
                cam_point = np.array([x_cam, y_cam, z_cam])
                
                # 相机 → 世界 (w2c: cam = R @ world + t → world = R^T @ cam - R^T @ t)
                R = extrinsics[i, :3, :3]
                t_vec = extrinsics[i, :3, 3]
                world_point = R.T @ cam_point - R.T @ t_vec
                
                # 3D尺寸
                w_px = bbox_scaled[2] - bbox_scaled[0]
                h_px = bbox_scaled[3] - bbox_scaled[1]
                w_3d = w_px * depth / fx
                h_3d = h_px * depth / fy
                
                all_detections[raw_label].append({
                    'frame_idx': int(frame_indices[i]) if i < len(frame_indices) else i,
                    'frame_order': i,
                    'bbox': bbox,
                    'confidence': float(conf),
                    'position_3d': world_point,
                    'width_3d': float(w_3d),
                    'height_3d': float(h_3d),
                    'depth_value': float(depth),  # v4: 保存深度用于distance_scale校准
                    # Mind-of-Thought: 添加 uncertainty 信息
                    'depth_std': depth_std,
                    'position_uncertainty': position_uncertainty,
                })
        
        # 6. 聚合 → 先收集所有Entity的3D位置
        temp_entities = {}
        for label, dets in all_detections.items():
            entity = self._aggregate_to_entity(grid, label, dets)
            if entity is not None:
                temp_entities[entity.entity_id] = entity
        
        if not temp_entities:
            return grid
        
        # 7. 用entities + camera positions设置场景边界（更鲁棒）
        all_positions = [e.position_3d for e in temp_entities.values()]
        # 加入camera positions，帮助确定场景范围
        for cam in grid.camera_positions:
            all_positions.append(cam['world_pos'])
        all_positions = np.array(all_positions)
        grid.set_scene_bounds(all_positions)
        
        # 8. 重新分配Grid坐标
        for eid, entity in temp_entities.items():
            entity.grid_position = grid.world_to_grid(entity.position_3d)
            grid.entities[eid] = entity
        
        # 9. 校准尺度
        grid.calibrate_scale()
        
        logger.info(f"Grid64 built: {len(grid.entities)} entities, mpg={grid.meters_per_grid:.4f}m")
        return grid
    
    def _aggregate_to_entity(self, grid: Grid64, label: str, detections: List[Dict]) -> Optional[GridEntity]:
        if not detections:
            return None
        
        # 按帧统计数量
        frame_dets = defaultdict(list)
        for det in detections:
            frame_dets[det['frame_idx']].append(det)
        counts_per_frame = [len(fd) for fd in frame_dets.values()]
        # 用中位数帧计数（比max更鲁棒，抑制32帧下少数帧过度检测导致的overcount）
        max_count = max(1, int(np.median(counts_per_frame)))
        
        # Mind-of-Thought: 使用 summarize_detections 进行带权聚合
        # 创建临时 entity 来调用 summarize_detections
        temp_entity = GridEntity(
            entity_id="temp",
            category=label,
            grid_position=(0, 0, 0),
            position_3d=np.array([0.0, 0.0, 0.0]),
            detections=detections,
        )
        summary = temp_entity.summarize_detections(detections)
        
        if summary is None:
            # 回退到原始中位数方法
            positions = np.array([d['position_3d'] for d in detections])
            avg_pos = np.median(positions, axis=0)
            avg_conf = np.mean([d['confidence'] for d in detections])
            position_uncertainty = 1.0
            position_cov = None
        else:
            avg_pos = summary['position_3d']
            avg_conf = summary['confidence']
            position_uncertainty = summary['position_uncertainty']
            position_cov = summary['position_cov']
        
        best_det = max(detections, key=lambda x: x['confidence'])
        size_3d = np.array([best_det['width_3d'], best_det['height_3d'], 0.3])
        first_frame = min(d['frame_idx'] for d in detections)
        
        # 计算平均 depth_std
        depth_std_mean = np.mean([d.get('depth_std', 1.0) for d in detections])
        
        # 收集 support frames 和 frame confidences
        support_frames = sorted(set(d['frame_idx'] for d in detections))
        frame_confidences = {d['frame_idx']: d['confidence'] for d in detections}
        
        # 检查位置有效性
        if np.any(np.isnan(avg_pos)) or np.any(np.isinf(avg_pos)):
            return None
        
        grid_pos = grid.world_to_grid(avg_pos)
        eid = label.replace(' ', '_')
        
        return GridEntity(
            entity_id=eid,
            category=label,
            grid_position=grid_pos,
            position_3d=avg_pos,
            size_3d=size_3d,
            confidence=float(avg_conf),
            first_seen_frame=first_frame,
            count_in_frame=max_count,
            detections=detections,
            # Mind-of-Thought: 添加 uncertainty 字段
            position_cov=position_cov,
            position_uncertainty=position_uncertainty,
            depth_std_mean=float(depth_std_mean),
            obs_count=len(detections),
            support_frames=support_frames,
            frame_confidences=frame_confidences,
        )

    def search_and_add_entity(self, grid: Grid64, entity_name: str) -> Optional[GridEntity]:
        """ADD Evolution: 针对性搜索缺失entity并添加到Grid
        
        利用缓存的DA3结果(depth_maps, intrinsics, extrinsics)和采样帧,
        调用GroundingDINO针对性搜索指定entity, 做2D→3D投影, 添加到Grid中.
        
        Returns:
            添加的GridEntity, 或None(未找到)
        """
        if self._cached_da3_pred is None or self._cached_frames is None:
            logger.warning(f"  ADD: no cached DA3 results, cannot search for '{entity_name}'")
            return None
        
        if self._labeler is None:
            logger.warning(f"  ADD: GroundingDINO not loaded")
            return None
        
        da3_pred = self._cached_da3_pred
        frames = self._cached_frames
        proc_H, proc_W = self._cached_proc_shape
        orig_H, orig_W = self._cached_orig_shape
        
        depth_maps = da3_pred.depth_maps
        intrinsics = da3_pred.intrinsics
        extrinsics = da3_pred.extrinsics
        
        scale_x = proc_W / orig_W
        scale_y = proc_H / orig_H
        
        # 针对性搜索: 用entity_name作为GroundingDINO prompt
        search_prompt = entity_name.strip() + " ."
        
        all_dets = []
        for i, frame_rgb in enumerate(frames):
            dets = self._labeler.detect(frame_rgb, search_prompt)
            for det in dets:
                raw_label = det.label.strip().lower()
                bbox = det.bbox_pixels
                conf = det.confidence
                
                # 只接受和目标name匹配的检测
                if not _match_name(entity_name, raw_label):
                    continue
                
                # 坐标映射到DA3处理尺寸
                bbox_scaled = (
                    bbox[0] * scale_x, bbox[1] * scale_y,
                    bbox[2] * scale_x, bbox[3] * scale_y,
                )
                
                # 中值深度
                x1_int = int(np.clip(bbox_scaled[0], 0, proc_W - 1))
                y1_int = int(np.clip(bbox_scaled[1], 0, proc_H - 1))
                x2_int = int(np.clip(bbox_scaled[2], 0, proc_W - 1))
                y2_int = int(np.clip(bbox_scaled[3], 0, proc_H - 1))
                
                depth_region = depth_maps[i, y1_int:y2_int+1, x1_int:x2_int+1]
                if depth_region.size > 0:
                    depth = np.median(depth_region)
                else:
                    cx_int = int((x1_int + x2_int) / 2)
                    cy_int = int((y1_int + y2_int) / 2)
                    depth = depth_maps[i, cy_int, cx_int]
                
                if depth < 0.01:
                    continue
                
                # 像素 → 相机坐标
                center_u = (bbox_scaled[0] + bbox_scaled[2]) / 2
                center_v = (bbox_scaled[1] + bbox_scaled[3]) / 2
                K = intrinsics[i]
                fx, fy = K[0, 0], K[1, 1]
                cx_k, cy_k = K[0, 2], K[1, 2]
                
                x_cam = (center_u - cx_k) / fx * depth
                y_cam = (center_v - cy_k) / fy * depth
                z_cam = depth
                cam_point = np.array([x_cam, y_cam, z_cam])
                
                # 相机 → 世界
                R = extrinsics[i, :3, :3]
                t_vec = extrinsics[i, :3, 3]
                world_point = R.T @ cam_point - R.T @ t_vec
                
                # 3D尺寸
                w_px = bbox_scaled[2] - bbox_scaled[0]
                h_px = bbox_scaled[3] - bbox_scaled[1]
                w_3d = w_px * depth / fx
                h_3d = h_px * depth / fy
                
                all_dets.append({
                    'frame_idx': i,
                    'frame_order': i,
                    'bbox': bbox,
                    'confidence': float(conf),
                    'position_3d': world_point,
                    'width_3d': float(w_3d),
                    'height_3d': float(h_3d),
                    'depth_value': float(depth),
                })
        
        if not all_dets:
            logger.info(f"  ADD: '{entity_name}' not found in any frame")
            return None
        
        # 聚合: 多帧检测取position_3d中位数
        entity = self._aggregate_to_entity(grid, entity_name, all_dets)
        if entity is None:
            logger.info(f"  ADD: '{entity_name}' aggregation failed")
            return None
        
        # 更新grid_position (需要grid已有scene_bounds)
        if grid.scene_min is not None:
            entity.grid_position = grid.world_to_grid(entity.position_3d)
        
        # 添加到grid
        grid.entities[entity.entity_id] = entity
        
        logger.info(f"  ADD: '{entity_name}' added to grid at pos3d=({entity.position_3d[0]:.2f},{entity.position_3d[1]:.2f},{entity.position_3d[2]:.2f}), "
                     f"grid=({entity.grid_position[0]},{entity.grid_position[1]},{entity.grid_position[2]}), "
                     f"conf={entity.confidence:.2f}, seen_in={len(all_dets)} detections")
        
        return entity


# ============================================================================
# Grid工具回答各类问题 (确定性计算)
# ============================================================================

def grid_answer_counting(grid: Grid64, question: str) -> Tuple[str, str]:
    """v4: 改进counting - 更好的名称匹配（支持复数）"""
    # 匹配 "How many X" 或 "How many X(s/es)"
    match = re.search(r'How many (\w+(?:\s+\w+)*?)(?:\(s\)|\(es\))?\s+(?:are|is|do|does|can)', question, re.IGNORECASE)
    if not match:
        match = re.search(r'How many (\w+)', question, re.IGNORECASE)
    if not match:
        return "0", "no target"
    target = match.group(1).lower().strip()
    
    # 去掉复数后缀
    target_singular = target.rstrip('s')
    if target.endswith('es') and len(target) > 3:
        target_singular2 = target[:-2]
    else:
        target_singular2 = target_singular
    
    # 尝试多种匹配
    matched = grid.get_by_category(target)
    if not matched:
        matched = grid.get_by_category(target_singular)
    if not matched and target_singular2 != target_singular:
        matched = grid.get_by_category(target_singular2)
    
    if matched:
        best = max(matched, key=lambda e: e.count_in_frame)
        return str(best.count_in_frame), f"entity={best.entity_id} count={best.count_in_frame}"
    return "0", f"'{target}' not found"


def grid_answer_size(grid: Grid64, question: str) -> Tuple[str, str]:
    """v5: 尺寸估计 - 基于统一的meters_per_grid校准"""
    m = re.search(r'(?:size|height|length|width|dimension).*?(?:of|for)\s+(?:the\s+)?(.+?)[\?\.]', question.lower())
    if not m:
        m = re.search(r'of the (\w+)', question.lower())
    if not m:
        return "100", "no target"
    target = m.group(1).strip()
    
    matched = grid.get_by_category(target)
    if matched:
        entity = max(matched, key=lambda e: e.confidence)
        phys_size = grid.physical_size(entity.entity_id)
        if phys_size is not None and phys_size > 0:
            size_cm = int(round(phys_size * 100))
            # 合理性限制: 室内物体一般 5-500cm
            if 5 <= size_cm <= 500:
                return str(size_cm), f"entity={entity.entity_id} size={phys_size:.2f}m={size_cm}cm (mpg={grid.meters_per_grid:.4f})"
    
    # Fallback: 典型尺寸
    typical = {'chair':80,'table':150,'sofa':200,'bed':200,'tv':100,'door':200,'window':120,
               'toilet':60,'sink':50,'lamp':50,'bathtub':170,'refrigerator':180,'desk':120,
               'stove':65,'oven':65,'microwave':45,'pillow':40,'monitor':60,'nightstand':55,
               'bookshelf':180,'cabinet':150,'shelf':100,'counter':90,'curtain':200,
               'painting':60,'picture':50,'mirror':80,'plant':60,'rug':150,'carpet':200,
               'cup':10,'clock':30,'fan':40,'trash':50,'bin':50,'towel':60,'shoe':30,
               'fireplace':100,'blanket':200,'washer':85,'dryer':85,'vase':30,'bottle':25}
    for k, v in typical.items():
        if k in target.lower():
            return str(v), f"typical for {k}"
    return "80", "default"


def grid_answer_room_size(grid: Grid64, question: str) -> Tuple[str, str]:
    """v6b: room_size - 用entity+camera的raw位置范围 × scale_correction_factor
    
    entity位置只覆盖了有家具的区域，camera轨迹覆盖了人可以走到的区域。
    两者结合更能代表整个房间范围。
    """
    if not grid.entities:
        return "25", "no entities"
    
    # 收集entity + camera 位置
    all_positions = [e.position_3d for e in grid.entities.values()]
    if grid.camera_positions:
        for cam in grid.camera_positions:
            all_positions.append(cam['world_pos'])
    all_positions = np.array(all_positions)
    
    # 用百分位数避免异常值
    if len(all_positions) >= 10:
        p5 = np.percentile(all_positions, 5, axis=0)
        p95 = np.percentile(all_positions, 95, axis=0)
        ranges_raw = p95 - p5
    else:
        ranges_raw = np.ptp(all_positions, axis=0)
    
    # 乘以统一的scale_correction_factor
    scf = grid.scale_correction_factor
    physical_ranges = ranges_raw * scf
    
    # 取最大的两个维度作为地面面积（排除高度y）
    sorted_ranges = sorted(physical_ranges, reverse=True)
    
    # 最大两维 + 余量（1.0m each side for wall clearance）
    dim1 = sorted_ranges[0] + 2.0
    dim2 = sorted_ranges[1] + 2.0
    area = dim1 * dim2
    
    # 合理范围: 5-80m²
    area = max(5.0, min(area, 80.0))
    
    return str(int(round(area))), (
        f"area={area:.1f}m² (raw_ranges=[{ranges_raw[0]:.1f},{ranges_raw[1]:.1f},{ranges_raw[2]:.1f}]m, "
        f"phys=[{physical_ranges[0]:.1f},{physical_ranges[1]:.1f},{physical_ranges[2]:.1f}]m, "
        f"scf={scf:.3f}, mpg={grid.meters_per_grid:.4f})"
    )


def grid_answer_abs_distance(grid: Grid64, question: str) -> Tuple[str, str]:
    """v5: 绝对距离 - 基于Grid坐标 × meters_per_grid"""
    patterns = [
        r'distance between (?:the )?(.+?)\s+and\s+(?:the )?(.+?)[\s\(]',
        r'distance between (?:the )?(.+?)\s+and\s+(?:the )?(.+?)[\?\.]',
        r'from (?:the )?(\w+(?:\s+\w+)*) to (?:the )?(\w+(?:\s+\w+)*)',
        r'between (?:the )?(\w+) and (?:the )?(\w+)',
    ]
    
    obj1, obj2 = None, None
    for pattern in patterns:
        m = re.search(pattern, question.lower())
        if m:
            obj1, obj2 = m.group(1).strip(), m.group(2).strip()
            break
    
    if obj1 and obj2:
        e1_list = grid.get_by_category(obj1)
        e2_list = grid.get_by_category(obj2)
        if e1_list and e2_list:
            dist = grid.physical_distance(e1_list[0].entity_id, e2_list[0].entity_id)
            if dist is not None:
                # 室内距离一般 0.1-20m
                dist = max(0.1, min(dist, 20.0))
                return f"{dist:.2f}", f"dist({e1_list[0].entity_id},{e2_list[0].entity_id})={dist:.2f}m (mpg={grid.meters_per_grid:.4f})"
        else:
            missing = []
            if not e1_list:
                missing.append(f"'{obj1}'")
            if not e2_list:
                missing.append(f"'{obj2}'")
            return "2.0", f"not found: {', '.join(missing)}"
    return "2.0", "cannot parse distance question"


def grid_answer_direction(grid: Grid64, question: str, options: List[str]) -> Tuple[str, str]:
    """v5: Direction - 用Grid坐标计算(与meters_per_grid无关，只需相对位置)
    
    VSIBench方向题格式:
    - Easy: "If I am standing by X and facing Y, is Z to the left or the right of Y?"
    - Medium: "If I am standing by X and facing Y, is Z to my left, right, or back?"
    - Hard: "If I am standing by X and facing Y, is Z to my front-left, front-right, back-left, or back-right?"
    """
    if not options:
        return "A", "no options"
    
    q = question.lower()
    
    m = re.search(r'standing by (?:the )?(.+?)\s+and\s+facing (?:the )?(.+?),\s*is (?:the )?(.+?)\s+to\s', q)
    if not m:
        m = re.search(r'standing by (?:the )?(.+?)\s+and\s+facing (?:the )?(.+?),.*?(?:is|are)\s+(?:the )?(.+?)\s+(?:to|on)\s', q)
    
    if not m:
        return "A", f"cannot parse direction question"
    
    observer_name = m.group(1).strip()
    facing_name = m.group(2).strip()
    target_name = m.group(3).strip()
    
    e_obs = grid.get_by_category(observer_name)
    e_fac = grid.get_by_category(facing_name)
    e_tgt = grid.get_by_category(target_name)
    
    if not e_obs:
        return "A", f"observer '{observer_name}' not found"
    if not e_fac:
        return "A", f"facing '{facing_name}' not found"
    if not e_tgt:
        return "A", f"target '{target_name}' not found"
    
    # 用position_3d浮点世界坐标计算方位 — 比grid_position整数坐标精度更高
    obs_pos = np.array(e_obs[0].position_3d, dtype=float)
    fac_pos = np.array(e_fac[0].position_3d, dtype=float)
    tgt_pos = np.array(e_tgt[0].position_3d, dtype=float)
    
    # XZ平面 (世界坐标空间, Y轴通常是高度)
    facing_dir = np.array([fac_pos[0] - obs_pos[0], fac_pos[2] - obs_pos[2]])
    if np.linalg.norm(facing_dir) < 1e-8:
        return "A", "observer and facing at same 3d position"
    facing_dir = facing_dir / np.linalg.norm(facing_dir)
    
    target_dir = np.array([tgt_pos[0] - obs_pos[0], tgt_pos[2] - obs_pos[2]])
    if np.linalg.norm(target_dir) < 1e-8:
        return "A", "target at observer 3d position"
    
    forward = float(np.dot(target_dir, facing_dir))
    right = float(facing_dir[0] * target_dir[1] - facing_dir[1] * target_dir[0])
    
    is_front = forward >= 0
    is_right = right >= 0
    
    # Easy: left/right of facing object
    if "left or the right of" in q or "left or right of" in q:
        direction = "right" if is_right else "left"
    # Medium: left/right/back
    elif "left, right, or back" in q or "left, right or back" in q:
        # "back" = need to turn at least 135 degrees
        angle = np.arctan2(abs(right), forward)
        if not is_front and angle > np.pi * 0.5:
            direction = "back"
        elif is_right:
            direction = "right"
        else:
            direction = "left"
    # Hard: front-left/front-right/back-left/back-right
    else:
        fb = "front" if is_front else "back"
        lr = "right" if is_right else "left"
        direction = f"{fb}-{lr}"
    
    reasoning = f"obs={observer_name} fac={facing_name} tgt={target_name} | fwd={forward:.3f} right={right:.3f} → {direction} [pos3d]"
    
    # 匹配选项
    best_match = "A"
    best_score = -1
    
    for opt in options:
        letter = opt[0]
        opt_text = opt[2:].strip().lower() if len(opt) > 2 else ""
        
        score = 0
        # 精确匹配
        if direction in opt_text:
            score += 10
        # 部分匹配
        for kw in ["left", "right", "front", "back", "behind"]:
            if kw in direction and kw in opt_text:
                score += 2
            elif kw in opt_text and kw not in direction:
                score -= 2
        
        if score > best_score:
            best_score = score
            best_match = letter
    
    return best_match, reasoning


def grid_answer_rel_distance(grid: Grid64, question: str, options: List[str]) -> Tuple[str, str]:
    """Grid工具: 相对距离比较
    
    VSIBench格式:
    "which of these objects (A, B, C, D) is the closest to the X?"
    "which of these objects (A, B, C, D) is the farthest from the X?"
    """
    if not options:
        return "A", "no options"
    
    q = question.lower()
    is_farthest = 'farthest' in q or 'furthest' in q or 'farther' in q or 'further' in q
    
    # 解析参考物体
    m = re.search(r'(?:closest|nearest|farthest|furthest)\s+(?:to|from)\s+(?:the\s+)?(.+?)[\?\.]', q)
    if not m:
        return "A", "cannot parse rel_distance"
    ref_name = m.group(1).strip()
    
    # 从选项提取候选物体（而不是从括号里的A,B,C,D）
    candidates = []
    for opt in options:
        cand_name = re.sub(r'^[A-D]\.\s*', '', opt).strip().lower()
        candidates.append((opt[0], cand_name))  # (letter, name)
    
    # 找到参考物体
    e_ref = grid.get_by_category(ref_name)
    if not e_ref:
        return "A", f"ref '{ref_name}' not found"
    
    # 计算每个候选到ref的距离
    cand_distances = []
    for letter, cand_name in candidates:
        e_cand = grid.get_by_category(cand_name)
        if e_cand:
            d = grid.physical_distance(e_ref[0].entity_id, e_cand[0].entity_id)
            if d is not None:
                cand_distances.append((letter, cand_name, d))
    
    if not cand_distances:
        return "A", f"no candidates found in grid (ref='{ref_name}', candidates={[c[1] for c in candidates]})"
    
    # 排序
    if is_farthest:
        cand_distances.sort(key=lambda x: -x[2])  # 最远
    else:
        cand_distances.sort(key=lambda x: x[2])   # 最近
    
    answer_letter = cand_distances[0][0]
    dist_str = ", ".join([f"{name}={d:.2f}m" for _, name, d in cand_distances])
    
    return answer_letter, f"ref={ref_name}, {dist_str} → {cand_distances[0][1]}"


def grid_answer_appearance_order(grid: Grid64, question: str, options: List[str]) -> Tuple[str, str]:
    """Grid工具: 出现顺序 (直接从first_seen_frame排序)"""
    if not options:
        return "A", "no options"
    
    match = re.search(r'following categories.*?:\s*(.+?)\?', question, re.IGNORECASE)
    if match:
        target_objects = [obj.strip().lower() for obj in match.group(1).split(',')]
    else:
        opt_content = re.sub(r'^[A-D]\.\s*', '', options[0])
        target_objects = [obj.strip().lower() for obj in opt_content.split(',')]
    
    obj_frames = {}
    for target in target_objects:
        for eid, entity in grid.entities.items():
            if _match_name(target, entity.category):
                obj_frames[target] = entity.first_seen_frame
                break
    
    if len(obj_frames) >= len(target_objects) * 0.5:
        sorted_objs = sorted(obj_frames.keys(), key=lambda x: obj_frames.get(x, 99999))
        
        best_match = "A"
        best_score = -1
        for opt in options:
            letter = opt[0]
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt).lower()
            opt_items = [o.strip() for o in opt_content.split(',')]
            
            score = 0
            for i, item in enumerate(opt_items):
                if i < len(sorted_objs) and _match_name(item, sorted_objs[i]):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = letter
        
        return best_match, f"order by first_frame: {sorted_objs} ({obj_frames})"
    return "A", "insufficient data"


def grid_answer_route(grid: Grid64, question: str, options: List[str]) -> Tuple[str, str]:
    """v6: 路线规划 — 转向模拟算法
    
    VSIBench route_planning 格式:
    "You are a robot beginning at <START> facing <FACING>. You want to navigate to <END>.
     You will perform the following actions:
     1. [please fill in]
     2. Go forward until <WAYPOINT1>
     3. [please fill in]
     4. Go forward until <END>."
    
    选项: "A. Turn Left, Turn Right" 等 — 每个[please fill in]对应一个转向
    
    算法: 
    1. 解析起点、朝向、waypoint序列和fill-in位置
    2. 对每个选项的转向序列，模拟机器人行走
    3. 在每个fill-in位置，应用转向指令改变朝向
    4. 在Go forward步骤，更新位置到目标waypoint，朝向指向该waypoint
    5. 验证：在每个fill-in位置，计算下一个waypoint方向与当前朝向的关系，
       检查应用转向后朝向是否大致指向下一个waypoint
    6. 选择匹配得分最高的选项
    """
    if not options:
        return "A", "no options"
    
    q = question.lower()
    
    # 解析起点和朝向
    m_start = re.search(r'beginning at (?:the )?(.+?)\s+(?:and\s+)?facing (?:the )?(.+?)\.', q)
    if not m_start:
        m_start = re.search(r'beginning at (?:the )?(.+?)\s+facing (?:the )?(.+?)\.', q)
    if not m_start:
        return "A", "cannot parse start/facing"
    
    start_name = m_start.group(1).strip()
    facing_name = m_start.group(2).strip()
    
    # 解析步骤序列 — 提取 waypoints 和 fill-in 位置
    steps = re.findall(r'\d+\.\s+(.+?)(?=\d+\.|You have reached|$)', q, re.DOTALL)
    
    waypoints = []  # (type, name): 'fill_in' 或 'go_forward'
    for step in steps:
        step = step.strip().rstrip('.')
        if 'please fill in' in step:
            waypoints.append(('fill_in', None))
        elif 'go forward' in step:
            # 提取目标: "Go forward until the <X>" 或 "Go forward until <X>"
            m_fwd = re.search(r'go forward\s+until\s+(?:the\s+)?(.+?)(?:\s+is\s+on|\s*$)', step)
            if not m_fwd:
                m_fwd = re.search(r'go forward\s+until\s+(?:passing\s+)?(?:the\s+)?(.+?)(?:\s+on\s+|\s*$)', step)
            if m_fwd:
                wp_name = m_fwd.group(1).strip().rstrip('.')
                waypoints.append(('go_forward', wp_name))
            else:
                waypoints.append(('go_forward', None))
    
    if not waypoints:
        return "A", "no waypoints parsed"
    
    # 获取起点和朝向的3D位置
    e_start = grid.get_by_category(start_name)
    e_facing = grid.get_by_category(facing_name)
    
    if not e_start or not e_facing:
        missing = []
        if not e_start: missing.append(f"start='{start_name}'")
        if not e_facing: missing.append(f"facing='{facing_name}'")
        return "A", f"not found: {', '.join(missing)}"
    
    start_pos = np.array(e_start[0].position_3d, dtype=float)
    facing_pos = np.array(e_facing[0].position_3d, dtype=float)
    
    # 初始朝向 (XZ平面)
    init_facing = np.array([facing_pos[0] - start_pos[0], facing_pos[2] - start_pos[2]])
    if np.linalg.norm(init_facing) < 1e-8:
        # 起点和朝向是同一个物体 (e.g. "beginning at toilet facing toilet")
        # 尝试用第一个go_forward waypoint作为初始朝向
        first_wp_name = None
        for wtype, wname in waypoints:
            if wtype == 'go_forward' and wname:
                first_wp_name = wname
                break
        if first_wp_name:
            e_first_wp = grid.get_by_category(first_wp_name)
            if e_first_wp:
                first_wp_pos = np.array(e_first_wp[0].position_3d, dtype=float)
                init_facing = np.array([first_wp_pos[0] - start_pos[0], first_wp_pos[2] - start_pos[2]])
        if np.linalg.norm(init_facing) < 1e-8:
            # 仍然无法确定朝向，回退到fallback
            return "A", "start and facing at same position, no waypoint to infer direction"
    init_facing = init_facing / np.linalg.norm(init_facing)
    
    # 收集所有waypoint的位置
    wp_positions = {}  # name -> position_3d
    for wtype, wname in waypoints:
        if wname and wname not in wp_positions:
            e_wp = grid.get_by_category(wname)
            if e_wp:
                wp_positions[wname] = np.array(e_wp[0].position_3d, dtype=float)
    
    # 统计有多少个fill-in
    n_fill_ins = sum(1 for wtype, _ in waypoints if wtype == 'fill_in')
    
    # 对每个选项模拟
    best_score = -1
    best_opt = "A"
    score_details = []
    
    for opt in options:
        letter = opt[0]
        opt_content = re.sub(r'^[A-D]\.\s*', '', opt).strip()
        turns = [t.strip().lower() for t in opt_content.split(',')]
        
        # 如果转向数量不匹配fill-in数量，跳过
        if len(turns) != n_fill_ins:
            score_details.append(f"{letter}: turns={len(turns)} != fill_ins={n_fill_ins}")
            continue
        
        # 模拟行走
        current_pos = start_pos.copy()
        current_facing = init_facing.copy()  # XZ平面方向向量
        turn_idx = 0
        score = 0.0
        valid = True
        
        for wi, (wtype, wname) in enumerate(waypoints):
            if wtype == 'fill_in':
                # 应用转向
                turn_cmd = turns[turn_idx]
                turn_idx += 1
                
                if 'back' in turn_cmd:
                    current_facing = -current_facing
                elif 'left' in turn_cmd:
                    # 逆时针旋转90度: (x, z) → (-z, x)
                    current_facing = np.array([-current_facing[1], current_facing[0]])
                elif 'right' in turn_cmd:
                    # 顺时针旋转90度: (x, z) → (z, -x)
                    current_facing = np.array([current_facing[1], -current_facing[0]])
                
                # 检查转向后朝向是否大致指向下一个waypoint
                next_wp_name = None
                for future_type, future_name in waypoints[wi+1:]:
                    if future_type == 'go_forward' and future_name:
                        next_wp_name = future_name
                        break
                
                if next_wp_name and next_wp_name in wp_positions:
                    next_wp_pos = wp_positions[next_wp_name]
                    to_next = np.array([next_wp_pos[0] - current_pos[0], next_wp_pos[2] - current_pos[2]])
                    if np.linalg.norm(to_next) > 1e-8:
                        to_next_norm = to_next / np.linalg.norm(to_next)
                        dot = float(np.dot(current_facing, to_next_norm))
                        # dot > 0 表示大致指向目标（前方半球）
                        if dot > 0:
                            score += dot  # 越对齐得分越高
                        else:
                            score -= 0.5  # 背离目标扣分
                
            elif wtype == 'go_forward' and wname:
                # 移动到waypoint
                if wname in wp_positions:
                    new_pos = wp_positions[wname]
                    # 更新朝向为移动方向
                    move_dir = np.array([new_pos[0] - current_pos[0], new_pos[2] - current_pos[2]])
                    if np.linalg.norm(move_dir) > 1e-8:
                        current_facing = move_dir / np.linalg.norm(move_dir)
                    current_pos = new_pos.copy()
        
        score_details.append(f"{letter}: score={score:.3f} turns={opt_content}")
        if score > best_score:
            best_score = score
            best_opt = letter
    
    reasoning = f"route_sim: start={start_name} facing={facing_name}, {n_fill_ins} fill-ins, " + "; ".join(score_details)
    return best_opt, reasoning


# ============================================================================
# 评估指标
# ============================================================================

def mean_relative_accuracy(pred: float, target: float) -> float:
    if target == 0:
        return 1.0 if pred == 0 else 0.0
    rel_error = abs(pred - target) / (abs(target) + 1e-8)
    thresholds = np.arange(0.05, 0.50 + 0.025, 0.05)
    return float((rel_error < (1 - thresholds)).astype(float).mean())


def evaluate_sample(qt: str, pred: str, gt: str) -> float:
    if qt in ("object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"):
        try:
            p = float(re.search(r'[-+]?\d*\.?\d+', str(pred)).group()) if pred else 0
            g = float(re.search(r'[-+]?\d*\.?\d+', str(gt)).group()) if gt else 0
            return mean_relative_accuracy(p, g)
        except:
            return 0.0
    else:
        p = pred.strip().upper()[:1] if pred else ""
        g = gt.strip().upper()[:1] if gt else ""
        return 1.0 if p == g else 0.0


# ============================================================================
# 主逻辑
# ============================================================================

def select_test_samples(results: List[Dict], n_per_type: int = 10) -> List[Dict]:
    """从V7结果中选取测试样本 - 每种任务各选n个，优先选有视频的"""
    by_type = defaultdict(list)
    for r in results:
        by_type[r['question_type']].append(r)
    
    selected = []
    for qt, samples in sorted(by_type.items()):
        # 检查视频可用性
        available = []
        for s in samples:
            if find_video_path(s['scene_name']):
                available.append(s)
        
        if len(available) < n_per_type:
            logger.warning(f"{qt}: only {len(available)} samples with video (need {n_per_type})")
        
        # 选取: 均匀采样
        n = min(n_per_type, len(available))
        indices = np.linspace(0, len(available) - 1, n, dtype=int)
        for idx in indices:
            selected.append(available[idx])
    
    return selected


def process_sample(grid: Grid64, sample: Dict) -> Dict:
    """用Grid工具回答一个样本"""
    qt = sample['question_type']
    question = sample['question']
    options = sample.get('options') or []
    gt = sample['ground_truth']
    
    if qt == 'object_counting':
        pred, reason = grid_answer_counting(grid, question)
    elif qt == 'object_size_estimation':
        pred, reason = grid_answer_size(grid, question)
    elif qt == 'room_size_estimation':
        pred, reason = grid_answer_room_size(grid, question)
    elif qt == 'object_abs_distance':
        pred, reason = grid_answer_abs_distance(grid, question)
    elif 'direction' in qt:
        pred, reason = grid_answer_direction(grid, question, options)
    elif qt == 'object_rel_distance':
        pred, reason = grid_answer_rel_distance(grid, question, options)
    elif qt == 'obj_appearance_order':
        pred, reason = grid_answer_appearance_order(grid, question, options)
    elif qt == 'route_planning':
        pred, reason = grid_answer_route(grid, question, options)
    else:
        pred, reason = "0", f"unknown task: {qt}"
    
    grid_score = evaluate_sample(qt, pred, gt)
    
    return {
        'scene_name': sample['scene_name'],
        'question_type': qt,
        'question': question,
        'ground_truth': gt,
        'grid_prediction': pred,
        'grid_reasoning': reason,
        'grid_score': grid_score,
        'v7_vl_score': sample.get('vl_score', 0),
        'v7_rule_score': sample.get('rule_score', 0),
        'v7_vl_pred': sample.get('vl_prediction', ''),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_per_type', type=int, default=10, help='Samples per task type')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    # 加载V7基准结果
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    logger.info(f"Loading V7 baseline: {v7_path}")
    with open(v7_path) as f:
        v7_results = json.load(f)
    logger.info(f"V7 baseline: {len(v7_results)} samples")
    
    # 选取测试样本
    test_samples = select_test_samples(v7_results, n_per_type=args.n_per_type)
    logger.info(f"Selected {len(test_samples)} test samples")
    
    # 按scene分组 (同scene的多个问题共享一次Grid构建)
    by_scene = defaultdict(list)
    for s in test_samples:
        by_scene[s['scene_name']].append(s)
    logger.info(f"Unique scenes: {len(by_scene)}")
    
    # 构建Grid并回答
    builder = Grid64Builder(device=args.device, num_frames=16)
    
    all_results = []
    scene_grids = {}  # 缓存
    
    total_scenes = len(by_scene)
    for si, (scene_name, samples) in enumerate(by_scene.items()):
        video_path = find_video_path(scene_name)
        if not video_path:
            logger.warning(f"[{si+1}/{total_scenes}] Video not found: {scene_name}")
            for s in samples:
                all_results.append({
                    'scene_name': scene_name,
                    'question_type': s['question_type'],
                    'question': s['question'],
                    'ground_truth': s['ground_truth'],
                    'grid_prediction': '0',
                    'grid_reasoning': 'no video',
                    'grid_score': 0.0,
                    'v7_vl_score': s.get('vl_score', 0),
                    'v7_rule_score': s.get('rule_score', 0),
                })
            continue
        
        logger.info(f"[{si+1}/{total_scenes}] Building Grid for {scene_name} ({len(samples)} questions)")
        t0 = time.time()
        
        try:
            grid = builder.build_grid(video_path)
            elapsed = time.time() - t0
            logger.info(f"  Grid built in {elapsed:.1f}s: {len(grid.entities)} entities")
            
            for s in samples:
                result = process_sample(grid, s)
                all_results.append(result)
                
                delta = result['grid_score'] - result['v7_vl_score']
                marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
                logger.info(f"  {s['question_type'][:20]:20s} | Grid={result['grid_score']:.3f} VL={result['v7_vl_score']:.3f} {marker} | {result['grid_prediction'][:30]}")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            for s in samples:
                all_results.append({
                    'scene_name': scene_name,
                    'question_type': s['question_type'],
                    'question': s['question'],
                    'ground_truth': s['ground_truth'],
                    'grid_prediction': '0',
                    'grid_reasoning': f'error: {str(e)[:100]}',
                    'grid_score': 0.0,
                    'v7_vl_score': s.get('vl_score', 0),
                    'v7_rule_score': s.get('rule_score', 0),
                })
    
    # 卸载模型
    builder.unload()
    
    # ========================================================================
    # 汇总结果
    # ========================================================================
    print("\n" + "=" * 100)
    print("64³ Grid Mind Map 小样本测试结果")
    print(f"基准: V7 VL Overall = 63.61%")
    print(f"测试样本: {len(all_results)}")
    print("=" * 100)
    
    task_types = sorted(set(r['question_type'] for r in all_results))
    
    print(f"\n{'Task':<35} {'N':>4} {'V7_VL':>7} {'V7_Rule':>7} {'Grid64':>7} {'Delta':>7}")
    print(f"{'-'*35} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    
    all_v7_vl, all_v7_rule, all_grid = [], [], []
    
    for qt in task_types:
        qt_results = [r for r in all_results if r['question_type'] == qt]
        v7_vl = [r['v7_vl_score'] for r in qt_results]
        v7_rule = [r.get('v7_rule_score', 0) for r in qt_results]
        grid_scores = [r['grid_score'] for r in qt_results]
        
        v7_vl_mean = np.mean(v7_vl)
        v7_rule_mean = np.mean(v7_rule)
        grid_mean = np.mean(grid_scores)
        delta = grid_mean - v7_vl_mean
        
        marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
        print(f"  {qt:<35} {len(qt_results):>4} {v7_vl_mean:>6.3f} {v7_rule_mean:>6.3f} {grid_mean:>6.3f} {delta:>+6.3f} {marker}")
        
        all_v7_vl.extend(v7_vl)
        all_v7_rule.extend(v7_rule)
        all_grid.extend(grid_scores)
    
    print(f"{'-'*35} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    overall_v7 = np.mean(all_v7_vl)
    overall_rule = np.mean(all_v7_rule)
    overall_grid = np.mean(all_grid)
    overall_delta = overall_grid - overall_v7
    print(f"  {'Overall':<35} {len(all_results):>4} {overall_v7:>6.3f} {overall_rule:>6.3f} {overall_grid:>6.3f} {overall_delta:>+6.3f}")
    
    # 逐样本分析
    print(f"\n{'='*100}")
    print("Grid比VL好的样本:")
    improvements = [r for r in all_results if r['grid_score'] > r['v7_vl_score'] + 0.05]
    for r in improvements[:20]:
        print(f"  [{r['question_type'][:20]}] Grid={r['grid_score']:.3f} VL={r['v7_vl_score']:.3f} | "
              f"pred={r['grid_prediction'][:25]} gt={r['ground_truth'][:25]} | {r['grid_reasoning'][:50]}")
    
    print(f"\nGrid比VL差的样本:")
    degradations = [r for r in all_results if r['grid_score'] < r['v7_vl_score'] - 0.05]
    for r in degradations[:20]:
        print(f"  [{r['question_type'][:20]}] Grid={r['grid_score']:.3f} VL={r['v7_vl_score']:.3f} | "
              f"pred={r['grid_prediction'][:25]} gt={r['ground_truth'][:25]} | {r['grid_reasoning'][:50]}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = PROJECT_ROOT / "outputs" / f"grid64_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果
    with open(output_dir / "detailed_results.json", 'w') as f:
        # Convert numpy types
        clean_results = []
        for r in all_results:
            cr = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    cr[k] = float(v)
                else:
                    cr[k] = v
            clean_results.append(cr)
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    
    # 保存summary
    summary = {
        'timestamp': timestamp,
        'n_samples': len(all_results),
        'n_scenes': len(by_scene),
        'overall': {
            'v7_vl': float(overall_v7),
            'v7_rule': float(overall_rule),
            'grid64': float(overall_grid),
            'delta_vs_vl': float(overall_delta),
        },
        'by_task': {},
    }
    for qt in task_types:
        qt_r = [r for r in all_results if r['question_type'] == qt]
        summary['by_task'][qt] = {
            'n': len(qt_r),
            'v7_vl': float(np.mean([r['v7_vl_score'] for r in qt_r])),
            'grid64': float(np.mean([r['grid_score'] for r in qt_r])),
            'delta': float(np.mean([r['grid_score'] for r in qt_r]) - np.mean([r['v7_vl_score'] for r in qt_r])),
        }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\n总结: Grid64 Overall={overall_grid:.4f} vs V7 VL={overall_v7:.4f} (Delta={overall_delta:+.4f})")


if __name__ == "__main__":
    main()
