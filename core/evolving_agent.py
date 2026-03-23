#!/usr/bin/env python3
"""
Self-Evolving Mind Map Agent Framework (自进化心智地图智能体框架)

核心架构：
┌─────────────────────────────────────────────────────────────────────────┐
│                         Self-Evolving Agent Loop                        │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────┐ │
│  │ 感知    │ -> │ Manager │ -> │ Reasoner │ -> │ Critic  │ -> │Evolver│ │
│  │DA3+DINO │    │任务分析  │    │  CoST    │    │一致性检查│    │VL回溯 │ │
│  └─────────┘    └─────────┘    └──────────┘    └─────────┘    └──────┘ │
│       │              │              │               │             │     │
│       v              v              v               v             v     │
│  MindMapV5 ──> 提取策略 ──> 推理答案 ──> 验证/演化 ──> 更新地图   │
└─────────────────────────────────────────────────────────────────────────┘

作者: tianjungu
日期: 2026-01-30
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
import cv2
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mind_map_v5 import (
    MindMapEntityV5, SparseVoxelMap, GaussianPosition3D,
    MindMapBuilderV5
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 枚举和数据结构
# ============================================================================

class TaskType(Enum):
    """VSI-Bench 任务类型"""
    OBJECT_COUNTING = "object_counting"
    OBJECT_ABS_DISTANCE = "object_abs_distance"
    OBJECT_SIZE = "object_size_estimation"
    ROOM_SIZE = "room_size_estimation"
    REL_DIRECTION_EASY = "object_rel_direction_easy"
    REL_DIRECTION_MEDIUM = "object_rel_direction_medium"
    REL_DIRECTION_HARD = "object_rel_direction_hard"
    REL_DISTANCE = "object_rel_distance"
    APPEARANCE_ORDER = "obj_appearance_order"
    ROUTE_PLANNING = "route_planning"


class CriticVerdict(Enum):
    """Critic 判定结果"""
    PASS = "pass"                      # 通过验证
    LOW_CONFIDENCE = "low_confidence"  # 置信度过低，需要演化
    SPATIAL_CONFLICT = "spatial_conflict"  # 空间冲突
    LOGIC_ERROR = "logic_error"        # 逻辑矛盾
    RE_OBSERVE = "re_observe"          # 需要重新观察


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_name: str
    description: str
    result: Any
    confidence: float = 1.0


@dataclass 
class ReasoningResult:
    """推理结果"""
    answer: str
    steps: List[ReasoningStep] = field(default_factory=list)
    confidence: float = 1.0
    raw_reasoning: str = ""
    extracted_entities: List[str] = field(default_factory=list)


@dataclass
class CriticFeedback:
    """Critic 反馈"""
    verdict: CriticVerdict
    issues: List[str] = field(default_factory=list)
    suggested_entities_to_reobserve: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.4


@dataclass
class EvolutionResult:
    """演化结果"""
    updated_entities: Dict[str, MindMapEntityV5]
    corrections: List[str] = field(default_factory=list)
    confidence_boost: float = 0.0


# ============================================================================
# 1. Manager 模块 - 任务分析与策略决定
# ============================================================================

class TaskManager:
    """任务管理器：分析问题类型，决定心智地图提取策略"""
    
    # 任务类型匹配模式
    TASK_PATTERNS = {
        TaskType.OBJECT_COUNTING: [
            r"how many", r"count", r"number of"
        ],
        TaskType.OBJECT_ABS_DISTANCE: [
            r"distance between", r"how far.*from", r"meters.*between"
        ],
        TaskType.OBJECT_SIZE: [
            r"size of", r"how (big|large|tall|wide)", r"dimension"
        ],
        TaskType.ROOM_SIZE: [
            r"room.*size", r"area of.*room", r"square (meter|feet)"
        ],
        TaskType.REL_DIRECTION_EASY: [
            r"left or right", r"to (my|the) (left|right)"
        ],
        TaskType.REL_DIRECTION_MEDIUM: [
            r"(front|back|left|right).*facing"
        ],
        TaskType.REL_DIRECTION_HARD: [
            r"front.*(left|right)", r"back.*(left|right)"
        ],
        TaskType.REL_DISTANCE: [
            r"(closer|closer to|nearest|farthest)", r"which.*closest"
        ],
        TaskType.APPEARANCE_ORDER: [
            r"first.*appear", r"order.*appear", r"sequence"
        ],
        TaskType.ROUTE_PLANNING: [
            r"how.*get.*from.*to", r"path.*from", r"route", r"walk.*from"
        ],
    }
    
    # 每种任务需要的心智地图信息
    TASK_REQUIREMENTS = {
        TaskType.OBJECT_COUNTING: {
            "need_count": True,
            "need_position": False,
            "need_size": False,
            "need_features": False,
            "need_voxel_map": False,
            "need_frame_info": False,
        },
        TaskType.OBJECT_ABS_DISTANCE: {
            "need_count": False,
            "need_position": True,
            "need_size": False,
            "need_features": False,
            "need_voxel_map": False,
            "need_frame_info": False,
        },
        TaskType.OBJECT_SIZE: {
            "need_count": False,
            "need_position": False,
            "need_size": True,
            "need_features": False,
            "need_voxel_map": False,
            "need_frame_info": False,
        },
        TaskType.ROOM_SIZE: {
            "need_count": False,
            "need_position": True,
            "need_size": False,
            "need_features": False,
            "need_voxel_map": True,
            "need_frame_info": False,
        },
        TaskType.REL_DIRECTION_EASY: {
            "need_count": False,
            "need_position": True,
            "need_size": False,
            "need_features": False,
            "need_voxel_map": False,
            "need_frame_info": True,  # 需要相机朝向
        },
        TaskType.REL_DIRECTION_MEDIUM: {
            "need_count": False,
            "need_position": True,
            "need_size": False,
            "need_features": False,
            "need_voxel_map": False,
            "need_frame_info": True,
        },
        TaskType.REL_DIRECTION_HARD: {
            "need_count": False,
            "need_position": True,
            "need_size": False,
            "need_features": False,
            "need_voxel_map": False,
            "need_frame_info": True,
        },
        TaskType.REL_DISTANCE: {
            "need_count": False,
            "need_position": True,
            "need_size": False,
            "need_features": False,
            "need_voxel_map": False,
            "need_frame_info": False,
        },
        TaskType.APPEARANCE_ORDER: {
            "need_count": False,
            "need_position": False,
            "need_size": False,
            "need_features": False,
            "need_voxel_map": False,
            "need_frame_info": True,
        },
        TaskType.ROUTE_PLANNING: {
            "need_count": False,
            "need_position": True,
            "need_size": True,
            "need_voxel_map": True,
            "need_features": False,
            "need_frame_info": False,
        },
    }
    
    def __init__(self):
        pass
    
    def analyze_task(self, question: str, question_type: str = None) -> TaskType:
        """分析任务类型"""
        # 如果提供了明确的类型，直接映射
        if question_type:
            type_map = {
                "object_counting": TaskType.OBJECT_COUNTING,
                "object_abs_distance": TaskType.OBJECT_ABS_DISTANCE,
                "object_size_estimation": TaskType.OBJECT_SIZE,
                "room_size_estimation": TaskType.ROOM_SIZE,
                "object_rel_direction_easy": TaskType.REL_DIRECTION_EASY,
                "object_rel_direction_medium": TaskType.REL_DIRECTION_MEDIUM,
                "object_rel_direction_hard": TaskType.REL_DIRECTION_HARD,
                "object_rel_distance": TaskType.REL_DISTANCE,
                "obj_appearance_order": TaskType.APPEARANCE_ORDER,
                "route_planning": TaskType.ROUTE_PLANNING,
            }
            if question_type in type_map:
                return type_map[question_type]
        
        # 否则通过模式匹配
        q_lower = question.lower()
        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q_lower):
                    return task_type
        
        # 默认返回计数任务
        return TaskType.OBJECT_COUNTING
    
    def get_extraction_strategy(self, task_type: TaskType) -> Dict[str, bool]:
        """获取心智地图提取策略"""
        return self.TASK_REQUIREMENTS.get(task_type, {
            "need_count": True,
            "need_position": True,
            "need_size": True,
            "need_features": True,
            "need_voxel_map": True,
            "need_frame_info": True,
        })
    
    def extract_target_objects(self, question: str) -> List[str]:
        """从问题中提取目标物体"""
        targets = []
        
        # 常见物体名称
        common_objects = [
            "chair", "table", "sofa", "couch", "bed", "desk", "door", "window",
            "tv", "television", "monitor", "refrigerator", "fridge", "lamp",
            "plant", "picture", "painting", "shelf", "cabinet", "toilet", "sink",
            "bathtub", "mirror", "rug", "carpet", "curtain", "pillow", "cushion"
        ]
        
        q_lower = question.lower()
        for obj in common_objects:
            if obj in q_lower:
                targets.append(obj)
        
        # 提取 "the X" 模式
        the_pattern = re.findall(r"the (\w+)", q_lower)
        for match in the_pattern:
            if match not in ['room', 'distance', 'size', 'area', 'left', 'right', 
                            'front', 'back', 'first', 'second', 'my']:
                targets.append(match)
        
        return list(set(targets))
    
    def serialize_mind_map(self, 
                           entities: Dict[str, MindMapEntityV5],
                           voxel_map: SparseVoxelMap,
                           task_type: TaskType,
                           camera_info: Dict = None) -> str:
        """根据任务类型序列化心智地图"""
        strategy = self.get_extraction_strategy(task_type)
        
        serialized = []
        serialized.append("=== 心智地图数据 ===\n")
        
        # 实体信息
        serialized.append("【检测到的物体】")
        for label, entity in entities.items():
            entity_info = [f"- {label}:"]
            
            if strategy.get("need_count"):
                entity_info.append(f"  数量: {entity.max_single_frame_count}")
            
            if strategy.get("need_position") and entity.position:
                pos = entity.position.mean
                conf = entity.position.confidence
                entity_info.append(f"  位置 (x,y,z): ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) 米")
                entity_info.append(f"  位置置信度: {conf:.2f}")
            
            if strategy.get("need_size") and entity.size_mean is not None:
                size = entity.size_mean
                entity_info.append(f"  尺寸 (宽,高,深): ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}) 米")
            
            if strategy.get("need_frame_info"):
                entity_info.append(f"  首次出现帧: {entity.first_seen_frame}")
                entity_info.append(f"  最后出现帧: {entity.last_seen_frame}")
            
            serialized.append("\n".join(entity_info))
        
        # 体素地图信息
        if strategy.get("need_voxel_map") and voxel_map:
            serialized.append("\n【空间信息】")
            dims = voxel_map.get_room_dimensions()
            serialized.append(f"- 房间尺寸: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} 米")
            serialized.append(f"- 占据体积: {voxel_map.get_occupied_volume():.2f} 立方米")
            serialized.append(f"- 地面面积: {voxel_map.get_floor_area():.2f} 平方米")
        
        # 相机信息（用于相对方向任务）
        if strategy.get("need_frame_info") and camera_info:
            serialized.append("\n【观察者视角】")
            if "forward_vector" in camera_info:
                fwd = camera_info["forward_vector"]
                serialized.append(f"- 相机朝向: ({fwd[0]:.2f}, {fwd[1]:.2f}, {fwd[2]:.2f})")
            if "position" in camera_info:
                pos = camera_info["position"]
                serialized.append(f"- 相机位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        return "\n".join(serialized)
    
    def generate_grid_map(self, voxel_map: SparseVoxelMap, 
                          resolution: float = 0.5) -> str:
        """生成 2D Grid Map（用于路径规划）"""
        if not voxel_map or not voxel_map.voxels:
            return "无法生成地图"
        
        # 获取边界
        min_bounds = voxel_map.min_bounds
        max_bounds = voxel_map.max_bounds
        
        # 计算网格大小
        x_size = int((max_bounds[0] - min_bounds[0]) / resolution) + 3
        z_size = int((max_bounds[2] - min_bounds[2]) / resolution) + 3
        
        # 限制大小
        x_size = min(x_size, 30)
        z_size = min(z_size, 30)
        
        # 初始化网格
        grid = [['.' for _ in range(x_size)] for _ in range(z_size)]
        
        # 填充障碍物
        for (vx, vy, vz), voxel in voxel_map.voxels.items():
            if voxel.occupancy_prob > 0.3:
                # 转换到网格坐标
                world_x = vx * voxel_map.voxel_size
                world_z = vz * voxel_map.voxel_size
                
                grid_x = int((world_x - min_bounds[0]) / resolution) + 1
                grid_z = int((world_z - min_bounds[2]) / resolution) + 1
                
                if 0 <= grid_x < x_size and 0 <= grid_z < z_size:
                    # 用物体首字母标记
                    label = voxel.semantic_label
                    grid[grid_z][grid_x] = label[0].upper() if label else '#'
        
        # 转换为字符串
        lines = []
        lines.append("2D 地面投影图 (. = 通路, 字母 = 物体):")
        for row in grid:
            lines.append(''.join(row))
        
        return '\n'.join(lines)


# ============================================================================
# 2. Reasoner 模块 - Chain-of-Spatial-Thought 推理
# ============================================================================

class SpatialReasoner:
    """空间推理器：执行 Chain-of-Spatial-Thought (CoST) 推理"""
    
    # CoST Prompt 模板
    COST_PROMPT_TEMPLATE = """你是一个 3D 空间推理专家。基于提供的心智地图，请通过以下步骤回答问题：

**步骤 1 - 定位**：找到问题提及的物体在心智地图中的坐标位置。
**步骤 2 - 空间映射**：在 3D 坐标系中建立它们的相对位置关系。
**步骤 3 - 遮挡与距离校验**：利用尺寸和位置计算是否存在遮挡或距离是否合理。
**步骤 4 - 结论**：给出最终答案。

{mind_map_data}

【问题】
{question}

{options_text}

请按步骤推理，最后给出答案。答案格式：
最终答案: [你的答案]"""

    # 任务特定的推理提示
    TASK_SPECIFIC_PROMPTS = {
        TaskType.OBJECT_COUNTING: """
注意：
- 关注心智地图中的"数量"字段
- 这个数量代表单帧中检测到的最大数量""",
        
        TaskType.OBJECT_ABS_DISTANCE: """
注意：
- 使用欧氏距离公式计算两点间距离：d = sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
- 答案应为米，保留一位小数""",
        
        TaskType.OBJECT_SIZE: """
注意：
- 答案为物体最大维度的尺寸（厘米）
- 取尺寸的宽、高、深三个维度中的最大值""",
        
        TaskType.ROOM_SIZE: """
注意：
- 答案为房间面积（平方米）
- 基于物体分布的包围盒估算""",
        
        TaskType.REL_DIRECTION_EASY: """
注意（自我中心方向判断）：
- 你需要判断目标物体相对于"站在A处面向B"这个观察者的左右
- 使用向量叉积判断：如果 (forward × to_target) 的 Y 分量 > 0，目标在右边
- 坐标系：X 向右，Y 向上，Z 向前""",
        
        TaskType.REL_DIRECTION_MEDIUM: """
注意（自我中心方向判断）：
- 前方/后方：to_target · forward > 0 为前方
- 左右：使用叉积的 Y 分量判断""",
        
        TaskType.REL_DIRECTION_HARD: """
注意（八方向判断）：
- 结合前后和左右判断
- 例如：前方+左边 = front-left""",
        
        TaskType.REL_DISTANCE: """
注意：
- 计算每个候选物体到参考物体的距离
- 选择最近/最远的那个""",
        
        TaskType.APPEARANCE_ORDER: """
注意：
- 按 first_seen_frame 排序
- 帧号越小，出现越早""",
        
        TaskType.ROUTE_PLANNING: """
注意：
- 分析路径中每个转向点
- 使用向量叉积判断左转/右转
- 直行：方向变化 < 30°""",
    }
    
    def __init__(self, llm_client=None, use_rule_based: bool = True):
        """
        Args:
            llm_client: LLM 客户端（如 Qwen3-VL）
            use_rule_based: 是否使用规则推理作为备选
        """
        self.llm_client = llm_client
        self.use_rule_based = use_rule_based
        self.manager = TaskManager()
    
    def reason(self, 
               entities: Dict[str, MindMapEntityV5],
               voxel_map: SparseVoxelMap,
               question: str,
               task_type: TaskType,
               options: List[str] = None,
               camera_info: Dict = None) -> ReasoningResult:
        """执行推理"""
        
        # 序列化心智地图
        mind_map_data = self.manager.serialize_mind_map(
            entities, voxel_map, task_type, camera_info
        )
        
        # 对于路径规划，添加 Grid Map
        if task_type == TaskType.ROUTE_PLANNING and voxel_map:
            mind_map_data += "\n\n" + self.manager.generate_grid_map(voxel_map)
        
        # 构建 Prompt
        options_text = ""
        if options:
            options_text = "【选项】\n" + "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        
        task_hint = self.TASK_SPECIFIC_PROMPTS.get(task_type, "")
        
        prompt = self.COST_PROMPT_TEMPLATE.format(
            mind_map_data=mind_map_data,
            question=question,
            options_text=options_text
        ) + task_hint
        
        # 如果有 LLM，使用 LLM 推理
        if self.llm_client:
            try:
                response = self._llm_inference(prompt)
                return self._parse_llm_response(response, options)
            except Exception as e:
                logger.warning(f"LLM 推理失败: {e}，回退到规则推理")
        
        # 否则使用规则推理
        return self._rule_based_inference(entities, voxel_map, question, task_type, options)
    
    def _llm_inference(self, prompt: str) -> str:
        """调用 LLM 推理"""
        if self.llm_client is None:
            raise ValueError("LLM 客户端未初始化")
        
        # 这里假设 llm_client 有 generate 方法
        response = self.llm_client.generate(prompt)
        return response
    
    def _parse_llm_response(self, response: str, options: List[str] = None) -> ReasoningResult:
        """解析 LLM 响应"""
        # 提取最终答案
        answer_match = re.search(r"最终答案[：:]\s*(.+?)(?:\n|$)", response)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # 尝试其他格式
            answer_match = re.search(r"答案[：:]\s*(.+?)(?:\n|$)", response)
            answer = answer_match.group(1).strip() if answer_match else ""
        
        # 如果有选项，匹配选项
        if options and answer:
            for i, opt in enumerate(options):
                if opt.lower() in answer.lower() or chr(65+i) in answer:
                    answer = chr(65+i)
                    break
        
        # 提取推理步骤
        steps = []
        for step_name in ["定位", "空间映射", "遮挡与距离校验", "结论"]:
            step_match = re.search(rf"{step_name}[：:]\s*(.+?)(?=步骤|\n\n|最终答案|$)", response, re.DOTALL)
            if step_match:
                steps.append(ReasoningStep(
                    step_name=step_name,
                    description=step_match.group(1).strip(),
                    result=None
                ))
        
        return ReasoningResult(
            answer=answer,
            steps=steps,
            raw_reasoning=response,
            confidence=0.8 if answer else 0.3
        )
    
    def _rule_based_inference(self,
                              entities: Dict[str, MindMapEntityV5],
                              voxel_map: SparseVoxelMap,
                              question: str,
                              task_type: TaskType,
                              options: List[str] = None) -> ReasoningResult:
        """规则推理（作为 LLM 的备选）"""
        
        steps = []
        
        # 步骤 1: 定位
        target_objects = self.manager.extract_target_objects(question)
        located_entities = {}
        for target in target_objects:
            for label, entity in entities.items():
                if target in label.lower() or label.lower() in target:
                    located_entities[target] = entity
                    break
        
        steps.append(ReasoningStep(
            step_name="定位",
            description=f"找到物体: {list(located_entities.keys())}",
            result=located_entities
        ))
        
        # 根据任务类型执行不同推理
        if task_type == TaskType.OBJECT_COUNTING:
            answer, conf = self._reason_counting(entities, question)
        elif task_type == TaskType.OBJECT_ABS_DISTANCE:
            answer, conf = self._reason_abs_distance(located_entities, question)
        elif task_type == TaskType.OBJECT_SIZE:
            answer, conf = self._reason_object_size(entities, question)
        elif task_type == TaskType.ROOM_SIZE:
            answer, conf = self._reason_room_size(entities, voxel_map, question)
        elif task_type in [TaskType.REL_DIRECTION_EASY, TaskType.REL_DIRECTION_MEDIUM, TaskType.REL_DIRECTION_HARD]:
            answer, conf = self._reason_rel_direction(entities, question, task_type, options)
        elif task_type == TaskType.REL_DISTANCE:
            answer, conf = self._reason_rel_distance(entities, question, options)
        elif task_type == TaskType.APPEARANCE_ORDER:
            answer, conf = self._reason_appearance_order(entities, question, options)
        elif task_type == TaskType.ROUTE_PLANNING:
            answer, conf = self._reason_route_planning(entities, question, options)
        else:
            answer, conf = "unknown", 0.1
        
        steps.append(ReasoningStep(
            step_name="结论",
            description=f"最终答案: {answer}",
            result=answer,
            confidence=conf
        ))
        
        return ReasoningResult(
            answer=str(answer),
            steps=steps,
            confidence=conf,
            extracted_entities=target_objects
        )
    
    # ================== 各任务的规则推理实现 ==================
    
    def _reason_counting(self, entities: Dict[str, MindMapEntityV5], question: str) -> Tuple[str, float]:
        """计数推理"""
        from tests.test_vsibench_directqa import get_synonyms
        
        q_lower = question.lower()
        
        for label, entity in entities.items():
            if label.lower() in q_lower or any(s in q_lower for s in get_synonyms(label)):
                return str(entity.max_single_frame_count), min(1.0, entity.avg_confidence + 0.2)
        
        return "0", 0.3
    
    def _reason_abs_distance(self, located_entities: Dict, question: str) -> Tuple[str, float]:
        """绝对距离推理"""
        if len(located_entities) < 2:
            return "2.0", 0.3
        
        entities_list = list(located_entities.values())
        if entities_list[0].position and entities_list[1].position:
            pos1 = entities_list[0].position.mean
            pos2 = entities_list[1].position.mean
            dist = float(np.linalg.norm(pos1 - pos2))
            conf = min(entities_list[0].position.confidence, entities_list[1].position.confidence)
            return f"{dist:.1f}", conf
        
        return "2.0", 0.3
    
    def _reason_object_size(self, entities: Dict[str, MindMapEntityV5], question: str) -> Tuple[str, float]:
        """物体尺寸推理"""
        from tests.test_vsibench_directqa import get_synonyms
        
        q_lower = question.lower()
        
        for label, entity in entities.items():
            if label.lower() in q_lower or any(s in q_lower for s in get_synonyms(label)):
                if entity.size_mean is not None:
                    max_dim = float(np.max(entity.size_mean)) * 100  # cm
                    conf = min(1.0, entity.avg_confidence * entity.detection_count / 10)
                    return str(int(max_dim)), conf
        
        return "50", 0.3
    
    def _reason_room_size(self, entities: Dict[str, MindMapEntityV5], 
                          voxel_map: SparseVoxelMap, question: str) -> Tuple[str, float]:
        """房间面积推理"""
        positions = []
        for entity in entities.values():
            if entity.position:
                positions.append(entity.position.mean)
        
        if len(positions) < 2:
            return str(12 + len(entities) * 2), 0.3
        
        positions = np.array(positions)
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        estimated_area = (x_range + 1.5) * (y_range + 1.5)
        estimated_area = max(8, min(80, estimated_area))
        
        return f"{estimated_area:.1f}", min(1.0, len(positions) / 5)
    
    def _reason_rel_direction(self, entities: Dict[str, MindMapEntityV5],
                              question: str, task_type: TaskType,
                              options: List[str]) -> Tuple[str, float]:
        """相对方向推理"""
        # 解析问题中的物体
        q_lower = question.lower()
        
        # 解析 "standing by X and facing Y, is Z on my left/right?"
        standing_match = re.search(r"standing by (?:the )?(\w+)", q_lower)
        facing_match = re.search(r"facing (?:the )?(\w+)", q_lower)
        target_match = re.search(r"is (?:the )?(\w+) (?:on|to)", q_lower)
        
        if not all([standing_match, facing_match, target_match]):
            return options[0] if options else "left", 0.3
        
        standing_name = standing_match.group(1)
        facing_name = facing_match.group(1)
        target_name = target_match.group(1)
        
        # 找到对应实体
        def find_entity(name):
            for label, ent in entities.items():
                if name in label.lower() or label.lower() in name:
                    return ent
            return None
        
        standing_ent = find_entity(standing_name)
        facing_ent = find_entity(facing_name)
        target_ent = find_entity(target_name)
        
        if not all([standing_ent, facing_ent, target_ent]):
            return options[0] if options else "left", 0.3
        
        # 获取位置
        standing_pos = standing_ent.position.mean if standing_ent.position else np.zeros(3)
        facing_pos = facing_ent.position.mean if facing_ent.position else np.zeros(3)
        target_pos = target_ent.position.mean if target_ent.position else np.zeros(3)
        
        # 计算方向向量（在 XZ 平面）
        forward = facing_pos - standing_pos
        forward_2d = np.array([forward[0], forward[2]])
        forward_2d = forward_2d / (np.linalg.norm(forward_2d) + 1e-8)
        
        # 右向量（顺时针旋转90度）
        right_2d = np.array([forward_2d[1], -forward_2d[0]])
        
        # 目标相对向量
        to_target = target_pos - standing_pos
        to_target_2d = np.array([to_target[0], to_target[2]])
        to_target_2d = to_target_2d / (np.linalg.norm(to_target_2d) + 1e-8)
        
        # 点积判断
        right_dot = np.dot(to_target_2d, right_2d)
        front_dot = np.dot(to_target_2d, forward_2d)
        
        # 置信度
        conf = min(standing_ent.position.confidence if standing_ent.position else 0.5,
                   facing_ent.position.confidence if facing_ent.position else 0.5,
                   target_ent.position.confidence if target_ent.position else 0.5)
        
        # 根据难度返回
        if task_type == TaskType.REL_DIRECTION_EASY:
            direction = "right" if right_dot > 0 else "left"
        elif task_type == TaskType.REL_DIRECTION_MEDIUM:
            if front_dot < -0.5:
                direction = "back"
            else:
                direction = "right" if right_dot > 0 else "left"
        else:  # hard
            front_back = "front" if front_dot > 0 else "back"
            left_right = "right" if right_dot > 0 else "left"
            direction = f"{front_back}-{left_right}"
        
        # 匹配选项
        if options:
            for opt in options:
                if direction in opt.lower():
                    return opt, conf
            return options[0], conf * 0.5
        
        return direction, conf
    
    def _reason_rel_distance(self, entities: Dict[str, MindMapEntityV5],
                             question: str, options: List[str]) -> Tuple[str, float]:
        """相对距离推理"""
        # 简化实现
        if not options:
            return "unknown", 0.3
        
        # 找参考物体
        ref_match = re.search(r"(closest|nearest|farthest) to (?:the )?(\w+)", question.lower())
        if not ref_match:
            return options[0], 0.3
        
        direction = ref_match.group(1)
        ref_name = ref_match.group(2)
        
        # 找参考实体
        ref_entity = None
        for label, ent in entities.items():
            if ref_name in label.lower():
                ref_entity = ent
                break
        
        if not ref_entity or not ref_entity.position:
            return options[0], 0.3
        
        ref_pos = ref_entity.position.mean
        
        # 计算每个候选物体的距离
        distances = {}
        for opt in options:
            opt_lower = opt.lower()
            for label, ent in entities.items():
                if label.lower() in opt_lower and ent.position:
                    dist = np.linalg.norm(ent.position.mean - ref_pos)
                    distances[opt] = dist
                    break
        
        if not distances:
            return options[0], 0.3
        
        # 选择最近/最远
        if "closest" in direction or "nearest" in direction:
            return min(distances, key=distances.get), 0.7
        else:
            return max(distances, key=distances.get), 0.7
    
    def _reason_appearance_order(self, entities: Dict[str, MindMapEntityV5],
                                 question: str, options: List[str]) -> Tuple[str, float]:
        """出现顺序推理"""
        # 提取问题中的物体
        object_frames = {}
        for label, ent in entities.items():
            object_frames[label] = ent.first_seen_frame
        
        # 排序
        sorted_objects = sorted(object_frames, key=lambda x: object_frames[x])
        
        # 匹配最佳选项
        if options:
            best_match = options[0]
            best_score = 0
            
            for opt in options:
                score = 0
                opt_lower = opt.lower()
                for i, obj in enumerate(sorted_objects):
                    if obj.lower() in opt_lower:
                        # 位置越靠前，分数越高
                        score += (len(sorted_objects) - i)
                
                if score > best_score:
                    best_score = score
                    best_match = opt
            
            return best_match, 0.6
        
        return ", ".join(sorted_objects[:3]), 0.6
    
    def _reason_route_planning(self, entities: Dict[str, MindMapEntityV5],
                               question: str, options: List[str]) -> Tuple[str, float]:
        """路径规划推理"""
        if not options:
            return "unknown", 0.3
        
        # 提取路径点
        path_pattern = re.findall(r"from (?:the )?(\w+) to (?:the )?(\w+)", question.lower())
        if not path_pattern:
            return options[0], 0.3
        
        # 获取路径点位置
        positions = []
        for start, end in path_pattern:
            for label, ent in entities.items():
                if start in label.lower() and ent.position:
                    positions.append(ent.position.mean)
                if end in label.lower() and ent.position:
                    positions.append(ent.position.mean)
        
        if len(positions) < 2:
            return options[0], 0.3
        
        # 简化：分析转向
        turns = []
        for i in range(1, len(positions) - 1):
            prev_dir = positions[i] - positions[i-1]
            next_dir = positions[i+1] - positions[i]
            
            # 叉积判断转向
            cross = prev_dir[0] * next_dir[2] - prev_dir[2] * next_dir[0]
            
            if cross > 0.3:
                turns.append("right")
            elif cross < -0.3:
                turns.append("left")
            else:
                turns.append("forward")
        
        # 匹配选项
        best_match = options[0]
        best_score = 0
        
        for opt in options:
            score = 0
            opt_lower = opt.lower()
            for turn in turns:
                if turn in opt_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = opt
        
        return best_match, 0.5


# ============================================================================
# 3. Critic 模块 - 一致性检查与验证
# ============================================================================

class SpatialCritic:
    """空间推理批判器：验证推理结果的一致性"""
    
    # 物理约束
    PHYSICAL_CONSTRAINTS = {
        "min_object_distance": 0.05,   # 物体间最小距离 (m)
        "max_room_size": 200,          # 最大房间面积 (m²)
        "min_room_size": 4,            # 最小房间面积 (m²)
        "max_object_size": 5.0,        # 最大物体尺寸 (m)
        "min_object_size": 0.01,       # 最小物体尺寸 (m)
    }
    
    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
    
    def evaluate(self, 
                 reasoning_result: ReasoningResult,
                 entities: Dict[str, MindMapEntityV5],
                 voxel_map: SparseVoxelMap,
                 question: str,
                 task_type: TaskType) -> CriticFeedback:
        """评估推理结果"""
        issues = []
        entities_to_reobserve = []
        
        # 检查 1: 置信度检查
        low_conf_entities = self._check_confidence(entities)
        if low_conf_entities:
            issues.append(f"低置信度实体: {low_conf_entities}")
            entities_to_reobserve.extend(low_conf_entities)
        
        # 检查 2: 空间冲突检查
        conflicts = self._check_spatial_conflicts(entities)
        if conflicts:
            issues.append(f"空间冲突: {conflicts}")
            for e1, e2 in conflicts:
                entities_to_reobserve.extend([e1, e2])
        
        # 检查 3: 物理常识检查
        physics_issues = self._check_physics(entities, reasoning_result)
        if physics_issues:
            issues.extend(physics_issues)
        
        # 检查 4: 推理逻辑一致性
        logic_issues = self._check_logic_consistency(reasoning_result, entities, task_type)
        if logic_issues:
            issues.extend(logic_issues)
        
        # 确定判定结果
        if not issues:
            verdict = CriticVerdict.PASS
        elif entities_to_reobserve:
            verdict = CriticVerdict.RE_OBSERVE
        elif any("置信度" in issue for issue in issues):
            verdict = CriticVerdict.LOW_CONFIDENCE
        elif any("冲突" in issue for issue in issues):
            verdict = CriticVerdict.SPATIAL_CONFLICT
        else:
            verdict = CriticVerdict.LOGIC_ERROR
        
        return CriticFeedback(
            verdict=verdict,
            issues=issues,
            suggested_entities_to_reobserve=list(set(entities_to_reobserve)),
            confidence_threshold=self.confidence_threshold
        )
    
    def _check_confidence(self, entities: Dict[str, MindMapEntityV5]) -> List[str]:
        """检查低置信度实体"""
        low_conf = []
        for label, entity in entities.items():
            if entity.position and entity.position.confidence < self.confidence_threshold:
                low_conf.append(label)
        return low_conf
    
    def _check_spatial_conflicts(self, entities: Dict[str, MindMapEntityV5]) -> List[Tuple[str, str]]:
        """检查空间冲突（物体重叠）"""
        conflicts = []
        entities_list = list(entities.items())
        
        for i in range(len(entities_list)):
            for j in range(i + 1, len(entities_list)):
                label1, ent1 = entities_list[i]
                label2, ent2 = entities_list[j]
                
                if not ent1.position or not ent2.position:
                    continue
                
                # 计算距离
                dist = np.linalg.norm(ent1.position.mean - ent2.position.mean)
                
                # 计算最小安全距离（基于尺寸）
                min_dist = self.PHYSICAL_CONSTRAINTS["min_object_distance"]
                if ent1.size_mean is not None and ent2.size_mean is not None:
                    min_dist = (np.min(ent1.size_mean) + np.min(ent2.size_mean)) / 4
                
                if dist < min_dist:
                    conflicts.append((label1, label2))
        
        return conflicts
    
    def _check_physics(self, entities: Dict[str, MindMapEntityV5],
                       reasoning_result: ReasoningResult) -> List[str]:
        """检查物理常识"""
        issues = []
        
        # 检查物体尺寸是否合理
        for label, entity in entities.items():
            if entity.size_mean is not None:
                max_dim = np.max(entity.size_mean)
                if max_dim > self.PHYSICAL_CONSTRAINTS["max_object_size"]:
                    issues.append(f"{label} 尺寸过大: {max_dim:.2f}m")
                elif max_dim < self.PHYSICAL_CONSTRAINTS["min_object_size"]:
                    issues.append(f"{label} 尺寸过小: {max_dim:.4f}m")
        
        # 检查答案是否合理
        try:
            answer = float(reasoning_result.answer)
            # 如果是房间面积
            if answer > self.PHYSICAL_CONSTRAINTS["max_room_size"]:
                issues.append(f"房间面积过大: {answer}m²")
            elif answer < self.PHYSICAL_CONSTRAINTS["min_room_size"] and answer > 0:
                issues.append(f"房间面积过小: {answer}m²")
        except ValueError:
            pass
        
        return issues
    
    def _check_logic_consistency(self, reasoning_result: ReasoningResult,
                                 entities: Dict[str, MindMapEntityV5],
                                 task_type: TaskType) -> List[str]:
        """检查逻辑一致性"""
        issues = []
        
        # 检查相对方向的逻辑一致性
        if task_type in [TaskType.REL_DIRECTION_EASY, TaskType.REL_DIRECTION_MEDIUM, TaskType.REL_DIRECTION_HARD]:
            # 如果 Reasoner 说"A在B左边"，验证坐标
            answer = reasoning_result.answer.lower()
            
            # 这里可以添加更复杂的逻辑验证
            # 例如：检查答案是否与心智地图中的坐标一致
        
        return issues
    
    def needs_evolution(self, feedback: CriticFeedback) -> bool:
        """判断是否需要演化"""
        return feedback.verdict in [
            CriticVerdict.RE_OBSERVE,
            CriticVerdict.LOW_CONFIDENCE,
            CriticVerdict.SPATIAL_CONFLICT
        ]


# ============================================================================
# 4. Evolver 模块 - 视觉回溯与地图更新
# ============================================================================

class MindMapEvolver:
    """心智地图演化器：通过视觉回溯更新地图"""
    
    # 视觉验证 Prompt
    VISUAL_VERIFICATION_PROMPT = """请仔细观察图像中红框标记的区域。

问题：{question}

选项：
{options}

请直接给出你的答案，格式为：
答案: [你的选择]
置信度: [1-10]
原因: [简要说明]"""
    
    def __init__(self, vl_model=None, device: str = 'cuda'):
        """
        Args:
            vl_model: Vision-Language 模型（如 Qwen3-VL）
            device: 计算设备
        """
        self.vl_model = vl_model
        self.device = device
        self._model_loaded = False
    
    def load_qwen_vl(self, model_path: str = None):
        """加载 Qwen3-VL 模型"""
        if self._model_loaded:
            return
        
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            if model_path is None:
                model_path = os.environ.get("QWEN3_VL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
            
            logger.info(f"加载 Qwen3-VL 模型: {model_path}")
            
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            self._model_loaded = True
            logger.info("Qwen3-VL 模型加载完成")
            
        except Exception as e:
            logger.error(f"加载 Qwen3-VL 失败: {e}")
            self.vl_model = None
    
    def refine(self,
               entities: Dict[str, MindMapEntityV5],
               voxel_map: SparseVoxelMap,
               video_path: str,
               entities_to_check: List[str],
               feedback: CriticFeedback) -> EvolutionResult:
        """精炼心智地图"""
        
        corrections = []
        confidence_boost = 0.0
        
        # 提取关键帧
        frames = self._extract_key_frames(video_path, entities, entities_to_check)
        
        for entity_label in entities_to_check:
            if entity_label not in entities:
                continue
            
            entity = entities[entity_label]
            
            # 获取该实体首次出现的帧
            frame_idx = entity.first_seen_frame
            if frame_idx not in frames:
                continue
            
            frame = frames[frame_idx]
            
            # 视觉验证
            if self.vl_model:
                correction = self._visual_verify(frame, entity, entity_label)
                if correction:
                    # 应用修正
                    if "label" in correction:
                        old_label = entity.label
                        entity.label = correction["label"]
                        corrections.append(f"{old_label} -> {correction['label']}")
                    
                    if "confidence_boost" in correction:
                        if entity.position:
                            # 增加位置置信度
                            entity.position.covariance *= 0.8  # 减小不确定性
                        confidence_boost += correction["confidence_boost"]
            else:
                # 无 VL 模型时，使用启发式修正
                correction = self._heuristic_correction(entity, entity_label, feedback)
                if correction:
                    corrections.append(correction)
                    confidence_boost += 0.1
        
        return EvolutionResult(
            updated_entities=entities,
            corrections=corrections,
            confidence_boost=confidence_boost
        )
    
    def _extract_key_frames(self, video_path: str,
                            entities: Dict[str, MindMapEntityV5],
                            entities_to_check: List[str]) -> Dict[int, np.ndarray]:
        """提取关键帧"""
        frames = {}
        
        # 获取需要的帧索引
        frame_indices = set()
        for label in entities_to_check:
            if label in entities:
                frame_indices.add(entities[label].first_seen_frame)
                frame_indices.add(entities[label].last_seen_frame)
        
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for idx in frame_indices:
            if 0 <= idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cap.release()
        return frames
    
    def _visual_verify(self, frame: np.ndarray, 
                       entity: MindMapEntityV5,
                       entity_label: str) -> Optional[Dict]:
        """使用 VL 模型进行视觉验证"""
        if self.vl_model is None:
            return None
        
        try:
            # 构建提示
            question = f"图像中是否存在 {entity_label}？如果存在，请描述它的位置和状态。"
            
            # 准备输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # 生成
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[frame], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.vl_model.generate(**inputs, max_new_tokens=256)
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # 解析响应
            return self._parse_vl_response(response, entity_label)
            
        except Exception as e:
            logger.warning(f"视觉验证失败: {e}")
            return None
    
    def _parse_vl_response(self, response: str, expected_label: str) -> Optional[Dict]:
        """解析 VL 模型响应"""
        response_lower = response.lower()
        
        correction = {}
        
        # 检查是否确认物体存在
        if "yes" in response_lower or "存在" in response_lower or expected_label in response_lower:
            correction["confidence_boost"] = 0.2
        
        # 检查是否建议修正标签
        if "not" in response_lower and expected_label in response_lower:
            # 尝试提取正确标签
            correction["confidence_boost"] = -0.1
        
        return correction if correction else None
    
    def _heuristic_correction(self, entity: MindMapEntityV5,
                              entity_label: str,
                              feedback: CriticFeedback) -> Optional[str]:
        """启发式修正（无 VL 模型时）"""
        corrections = []
        
        # 如果置信度过低，增加位置不确定性
        if entity.position and entity.position.confidence < feedback.confidence_threshold:
            # 增大协方差
            entity.position.covariance *= 1.2
            corrections.append(f"{entity_label}: 增大位置不确定性")
        
        return corrections[0] if corrections else None


# ============================================================================
# 5. 自进化 Agent 主循环
# ============================================================================

class SelfEvolvingAgent:
    """自进化心智地图智能体"""
    
    def __init__(self,
                 device: str = 'cuda',
                 num_frames: int = 32,
                 confidence_threshold: float = 0.4,
                 max_evolution_rounds: int = 2,
                 use_vl_model: bool = True,
                 vl_model_path: str = None):
        """
        Args:
            device: 计算设备
            num_frames: 视频采样帧数
            confidence_threshold: 置信度阈值（低于此值触发演化）
            max_evolution_rounds: 最大演化轮数
            use_vl_model: 是否使用 VL 模型进行视觉回溯
            vl_model_path: VL 模型路径
        """
        self.device = device
        self.num_frames = num_frames
        self.confidence_threshold = confidence_threshold
        self.max_evolution_rounds = max_evolution_rounds
        
        # 初始化各模块
        self.builder = MindMapBuilderV5(
            device=device,
            num_frames=num_frames
        )
        self.manager = TaskManager()
        self.reasoner = SpatialReasoner(use_rule_based=True)
        self.critic = SpatialCritic(confidence_threshold=confidence_threshold)
        self.evolver = MindMapEvolver(device=device)
        
        # 可选：加载 VL 模型
        if use_vl_model and vl_model_path:
            self.evolver.load_qwen_vl(vl_model_path)
        
        # 统计
        self.stats = {
            "total_queries": 0,
            "evolution_triggered": 0,
            "answer_improved": 0,
        }
    
    def process(self,
                video_path: str,
                question: str,
                question_type: str = None,
                options: List[str] = None) -> Dict[str, Any]:
        """处理单个查询
        
        Args:
            video_path: 视频路径
            question: 问题文本
            question_type: 问题类型（可选）
            options: 选项列表（选择题）
        
        Returns:
            {
                "answer": str,
                "confidence": float,
                "evolved": bool,
                "reasoning_steps": List,
                "critic_feedback": Dict,
            }
        """
        self.stats["total_queries"] += 1
        
        # 1. 分析任务类型
        task_type = self.manager.analyze_task(question, question_type)
        logger.info(f"任务类型: {task_type.value}")
        
        # 2. 确定是否使用动态尺度校准
        use_dynamic_scale = (task_type != TaskType.ROOM_SIZE)
        
        # 3. 构建心智地图
        logger.info("构建心智地图...")
        target_objects = self.manager.extract_target_objects(question)
        entities, voxel_map = self.builder.build_from_video(
            video_path,
            target_objects=target_objects,
            use_dynamic_scale=use_dynamic_scale
        )
        
        if not entities:
            return {
                "answer": self._get_default_answer(task_type, options),
                "confidence": 0.1,
                "evolved": False,
                "reasoning_steps": [],
                "critic_feedback": None,
            }
        
        # 4. 初次推理
        logger.info("执行初次推理...")
        reasoning_result = self.reasoner.reason(
            entities, voxel_map, question, task_type, options
        )
        
        # 5. 批判评估
        logger.info("评估推理结果...")
        feedback = self.critic.evaluate(
            reasoning_result, entities, voxel_map, question, task_type
        )
        
        # 6. 演化循环
        evolved = False
        evolution_round = 0
        
        while (self.critic.needs_evolution(feedback) and 
               evolution_round < self.max_evolution_rounds):
            
            self.stats["evolution_triggered"] += 1
            evolution_round += 1
            logger.info(f"触发演化 (第 {evolution_round} 轮): {feedback.verdict.value}")
            
            # 演化：视觉回溯
            evolution_result = self.evolver.refine(
                entities, voxel_map, video_path,
                feedback.suggested_entities_to_reobserve,
                feedback
            )
            
            if evolution_result.corrections:
                logger.info(f"修正: {evolution_result.corrections}")
                evolved = True
                
                # 重新推理
                reasoning_result = self.reasoner.reason(
                    evolution_result.updated_entities, 
                    voxel_map, question, task_type, options
                )
                
                # 重新评估
                feedback = self.critic.evaluate(
                    reasoning_result, evolution_result.updated_entities,
                    voxel_map, question, task_type
                )
            else:
                break
        
        if evolved:
            self.stats["answer_improved"] += 1
        
        return {
            "answer": reasoning_result.answer,
            "confidence": reasoning_result.confidence,
            "evolved": evolved,
            "evolution_rounds": evolution_round,
            "reasoning_steps": [s.to_dict() if hasattr(s, 'to_dict') else {
                "step_name": s.step_name,
                "description": s.description,
                "confidence": s.confidence
            } for s in reasoning_result.steps],
            "critic_feedback": {
                "verdict": feedback.verdict.value,
                "issues": feedback.issues,
            },
        }
    
    def _get_default_answer(self, task_type: TaskType, options: List[str] = None) -> str:
        """获取默认答案"""
        defaults = {
            TaskType.OBJECT_COUNTING: "1",
            TaskType.OBJECT_ABS_DISTANCE: "2.0",
            TaskType.OBJECT_SIZE: "50",
            TaskType.ROOM_SIZE: "20",
        }
        
        if options:
            return options[0]
        
        return defaults.get(task_type, "unknown")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "evolution_rate": (self.stats["evolution_triggered"] / max(1, self.stats["total_queries"])),
            "improvement_rate": (self.stats["answer_improved"] / max(1, self.stats["evolution_triggered"])) 
                               if self.stats["evolution_triggered"] > 0 else 0,
        }


# ============================================================================
# 6. 针对短板任务的专项优化
# ============================================================================

class DirectionOptimizer:
    """相对方向任务优化器"""
    
    @staticmethod
    def compute_ego_centric_direction(observer_pos: np.ndarray,
                                       observer_facing: np.ndarray,
                                       target_pos: np.ndarray) -> Dict[str, Any]:
        """计算自我中心坐标系下的方向
        
        Args:
            observer_pos: 观察者位置 [x, y, z]
            observer_facing: 观察者朝向位置 [x, y, z]
            target_pos: 目标位置 [x, y, z]
        
        Returns:
            {
                "direction": str,  # "left", "right", "front", "back", etc.
                "angle": float,    # 角度（度）
                "confidence": float
            }
        """
        # 计算前向向量（XZ 平面）
        forward = observer_facing - observer_pos
        forward_2d = np.array([forward[0], forward[2]])
        forward_norm = np.linalg.norm(forward_2d)
        
        if forward_norm < 1e-6:
            return {"direction": "unknown", "angle": 0, "confidence": 0}
        
        forward_2d = forward_2d / forward_norm
        
        # 右向量（顺时针旋转90度）
        right_2d = np.array([forward_2d[1], -forward_2d[0]])
        
        # 目标相对向量
        to_target = target_pos - observer_pos
        to_target_2d = np.array([to_target[0], to_target[2]])
        to_target_norm = np.linalg.norm(to_target_2d)
        
        if to_target_norm < 1e-6:
            return {"direction": "same_position", "angle": 0, "confidence": 0.5}
        
        to_target_2d = to_target_2d / to_target_norm
        
        # 计算点积
        front_dot = np.dot(to_target_2d, forward_2d)
        right_dot = np.dot(to_target_2d, right_2d)
        
        # 计算角度
        angle = np.arctan2(right_dot, front_dot) * 180 / np.pi
        
        # 确定方向
        if angle > 157.5 or angle < -157.5:
            direction = "back"
        elif -157.5 <= angle < -112.5:
            direction = "back-left"
        elif -112.5 <= angle < -67.5:
            direction = "left"
        elif -67.5 <= angle < -22.5:
            direction = "front-left"
        elif -22.5 <= angle < 22.5:
            direction = "front"
        elif 22.5 <= angle < 67.5:
            direction = "front-right"
        elif 67.5 <= angle < 112.5:
            direction = "right"
        elif 112.5 <= angle < 157.5:
            direction = "back-right"
        else:
            direction = "unknown"
        
        # 置信度：角度越接近边界，置信度越低
        boundary_distances = [
            abs(angle - 0),
            abs(angle - 45),
            abs(angle - 90),
            abs(angle - 135),
            abs(angle - 180),
            abs(angle + 45),
            abs(angle + 90),
            abs(angle + 135),
        ]
        min_boundary_dist = min(boundary_distances)
        confidence = min(1.0, min_boundary_dist / 22.5)
        
        return {
            "direction": direction,
            "angle": angle,
            "confidence": confidence,
            "front_dot": front_dot,
            "right_dot": right_dot,
        }


class RouteOptimizer:
    """路径规划任务优化器"""
    
    @staticmethod
    def generate_navigable_grid(voxel_map: SparseVoxelMap,
                                 resolution: float = 0.3) -> np.ndarray:
        """生成可导航网格
        
        Returns:
            2D numpy array，0 = 可通行，1 = 障碍物
        """
        if not voxel_map or not voxel_map.voxels:
            return np.zeros((10, 10))
        
        min_bounds = voxel_map.min_bounds
        max_bounds = voxel_map.max_bounds
        
        # 计算网格大小
        x_size = int((max_bounds[0] - min_bounds[0]) / resolution) + 3
        z_size = int((max_bounds[2] - min_bounds[2]) / resolution) + 3
        
        x_size = min(x_size, 50)
        z_size = min(z_size, 50)
        
        # 初始化为可通行
        grid = np.zeros((z_size, x_size), dtype=np.int8)
        
        # 标记障碍物
        for (vx, vy, vz), voxel in voxel_map.voxels.items():
            if voxel.occupancy_prob > 0.3:
                world_x = vx * voxel_map.voxel_size
                world_z = vz * voxel_map.voxel_size
                
                grid_x = int((world_x - min_bounds[0]) / resolution) + 1
                grid_z = int((world_z - min_bounds[2]) / resolution) + 1
                
                if 0 <= grid_x < x_size and 0 <= grid_z < z_size:
                    grid[grid_z, grid_x] = 1
        
        return grid
    
    @staticmethod
    def analyze_path_turns(positions: List[np.ndarray]) -> List[str]:
        """分析路径中的转向
        
        Returns:
            转向列表 ["left", "right", "forward", ...]
        """
        if len(positions) < 3:
            return ["forward"]
        
        turns = []
        for i in range(1, len(positions) - 1):
            prev_dir = positions[i] - positions[i-1]
            next_dir = positions[i+1] - positions[i]
            
            # 在 XZ 平面计算叉积
            cross = prev_dir[0] * next_dir[2] - prev_dir[2] * next_dir[0]
            
            # 计算角度变化
            prev_angle = np.arctan2(prev_dir[2], prev_dir[0])
            next_angle = np.arctan2(next_dir[2], next_dir[0])
            angle_diff = (next_angle - prev_angle) * 180 / np.pi
            
            # 归一化到 [-180, 180]
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            
            if abs(angle_diff) < 30:
                turns.append("go forward")
            elif angle_diff > 0:
                turns.append("turn left")
            else:
                turns.append("turn right")
        
        return turns


# ============================================================================
# 主函数和测试
# ============================================================================

def main():
    """测试自进化 Agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Evolving Mind Map Agent")
    parser.add_argument("--video", type=str, required=True, help="视频路径")
    parser.add_argument("--question", type=str, required=True, help="问题")
    parser.add_argument("--question-type", type=str, default=None, help="问题类型")
    parser.add_argument("--options", type=str, nargs="+", default=None, help="选项")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--use-vl", action="store_true", help="使用 VL 模型")
    parser.add_argument("--vl-model", type=str, default=None, help="VL 模型路径")
    
    args = parser.parse_args()
    
    # 创建 Agent
    agent = SelfEvolvingAgent(
        device=args.device,
        use_vl_model=args.use_vl,
        vl_model_path=args.vl_model
    )
    
    # 处理查询
    result = agent.process(
        video_path=args.video,
        question=args.question,
        question_type=args.question_type,
        options=args.options
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("【推理结果】")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"是否演化: {result['evolved']}")
    if result['evolved']:
        print(f"演化轮数: {result['evolution_rounds']}")
    print(f"\n批判反馈: {result['critic_feedback']}")
    print("=" * 60)
    
    # 输出统计
    print(f"\n统计: {agent.get_stats()}")


if __name__ == "__main__":
    main()
