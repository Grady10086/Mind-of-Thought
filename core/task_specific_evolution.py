#!/usr/bin/env python3
"""
任务特定的Evolution策略设计

核心思想：每种任务有不同的"不确定性来源"，Evolution应该针对这些来源进行修正

任务类型分析：
1. Counting: 不确定性来自多帧重复检测 → 去重
2. Size: 不确定性来自深度估计误差 → 标定物校准
3. Distance: 不确定性来自深度估计误差 → 标定物校准 + 物理约束
4. Direction: 不确定性来自坐标系和位置精度 → 多帧位置聚合 + 空间一致性
5. Appearance Order: 不确定性来自检测时序 → 检测置信度加权
6. Route Planning: 不确定性来自拓扑结构 → 连通性分析
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class MindMapEntity:
    label: str
    detections: List[Any]  # 所有帧的检测
    count: int
    avg_confidence: float
    position_3d: np.ndarray  # (x, y, z)
    size_3d: np.ndarray  # (width, height, depth)
    first_seen_frame: int
    frame_positions: Dict[int, np.ndarray] = None  # 每帧的位置
    frame_confidences: Dict[int, float] = None  # 每帧的置信度


@dataclass
class EvolutionAction:
    action_type: str
    target: str
    before: Any
    after: Any
    reasoning: str
    confidence: float


class TaskSpecificEvolver:
    """
    任务特定的Evolution策略
    
    设计原则：
    1. 每种任务的Evolution应该针对该任务的核心挑战
    2. Evolution应该可解释，有明确的reasoning
    3. Evolution应该有置信度，低置信度的修正要谨慎
    """
    
    def __init__(self):
        # 标定物参考尺寸 (米)
        self.calibration_refs = {
            'door': 2.0,
            'chair': 0.8,
            'table': 0.75,
            'bed': 2.0,
            'toilet': 0.4,
            'refrigerator': 1.7,
            'fridge': 1.7,
            'sofa': 0.85,
            'couch': 0.85,
            'desk': 0.75,
            'window': 1.2,
            'sink': 0.85,
        }
        
        # 物体典型尺寸范围 (min, max) 米
        self.typical_sizes = {
            'chair': (0.4, 1.0),
            'table': (0.6, 1.5),
            'bed': (1.8, 2.2),
            'door': (1.8, 2.2),
            'window': (0.6, 2.0),
            'sofa': (0.6, 1.2),
            'desk': (0.6, 1.0),
            'toilet': (0.3, 0.5),
            'refrigerator': (1.5, 2.0),
            'lamp': (0.2, 1.8),
            'tv': (0.3, 1.5),
        }
    
    # =========================================================================
    # 1. COUNTING TASK EVOLUTION
    # =========================================================================
    
    def evolve_counting(
        self, 
        mind_map: Dict[str, MindMapEntity],
        question: str,
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """
        Counting任务Evolution策略
        
        核心挑战：同一物体在多帧中被重复检测
        
        策略：
        1. 空间聚类：位置相近的检测视为同一物体
        2. 时序一致性：计算每帧的检测数，取中位数
        3. 置信度加权：低置信度检测给予较低权重
        """
        actions = []
        
        # 提取目标物体
        match = re.search(r'How many (\w+)', question, re.IGNORECASE)
        target_obj = match.group(1).lower() if match else None
        
        for label, entity in mind_map.items():
            # 只处理目标物体或所有物体
            if target_obj and target_obj not in label.lower():
                continue
            
            if not entity.detections or len(entity.detections) < 2:
                continue
            
            # 策略1: 基于帧的计数分析
            frame_counts = defaultdict(int)
            for det in entity.detections:
                frame_counts[det.frame_idx] += 1
            
            counts = list(frame_counts.values())
            original_count = entity.count
            
            # 策略2: 使用稳健统计
            if len(counts) >= 3:
                # 使用中位数，对异常值鲁棒
                median_count = int(np.median(counts))
                # 使用众数，最常见的计数
                mode_count = max(set(counts), key=counts.count)
                
                # 如果中位数和众数一致，更有信心
                if median_count == mode_count:
                    evolved_count = median_count
                    confidence = 0.9
                else:
                    # 取较小值，保守估计
                    evolved_count = min(median_count, mode_count)
                    confidence = 0.7
            else:
                evolved_count = int(np.median(counts)) if counts else original_count
                confidence = 0.6
            
            # 策略3: 置信度过滤
            high_conf_dets = [d for d in entity.detections if d.confidence > 0.5]
            if high_conf_dets:
                high_conf_frame_counts = defaultdict(int)
                for det in high_conf_dets:
                    high_conf_frame_counts[det.frame_idx] += 1
                high_conf_median = int(np.median(list(high_conf_frame_counts.values())))
                
                # 如果高置信度计数更低，采用它
                if high_conf_median < evolved_count:
                    evolved_count = high_conf_median
                    confidence = min(confidence + 0.1, 0.95)
            
            if evolved_count != original_count:
                entity.count = evolved_count
                actions.append(EvolutionAction(
                    action_type='count_correction',
                    target=label,
                    before=original_count,
                    after=evolved_count,
                    reasoning=f"Multi-frame analysis: median={np.median(counts):.1f}, "
                              f"mode={max(set(counts), key=counts.count)}, "
                              f"frame_variance={np.std(counts):.2f}",
                    confidence=confidence,
                ))
        
        return mind_map, actions
    
    # =========================================================================
    # 2. SIZE TASK EVOLUTION
    # =========================================================================
    
    def evolve_size(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """
        Size任务Evolution策略
        
        核心挑战：单目深度估计有尺度歧义
        
        策略：
        1. 标定物校准：找到已知尺寸的物体，计算校准系数
        2. 物理约束：检查尺寸是否在合理范围内
        3. 相对尺寸一致性：场景中物体的相对大小应合理
        """
        actions = []
        
        # 策略1: 寻找最佳标定物
        calibration_factor = 1.0
        calibration_source = None
        best_calibration_conf = 0
        
        for label, entity in mind_map.items():
            if entity.size_3d is None:
                continue
            
            for cal_obj, standard_size in self.calibration_refs.items():
                if cal_obj in label.lower():
                    estimated_size = max(entity.size_3d[0], entity.size_3d[1])
                    if estimated_size > 0.01:  # 避免除零
                        factor = standard_size / estimated_size
                        # 只接受合理范围内的校准
                        if 0.2 < factor < 5.0 and entity.avg_confidence > best_calibration_conf:
                            calibration_factor = factor
                            calibration_source = cal_obj
                            best_calibration_conf = entity.avg_confidence
                    break
        
        if calibration_source:
            actions.append(EvolutionAction(
                action_type='calibration_applied',
                target='global_scale',
                before=1.0,
                after=calibration_factor,
                reasoning=f"Using {calibration_source} as reference "
                          f"(confidence: {best_calibration_conf:.2f})",
                confidence=best_calibration_conf,
            ))
        
        # 应用校准并检查物理约束
        for label, entity in mind_map.items():
            if entity.size_3d is None:
                continue
            
            original_size = entity.size_3d.copy()
            
            # 应用校准
            if calibration_factor != 1.0:
                entity.size_3d = entity.size_3d * calibration_factor
                if entity.position_3d is not None:
                    entity.position_3d = entity.position_3d * calibration_factor
            
            # 策略2: 物理约束检查
            for obj_type, (min_size, max_size) in self.typical_sizes.items():
                if obj_type in label.lower():
                    current_height = max(entity.size_3d[0], entity.size_3d[1])
                    
                    if current_height < min_size * 0.3 or current_height > max_size * 3:
                        # 尺寸明显不合理，进行修正
                        target_size = (min_size + max_size) / 2
                        correction = target_size / current_height if current_height > 0 else 1
                        entity.size_3d = entity.size_3d * correction
                        
                        actions.append(EvolutionAction(
                            action_type='physics_constraint',
                            target=label,
                            before=f"{current_height:.2f}m",
                            after=f"{max(entity.size_3d):.2f}m",
                            reasoning=f"Size {current_height:.2f}m outside typical range "
                                      f"[{min_size:.1f}, {max_size:.1f}]m for {obj_type}",
                            confidence=0.7,
                        ))
                    break
        
        return mind_map, actions
    
    # =========================================================================
    # 3. DISTANCE TASK EVOLUTION  
    # =========================================================================
    
    def evolve_distance(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """
        Distance任务Evolution策略
        
        核心挑战：深度估计误差累积到距离计算
        
        策略：
        1. 继承Size的校准（已在evolve_size中处理）
        2. 室内距离约束：室内场景距离通常<10m
        3. 相对深度一致性：检查物体间深度顺序是否合理
        4. 遮挡关系验证：被遮挡物体应该在更远处
        """
        actions = []
        
        # 先应用Size的校准
        mind_map, size_actions = self.evolve_size(mind_map, question)
        actions.extend(size_actions)
        
        # 策略2: 室内距离约束
        max_indoor_dist = 10.0  # 典型室内最大距离
        
        positions = []
        labels = []
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                positions.append(entity.position_3d)
                labels.append(label)
        
        if len(positions) >= 2:
            positions = np.array(positions)
            
            # 检查最大距离
            max_depth = np.max(positions[:, 2])
            if max_depth > max_indoor_dist:
                # 需要缩放
                scale = max_indoor_dist / max_depth * 0.9
                for label, entity in mind_map.items():
                    if entity.position_3d is not None:
                        entity.position_3d = entity.position_3d * scale
                
                actions.append(EvolutionAction(
                    action_type='indoor_constraint',
                    target='all_positions',
                    before=f"max_depth={max_depth:.2f}m",
                    after=f"max_depth={max_depth*scale:.2f}m",
                    reasoning=f"Scaled positions to fit indoor constraint (<{max_indoor_dist}m)",
                    confidence=0.75,
                ))
            
            # 策略3: 深度一致性检查
            depth_std = np.std(positions[:, 2])
            depth_range = np.max(positions[:, 2]) - np.min(positions[:, 2])
            
            if depth_range < 0.5 and len(positions) > 5:
                # 深度变化太小，可能深度估计有问题
                actions.append(EvolutionAction(
                    action_type='depth_warning',
                    target='scene',
                    before=f"depth_range={depth_range:.2f}m",
                    after="flagged",
                    reasoning="Low depth variation may indicate depth estimation issues",
                    confidence=0.5,
                ))
        
        return mind_map, actions
    
    # =========================================================================
    # 4. DIRECTION TASK EVOLUTION
    # =========================================================================
    
    def evolve_direction(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """
        Direction任务Evolution策略
        
        核心挑战：
        1. 单帧位置可能有误差
        2. 需要准确的相对位置关系
        3. 坐标系理解（左右、前后）
        
        策略：
        1. 多帧位置聚合：使用中位数位置，减少单帧误差
        2. 位置稳定性检查：位置方差大的物体标记低置信度
        3. 空间关系验证：检查物体间关系是否符合常识
        """
        actions = []
        
        # 先应用基础校准
        mind_map, _ = self.evolve_size(mind_map, question)
        
        # 策略1: 多帧位置聚合
        for label, entity in mind_map.items():
            if not entity.detections or len(entity.detections) < 2:
                continue
            
            # 收集所有帧的位置
            frame_positions = []
            for det in entity.detections:
                if hasattr(det, 'position_3d') and det.position_3d is not None:
                    frame_positions.append(det.position_3d)
            
            if len(frame_positions) >= 3:
                positions_array = np.array(frame_positions)
                
                # 使用中位数位置（对异常值鲁棒）
                median_position = np.median(positions_array, axis=0)
                original_position = entity.position_3d.copy() if entity.position_3d is not None else None
                
                # 计算位置稳定性
                position_std = np.std(positions_array, axis=0)
                stability = 1.0 / (1.0 + np.mean(position_std))  # 稳定性分数
                
                # 更新位置
                entity.position_3d = median_position
                
                if original_position is not None:
                    position_change = np.linalg.norm(median_position - original_position)
                    if position_change > 0.1:  # 显著变化
                        actions.append(EvolutionAction(
                            action_type='position_aggregation',
                            target=label,
                            before=f"({original_position[0]:.2f}, {original_position[1]:.2f}, {original_position[2]:.2f})",
                            after=f"({median_position[0]:.2f}, {median_position[1]:.2f}, {median_position[2]:.2f})",
                            reasoning=f"Aggregated from {len(frame_positions)} frames, "
                                      f"position_std=({position_std[0]:.2f}, {position_std[1]:.2f}, {position_std[2]:.2f}), "
                                      f"stability={stability:.2f}",
                            confidence=stability,
                        ))
        
        # 策略2: 提取问题中的关键物体并验证
        # 解析问题中的物体
        obj_patterns = [
            r'from (?:the )?(\w+)',
            r'of (?:the )?(\w+)',
            r'at (?:the )?(\w+)',
            r'toward (?:the )?(\w+)',
            r'facing (?:the )?(\w+)',
        ]
        
        mentioned_objects = set()
        for pattern in obj_patterns:
            matches = re.findall(pattern, question.lower())
            mentioned_objects.update(matches)
        
        # 检查关键物体是否存在
        found_objects = []
        for obj in mentioned_objects:
            for label in mind_map.keys():
                if obj in label.lower():
                    found_objects.append(label)
                    break
        
        if len(mentioned_objects) > 0 and len(found_objects) < len(mentioned_objects):
            missing = mentioned_objects - {o.lower() for o in found_objects}
            actions.append(EvolutionAction(
                action_type='missing_object_warning',
                target='question_objects',
                before=str(mentioned_objects),
                after=str(found_objects),
                reasoning=f"Some objects from question not found: {missing}",
                confidence=0.5,
            ))
        
        return mind_map, actions
    
    # =========================================================================
    # 5. APPEARANCE ORDER TASK EVOLUTION
    # =========================================================================
    
    def evolve_appearance_order(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """
        Appearance Order任务Evolution策略
        
        核心挑战：
        1. 检测可能漏检（物体存在但未被检测到）
        2. 检测置信度随时间变化
        3. 遮挡导致的误检
        
        策略：
        1. 置信度加权的首现帧：使用高置信度检测确定首现
        2. 检测连续性分析：检查检测是否连续，不连续可能是误检
        3. 空间位置辅助：靠近相机的物体通常先被检测到
        """
        actions = []
        
        # 收集时序信息
        temporal_info = []
        for label, entity in mind_map.items():
            if not entity.detections:
                continue
            
            # 策略1: 置信度加权的首现帧
            frame_conf = defaultdict(list)
            for det in entity.detections:
                frame_conf[det.frame_idx].append(det.confidence)
            
            # 找到第一个高置信度检测
            first_high_conf_frame = None
            for frame_idx in sorted(frame_conf.keys()):
                max_conf = max(frame_conf[frame_idx])
                if max_conf > 0.4:
                    first_high_conf_frame = frame_idx
                    break
            
            if first_high_conf_frame is None:
                first_high_conf_frame = min(frame_conf.keys())
            
            # 策略2: 检测连续性分析
            frames = sorted(frame_conf.keys())
            if len(frames) >= 2:
                gaps = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
                avg_gap = np.mean(gaps)
                max_gap = max(gaps)
                continuity = 1.0 if max_gap < avg_gap * 3 else 0.5
            else:
                continuity = 0.5
            
            # 更新首现帧
            original_first_frame = entity.first_seen_frame
            if first_high_conf_frame != original_first_frame:
                entity.first_seen_frame = first_high_conf_frame
                actions.append(EvolutionAction(
                    action_type='first_frame_correction',
                    target=label,
                    before=original_first_frame,
                    after=first_high_conf_frame,
                    reasoning=f"Using first high-confidence (>0.4) detection, "
                              f"continuity={continuity:.2f}",
                    confidence=continuity * entity.avg_confidence,
                ))
            
            temporal_info.append({
                'label': label,
                'first_frame': entity.first_seen_frame,
                'confidence': entity.avg_confidence,
                'continuity': continuity,
                'depth': entity.position_3d[2] if entity.position_3d is not None else 999,
            })
        
        # 策略3: 空间位置辅助排序
        # 深度较小（靠近相机）的物体如果首现帧较晚，可能是漏检
        temporal_info.sort(key=lambda x: x['first_frame'])
        
        for i, info in enumerate(temporal_info):
            # 检查是否有更近的物体出现更晚
            for j, other in enumerate(temporal_info[i+1:], i+1):
                if other['depth'] < info['depth'] - 1.0:  # 显著更近
                    if other['first_frame'] > info['first_frame'] + 5:  # 显著更晚
                        actions.append(EvolutionAction(
                            action_type='temporal_anomaly',
                            target=f"{info['label']} vs {other['label']}",
                            before=f"{info['label']}@{info['first_frame']} (depth={info['depth']:.1f})",
                            after=f"{other['label']}@{other['first_frame']} (depth={other['depth']:.1f})",
                            reasoning="Closer object appears later - possible detection miss",
                            confidence=0.6,
                        ))
        
        return mind_map, actions
    
    # =========================================================================
    # 6. ROUTE PLANNING TASK EVOLUTION
    # =========================================================================
    
    def evolve_route(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """
        Route Planning任务Evolution策略
        
        核心挑战：
        1. 需要理解场景的空间布局
        2. 需要识别可通行区域
        3. 需要理解物体间的连接关系
        
        策略：
        1. 构建空间拓扑图：计算物体间距离矩阵
        2. 识别功能区域：基于物体类型推断区域（厨房、卧室等）
        3. 路径可行性：检查路径上是否有障碍
        """
        actions = []
        
        # 先应用基础校准
        mind_map, _ = self.evolve_size(mind_map, question)
        
        # 收集位置信息
        objects_with_pos = []
        for label, entity in mind_map.items():
            if entity.position_3d is not None:
                objects_with_pos.append({
                    'label': label,
                    'position': entity.position_3d,
                    'size': entity.size_3d if entity.size_3d is not None else np.array([0.5, 0.5, 0.5]),
                })
        
        if len(objects_with_pos) < 2:
            return mind_map, actions
        
        # 策略1: 构建距离矩阵
        n = len(objects_with_pos)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(
                    objects_with_pos[i]['position'] - objects_with_pos[j]['position']
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # 找到最近邻
        nearest_neighbors = []
        for i in range(n):
            distances = distance_matrix[i].copy()
            distances[i] = np.inf  # 排除自己
            nearest_idx = np.argmin(distances)
            nearest_neighbors.append({
                'object': objects_with_pos[i]['label'],
                'nearest': objects_with_pos[nearest_idx]['label'],
                'distance': distances[nearest_idx],
            })
        
        # 策略2: 识别功能区域
        kitchen_objects = ['stove', 'refrigerator', 'fridge', 'sink', 'microwave', 'oven']
        bedroom_objects = ['bed', 'nightstand', 'dresser', 'closet']
        living_objects = ['sofa', 'couch', 'tv', 'television', 'coffee table']
        bathroom_objects = ['toilet', 'bathtub', 'shower', 'sink']
        
        detected_areas = defaultdict(list)
        for obj in objects_with_pos:
            label_lower = obj['label'].lower()
            if any(k in label_lower for k in kitchen_objects):
                detected_areas['kitchen'].append(obj['label'])
            elif any(k in label_lower for k in bedroom_objects):
                detected_areas['bedroom'].append(obj['label'])
            elif any(k in label_lower for k in living_objects):
                detected_areas['living_room'].append(obj['label'])
            elif any(k in label_lower for k in bathroom_objects):
                detected_areas['bathroom'].append(obj['label'])
        
        if detected_areas:
            actions.append(EvolutionAction(
                action_type='area_detection',
                target='scene_layout',
                before='unknown',
                after=dict(detected_areas),
                reasoning=f"Detected {len(detected_areas)} functional areas based on objects",
                confidence=0.7,
            ))
        
        # 策略3: 记录空间拓扑信息
        avg_distance = np.mean(distance_matrix[distance_matrix > 0])
        max_distance = np.max(distance_matrix)
        
        actions.append(EvolutionAction(
            action_type='topology_analysis',
            target='scene',
            before='raw_positions',
            after=f"avg_dist={avg_distance:.2f}m, max_dist={max_distance:.2f}m, objects={n}",
            reasoning=f"Built spatial topology with {n} objects",
            confidence=0.8,
        ))
        
        return mind_map, actions
    
    # =========================================================================
    # 主Evolution函数
    # =========================================================================
    
    def evolve(
        self,
        mind_map: Dict[str, MindMapEntity],
        question: str,
        question_type: str,
    ) -> Tuple[Dict[str, MindMapEntity], List[EvolutionAction]]:
        """
        根据任务类型选择合适的Evolution策略
        """
        all_actions = []
        
        # 1. 通用预处理：低置信度过滤
        low_conf_labels = [
            label for label, entity in mind_map.items() 
            if entity.avg_confidence < 0.25
        ]
        for label in low_conf_labels:
            all_actions.append(EvolutionAction(
                action_type='low_confidence_filter',
                target=label,
                before=f"conf={mind_map[label].avg_confidence:.2f}",
                after='removed',
                reasoning="Confidence below threshold (0.25)",
                confidence=0.9,
            ))
            del mind_map[label]
        
        # 2. 任务特定Evolution
        if 'counting' in question_type.lower():
            mind_map, actions = self.evolve_counting(mind_map, question)
        elif 'size' in question_type.lower():
            mind_map, actions = self.evolve_size(mind_map, question)
        elif 'distance' in question_type.lower():
            mind_map, actions = self.evolve_distance(mind_map, question)
        elif 'direction' in question_type.lower():
            mind_map, actions = self.evolve_direction(mind_map, question)
        elif 'appearance' in question_type.lower() or 'order' in question_type.lower():
            mind_map, actions = self.evolve_appearance_order(mind_map, question)
        elif 'route' in question_type.lower():
            mind_map, actions = self.evolve_route(mind_map, question)
        else:
            # 默认：只做基础校准
            mind_map, actions = self.evolve_size(mind_map, question)
        
        all_actions.extend(actions)
        
        return mind_map, all_actions


# =========================================================================
# 使用示例
# =========================================================================

if __name__ == '__main__':
    evolver = TaskSpecificEvolver()
    
    print("=" * 80)
    print("任务特定Evolution策略设计")
    print("=" * 80)
    
    print("""
每种任务的Evolution策略总结：

1. COUNTING
   - 核心问题：多帧重复检测
   - 策略：中位数计数 + 置信度加权 + 空间去重
   
2. SIZE  
   - 核心问题：深度尺度歧义
   - 策略：标定物校准 + 物理约束检查
   
3. DISTANCE
   - 核心问题：深度误差累积
   - 策略：继承Size校准 + 室内距离约束 + 深度一致性
   
4. DIRECTION
   - 核心问题：单帧位置误差
   - 策略：多帧位置聚合 + 位置稳定性分析 + 关键物体验证
   
5. APPEARANCE ORDER
   - 核心问题：检测漏检/误检
   - 策略：置信度加权首现帧 + 检测连续性 + 空间位置辅助
   
6. ROUTE PLANNING
   - 核心问题：空间布局理解
   - 策略：空间拓扑图 + 功能区域识别 + 路径可行性
""")
