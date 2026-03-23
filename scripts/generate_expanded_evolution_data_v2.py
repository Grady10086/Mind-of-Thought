#!/usr/bin/env python3
"""
阶段2: 扩大Evolution覆盖范围 - 生成训练数据
目标: 将Evolution覆盖率从3%提升到80%+
"""
import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')

import json
import os
import gc
import torch
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import traceback
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvolutionAction:
    """演化动作"""
    action_type: str
    target_entity: str
    old_value: Any
    new_value: Any
    reasoning: str
    confidence: float = 0.8


class ExpandedMindMapEvolver:
    """扩展的心智地图演化器 - 支持所有任务类型"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.evolution_history = []
    
    def evolve_for_counting(
        self, 
        mind_map: Dict,
        target_object: str,
        frames: List[np.ndarray],
        frame_indices: List[int],
    ) -> Tuple[Dict, List[EvolutionAction]]:
        """针对 counting 任务演化心智地图 - 去重"""
        actions = []
        
        # 找到目标物体
        target_entity = None
        target_label = None
        for label, entity in mind_map.items():
            if self._match_object_name(target_object, label):
                target_entity = entity
                target_label = label
                break
        
        if target_entity is None:
            return mind_map, actions
        
        # 检查是否有重复计数的可能
        frame_dets = defaultdict(list)
        for det in target_entity.detections:
            frame_dets[det.frame_idx].append(det)
        
        counts = [len(dets) for dets in frame_dets.values()]
        if len(counts) > 1:
            max_count = max(counts)
            median_count = np.median(counts)
            
            if max_count > median_count * 2 and max_count > 2:
                new_count = int(np.ceil(median_count))
                if new_count != target_entity.count:
                    actions.append(EvolutionAction(
                        action_type='correct_count',
                        target_entity=target_label,
                        old_value=target_entity.count,
                        new_value=new_count,
                        reasoning=f"Deduplication: max({max_count}) >> median({median_count:.1f})",
                        confidence=0.7,
                    ))
                    target_entity.count = new_count
        
        return mind_map, actions
    
    def evolve_for_size_estimation(
        self,
        mind_map: Dict,
        calibration,
        question: str,
    ) -> Tuple[Dict, List[EvolutionAction]]:
        """针对 size_estimation 任务演化心智地图 - 尺度修正"""
        actions = []
        
        # 提取目标物体
        match = re.search(r'(size|dimensions?|length|width|height)\s+(?:of\s+)?(?:the\s+)?(\w+)', question, re.IGNORECASE)
        if not match:
            match = re.search(r'How\s+(long|wide|tall|big)\s+is\s+(?:the\s+)?(\w+)', question, re.IGNORECASE)
        
        if not match:
            return mind_map, actions
        
        target_object = match.group(2).lower() if match else None
        
        # 找到目标物体
        for label, entity in mind_map.items():
            if target_object and self._match_object_name(target_object, label):
                # 应用尺度校准修正
                if calibration and calibration.scale_factor != 1.0 and entity.size_3d is not None:
                    old_size = entity.size_3d.copy() if hasattr(entity.size_3d, 'copy') else list(entity.size_3d)
                    
                    # 演化：根据校准因子和物理常识修正尺寸
                    # 例如：如果检测到的椅子高度>2米，很可能是错误的
                    corrected = False
                    
                    # 物理常识检查
                    typical_sizes = {
                        'chair': (0.4, 0.8),  # 高度范围
                        'table': (0.7, 1.2),
                        'bed': (0.4, 0.6),
                        'door': (1.8, 2.2),
                        'window': (0.8, 1.5),
                        'sofa': (0.7, 1.0),
                    }
                    
                    for obj_type, (min_h, max_h) in typical_sizes.items():
                        if obj_type in label.lower():
                            size_3d = entity.size_3d
                            height = size_3d[2] if len(size_3d) > 2 else size_3d[1]
                            if height < min_h * 0.5 or height > max_h * 2:
                                # 尺寸异常，进行修正
                                correction_factor = (min_h + max_h) / 2 / height
                                new_size = np.array([s * correction_factor for s in size_3d])
                                
                                actions.append(EvolutionAction(
                                    action_type='correct_size',
                                    target_entity=label,
                                    old_value=list(old_size),
                                    new_value=list(new_size),
                                    reasoning=f"Size correction: {label} height {height:.2f}m outside typical range [{min_h}, {max_h}]m",
                                    confidence=0.6,
                                ))
                                entity.size_3d = new_size
                                corrected = True
                                break
                    
                    if not corrected and calibration.confidence > 0.7:
                        # 应用校准因子
                        actions.append(EvolutionAction(
                            action_type='apply_calibration',
                            target_entity=label,
                            old_value=list(old_size),
                            new_value=list(entity.size_3d),
                            reasoning=f"Applied calibration factor {calibration.scale_factor:.3f} from {calibration.calibration_object}",
                            confidence=calibration.confidence,
                        ))
                break
        
        return mind_map, actions
    
    def evolve_for_distance(
        self,
        mind_map: Dict,
        calibration,
        question: str,
    ) -> Tuple[Dict, List[EvolutionAction]]:
        """针对 distance 任务演化心智地图 - 距离修正"""
        actions = []
        
        # 提取两个物体
        match = re.search(r'distance\s+(?:between|from)\s+(?:the\s+)?(\w+)\s+(?:and|to)\s+(?:the\s+)?(\w+)', question, re.IGNORECASE)
        if not match:
            return mind_map, actions
        
        obj1_name = match.group(1).lower()
        obj2_name = match.group(2).lower()
        
        # 找到两个物体
        obj1_entity = obj2_entity = None
        obj1_label = obj2_label = None
        
        for label, entity in mind_map.items():
            if self._match_object_name(obj1_name, label):
                obj1_entity = entity
                obj1_label = label
            if self._match_object_name(obj2_name, label):
                obj2_entity = entity
                obj2_label = label
        
        if obj1_entity and obj2_entity and obj1_entity.position_3d is not None and obj2_entity.position_3d is not None:
            # 计算当前距离
            pos1 = np.array(obj1_entity.position_3d)
            pos2 = np.array(obj2_entity.position_3d)
            current_dist = np.linalg.norm(pos1 - pos2)
            
            # 物理常识检查：室内距离通常不会超过20米
            if current_dist > 20:
                # 可能是深度估计错误
                correction_factor = 10 / current_dist  # 假设合理距离为10米
                
                # 修正位置
                new_pos1 = pos1 * correction_factor
                new_pos2 = pos2 * correction_factor
                
                actions.append(EvolutionAction(
                    action_type='correct_distance',
                    target_entity=f"{obj1_label}-{obj2_label}",
                    old_value=current_dist,
                    new_value=np.linalg.norm(new_pos1 - new_pos2),
                    reasoning=f"Distance {current_dist:.2f}m exceeds indoor limit, applying correction",
                    confidence=0.5,
                ))
                
                obj1_entity.position_3d = new_pos1
                obj2_entity.position_3d = new_pos2
            
            # 应用三角不等式检查
            # 如果有第三个物体C，检查 |AC| + |BC| >= |AB|
            for label, entity in mind_map.items():
                if label not in [obj1_label, obj2_label] and entity.position_3d is not None:
                    pos3 = np.array(entity.position_3d)
                    d13 = np.linalg.norm(pos1 - pos3)
                    d23 = np.linalg.norm(pos2 - pos3)
                    d12 = np.linalg.norm(pos1 - pos2)
                    
                    # 如果违反三角不等式，修正
                    if d13 + d23 < d12 * 0.9:  # 留一些余量
                        actions.append(EvolutionAction(
                            action_type='triangle_check',
                            target_entity=label,
                            old_value={'d13': d13, 'd23': d23, 'd12': d12},
                            new_value='flagged',
                            reasoning=f"Triangle inequality violated: {d13:.2f} + {d23:.2f} < {d12:.2f}",
                            confidence=0.4,
                        ))
                    break  # 只检查一个第三方物体
        
        return mind_map, actions
    
    def evolve_for_direction(
        self,
        mind_map: Dict,
        question: str,
        options: List[str],
    ) -> Tuple[Dict, List[EvolutionAction]]:
        """针对 direction 任务演化心智地图 - 空间关系修正"""
        actions = []
        
        # 提取物体和方向关系
        match = re.search(r'(\w+)\s+(?:is\s+)?(?:to\s+the\s+)?(left|right|front|behind|above|below)\s+of\s+(?:the\s+)?(\w+)', 
                         question, re.IGNORECASE)
        if not match:
            return mind_map, actions
        
        obj1_name = match.group(1).lower()
        direction = match.group(2).lower()
        obj2_name = match.group(3).lower()
        
        # 找到两个物体
        obj1_entity = obj2_entity = None
        obj1_label = obj2_label = None
        
        for label, entity in mind_map.items():
            if self._match_object_name(obj1_name, label):
                obj1_entity = entity
                obj1_label = label
            if self._match_object_name(obj2_name, label):
                obj2_entity = entity
                obj2_label = label
        
        if obj1_entity and obj2_entity and obj1_entity.position_3d is not None and obj2_entity.position_3d is not None:
            pos1 = np.array(obj1_entity.position_3d)
            pos2 = np.array(obj2_entity.position_3d)
            
            # 计算相对方向
            diff = pos1 - pos2
            
            # 检查方向是否一致
            computed_direction = self._compute_direction(diff)
            
            if computed_direction != direction:
                # 方向不一致，记录但不修改（可能是视角问题）
                actions.append(EvolutionAction(
                    action_type='direction_check',
                    target_entity=f"{obj1_label}-{obj2_label}",
                    old_value=computed_direction,
                    new_value=direction,
                    reasoning=f"Computed direction '{computed_direction}' differs from question '{direction}'",
                    confidence=0.5,
                ))
        
        return mind_map, actions
    
    def evolve_for_appearance_order(
        self,
        mind_map: Dict,
        question: str,
        options: List[str],
    ) -> Tuple[Dict, List[EvolutionAction]]:
        """针对 appearance_order 任务演化心智地图 - 时序修正"""
        actions = []
        
        # 检查时序一致性
        # 按first_seen_frame排序的物体应该形成合理的空间轨迹
        entities_by_time = sorted(
            [(label, entity) for label, entity in mind_map.items()],
            key=lambda x: x[1].first_seen_frame
        )
        
        if len(entities_by_time) < 2:
            return mind_map, actions
        
        # 检查相邻帧的物体位置是否合理（相机移动不应该太剧烈）
        prev_label, prev_entity = entities_by_time[0]
        
        for label, entity in entities_by_time[1:]:
            frame_diff = entity.first_seen_frame - prev_entity.first_seen_frame
            
            if frame_diff > 0 and entity.position_3d is not None and prev_entity.position_3d is not None:
                pos_diff = np.linalg.norm(
                    np.array(entity.position_3d) - np.array(prev_entity.position_3d)
                )
                
                # 假设相机移动速度不超过1米/帧
                max_reasonable_diff = frame_diff * 1.0
                
                if pos_diff > max_reasonable_diff * 3:  # 宽松阈值
                    actions.append(EvolutionAction(
                        action_type='temporal_consistency',
                        target_entity=label,
                        old_value={'frame': entity.first_seen_frame, 'pos_diff': pos_diff},
                        new_value={'expected_max': max_reasonable_diff},
                        reasoning=f"Large position jump ({pos_diff:.2f}m) between frames {prev_entity.first_seen_frame} and {entity.first_seen_frame}",
                        confidence=0.4,
                    ))
            
            prev_label, prev_entity = label, entity
        
        return mind_map, actions
    
    def evolve_for_route_planning(
        self,
        mind_map: Dict,
        question: str,
        options: List[str],
    ) -> Tuple[Dict, List[EvolutionAction]]:
        """针对 route_planning 任务演化心智地图 - 拓扑修正"""
        actions = []
        
        # 构建简单的可达性图
        # 检查是否有障碍物阻挡
        
        entities = list(mind_map.items())
        if len(entities) < 3:
            return mind_map, actions
        
        # 检查物体之间是否有遮挡
        for i, (label1, entity1) in enumerate(entities):
            if entity1.position_3d is None:
                continue
            for j, (label2, entity2) in enumerate(entities[i+1:], i+1):
                if entity2.position_3d is None:
                    continue
                pos1 = np.array(entity1.position_3d)
                pos2 = np.array(entity2.position_3d)
                
                # 检查是否有第三个物体在它们之间
                for k, (label3, entity3) in enumerate(entities):
                    if k == i or k == j or entity3.position_3d is None:
                        continue
                    
                    pos3 = np.array(entity3.position_3d)
                    
                    # 简单的遮挡检测：pos3是否在pos1-pos2连线附近
                    if self._is_between(pos1, pos2, pos3, threshold=0.5):
                        actions.append(EvolutionAction(
                            action_type='occlusion_detected',
                            target_entity=f"{label1}-{label2}",
                            old_value={'blocker': label3},
                            new_value='path_blocked',
                            reasoning=f"Object '{label3}' may block path between '{label1}' and '{label2}'",
                            confidence=0.3,
                        ))
                        break
        
        return mind_map, actions
    
    def _match_object_name(self, target: str, label: str) -> bool:
        """匹配物体名称"""
        target = target.lower().strip()
        label = label.lower().strip()
        
        # 直接匹配
        if target in label or label in target:
            return True
        
        # 复数形式
        if target.endswith('s') and target[:-1] in label:
            return True
        if label.endswith('s') and label[:-1] in target:
            return True
        
        return False
    
    def _compute_direction(self, diff: np.ndarray) -> str:
        """计算方向"""
        x, y, z = diff[0], diff[1], diff[2] if len(diff) > 2 else 0
        
        if abs(x) > abs(y) and abs(x) > abs(z):
            return 'right' if x > 0 else 'left'
        elif abs(y) > abs(x) and abs(y) > abs(z):
            return 'front' if y > 0 else 'behind'
        else:
            return 'above' if z > 0 else 'below'
    
    def _is_between(self, pos1: np.ndarray, pos2: np.ndarray, pos3: np.ndarray, threshold: float) -> bool:
        """检查pos3是否在pos1和pos2之间"""
        # 计算pos3到直线pos1-pos2的距离
        line_vec = pos2 - pos1
        line_len = np.linalg.norm(line_vec)
        if line_len < 0.01:
            return False
        
        line_unit = line_vec / line_len
        point_vec = pos3 - pos1
        
        # 投影长度
        proj_len = np.dot(point_vec, line_unit)
        
        # 检查是否在线段范围内
        if proj_len < 0 or proj_len > line_len:
            return False
        
        # 计算垂直距离
        proj_point = pos1 + proj_len * line_unit
        dist = np.linalg.norm(pos3 - proj_point)
        
        return dist < threshold


def format_mind_map_with_evolution(mind_map: Dict, calibration, evolution_actions: List[EvolutionAction]) -> str:
    """格式化Mind Map为文本，包含演化信息"""
    lines = []
    
    # 校准信息
    if calibration and calibration.calibration_object:
        lines.append(f"Scale Calibration: {calibration.calibration_object} (confidence: {calibration.confidence:.2f}, factor: {calibration.scale_factor:.3f})")
    
    # 演化信息
    if evolution_actions:
        lines.append(f"\n[EVOLUTION APPLIED: {len(evolution_actions)} actions]")
        for action in evolution_actions[:3]:  # 只显示前3个
            lines.append(f"  - {action.action_type}: {action.reasoning[:80]}...")
    
    lines.append("\nDetected Objects:")
    
    for label, entity in sorted(mind_map.items(), key=lambda x: x[1].first_seen_frame):
        pos_3d = entity.position_3d
        size_3d = entity.size_3d
        
        lines.append(f"- {label}")
        lines.append(f"  Count: {entity.count}, Confidence: {entity.avg_confidence:.2f}")
        if pos_3d is not None:
            lines.append(f"  Position: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f}) m")
        if size_3d is not None:
            lines.append(f"  Size: {size_3d[0]:.2f} x {size_3d[1]:.2f} x {size_3d[2]:.2f} m")
        lines.append(f"  First seen: frame {entity.first_seen_frame}")
    
    return '\n'.join(lines)


def worker_process(gpu_id: int, samples: List[Dict], output_file: str):
    """GPU Worker进程"""
    # 添加路径
    import sys
    sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
    sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')
    
    import torch
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    logger.info(f"GPU {gpu_id}: 初始化组件...")
    
    from tests.test_v7_with_finetuned_vl import MindMapBuilder, ScaleCalibrator
    
    builder = MindMapBuilder(device=device, num_frames=32, box_threshold=0.25)
    calibrator = ScaleCalibrator()
    evolver = ExpandedMindMapEvolver(device=device)
    
    builder.load_models()
    logger.info(f"GPU {gpu_id}: 组件初始化完成")
    
    results = []
    evolution_stats = defaultdict(int)
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        try:
            video_path = sample['video_path']
            question = sample['question']
            question_type = sample['question_type']
            answer = sample['ground_truth']
            options = sample.get('options', [])
            
            # 1. 构建Mind Map
            target_objects = []
            if 'counting' in question_type:
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map, total_frames, sampled_frames = builder.build_from_video(
                video_path, target_objects
            )
            
            # 2. 校准
            calibration = calibrator.calculate_scale_factor(mind_map)
            mind_map = calibrator.apply_calibration(mind_map, calibration)
            
            # 3. 🔥 扩大Evolution - 根据任务类型应用不同的演化策略
            all_actions = []
            
            # Counting任务
            if question_type == 'object_counting' and target_objects:
                frame_indices = list(range(len(sampled_frames)))
                mind_map, actions = evolver.evolve_for_counting(
                    mind_map, target_objects[0], sampled_frames, frame_indices
                )
                all_actions.extend(actions)
            
            # Size Estimation任务
            elif question_type in ['object_size_estimation', 'room_size_estimation']:
                mind_map, actions = evolver.evolve_for_size_estimation(
                    mind_map, calibration, question
                )
                all_actions.extend(actions)
            
            # Distance任务
            elif question_type in ['object_abs_distance', 'object_rel_distance']:
                mind_map, actions = evolver.evolve_for_distance(
                    mind_map, calibration, question
                )
                all_actions.extend(actions)
            
            # Direction任务
            elif 'direction' in question_type:
                mind_map, actions = evolver.evolve_for_direction(
                    mind_map, question, options
                )
                all_actions.extend(actions)
            
            # Appearance Order任务
            elif question_type == 'obj_appearance_order':
                mind_map, actions = evolver.evolve_for_appearance_order(
                    mind_map, question, options
                )
                all_actions.extend(actions)
            
            # Route Planning任务
            elif question_type == 'route_planning':
                mind_map, actions = evolver.evolve_for_route_planning(
                    mind_map, question, options
                )
                all_actions.extend(actions)
            
            # 统计
            evolution_applied = len(all_actions) > 0
            if evolution_applied:
                evolution_stats[question_type] += 1
            
            # 4. 格式化Mind Map
            mind_map_text = format_mind_map_with_evolution(mind_map, calibration, all_actions)
            
            # 5. 生成训练样本
            evolution_tag = "WITH EVOLUTION" if evolution_applied else "PERCEPTION ONLY"
            
            training_sample = {
                'conversations': [
                    {
                        'from': 'human',
                        'value': f"""You are a spatial intelligence assistant analyzing a video of an indoor scene.

=== 3D SCENE UNDERSTANDING ({evolution_tag}) ===
{mind_map_text}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Based on the 3D spatial information above and the video frames, answer the question accurately.
"""
                    },
                    {'from': 'gpt', 'value': answer}
                ],
                'videos': [video_path],
                'question_type': question_type,
                'evolution_applied': evolution_applied,
                'evolution_actions': len(all_actions),
            }
            
            results.append(training_sample)
            
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Error processing sample: {e}")
            traceback.print_exc()
            continue
    
    # 保存结果
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # 清理
    builder.unload()
    gc.collect()
    torch.cuda.empty_cache()
    
    total_evolved = sum(evolution_stats.values())
    logger.info(f"GPU {gpu_id}: 完成 {len(results)} 样本, Evolution应用于 {total_evolved} ({total_evolved/len(results)*100:.1f}%)")
    logger.info(f"GPU {gpu_id}: Evolution分布: {dict(evolution_stats)}")


def main():
    print("=" * 80)
    print("阶段2: 扩大Evolution覆盖范围 - 生成训练数据")
    print("目标: Evolution覆盖率 80%+")
    print("=" * 80)
    
    # 加载原始数据
    input_file = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train_balanced_1k_scannet_vsibench.json"
    print(f"\n加载数据: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 统计任务类型分布
    from collections import Counter
    task_dist = Counter(d['question_type'] for d in data)
    print("\n任务类型分布:")
    for task, count in sorted(task_dist.items()):
        print(f"  {task}: {count}")
    
    # 多GPU并行处理
    num_gpus = 8
    samples_per_gpu = len(data) // num_gpus
    
    output_dir = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    processes = []
    output_files = []
    
    print(f"\n启动 {num_gpus} GPU 并行处理...")
    
    for gpu_id in range(num_gpus):
        start = gpu_id * samples_per_gpu
        end = start + samples_per_gpu if gpu_id < num_gpus - 1 else len(data)
        gpu_samples = data[start:end]
        
        output_file = f"{output_dir}/mindmap_expanded_evo_gpu{gpu_id}.jsonl"
        output_files.append(output_file)
        
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_samples, output_file)
        )
        p.start()
        processes.append(p)
    
    # 等待完成
    for p in processes:
        p.join()
    
    # 合并结果
    print("\n合并结果...")
    final_output = f"{output_dir}/mindmap_9908_expanded_evolution.jsonl"
    
    all_samples = []
    evolution_count = 0
    task_evolution = defaultdict(int)
    
    for output_file in output_files:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    all_samples.append(sample)
                    if sample.get('evolution_applied', False):
                        evolution_count += 1
                        task_evolution[sample['question_type']] += 1
    
    with open(final_output, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\n{'=' * 80}")
    print("✅ 完成！")
    print(f"{'=' * 80}")
    print(f"总样本数: {len(all_samples)}")
    print(f"Evolution应用数: {evolution_count} ({evolution_count/len(all_samples)*100:.1f}%)")
    print(f"\nEvolution按任务类型分布:")
    for task, count in sorted(task_evolution.items()):
        total = task_dist[task]
        print(f"  {task}: {count}/{total} ({count/total*100:.1f}%)")
    print(f"\n输出文件: {final_output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
