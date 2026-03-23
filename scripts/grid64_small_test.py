#!/usr/bin/env python3
"""
64³ Grid Mind Map + Evolution 小样本测试

对比基准: V7原始 VL Overall = 63.61%
         outputs/evolving_agent_v7_20260203_134612/

测试方案:
1. 从V7原始结果中取100个样本(每种任务10个)
2. 用已有的mind_map_summary + DA3信息构建64³ Grid
3. 直接用Grid工具回答(无VL,无Evolution) = Grid baseline
4. 分析哪些样本Grid工具能答对但VL答错,反之亦然
5. 模拟Oracle Evolution后Grid工具的效果

目的: 确定64³ Grid本身(DA3精度下)的baseline,以及Evolution的修正空间
"""

import json
import re
import sys
import os
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# ============================================================================
# 64³ Grid 数据结构
# ============================================================================

@dataclass
class GridEntity:
    """64³ Grid中的Entity"""
    entity_id: str
    category: str
    grid_position: Tuple[int, int, int]  # (x, y, z) in [0, 63]
    grid_size: Tuple[int, int, int] = (1, 1, 1)
    confidence: float = 0.5
    first_seen_frame: int = 0
    count_in_frame: int = 1
    size_cm: float = 0.0  # 物理尺寸(cm)
    position_3d: Optional[np.ndarray] = None  # 原始3D位置(m)


class Grid64:
    """64×64×64 场景Grid"""
    
    def __init__(self):
        self.entities: Dict[str, GridEntity] = {}
        self.scene_min = np.array([0.0, 0.0, 0.0])
        self.scene_max = np.array([5.0, 5.0, 3.0])
        self.scale_factor = 1.0
        self.grid_size = 64
    
    @property
    def scale_per_grid(self) -> float:
        """每格的物理尺寸(m)"""
        ranges = self.scene_max - self.scene_min
        return float(np.max(ranges)) / self.grid_size
    
    def world_to_grid(self, pos_3d: np.ndarray) -> Tuple[int, int, int]:
        """世界坐标 → Grid坐标"""
        normalized = (pos_3d - self.scene_min) / (self.scene_max - self.scene_min + 1e-6)
        normalized = np.clip(normalized, 0, 1)
        grid_coord = (normalized * (self.grid_size - 1)).astype(int)
        grid_coord = np.clip(grid_coord, 0, self.grid_size - 1)
        return tuple(grid_coord)
    
    def grid_distance(self, id1: str, id2: str) -> Optional[float]:
        """两Entity间的grid距离 → 物理距离(m)"""
        if id1 not in self.entities or id2 not in self.entities:
            return None
        e1, e2 = self.entities[id1], self.entities[id2]
        
        # 优先用原始3D位置（更精确）
        if e1.position_3d is not None and e2.position_3d is not None:
            return float(np.linalg.norm(e1.position_3d - e2.position_3d)) * self.scale_factor
        
        # 否则用grid坐标
        p1 = np.array(e1.grid_position, dtype=float)
        p2 = np.array(e2.grid_position, dtype=float)
        return float(np.linalg.norm(p1 - p2)) * self.scale_per_grid * self.scale_factor
    
    def get_by_category(self, category: str) -> List[GridEntity]:
        return [e for e in self.entities.values() if match_name(category, e.category)]
    
    def get_relative_direction(self, observer_id: str, facing_id: str, target_id: str) -> str:
        """计算相对方向: 站在observer面向facing,target在哪个方位"""
        if observer_id not in self.entities or facing_id not in self.entities or target_id not in self.entities:
            return "unknown"
        
        obs = np.array(self.entities[observer_id].grid_position, dtype=float)
        fac = np.array(self.entities[facing_id].grid_position, dtype=float)
        tgt = np.array(self.entities[target_id].grid_position, dtype=float)
        
        # 面向向量 (xz平面)
        facing_dir = fac[:2] - obs[:2]  # 只看x,z (水平面)
        if np.linalg.norm(facing_dir) < 0.01:
            facing_dir = np.array([0, 1])  # 默认朝前
        facing_dir = facing_dir / (np.linalg.norm(facing_dir) + 1e-6)
        
        # target相对observer的向量
        target_dir = tgt[:2] - obs[:2]
        if np.linalg.norm(target_dir) < 0.01:
            return "same_position"
        
        # 前后: dot product with facing
        forward = np.dot(target_dir, facing_dir)
        # 左右: cross product (2D)
        right = facing_dir[0] * target_dir[1] - facing_dir[1] * target_dir[0]
        
        if forward >= 0 and right >= 0:
            return "front-right"
        elif forward >= 0 and right < 0:
            return "front-left"
        elif forward < 0 and right >= 0:
            return "back-right"
        else:
            return "back-left"
    
    def get_left_right(self, observer_id: str, facing_id: str, target_id: str) -> str:
        """简单左右判断"""
        direction = self.get_relative_direction(observer_id, facing_id, target_id)
        if "left" in direction:
            return "left"
        elif "right" in direction:
            return "right"
        return "unknown"


# ============================================================================
# 工具函数
# ============================================================================

def match_name(target: str, label: str) -> bool:
    """模糊匹配"""
    target = target.lower().strip()
    label = label.lower().strip()
    if target == label or target in label or label in target:
        return True
    target_words = set(target.replace('_', ' ').replace('-', ' ').split())
    label_words = set(label.replace('_', ' ').replace('-', ' ').split())
    if target_words & label_words:
        return True
    synonyms = {
        'sofa': ['couch'], 'couch': ['sofa'],
        'tv': ['television', 'monitor'], 'television': ['tv'],
        'fridge': ['refrigerator'], 'refrigerator': ['fridge'],
        'desk': ['table'], 'stool': ['chair'],
        'trash': ['bin', 'garbage', 'trashcan'], 'bin': ['trash', 'garbage'],
        'rug': ['carpet'], 'carpet': ['rug'],
        'wardrobe': ['closet', 'cabinet'], 'closet': ['wardrobe'],
        'backpack': ['bag'], 'bag': ['backpack'],
    }
    for t_word in target_words:
        if t_word in synonyms:
            if any(s in label_words for s in synonyms[t_word]):
                return True
    return False


def build_grid_from_v7_result(sample: Dict) -> Grid64:
    """从V7结果中的mind_map_summary构建64³ Grid"""
    grid = Grid64()
    mm = sample.get('mind_map_summary', {})
    cal = sample.get('calibration', {})
    
    grid.scale_factor = cal.get('scale_factor', 1.0)
    
    # 给每个Entity分配grid位置
    # 由于V7 summary只保存了count/confidence/first_frame,没有精确3D位置
    # 我们用hash + 一些启发式分配位置
    n_entities = len(mm)
    for i, (label, info) in enumerate(mm.items()):
        # 均匀分布在grid空间中
        angle = 2 * np.pi * i / max(n_entities, 1)
        radius = 20  # 距中心20格
        x = int(32 + radius * np.cos(angle))
        y = int(32 + radius * np.sin(angle))
        z = 8  # 地面层
        
        x = max(0, min(63, x))
        y = max(0, min(63, y))
        
        eid = f"{label.replace(' ', '_')}"
        grid.entities[eid] = GridEntity(
            entity_id=eid,
            category=label,
            grid_position=(x, y, z),
            confidence=info.get('confidence', 0.5),
            first_seen_frame=info.get('first_seen_frame', 0),
            count_in_frame=info.get('count', 1),
        )
    
    return grid


def mean_relative_accuracy(pred: float, target: float) -> float:
    """MRA评估"""
    if target == 0:
        return 1.0 if pred == 0 else 0.0
    abs_dist_norm = abs(pred - target) / target
    conf_intervs = np.linspace(0.5, 0.95, 11)
    accuracy = (abs_dist_norm <= (1 - conf_intervs)).astype(float)
    return float(accuracy.mean())


# ============================================================================
# Grid工具回答各类问题
# ============================================================================

def grid_answer_counting(grid: Grid64, question: str) -> Tuple[str, str]:
    """Grid工具: Counting"""
    match = re.search(r'How many (\w+)', question, re.IGNORECASE)
    if not match:
        return "0", "no target"
    target = match.group(1).lower()
    
    matched = grid.get_by_category(target)
    if matched:
        count = matched[0].count_in_frame
        return str(count), f"Grid: {matched[0].entity_id} count={count}"
    return "0", f"Grid: '{target}' not found"


def grid_answer_size(grid: Grid64, question: str) -> Tuple[str, str]:
    """Grid工具: Size estimation"""
    match = re.search(r'of the (\w+)', question.lower())
    if not match:
        return "100", "no target"
    target = match.group(1)
    
    matched = grid.get_by_category(target)
    if matched:
        entity = matched[0]
        if entity.size_cm > 0:
            return str(int(entity.size_cm)), f"Grid: size={entity.size_cm:.0f}cm"
    
    # 用典型值
    typical = {'chair':80,'table':150,'sofa':200,'bed':200,'tv':100,'door':200,'window':120,
               'toilet':60,'sink':50,'lamp':50,'bathtub':170,'refrigerator':180,'desk':120,
               'stove':65,'oven':65,'microwave':45,'pillow':40,'monitor':60}
    for k, v in typical.items():
        if k in target.lower():
            return str(v), f"Grid: typical size for {k}"
    return "100", "Grid: default"


def grid_answer_room_size(grid: Grid64, question: str) -> Tuple[str, str]:
    """Grid工具: Room size"""
    if not grid.entities:
        return "25", "no entities"
    
    positions = []
    for e in grid.entities.values():
        if e.position_3d is not None:
            positions.append(e.position_3d[:2])
        else:
            # 从grid坐标反推
            g = np.array(e.grid_position, dtype=float)
            world = grid.scene_min + g / 63.0 * (grid.scene_max - grid.scene_min)
            positions.append(world[:2])
    
    if len(positions) >= 2:
        positions = np.array(positions)
        x_range = (positions[:, 0].max() - positions[:, 0].min()) * grid.scale_factor
        y_range = (positions[:, 1].max() - positions[:, 1].min()) * grid.scale_factor
        area = max((x_range + 2) * (y_range + 2), 10)
        return str(int(area)), f"Grid: area={area:.1f}m²"
    return "25", "Grid: default"


def grid_answer_distance(grid: Grid64, question: str) -> Tuple[str, str]:
    """Grid工具: Absolute distance"""
    patterns = [
        r'distance between (?:the )?([a-z]+(?:\s+[a-z]+)*)\s+and\s+(?:the )?([a-z]+(?:\s+[a-z]+)*)',
        r'from (?:the )?(\w+) to (?:the )?(\w+)',
        r'between (?:the )?(\w+) and (?:the )?(\w+)',
    ]
    
    obj1, obj2 = None, None
    for pattern in patterns:
        m = re.search(pattern, question.lower())
        if m:
            obj1, obj2 = m.groups()[:2]
            break
    
    if obj1 and obj2:
        e1_list = grid.get_by_category(obj1)
        e2_list = grid.get_by_category(obj2)
        if e1_list and e2_list:
            dist = grid.grid_distance(e1_list[0].entity_id, e2_list[0].entity_id)
            if dist is not None:
                return f"{dist:.2f}", f"Grid: dist={dist:.2f}m"
    return "2.0", "Grid: default"


def grid_answer_appearance_order(grid: Grid64, question: str, options: List[str]) -> Tuple[str, str]:
    """Grid工具: Appearance order (直接从first_seen_frame排序)"""
    if not options:
        return "A", "no options"
    
    # 提取目标物体
    match = re.search(r'following categories.*?:\s*(.+?)\?', question, re.IGNORECASE)
    if match:
        target_objects = [obj.strip().lower() for obj in match.group(1).split(',')]
    else:
        opt_content = re.sub(r'^[A-D]\.\s*', '', options[0])
        target_objects = [obj.strip().lower() for obj in opt_content.split(',')]
    
    # 获取每个物体的first_seen_frame
    obj_frames = {}
    for target in target_objects:
        for eid, entity in grid.entities.items():
            if match_name(target, entity.category):
                obj_frames[target] = entity.first_seen_frame
                break
    
    if len(obj_frames) >= len(target_objects) * 0.5:
        # 按frame排序
        sorted_objs = sorted(obj_frames.keys(), key=lambda x: obj_frames.get(x, 99999))
        sorted_str = ", ".join(sorted_objs)
        
        # 匹配选项
        best_match = "A"
        best_score = -1
        for opt in options:
            letter = opt[0]
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt).lower()
            opt_items = [o.strip() for o in opt_content.split(',')]
            
            score = 0
            for i, item in enumerate(opt_items):
                if i < len(sorted_objs) and match_name(item, sorted_objs[i]):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = letter
        
        return best_match, f"Grid: order by first_frame: {sorted_str}"
    return "A", "Grid: insufficient data"


def grid_answer_direction(grid: Grid64, question: str, options: List[str], question_type: str) -> Tuple[str, str]:
    """Grid工具: Direction (从Grid坐标直接计算)"""
    if not options:
        return "A", "no options"
    
    # 这里Grid坐标不精确(从summary构建的),所以返回当前Rule的结果
    # 在实际DA3构建的Grid中,坐标是精确的
    return "A", "Grid: need precise coordinates for direction"


def grid_answer_rel_distance(grid: Grid64, question: str, options: List[str]) -> Tuple[str, str]:
    """Grid工具: Relative distance comparison"""
    return "A", "Grid: need precise coordinates for distance comparison"


def grid_answer_route(grid: Grid64, question: str, options: List[str]) -> Tuple[str, str]:
    """Grid工具: Route planning"""
    return "A", "Grid: need spatial topology for route"


# ============================================================================
# 评估
# ============================================================================

def evaluate_sample(qt: str, pred: str, gt: str) -> float:
    """评估单个样本"""
    if qt in ("object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"):
        try:
            return mean_relative_accuracy(float(pred), float(gt))
        except:
            return 0.0
    else:
        p = pred.strip().upper()[:1] if pred else ""
        g = gt.strip().upper()[:1]
        return 1.0 if p == g else 0.0


# ============================================================================
# 主逻辑
# ============================================================================

def main():
    results_path = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/evolving_agent_v7_20260203_134612/detailed_results.json"
    
    with open(results_path) as f:
        results = json.load(f)
    
    print("=" * 80)
    print("64³ Grid 小样本测试")
    print(f"基准: V7原始 VL Overall = 63.61%")
    print(f"总样本: {len(results)}")
    print("=" * 80)
    
    # 每种任务类型的VL MRA (V7基准)
    task_types = sorted(set(r['question_type'] for r in results))
    
    print(f"\n{'Task':<35} {'N':>5} {'V7 VL':>7} {'V7 Rule':>7} {'Grid工具':>8} {'Oracle':>7}")
    print(f"{'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
    
    all_v7_vl = []
    all_v7_rule = []
    all_grid = []
    all_oracle = []
    
    # 逐样本分析
    improvement_examples = []
    degradation_examples = []
    
    for qt in task_types:
        samples = [r for r in results if r['question_type'] == qt]
        
        v7_vl_scores = []
        v7_rule_scores = []
        grid_scores = []
        oracle_scores = []
        
        for r in samples:
            gt = r['ground_truth']
            v7_vl = r.get('vl_score', 0)
            v7_rule = r.get('rule_score', 0)
            
            # 构建Grid
            grid = build_grid_from_v7_result(r)
            
            # Grid工具回答
            if qt == 'object_counting':
                grid_pred, grid_reason = grid_answer_counting(grid, r['question'])
            elif qt == 'object_size_estimation':
                grid_pred, grid_reason = grid_answer_size(grid, r['question'])
            elif qt == 'room_size_estimation':
                grid_pred, grid_reason = grid_answer_room_size(grid, r['question'])
            elif qt == 'object_abs_distance':
                grid_pred, grid_reason = grid_answer_distance(grid, r['question'])
            elif qt == 'obj_appearance_order':
                grid_pred, grid_reason = grid_answer_appearance_order(grid, r['question'], r.get('options', []))
            elif 'direction' in qt:
                grid_pred, grid_reason = grid_answer_direction(grid, r['question'], r.get('options', []), qt)
            elif qt == 'object_rel_distance':
                grid_pred, grid_reason = grid_answer_rel_distance(grid, r['question'], r.get('options', []))
            elif qt == 'route_planning':
                grid_pred, grid_reason = grid_answer_route(grid, r['question'], r.get('options', []))
            else:
                grid_pred = ""
                grid_reason = "unknown task"
            
            grid_score = evaluate_sample(qt, grid_pred, gt)
            
            # Oracle = 如果Grid能被完美Evolution修正 → 正确答案
            oracle_score = 1.0  # 完美Evolution
            
            v7_vl_scores.append(v7_vl)
            v7_rule_scores.append(v7_rule)
            grid_scores.append(grid_score)
            oracle_scores.append(oracle_score)
            
            # 记录有趣的案例
            if grid_score > v7_vl + 0.1:
                improvement_examples.append({
                    'qt': qt, 'q': r['question'][:60], 'gt': gt,
                    'vl': v7_vl, 'grid': grid_score, 'reason': grid_reason
                })
            elif grid_score < v7_vl - 0.1:
                degradation_examples.append({
                    'qt': qt, 'q': r['question'][:60], 'gt': gt,
                    'vl': v7_vl, 'grid': grid_score, 'grid_pred': grid_pred, 'reason': grid_reason
                })
        
        v7_vl_mean = np.mean(v7_vl_scores)
        v7_rule_mean = np.mean(v7_rule_scores)
        grid_mean = np.mean(grid_scores)
        oracle_mean = np.mean(oracle_scores)
        
        all_v7_vl.extend(v7_vl_scores)
        all_v7_rule.extend(v7_rule_scores)
        all_grid.extend(grid_scores)
        all_oracle.extend(oracle_scores)
        
        delta = grid_mean - v7_vl_mean
        marker = "✓" if delta > 0 else "✗"
        print(f"  {qt:<35} {len(samples):>5} {v7_vl_mean:>6.3f} {v7_rule_mean:>6.3f} {grid_mean:>7.3f} {oracle_mean:>6.3f} {marker}")
    
    print(f"{'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
    print(f"  {'Overall':<35} {len(results):>5} {np.mean(all_v7_vl):>6.3f} {np.mean(all_v7_rule):>6.3f} {np.mean(all_grid):>7.3f} {np.mean(all_oracle):>6.3f}")
    
    # 关键分析
    print(f"\n{'='*80}")
    print("关键分析")
    print(f"{'='*80}")
    
    print(f"\n  V7 VL Overall:     {np.mean(all_v7_vl):.4f} (63.61% 基准)")
    print(f"  V7 Rule Overall:   {np.mean(all_v7_rule):.4f}")
    print(f"  Grid工具 Overall:  {np.mean(all_grid):.4f}")
    print(f"  Oracle Overall:    {np.mean(all_oracle):.4f}")
    
    # 真正有意义的比较: Grid工具=当前Rule (因为都是从同一个mind_map读取)
    print(f"\n  说明: 当前Grid工具 ≈ V7 Rule")
    print(f"  因为Grid是从V7的mind_map_summary构建的,信息相同")
    print(f"  真正的提升来自:")
    print(f"    1. DA3构建的64³ Grid精度 >> V7的简化summary")
    print(f"    2. 选择题由Grid确定性计算 >> VL猜测")
    print(f"    3. Evolution修正Grid → 进一步提升")
    
    # Counting/Appearance 这两个用Grid信息就能直接回答
    print(f"\n  Grid工具直接有效的任务:")
    for qt in ['object_counting', 'obj_appearance_order']:
        samples = [r for r in results if r['question_type'] == qt]
        v7_vl = np.mean([r.get('vl_score', 0) for r in samples])
        v7_rule = np.mean([r.get('rule_score', 0) for r in samples])
        print(f"    {qt}: VL={v7_vl:.3f}, Rule={v7_rule:.3f}")
        print(f"      Counting: Grid直接读count → ≈Rule")
        print(f"      Appearance: Grid直接排序first_frame → ≈Rule") if 'appearance' in qt else None
    
    # 选择题: Grid坐标精确度是关键
    print(f"\n  选择题(direction/rel_distance/route):")
    print(f"    当前Grid无法回答 → 因为从summary构建的Grid没有精确3D坐标")
    print(f"    用DA3构建的Grid → 有精确世界坐标 → 确定性计算")
    print(f"    这是最大的提升空间!")
    
    direction_samples = [r for r in results if 'direction' in r['question_type']]
    print(f"    Direction样本数: {len(direction_samples)}")
    print(f"    当前VL准确率: {np.mean([r.get('vl_score',0) for r in direction_samples]):.3f}")
    print(f"    当前Rule准确率: {np.mean([r.get('rule_score',0) for r in direction_samples]):.3f}")
    print(f"    如果DA3坐标准确 → Grid确定性计算 → 接近100%")
    print(f"    如果DA3有误差 → Evolution修正位置 → 显著提升")
    
    # 距离任务
    dist_samples = [r for r in results if r['question_type'] == 'object_abs_distance']
    print(f"\n  距离任务(abs_distance): {len(dist_samples)}个样本")
    print(f"    VL={np.mean([r.get('vl_score',0) for r in dist_samples]):.3f}")
    print(f"    Rule={np.mean([r.get('rule_score',0) for r in dist_samples]):.3f}")
    print(f"    DA3有尺度 → Grid距离计算应更准确")
    print(f"    Evolution SET_SCALE → 进一步校准")
    
    # 总结
    print(f"\n{'='*80}")
    print("结论: 需要用DA3实际构建Grid才能验证真实效果")
    print(f"{'='*80}")
    print(f"""
  当前离线分析无法准确评估Grid方案的效果,因为:
  1. V7 summary中只保存了count/confidence/first_frame,没有3D坐标
  2. 选择题需要精确3D坐标才能计算,summary无法提供
  3. 距离/尺寸需要DA3的真实尺度,summary中的scale_factor精度有限
  
  → 必须跑一个真实的小样本测试:
     选50个样本 → DA3推理 → 构建64³ Grid → 工具回答 → 对比VL
  
  预估效果:
  - Counting: Grid ≈ Rule(0.771) > VL(0.859)? → 取决于Evolution能否修正count
  - Size: DA3尺度 + 校准 → 可能接近 VL(0.878)
  - Direction: DA3精确坐标 → 显著优于 VL(0.33~0.49)
  - Distance: DA3精确坐标+尺度 → 显著优于 VL(0.756)的下限
  - Appearance: Grid first_frame → ≈ Rule(0.468)
  - Route: 需要空间拓扑 → 待验证
    """)


if __name__ == "__main__":
    main()
