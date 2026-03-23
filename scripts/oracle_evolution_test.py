#!/usr/bin/env python3
"""
Oracle Evolution 测试 - 验证"完美Evolution"的推理上限

核心思路：
1. 读取V7已有的5130个样本结果(mind_map_summary + GT)
2. 用GT反推每个样本应该执行的Evolution指令
3. 模拟执行指令修正mind_map
4. 用规则推理看修正后的准确率
5. 对比修正前后，验证Evolution方案的理论上限

这个测试不需要GPU，纯离线分析。
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
# 数据结构
# ============================================================================

@dataclass
class GridEntity:
    """64×64×64 Grid中的Entity"""
    entity_id: str           # 如 "chair_1"
    category: str            # 如 "chair"
    grid_position: Tuple[int, int, int]  # (x, y, z) in [0, 63]
    grid_size: Tuple[int, int, int] = (1, 1, 1)  # 占据的格数
    confidence: float = 0.5
    first_seen_frame: int = 0
    count_in_frame: int = 1  # 单帧最大检测数(用于counting)
    
    def physical_size_cm(self, scale_per_grid: float) -> float:
        """物理尺寸(cm) = 最大格数维度 × 每格物理尺寸"""
        return max(self.grid_size) * scale_per_grid * 100


@dataclass
class EvolutionInstruction:
    """Evolution指令"""
    op: str               # DELETE, ADD, MOVE, MERGE, RELABEL, RESIZE, SET_FIRST_FRAME, SET_BOUNDS, SET_SCALE, NOOP
    args: Dict[str, Any]  # 操作参数
    reason: str           # 理由
    task_type: str        # 关联的任务类型


@dataclass
class Grid64:
    """64×64×64 场景Grid"""
    entities: Dict[str, GridEntity]  # entity_id -> GridEntity
    scene_bounds_min: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    scene_bounds_max: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 3.0]))
    scale_factor: float = 1.0  # 校准系数
    
    @property
    def scale_per_grid(self) -> float:
        """每格的物理尺寸(m) - 取最大维度"""
        ranges = self.scene_bounds_max - self.scene_bounds_min
        return float(np.max(ranges)) / 64.0
    
    def get_entity_by_category(self, category: str) -> List[GridEntity]:
        return [e for e in self.entities.values() if match_name(category, e.category)]
    
    def distance_between(self, id1: str, id2: str) -> Optional[float]:
        if id1 not in self.entities or id2 not in self.entities:
            return None
        p1 = np.array(self.entities[id1].grid_position)
        p2 = np.array(self.entities[id2].grid_position)
        grid_dist = np.linalg.norm(p1 - p2)
        return grid_dist * self.scale_per_grid * self.scale_factor
    
    def apply_instruction(self, inst: EvolutionInstruction):
        """执行一条Evolution指令"""
        if inst.op == "DELETE":
            eid = inst.args["entity_id"]
            if eid in self.entities:
                del self.entities[eid]
        
        elif inst.op == "ADD":
            eid = inst.args["entity_id"]
            self.entities[eid] = GridEntity(
                entity_id=eid,
                category=inst.args.get("category", eid.rsplit("_", 1)[0]),
                grid_position=tuple(inst.args["position"]),
                confidence=inst.args.get("confidence", 0.5),
                first_seen_frame=inst.args.get("first_seen_frame", 0),
                count_in_frame=inst.args.get("count", 1),
            )
        
        elif inst.op == "MOVE":
            eid = inst.args["entity_id"]
            if eid in self.entities:
                self.entities[eid].grid_position = tuple(inst.args["position"])
        
        elif inst.op == "MERGE":
            id1, id2 = inst.args["id_1"], inst.args["id_2"]
            keep = inst.args["keep_id"]
            remove = id2 if keep == id1 else id1
            if remove in self.entities:
                del self.entities[remove]
        
        elif inst.op == "RELABEL":
            eid = inst.args["entity_id"]
            if eid in self.entities:
                new_label = inst.args["new_label"]
                old = self.entities[eid]
                new_id = new_label + "_1"
                self.entities[new_id] = GridEntity(
                    entity_id=new_id,
                    category=new_label,
                    grid_position=old.grid_position,
                    grid_size=old.grid_size,
                    confidence=old.confidence,
                    first_seen_frame=old.first_seen_frame,
                    count_in_frame=old.count_in_frame,
                )
                del self.entities[eid]
        
        elif inst.op == "RESIZE":
            eid = inst.args["entity_id"]
            if eid in self.entities:
                self.entities[eid].grid_size = tuple(inst.args["new_size"])
        
        elif inst.op == "SET_FIRST_FRAME":
            eid = inst.args["entity_id"]
            if eid in self.entities:
                self.entities[eid].first_seen_frame = inst.args["frame_idx"]
        
        elif inst.op == "SET_COUNT":
            cat = inst.args["category"]
            n = inst.args["count"]
            entities = self.get_entity_by_category(cat)
            if entities:
                # 修改第一个匹配entity的count
                entities[0].count_in_frame = n
        
        elif inst.op == "SET_SCALE":
            self.scale_factor = inst.args["factor"]
        
        elif inst.op == "SET_BOUNDS":
            self.scene_bounds_min = np.array(inst.args["min"])
            self.scene_bounds_max = np.array(inst.args["max"])
        
        elif inst.op == "NOOP":
            pass


# ============================================================================
# 工具函数
# ============================================================================

def match_name(target: str, label: str) -> bool:
    """模糊匹配物体名称"""
    target = target.lower().strip()
    label = label.lower().strip()
    if target == label:
        return True
    if target in label or label in target:
        return True
    # 处理复合名
    target_words = set(target.replace('_', ' ').replace('-', ' ').split())
    label_words = set(label.replace('_', ' ').replace('-', ' ').split())
    if target_words & label_words:
        return True
    # 常见同义词
    synonyms = {
        'sofa': ['couch'], 'couch': ['sofa'],
        'tv': ['television', 'monitor'], 'television': ['tv'],
        'fridge': ['refrigerator'], 'refrigerator': ['fridge'],
        'desk': ['table'], 'stool': ['chair'],
        'trash': ['bin', 'garbage'], 'bin': ['trash', 'garbage'],
        'wardrobe': ['closet', 'cabinet'], 'closet': ['wardrobe'],
        'rug': ['carpet'], 'carpet': ['rug'],
    }
    for t_word in target_words:
        if t_word in synonyms:
            if any(s in label_words for s in synonyms[t_word]):
                return True
    return False


def build_grid_from_mind_map_summary(
    mind_map_summary: Dict[str, Dict],
    calibration: Dict = None,
) -> Grid64:
    """从V7的mind_map_summary构建64³ Grid"""
    entities = {}
    idx = 0
    
    for label, info in mind_map_summary.items():
        count = info.get("count", 1)
        conf = info.get("confidence", 0.5)
        first_frame = info.get("first_seen_frame", 0)
        
        # 为每个entity分配一个grid位置（均匀分布）
        # 由于summary中没有精确3D位置，用hash分布
        hash_val = hash(label)
        x = (hash_val % 64)
        y = ((hash_val >> 6) % 64)
        z = ((hash_val >> 12) % 16) + 4  # z在4-19之间（地面层附近）
        
        eid = f"{label.replace(' ', '_')}_1"
        entities[eid] = GridEntity(
            entity_id=eid,
            category=label,
            grid_position=(x, y, z),
            confidence=conf,
            first_seen_frame=first_frame,
            count_in_frame=count,
        )
        idx += 1
    
    grid = Grid64(entities=entities)
    
    if calibration:
        grid.scale_factor = calibration.get("scale_factor", 1.0)
    
    return grid


# ============================================================================
# Oracle Evolution生成器 - 用GT反推正确的指令
# ============================================================================

def generate_oracle_instructions(
    sample: Dict,
    grid: Grid64,
) -> List[EvolutionInstruction]:
    """根据GT和mind_map，生成Oracle Evolution指令"""
    qt = sample["question_type"]
    gt = sample["ground_truth"]
    question = sample["question"]
    instructions = []
    
    if qt == "object_counting":
        instructions = oracle_counting(question, gt, grid)
    elif qt == "object_size_estimation":
        instructions = oracle_size(question, gt, grid, sample)
    elif qt == "room_size_estimation":
        instructions = oracle_room_size(question, gt, grid, sample)
    elif qt == "object_abs_distance":
        instructions = oracle_abs_distance(question, gt, grid, sample)
    elif qt in ("object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"):
        instructions = oracle_direction(question, gt, grid, sample)
    elif qt == "object_rel_distance":
        instructions = oracle_rel_distance(question, gt, grid, sample)
    elif qt == "obj_appearance_order":
        instructions = oracle_appearance_order(question, gt, grid, sample)
    elif qt == "route_planning":
        instructions = oracle_route(question, gt, grid, sample)
    
    if not instructions:
        instructions = [EvolutionInstruction(
            op="NOOP", args={}, reason="Grid consistent with GT", task_type=qt
        )]
    
    return instructions


def oracle_counting(question: str, gt: str, grid: Grid64) -> List[EvolutionInstruction]:
    """Counting: 对比grid中entity count和GT count"""
    match = re.search(r'How many (\w+)', question, re.IGNORECASE)
    if not match:
        return []
    
    target = match.group(1).lower()
    gt_count = int(gt)
    
    # 找到grid中匹配的entity
    matched = [(eid, e) for eid, e in grid.entities.items() if match_name(target, e.category)]
    
    if not matched:
        if gt_count > 0:
            return [EvolutionInstruction(
                op="ADD",
                args={"entity_id": f"{target}_1", "category": target, 
                      "position": (32, 32, 8), "count": gt_count},
                reason=f"GT={gt_count} but target '{target}' not in grid, add it",
                task_type="object_counting",
            )]
        return []
    
    eid, entity = matched[0]
    current_count = entity.count_in_frame
    
    if current_count != gt_count:
        return [EvolutionInstruction(
            op="SET_COUNT",
            args={"category": target, "count": gt_count},
            reason=f"Grid count={current_count}, GT count={gt_count}",
            task_type="object_counting",
        )]
    
    return []


def oracle_size(question: str, gt: str, grid: Grid64, sample: Dict) -> List[EvolutionInstruction]:
    """Size: 调整scale使得估计尺寸接近GT"""
    gt_cm = float(gt)
    
    # 从规则推理结果中提取当前预测
    rule_pred = sample.get("rule_prediction", "100")
    try:
        current_cm = float(rule_pred)
    except:
        current_cm = 100.0
    
    if current_cm > 0 and abs(current_cm - gt_cm) / max(gt_cm, 1) > 0.1:
        new_factor = gt_cm / max(current_cm, 1)
        return [EvolutionInstruction(
            op="SET_SCALE",
            args={"factor": new_factor * grid.scale_factor},
            reason=f"Current pred={current_cm:.0f}cm, GT={gt_cm:.0f}cm, adjust scale",
            task_type="object_size_estimation",
        )]
    return []


def oracle_room_size(question: str, gt: str, grid: Grid64, sample: Dict) -> List[EvolutionInstruction]:
    """Room Size: 调整bounds使面积接近GT"""
    gt_area = float(gt)
    
    rule_pred = sample.get("rule_prediction", "25")
    try:
        current_area = float(rule_pred)
    except:
        current_area = 25.0
    
    if abs(current_area - gt_area) / max(gt_area, 1) > 0.1:
        # 反推所需的边界范围
        target_side = np.sqrt(gt_area)
        return [EvolutionInstruction(
            op="SET_BOUNDS",
            args={"min": [0, 0, 0], "max": [target_side, target_side, 3.0]},
            reason=f"Current area≈{current_area:.1f}m², GT={gt_area:.1f}m²",
            task_type="room_size_estimation",
        )]
    return []


def oracle_abs_distance(question: str, gt: str, grid: Grid64, sample: Dict) -> List[EvolutionInstruction]:
    """Abs Distance: 校准使距离接近GT"""
    gt_dist = float(gt)
    
    rule_pred = sample.get("rule_prediction", "2.0")
    try:
        current_dist = float(rule_pred)
    except:
        current_dist = 2.0
    
    if current_dist > 0 and abs(current_dist - gt_dist) / max(gt_dist, 0.1) > 0.1:
        new_factor = gt_dist / max(current_dist, 0.01)
        return [EvolutionInstruction(
            op="SET_SCALE",
            args={"factor": new_factor * grid.scale_factor},
            reason=f"Current dist={current_dist:.2f}m, GT={gt_dist:.2f}m",
            task_type="object_abs_distance",
        )]
    return []


def oracle_direction(question: str, gt: str, grid: Grid64, sample: Dict) -> List[EvolutionInstruction]:
    """Direction: 这类问题依赖entity相对位置，Oracle直接标记正确选项"""
    # 对于direction，Oracle Evolution本质上是修正entity位置使得方向关系正确
    # 但由于summary中没有精确位置，我们记录需要的修正
    rule_pred = sample.get("rule_prediction", "")
    if rule_pred == gt:
        return []  # 规则推理已经正确
    
    # 标记需要位置修正（无法精确计算，但记录下来）
    return [EvolutionInstruction(
        op="NOOP",  # Direction的Oracle修正需要精确位置，这里只记录
        args={"note": f"rule_pred={rule_pred}, gt={gt}, need position correction"},
        reason=f"Direction mismatch: pred={rule_pred}, GT={gt}",
        task_type=sample["question_type"],
    )]


def oracle_rel_distance(question: str, gt: str, grid: Grid64, sample: Dict) -> List[EvolutionInstruction]:
    """Rel Distance: 类似Direction，依赖相对位置"""
    rule_pred = sample.get("rule_prediction", "")
    if rule_pred == gt:
        return []
    
    return [EvolutionInstruction(
        op="NOOP",
        args={"note": f"rule_pred={rule_pred}, gt={gt}, need distance ordering correction"},
        reason=f"Rel distance mismatch: pred={rule_pred}, GT={gt}",
        task_type="object_rel_distance",
    )]


def oracle_appearance_order(question: str, gt: str, grid: Grid64, sample: Dict) -> List[EvolutionInstruction]:
    """Appearance Order: 修正first_seen_frame使排序匹配GT选项"""
    rule_pred = sample.get("rule_prediction", "")
    if rule_pred == gt:
        return []
    
    # 需要修正frame顺序
    return [EvolutionInstruction(
        op="NOOP",
        args={"note": f"rule_pred={rule_pred}, gt={gt}, need frame order correction"},
        reason=f"Appearance order mismatch: pred={rule_pred}, GT={gt}",
        task_type="obj_appearance_order",
    )]


def oracle_route(question: str, gt: str, grid: Grid64, sample: Dict) -> List[EvolutionInstruction]:
    """Route: 路线规划修正"""
    rule_pred = sample.get("rule_prediction", "")
    if rule_pred == gt:
        return []
    
    return [EvolutionInstruction(
        op="NOOP",
        args={"note": f"rule_pred={rule_pred}, gt={gt}"},
        reason=f"Route mismatch: pred={rule_pred}, GT={gt}",
        task_type="route_planning",
    )]


# ============================================================================
# 评估指标
# ============================================================================

def mean_relative_accuracy(pred: float, target: float) -> float:
    """MRA评估 (VSIBench标准)"""
    if target == 0:
        return 1.0 if pred == 0 else 0.0
    
    abs_dist_norm = abs(pred - target) / target
    conf_intervs = np.linspace(0.5, 0.95, 11)  # [0.5, 0.55, ..., 0.95]
    accuracy = (abs_dist_norm <= (1 - conf_intervs)).astype(float)
    return float(accuracy.mean())


def evaluate_sample(question_type: str, prediction: str, ground_truth: str) -> float:
    """评估单个样本"""
    if question_type in ("object_counting", "object_size_estimation", 
                          "room_size_estimation", "object_abs_distance"):
        try:
            pred_val = float(prediction)
            gt_val = float(ground_truth)
            return mean_relative_accuracy(pred_val, gt_val)
        except:
            return 0.0
    else:
        # 多选题: exact match
        pred_letter = prediction.strip().upper()[:1]
        gt_letter = ground_truth.strip().upper()[:1]
        return 1.0 if pred_letter == gt_letter else 0.0


# ============================================================================
# 分析主逻辑
# ============================================================================

def analyze_evolution_potential(results_path: str):
    """分析Evolution的理论提升潜力"""
    
    with open(results_path) as f:
        results = json.load(f)
    
    print(f"=" * 80)
    print(f"Oracle Evolution 潜力分析")
    print(f"=" * 80)
    print(f"总样本数: {len(results)}")
    print()
    
    # ---- Part 1: 当前V7各方法准确率 ----
    print(f"{'='*80}")
    print(f"Part 1: 当前V7各方法准确率")
    print(f"{'='*80}")
    
    type_stats = defaultdict(lambda: {
        "total": 0, "rule_scores": [], "vl_scores": [], "final_scores": [],
        "rule_correct": 0, "vl_correct": 0, "final_correct": 0,
        "rule_wrong_samples": [], "vl_wrong_samples": [],
    })
    
    for r in results:
        qt = r["question_type"]
        stats = type_stats[qt]
        stats["total"] += 1
        
        rule_score = evaluate_sample(qt, r.get("rule_prediction", ""), r["ground_truth"])
        vl_score = evaluate_sample(qt, r.get("vl_prediction", ""), r["ground_truth"])
        final_score = r.get("score", 0)
        
        stats["rule_scores"].append(rule_score)
        stats["vl_scores"].append(vl_score)
        stats["final_scores"].append(final_score)
        
        if rule_score >= 0.5:
            stats["rule_correct"] += 1
        else:
            stats["rule_wrong_samples"].append(r)
        
        if vl_score >= 0.5:
            stats["vl_correct"] += 1
        else:
            stats["vl_wrong_samples"].append(r)
        
        if final_score >= 0.5:
            stats["final_correct"] += 1
    
    print(f"\n{'Task Type':<35} {'N':>5} {'Rule':>8} {'VL':>8} {'Final':>8} {'Oracle↑':>8}")
    print(f"{'-'*35} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    all_rule = []
    all_vl = []
    all_final = []
    all_oracle_upper = []
    
    for qt in sorted(type_stats.keys()):
        stats = type_stats[qt]
        n = stats["total"]
        rule_acc = np.mean(stats["rule_scores"])
        vl_acc = np.mean(stats["vl_scores"])
        final_acc = np.mean(stats["final_scores"])
        
        # Oracle上限 = 每个样本取rule和vl的最优
        oracle_scores = [max(r, v) for r, v in zip(stats["rule_scores"], stats["vl_scores"])]
        oracle_acc = np.mean(oracle_scores)
        
        print(f"{qt:<35} {n:>5} {rule_acc:>7.1%} {vl_acc:>7.1%} {final_acc:>7.1%} {oracle_acc:>7.1%}")
        
        all_rule.extend(stats["rule_scores"])
        all_vl.extend(stats["vl_scores"])
        all_final.extend(stats["final_scores"])
        all_oracle_upper.extend(oracle_scores)
    
    print(f"{'-'*35} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'Overall':<35} {len(results):>5} {np.mean(all_rule):>7.1%} {np.mean(all_vl):>7.1%} {np.mean(all_final):>7.1%} {np.mean(all_oracle_upper):>7.1%}")
    
    # ---- Part 2: 错误分析 ----
    print(f"\n{'='*80}")
    print(f"Part 2: 规则推理错误分析 - Evolution能修正什么")
    print(f"{'='*80}")
    
    # 对数值任务: 分析误差分布
    numeric_tasks = ["object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"]
    
    for qt in numeric_tasks:
        stats = type_stats[qt]
        if not stats["rule_wrong_samples"]:
            continue
        
        errors = []
        for r in stats["rule_wrong_samples"]:
            try:
                pred = float(r.get("rule_prediction", 0))
                gt = float(r["ground_truth"])
                errors.append({
                    "pred": pred, "gt": gt,
                    "abs_error": abs(pred - gt),
                    "rel_error": abs(pred - gt) / max(gt, 0.01),
                    "direction": "over" if pred > gt else "under",
                })
            except:
                pass
        
        if errors:
            over = sum(1 for e in errors if e["direction"] == "over")
            under = len(errors) - over
            median_rel = np.median([e["rel_error"] for e in errors])
            
            print(f"\n  {qt}: {len(errors)} wrong ({over} overestimate, {under} underestimate)")
            print(f"    Median relative error: {median_rel:.1%}")
            
            # Counting特殊分析
            if qt == "object_counting":
                off_by_1 = sum(1 for e in errors if e["abs_error"] <= 1)
                off_by_2 = sum(1 for e in errors if 1 < e["abs_error"] <= 2)
                off_by_more = sum(1 for e in errors if e["abs_error"] > 2)
                print(f"    Off by ±1: {off_by_1}, ±2: {off_by_2}, >2: {off_by_more}")
    
    # 对选择题: 分析VL和Rule的互补性
    choice_tasks = ["object_rel_direction_easy", "object_rel_direction_medium", 
                     "object_rel_direction_hard", "object_rel_distance",
                     "obj_appearance_order", "route_planning"]
    
    print(f"\n  选择题互补性分析:")
    for qt in choice_tasks:
        stats = type_stats[qt]
        n = stats["total"]
        both_right = sum(1 for r, v in zip(stats["rule_scores"], stats["vl_scores"]) if r >= 0.5 and v >= 0.5)
        only_rule = sum(1 for r, v in zip(stats["rule_scores"], stats["vl_scores"]) if r >= 0.5 and v < 0.5)
        only_vl = sum(1 for r, v in zip(stats["rule_scores"], stats["vl_scores"]) if r < 0.5 and v >= 0.5)
        both_wrong = sum(1 for r, v in zip(stats["rule_scores"], stats["vl_scores"]) if r < 0.5 and v < 0.5)
        
        print(f"    {qt:<35}: both✓={both_right:>3} rule_only={only_rule:>3} vl_only={only_vl:>3} both✗={both_wrong:>3}")
    
    # ---- Part 3: Counting任务 Oracle Evolution详细分析 ----
    print(f"\n{'='*80}")
    print(f"Part 3: Counting任务 - Oracle SET_COUNT 效果")
    print(f"{'='*80}")
    
    counting_results = [r for r in results if r["question_type"] == "object_counting"]
    
    original_scores = []
    oracle_scores = []
    evolution_applied = 0
    
    for r in counting_results:
        gt = r["ground_truth"]
        rule_pred = r.get("rule_prediction", "0")
        original_score = evaluate_sample("object_counting", rule_pred, gt)
        original_scores.append(original_score)
        
        # Oracle: 直接将count设为GT
        oracle_score = evaluate_sample("object_counting", gt, gt)
        oracle_scores.append(oracle_score)
        
        if original_score < 1.0:
            evolution_applied += 1
    
    print(f"  Counting样本数: {len(counting_results)}")
    print(f"  规则推理 MRA: {np.mean(original_scores):.4f}")
    print(f"  Oracle (完美count) MRA: {np.mean(oracle_scores):.4f}")
    print(f"  需要Evolution的样本: {evolution_applied}/{len(counting_results)}")
    
    # ---- Part 4: Size任务 Oracle Evolution详细分析 ----
    print(f"\n{'='*80}")
    print(f"Part 4: Size任务 - Oracle SET_SCALE 效果")
    print(f"{'='*80}")
    
    size_results = [r for r in results if r["question_type"] == "object_size_estimation"]
    
    original_scores = []
    oracle_scores = []
    
    for r in size_results:
        gt = r["ground_truth"]
        rule_pred = r.get("rule_prediction", "100")
        original_score = evaluate_sample("object_size_estimation", rule_pred, gt)
        original_scores.append(original_score)
        
        oracle_score = evaluate_sample("object_size_estimation", gt, gt)
        oracle_scores.append(oracle_score)
    
    print(f"  Size样本数: {len(size_results)}")
    print(f"  规则推理 MRA: {np.mean(original_scores):.4f}")
    print(f"  Oracle (完美size) MRA: {np.mean(oracle_scores):.4f}")
    
    # 分析校准误差分布
    scale_errors = []
    for r in size_results:
        try:
            pred = float(r.get("rule_prediction", "100"))
            gt = float(r["ground_truth"])
            if gt > 0:
                scale_errors.append(pred / gt)
        except:
            pass
    
    if scale_errors:
        print(f"  Scale ratio (pred/GT) 分布:")
        print(f"    Mean: {np.mean(scale_errors):.2f}, Median: {np.median(scale_errors):.2f}")
        print(f"    P10: {np.percentile(scale_errors, 10):.2f}, P90: {np.percentile(scale_errors, 90):.2f}")
        print(f"    在0.5x-2x范围内: {sum(1 for s in scale_errors if 0.5 <= s <= 2.0)}/{len(scale_errors)}")
    
    # ---- Part 5: 理论上限计算 ----
    print(f"\n{'='*80}")
    print(f"Part 5: Evolution理论上限")
    print(f"{'='*80}")
    
    # 场景1: 仅对数值任务做Oracle Evolution（Rule端直接用GT）
    scenario1_scores = []
    for r in results:
        qt = r["question_type"]
        if qt in numeric_tasks:
            # Oracle: 直接用GT
            scenario1_scores.append(1.0)
        else:
            # 保持原始final score
            scenario1_scores.append(r.get("score", 0))
    
    # 场景2: 数值任务Oracle + 选择题取Rule/VL最优
    scenario2_scores = []
    for r in results:
        qt = r["question_type"]
        if qt in numeric_tasks:
            scenario2_scores.append(1.0)
        else:
            rule_s = evaluate_sample(qt, r.get("rule_prediction", ""), r["ground_truth"])
            vl_s = evaluate_sample(qt, r.get("vl_prediction", ""), r["ground_truth"])
            scenario2_scores.append(max(rule_s, vl_s))
    
    # 场景3: 仅对Rule推理做Oracle（VL不变）
    scenario3_rule_scores = []
    scenario3_vl_scores = []
    for r in results:
        qt = r["question_type"]
        if qt in numeric_tasks:
            scenario3_rule_scores.append(1.0)  # Rule Oracle
        else:
            scenario3_rule_scores.append(evaluate_sample(qt, r.get("rule_prediction", ""), r["ground_truth"]))
        scenario3_vl_scores.append(evaluate_sample(qt, r.get("vl_prediction", ""), r["ground_truth"]))
    
    # 场景4: 对数值任务做±20%误差的近似Oracle（模拟现实中Evolution不完美的情况）
    scenario4_scores = []
    for r in results:
        qt = r["question_type"]
        if qt in numeric_tasks:
            gt = float(r["ground_truth"])
            # 模拟 Evolution后仍有10-20%误差
            noisy_pred = gt * (1 + np.random.uniform(-0.15, 0.15))
            scenario4_scores.append(evaluate_sample(qt, str(noisy_pred), r["ground_truth"]))
        else:
            scenario4_scores.append(r.get("score", 0))
    
    print(f"\n  当前V7 Overall:                         {np.mean(all_final):.4f}")
    print(f"  场景1 (数值Oracle + 选择不变):          {np.mean(scenario1_scores):.4f}")
    print(f"  场景2 (数值Oracle + 选择取最优):        {np.mean(scenario2_scores):.4f}")
    print(f"  场景3 (Rule Oracle + VL不变 取最优):    {np.mean([max(r,v) for r,v in zip(scenario3_rule_scores, scenario3_vl_scores)]):.4f}")
    print(f"  场景4 (数值±15%噪声Oracle + 选择不变):  {np.mean(scenario4_scores):.4f}")
    
    # ---- Part 6: 每种指令类型的适用统计 ----
    print(f"\n{'='*80}")
    print(f"Part 6: Evolution指令适用统计")
    print(f"{'='*80}")
    
    instruction_stats = Counter()
    task_instruction_map = defaultdict(Counter)
    
    for r in results:
        qt = r["question_type"]
        grid = build_grid_from_mind_map_summary(r.get("mind_map_summary", {}), r.get("calibration"))
        instructions = generate_oracle_instructions(r, grid)
        
        for inst in instructions:
            instruction_stats[inst.op] += 1
            task_instruction_map[qt][inst.op] += 1
    
    print(f"\n  指令分布:")
    for op, count in instruction_stats.most_common():
        print(f"    {op:<20}: {count:>5} ({count/len(results)*100:.1f}%)")
    
    print(f"\n  按任务类型×指令:")
    for qt in sorted(task_instruction_map.keys()):
        ops = task_instruction_map[qt]
        ops_str = ", ".join(f"{op}={c}" for op, c in ops.most_common())
        print(f"    {qt:<35}: {ops_str}")
    
    # ---- Part 7: 关键结论 ----
    print(f"\n{'='*80}")
    print(f"Part 7: 关键结论")
    print(f"{'='*80}")
    
    vl_overall = np.mean(all_vl)
    rule_overall = np.mean(all_rule)
    final_overall = np.mean(all_final)
    oracle_overall = np.mean(all_oracle_upper)
    
    print(f"""
  1. 当前V7:
     - Rule推理: {rule_overall:.4f}
     - VL推理:   {vl_overall:.4f}
     - Final:    {final_overall:.4f}
     - Oracle(取最优): {oracle_overall:.4f}
  
  2. Evolution提升空间:
     - 数值任务(counting/size/room/distance)占 {sum(type_stats[t]['total'] for t in numeric_tasks)}/{len(results)} 样本
     - 选择题(direction/distance_rel/appearance/route)占 {sum(type_stats[t]['total'] for t in choice_tasks)}/{len(results)} 样本
     
  3. 指令集覆盖:
     - SET_COUNT: 直接修正counting，效果确定性最高
     - SET_SCALE: 修正size/distance，依赖校准精度
     - SET_FIRST_FRAME: 修正appearance_order
     - MOVE: 修正direction/rel_distance，需要精确位置（最难）
     - NOOP: Grid本身正确，无需修正
    """)


if __name__ == "__main__":
    results_path = "/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/evolving_agent_v7_20260220_201744/detailed_results.json"
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        sys.exit(1)
    
    np.random.seed(42)
    analyze_evolution_potential(results_path)
