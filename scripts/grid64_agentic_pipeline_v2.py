#!/usr/bin/env python3
"""
64³ Grid Mind Map — Agentic Pipeline V4 (ADD Evolution + Position3D Precision)

核心理念: 所有任务都参考Grid(MindMap), 但通过多Agent分工协作来精进Grid质量
  Phase 1: Manager       — 问题解耦 → 提取key_entities + checkpoints
  Phase 2: Critic        — VL(16帧)审查Grid, 只输出诊断(issues+原因), 不执行任何修正
  Phase 3: Manager审核   — 根据Critic诊断决定Evolution类型: DELETE误检 / ADD缺失
  Phase 4: Evolutor      — 执行Grid修正: DELETE误检 + ADD缺失(针对性GroundingDINO搜索+DA3投影)
  Phase 5: Reasoner      — VL(16帧)最终回答, 所有任务参考Grid

V4新增 (vs V3):
  1. grid_answer_direction: 改用position_3d浮点世界坐标(替代grid_position整数坐标), 提升方向计算精度
  2. Manager区分DELETE(误检) vs ADD(缺失), ADD仅对direction/rel_distance的key entity触发
  3. Evolutor支持ADD操作: 调用Grid64Builder.search_and_add_entity()针对性搜索缺失entity
     - 复用build_grid已有的DA3结果(depth_maps, intrinsics, extrinsics), 不重复推理
     - GroundingDINO精确搜索指定entity, DA3 2D→3D投影获取世界坐标

对比基准: V7 VL Overall = 63.61%
"""

import os
import sys
import json
import re
import gc
import copy
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

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

# ============================================================================
# 从grid64_real_test.py导入Grid核心组件
# ============================================================================
from scripts.grid64_real_test import (
    Grid64, GridEntity, Grid64Builder,
    EXTENDED_VOCABULARY, SYNONYMS, CALIBRATION_OBJECTS,
    _match_name, find_video_path, evaluate_sample, mean_relative_accuracy,
    grid_answer_counting, grid_answer_size, grid_answer_room_size,
    grid_answer_abs_distance, grid_answer_direction, grid_answer_rel_distance,
    grid_answer_appearance_order, grid_answer_route,
)

NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 'room_size_estimation', 'object_abs_distance']
CHOICE_TASKS = ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard',
                'object_rel_distance', 'obj_appearance_order', 'route_planning']

# ============================================================================
# P0: VL Model Wrapper — 16帧 + 更高分辨率
# ============================================================================

VL_DEFAULT_NFRAMES = 16        # P0: 从8帧提升到16帧
VL_DEFAULT_MAX_PIXELS = 480 * 560   # P0: 从360×420(151K)提升到480×560(269K)


class VLModel:
    """VL模型的统一封装 — 支持参数化帧数和分辨率"""

    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.processor = None

    def load(self, model_path: str):
        if self.model is not None:
            return
        logger.info(f"Loading VL model: {model_path}")
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            if 'qwen3' in model_path.lower() or 'Qwen3' in model_path:
                from transformers import Qwen3VLForConditionalGeneration
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True,
                )
                logger.info("VL model loaded (Qwen3-VL)")
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True,
                )
                logger.info("VL model loaded (Qwen2.5-VL)")
        except Exception as e:
            logger.error(f"Failed to load VL: {e}")
            import traceback
            traceback.print_exc()

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()

    def call(self, prompt: str, video_path: str, max_tokens: int = 512,
             nframes: int = VL_DEFAULT_NFRAMES,
             max_pixels: int = VL_DEFAULT_MAX_PIXELS) -> str:
        """调用VL模型 — P0: 支持参数化帧数和分辨率"""
        if self.model is None:
            return ""
        try:
            from qwen_vl_utils import process_vision_info
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path,
                     "max_pixels": max_pixels, "nframes": nframes},
                    {"type": "text", "text": prompt}
                ]
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                )
            response = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0]
            return response.strip()
        except Exception as e:
            logger.warning(f"VL call failed: {e}")
            return ""


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class DecomposedTask:
    """Manager输出: 问题解耦结果"""
    question_type: str
    question: str
    options: List[str]
    ground_truth: str
    checkpoints: List[Dict[str, str]]
    key_entities: List[str]
    grid_tool_answer: Optional[str]
    grid_tool_reasoning: Optional[str]


@dataclass
class CriticDiagnosis:
    """Critic输出: 只有诊断, 不含修正指令"""
    issues: List[Dict[str, str]]   # [{"entity": "chair", "problem": "not visible", "confidence": "high"}, ...]
    summary: str                    # 一句话总结
    raw_response: str


@dataclass
class ManagerDecision:
    """Manager审核输出: 是否需要Evolution"""
    should_evolve: bool
    approved_actions: List[Dict[str, str]]   # DELETE: {"action":"DELETE","entity_id":"xxx","reason":"..."} | ADD: {"action":"ADD","entity_name":"xxx","reason":"..."}
    reasoning: str


@dataclass
class EvolutionResult:
    """Evolutor输出: Grid修正结果"""
    actions_executed: List[str]
    grid_modified: bool
    details: str


# ============================================================================
# Phase 1: Manager — 问题解耦
# ============================================================================

def phase1_manager(question: str, question_type: str, options: List[str],
                   grid: Grid64) -> DecomposedTask:
    """Phase 1: 解耦问题 → 提取关注点列表"""
    q = question.lower()
    key_entities = []
    checkpoints = []

    if 'direction' in question_type:
        m = re.search(r'standing by (?:the )?(.+?)\s+and\s+facing (?:the )?(.+?),\s*is (?:the )?(.+?)\s+to\s', q)
        if m:
            observer, facing, target = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            key_entities = [observer, facing, target]
            checkpoints.append({"aspect": "entity_presence", "target": f"{observer}, {facing}, {target}",
                                "description": f"Verify all three objects exist: observer='{observer}', facing='{facing}', target='{target}'"})
            checkpoints.append({"aspect": "position_accuracy", "target": observer,
                                "description": f"Is '{observer}' position correct relative to video?"})
            checkpoints.append({"aspect": "position_accuracy", "target": target,
                                "description": f"Is '{target}' position correct relative to '{observer}' and '{facing}'?"})
        else:
            checkpoints.append({"aspect": "general", "target": "", "description": "Check spatial layout for direction reasoning"})

    elif question_type == 'object_counting':
        m = re.search(r'how many (\w[\w\s]*?)(?:\(s\))?\s+(?:are|is)', q)
        if m:
            obj_name = m.group(1).strip()
            key_entities = [obj_name]
            checkpoints.append({"aspect": "detection_completeness", "target": obj_name,
                                "description": f"Are all '{obj_name}' instances detected? Any missed or duplicates?"})

    elif question_type == 'object_size_estimation':
        m = re.search(r'(?:size|height|length|width).*?(?:of|for)\s+(?:the\s+)?(.+?)[\?,\.]', q)
        if m:
            obj_name = m.group(1).strip()
            key_entities = [obj_name]
            checkpoints.append({"aspect": "entity_presence", "target": obj_name,
                                "description": f"Is '{obj_name}' detected and correctly identified?"})
            checkpoints.append({"aspect": "scale_calibration", "target": obj_name,
                                "description": f"Is the scale calibration reasonable for '{obj_name}'?"})

    elif question_type == 'room_size_estimation':
        checkpoints.append({"aspect": "scene_boundary", "target": "room",
                            "description": "Is the detected scene boundary consistent with the room in video?"})

    elif question_type == 'object_abs_distance':
        patterns = [
            r'distance between (?:the )?(.+?)\s+and\s+(?:the )?(.+?)[\s\(\?\.]',
            r'from (?:the )?(\w[\w\s]*?) to (?:the )?(\w[\w\s]*)',
        ]
        for p in patterns:
            m = re.search(p, q)
            if m:
                obj1, obj2 = m.group(1).strip(), m.group(2).strip()
                key_entities = [obj1, obj2]
                checkpoints.append({"aspect": "entity_presence", "target": f"{obj1}, {obj2}",
                                    "description": f"Are both '{obj1}' and '{obj2}' correctly detected?"})
                break

    elif question_type == 'object_rel_distance':
        m = re.search(r'(?:closest|nearest|farthest|furthest)\s+(?:to|from)\s+(?:the\s+)?(.+?)[\?\.]', q)
        if m:
            ref = m.group(1).strip()
            key_entities = [ref]
            for opt in options:
                cand = re.sub(r'^[A-D]\.\s*', '', opt).strip().lower()
                key_entities.append(cand)
            checkpoints.append({"aspect": "relative_positions", "target": ref,
                                "description": f"Are relative distances from '{ref}' to candidates consistent with video?"})

    elif question_type == 'obj_appearance_order':
        m = re.search(r'appearance order.*?:\s*(.+?)[\?]', q)
        if m:
            obj_list = [o.strip() for o in m.group(1).strip().split(',')]
            key_entities = obj_list
            checkpoints.append({"aspect": "temporal_order", "target": ", ".join(obj_list),
                                "description": f"Verify first-appearance frame order of: {', '.join(obj_list)}"})
        else:
            checkpoints.append({"aspect": "temporal_order", "target": "", "description": "Check temporal appearance order"})

    elif question_type == 'route_planning':
        checkpoints.append({"aspect": "spatial_layout", "target": "route",
                            "description": "Is spatial layout consistent with video for route reasoning?"})

    if not checkpoints:
        checkpoints.append({"aspect": "general", "target": "", "description": "General spatial consistency check"})

    grid_pred, grid_reason = _compute_grid_answer(grid, question, question_type, options)

    return DecomposedTask(
        question_type=question_type, question=question, options=options,
        ground_truth="", checkpoints=checkpoints, key_entities=key_entities,
        grid_tool_answer=grid_pred, grid_tool_reasoning=grid_reason,
    )


def _compute_grid_answer(grid: Grid64, question: str, qt: str, options: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """调用Grid工具预计算答案"""
    try:
        if qt == 'object_counting':
            return grid_answer_counting(grid, question)
        elif qt == 'object_size_estimation':
            return grid_answer_size(grid, question)
        elif qt == 'room_size_estimation':
            return grid_answer_room_size(grid, question)
        elif qt == 'object_abs_distance':
            return grid_answer_abs_distance(grid, question)
        elif 'direction' in qt:
            return grid_answer_direction(grid, question, options)
        elif qt == 'object_rel_distance':
            return grid_answer_rel_distance(grid, question, options)
        elif qt == 'obj_appearance_order':
            return grid_answer_appearance_order(grid, question, options)
        elif qt == 'route_planning':
            return grid_answer_route(grid, question, options)
    except Exception as e:
        logger.warning(f"Grid tool error: {e}")
    return None, None


# ============================================================================
# Phase 2: Critic — VL(16帧)审查Grid, 只输出诊断
# ============================================================================

def _build_grid_evidence(grid: Grid64, task: DecomposedTask) -> str:
    """构建Grid证据 — 给Critic看的完整信息"""
    lines = []
    lines.append(f"Scene: {len(grid.entities)} objects, meters_per_grid={grid.meters_per_grid:.4f}m, "
                 f"scene_span≈{grid.meters_per_grid * 64:.1f}m")

    # key entities详细信息
    relevant_eids = set()
    for name in task.key_entities:
        found = grid.get_by_category(name)
        if found:
            for e in found:
                relevant_eids.add(e.entity_id)
        else:
            lines.append(f"  [NOT FOUND] '{name}' not detected in grid")

    if relevant_eids:
        lines.append("\nKey objects for this question:")
        for eid in sorted(relevant_eids):
            e = grid.entities[eid]
            phys = grid.grid_to_physical(e.grid_position)
            size_str = ""
            ps = grid.physical_size(eid)
            if ps:
                size_str = f", size≈{ps:.2f}m"
            n_frames = len(set(d['frame_order'] for d in e.detections)) if e.detections else 0
            lines.append(
                f"  {eid}: pos=({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m{size_str}, "
                f"conf={e.confidence:.2f}, seen_in={n_frames}/16 frames"
            )

    # 全局entity列表(简要)
    lines.append(f"\nAll detected objects ({len(grid.entities)}):")
    for eid, e in sorted(grid.entities.items()):
        phys = grid.grid_to_physical(e.grid_position)
        extra = ""
        if task.question_type == 'object_counting':
            extra = f", count={e.count_in_frame}"
        elif task.question_type == 'obj_appearance_order':
            extra = f", first_frame={e.first_seen_frame}"
        lines.append(f"  {eid}: ({phys[0]:.1f},{phys[1]:.1f},{phys[2]:.1f})m, conf={e.confidence:.2f}{extra}")

    return "\n".join(lines)


def phase2_critic(
    vl: VLModel,
    grid: Grid64,
    task: DecomposedTask,
    video_path: str,
) -> CriticDiagnosis:
    """
    Phase 2: Critic — 只诊断, 不修正

    VL用16帧审查Grid, 输出具体的issues及其原因和置信度.
    Critic不输出任何修正指令 — 这是Evolutor的职责.
    """
    grid_evidence = _build_grid_evidence(grid, task)

    # 构建checkpoints描述
    check_lines = []
    for i, cp in enumerate(task.checkpoints):
        check_lines.append(f"  {i+1}. [{cp['aspect']}] {cp['description']}")
    checkpoints_text = "\n".join(check_lines)

    prompt = f"""You are a quality reviewer for a 3D perception system. Your ONLY job is to find errors — do NOT suggest fixes.

=== PERCEPTION DATA FROM 3D SYSTEM ===
{grid_evidence}

=== QUESTION THIS DATA WILL ANSWER ===
{task.question}

=== VERIFICATION CHECKPOINTS ===
{checkpoints_text}

=== YOUR TASK ===
Watch the video carefully (you have 16 frames). For each checkpoint above, check if the perception data is consistent with the video.

Report ONLY concrete errors you are confident about. For each error, specify:
- Which object/entity is wrong
- What the problem is (not visible, wrong position, missing, etc.)
- Your confidence (high/medium/low)

Output format (one issue per line, or "none" if all checks pass):
ISSUE: entity=<name_or_id> | problem=<description> | confidence=<high/medium/low>
ISSUE: entity=<name_or_id> | problem=<description> | confidence=<high/medium/low>
SUMMARY: <one sentence overall assessment>"""

    response = vl.call(prompt, video_path, max_tokens=384)

    # 解析诊断结果
    issues = []
    summary = ""
    for line in response.split('\n'):
        line = line.strip()
        if line.upper().startswith('ISSUE:'):
            issue_text = line.split(':', 1)[1].strip()
            if issue_text.lower() in ('none', 'no issues', 'n/a', ''):
                continue
            # 解析结构化issue
            issue = {"raw": issue_text}
            for part in issue_text.split('|'):
                part = part.strip()
                if part.startswith('entity='):
                    issue['entity'] = part.split('=', 1)[1].strip()
                elif part.startswith('problem='):
                    issue['problem'] = part.split('=', 1)[1].strip()
                elif part.startswith('confidence='):
                    issue['confidence'] = part.split('=', 1)[1].strip().lower()
            issues.append(issue)
        elif line.upper().startswith('SUMMARY:'):
            summary = line.split(':', 1)[1].strip()

    logger.info(f"  Critic: {len(issues)} issue(s) found, summary={summary[:80]}")
    for iss in issues:
        logger.info(f"    ISSUE: {iss.get('entity','?')} — {iss.get('problem','?')} [{iss.get('confidence','?')}]")

    return CriticDiagnosis(
        issues=issues,
        summary=summary or (f"{len(issues)} issues found" if issues else "All checks passed"),
        raw_response=response[:500],
    )


# ============================================================================
# Phase 3: Manager审核 — 根据Critic诊断决定是否Evolution
# ============================================================================

def phase3_manager_review(
    task: DecomposedTask,
    critic: CriticDiagnosis,
    grid: Grid64,
) -> ManagerDecision:
    """
    Phase 3: Manager审核Critic诊断, 决定是否Evolution

    规则 (V4更新 — 支持DELETE + ADD):
      1. 只批准高置信度的issues
      2. DELETE: Grid中存在但视频中不可见 → 删除误检
      3. ADD: Critic报告key entity缺失(not detected in grid) → 针对性搜索添加
      4. 保护key entities不被DELETE
      5. 数值型任务/route/appearance_order: 只允许ADD(不允许DELETE)
      6. ADD仅对direction任务的key entity触发(最需要entity完整)
    """
    if not critic.issues:
        return ManagerDecision(
            should_evolve=False, approved_actions=[], reasoning="Critic found no issues"
        )

    qt = task.question_type
    key_names = set(n.lower().strip() for n in task.key_entities)
    key_eids = set()
    for name in task.key_entities:
        found = grid.get_by_category(name)
        if found:
            for e in found:
                key_eids.add(e.entity_id)

    approved = []
    rejected_reasons = []

    NO_DELETE_TASKS = set(NUMERICAL_TASKS) | {'route_planning', 'obj_appearance_order'}
    ADD_ELIGIBLE_TASKS = {'object_rel_direction_easy', 'object_rel_direction_medium',
                          'object_rel_direction_hard', 'object_rel_distance'}

    for iss in critic.issues:
        entity = iss.get('entity', '')
        problem = iss.get('problem', '').lower()
        confidence = iss.get('confidence', 'low')

        # 规则1: 只处理高置信度问题
        if confidence not in ('high',):
            rejected_reasons.append(f"low confidence: {entity}")
            continue

        # --- 分类: 是"误检(false detection)"还是"缺失(missing from grid)" ---
        is_false_detection = any(kw in problem for kw in
            ['not visible', 'not present', 'not in the video', 'false detection',
             'does not exist', 'not seen', 'absent', 'missing from video',
             'cannot be seen', 'not found in video', 'not appear'])

        is_missing_from_grid = any(kw in problem for kw in
            ['not detected in grid', 'not detected', 'not found', 'missing',
             'not in grid', 'not in the grid', 'absent from grid',
             'could not find', 'cannot find', 'not recognized'])

        # --- ADD逻辑: entity缺失 → 尝试针对性搜索添加 ---
        if is_missing_from_grid and qt in ADD_ELIGIBLE_TASKS:
            entity_name = entity.strip().lower()
            # 只对key entity做ADD(避免无关entity的噪声)
            is_key = entity_name in key_names or any(
                _match_name(entity_name, kn) for kn in key_names
            )
            # 确认entity确实不在Grid中
            already_in_grid = bool(grid.get_by_category(entity_name))

            if is_key and not already_in_grid:
                approved.append({
                    "action": "ADD",
                    "entity_name": entity_name,
                    "reason": iss.get('problem', 'missing from grid'),
                })
                logger.info(f"    Manager: ADD approved for '{entity_name}' (key entity missing)")
                continue
            elif already_in_grid:
                rejected_reasons.append(f"already in grid: {entity}")
                continue
            else:
                rejected_reasons.append(f"not a key entity for ADD: {entity}")
                continue

        # --- DELETE逻辑: Grid中存在但视频中不可见 → 删除误检 ---
        if not is_false_detection:
            rejected_reasons.append(f"not actionable: {entity} — {problem[:40]}")
            continue

        # 找到对应的entity_id
        eid = entity.replace(' ', '_')
        if eid not in grid.entities:
            candidates = grid.get_by_category(entity)
            if candidates:
                eid = candidates[0].entity_id
            else:
                rejected_reasons.append(f"entity not in grid: {entity}")
                continue

        # 保护key entities不被DELETE
        if eid in key_eids:
            rejected_reasons.append(f"key entity protected: {eid}")
            continue

        # 数值型/route/appearance_order不做DELETE
        if qt in NO_DELETE_TASKS:
            rejected_reasons.append(f"task type {qt} skip DELETE: {eid}")
            continue

        approved.append({
            "action": "DELETE",
            "entity_id": eid,
            "reason": iss.get('problem', 'false detection'),
        })

    # 限制: 最多2个DELETE + 最多3个ADD
    deletes = [a for a in approved if a['action'] == 'DELETE']
    adds = [a for a in approved if a['action'] == 'ADD']
    if len(deletes) > 2:
        deletes = deletes[:2]
    if len(adds) > 3:
        adds = adds[:3]
    approved = deletes + adds

    should_evolve = len(approved) > 0

    reasoning_parts = []
    n_del = len([a for a in approved if a['action'] == 'DELETE'])
    n_add = len([a for a in approved if a['action'] == 'ADD'])
    if n_del:
        reasoning_parts.append(f"Approved {n_del} DELETE(s)")
    if n_add:
        reasoning_parts.append(f"Approved {n_add} ADD(s): {[a['entity_name'] for a in approved if a['action'] == 'ADD']}")
    if rejected_reasons:
        reasoning_parts.append(f"Rejected {len(rejected_reasons)}: {'; '.join(rejected_reasons[:3])}")

    logger.info(f"  Manager: evolve={should_evolve}, DELETE={n_del}, ADD={n_add}, rejected={len(rejected_reasons)}")

    return ManagerDecision(
        should_evolve=should_evolve,
        approved_actions=approved,
        reasoning=" | ".join(reasoning_parts) or "No actionable issues",
    )


# ============================================================================
# Phase 4: Evolutor — 专门负责Grid修正
# ============================================================================

def phase4_evolutor(
    grid: Grid64,
    decision: ManagerDecision,
    builder: Grid64Builder = None,
) -> EvolutionResult:
    """
    Phase 4: Evolutor — 执行Manager批准的Grid修正

    支持两种操作:
      DELETE: 删除误检entity
      ADD: 针对性搜索缺失entity并添加到Grid (需要builder的缓存DA3结果)
    """
    if not decision.should_evolve:
        return EvolutionResult(actions_executed=[], grid_modified=False, details="No evolution needed")

    executed = []
    for action in decision.approved_actions:
        cmd = action['action'].upper()

        if cmd == 'DELETE':
            eid = action.get('entity_id', '')
            if eid in grid.entities:
                del grid.entities[eid]
                executed.append(f"DELETE {eid}")
                logger.info(f"  Evolutor: DELETE {eid} — {action.get('reason', '')[:60]}")

        elif cmd == 'ADD':
            entity_name = action.get('entity_name', '')
            if not entity_name:
                continue
            if builder is None:
                logger.warning(f"  Evolutor: ADD skipped (no builder) for '{entity_name}'")
                continue
            added = builder.search_and_add_entity(grid, entity_name)
            if added:
                executed.append(f"ADD {added.entity_id}")
                logger.info(f"  Evolutor: ADD {added.entity_id} — {action.get('reason', '')[:60]}")
            else:
                logger.info(f"  Evolutor: ADD failed for '{entity_name}' (not found in targeted search)")

    return EvolutionResult(
        actions_executed=executed,
        grid_modified=len(executed) > 0,
        details=f"Executed {len(executed)} action(s): {executed}" if executed else "No actions executed",
    )


# ============================================================================
# Phase 5: Reasoner — VL(16帧)最终回答 (所有任务参考Grid)
# ============================================================================

def phase5_reasoner(
    vl: VLModel,
    grid: Grid64,
    task: DecomposedTask,
    critic: CriticDiagnosis,
    evolution: EvolutionResult,
    video_path: str,
) -> Tuple[str, str]:
    """Phase 5: VL最终回答 — 所有任务参考Grid(MindMap)"""
    qt = task.question_type

    # 如果Evolution修改了Grid, 重新计算Grid参考答案
    if evolution.grid_modified:
        task.grid_tool_answer, task.grid_tool_reasoning = _compute_grid_answer(
            grid, task.question, qt, task.options
        )

    prompt = _build_reasoning_prompt(grid, task, critic, evolution)

    # 数值型任务(counting/size)用8帧 — 实验表明16帧会导致counting退化
    # 空间型任务用16帧 — 更多视角有助于空间推理
    if qt in ('object_counting', 'object_size_estimation'):
        nframes = 8
        max_pixels = 360 * 420
    else:
        nframes = VL_DEFAULT_NFRAMES
        max_pixels = VL_DEFAULT_MAX_PIXELS

    response = vl.call(prompt, video_path, max_tokens=256, nframes=nframes, max_pixels=max_pixels)

    # === Direction任务: 分难度策略 ===
    # easy: Grid preferred (position_3d精度足够, ~80%)
    # medium/hard: VL preferred (Grid精度不足, 30%/11%)
    if 'direction' in qt and task.grid_tool_answer:
        grid_pred = task.grid_tool_answer
        grid_reason = task.grid_tool_reasoning or ""
        vl_pred, vl_reasoning = _parse_choice(response, task.options)
        
        # Grid答案不可靠时回退VL: entity缺失/解析失败导致的fallback "A"
        grid_is_fallback = any(kw in grid_reason for kw in ['not found', 'cannot parse', 'same 3d position', 'no options'])
        
        if grid_is_fallback:
            return vl_pred, f"[vl_fallback] Grid={grid_pred}(reason:{grid_reason[:50]}) VL={vl_pred}"
        
        # easy: Grid方向计算更可靠 → Grid preferred
        if qt == 'object_rel_direction_easy':
            if grid_pred == vl_pred:
                return grid_pred, f"[grid+vl agree] Grid={grid_pred} VL={vl_pred}"
            else:
                return grid_pred, f"[grid_preferred] Grid={grid_pred} VL={vl_pred} ({vl_reasoning[:60]})"
        else:
            # medium/hard: VL更可靠 → VL preferred, Grid仅供参考
            if grid_pred == vl_pred:
                return grid_pred, f"[grid+vl agree] Grid={grid_pred} VL={vl_pred}"
            else:
                return vl_pred, f"[vl_preferred] Grid={grid_pred} VL={vl_pred} ({vl_reasoning[:60]})"

    # === Route/Appearance/RelDistance: Grid有答案时, VL和Grid不一致用VL ===
    # 注意: route的Grid答案质量差(总是A), rel_distance Grid也不可靠
    # 只对appearance_order使用Grid辅助(时序信息有价值)
    if qt == 'obj_appearance_order' and task.grid_tool_answer:
        grid_pred = task.grid_tool_answer
        vl_pred, vl_reasoning = _parse_choice(response, task.options)
        if grid_pred == vl_pred:
            return grid_pred, f"[grid+vl agree] Grid={grid_pred} VL={vl_pred}"
        else:
            return vl_pred, f"[vl_preferred] Grid={grid_pred} VL={vl_pred} ({vl_reasoning[:60]})"

    if qt in NUMERICAL_TASKS:
        prediction, reasoning = _parse_numerical(response, qt)
    else:
        prediction, reasoning = _parse_choice(response, task.options)

    log_parts = [f"[vl] {reasoning}"]
    if task.grid_tool_answer is not None:
        log_parts.append(f"grid_ref={task.grid_tool_answer}")
    if evolution.grid_modified:
        log_parts.append(f"evolved={evolution.actions_executed}")

    return prediction, " | ".join(log_parts)


def _build_reasoning_prompt(
    grid: Grid64,
    task: DecomposedTask,
    critic: CriticDiagnosis,
    evolution: EvolutionResult,
) -> str:
    """为VL最终回答构建prompt — 所有任务参考Grid"""
    qt = task.question_type
    q = task.question

    # Grid参考答案 — 只对abs_distance和空间选择题展示
    # counting/size/room_size有各自独立的参考信息, 不给grid_ref(实验证明会误导VL)
    grid_ref = ""
    if task.grid_tool_answer is not None and qt not in ('object_counting', 'object_size_estimation', 'room_size_estimation'):
        grid_ref = f"\n3D perception suggests: {task.grid_tool_answer}"
        if evolution.grid_modified:
            grid_ref += " (after correction)"

    # Critic审查摘要 — 只对空间型展示(数值型VL判断不可靠,避免干扰)
    review_note = ""
    if qt not in NUMERICAL_TASKS and critic.issues:
        high_conf = [iss for iss in critic.issues if iss.get('confidence') == 'high']
        if high_conf:
            findings = "; ".join(f"{iss.get('entity','?')}: {iss.get('problem','?')[:50]}" for iss in high_conf[:3])
            review_note = f"\nReview findings: {findings}"
            if evolution.grid_modified:
                review_note += " [corrections applied]"

    # === 数值型 ===

    if qt == 'object_counting':
        # counting: 给Grid完整entity列表(包含类别和检测帧数), 让VL交叉验证
        grid_lines = []
        for eid, e in sorted(grid.entities.items()):
            n_frames = len(set(d['frame_order'] for d in e.detections)) if e.detections else 0
            grid_lines.append(f"  {e.category} (conf={e.confidence:.2f}, seen_in={n_frames} frames)")
        grid_text = "\n".join(grid_lines) if grid_lines else "(none detected)"
        return f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to count specific objects in the scene.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{grid_text}

=== QUESTION ===
{q}

=== INSTRUCTIONS ===
1. Carefully examine the video frames.
2. Cross-reference with the detected objects list above.
3. If you see objects in the video that are not in the list, count them too.
4. If the perception count seems incorrect based on what you see, trust your visual observation.

Please respond with ONLY a single integer number representing the count.
Do not include any explanation or units, just the number.

Answer:"""

    elif qt == 'object_size_estimation':
        size_ref = ""
        for name in task.key_entities:
            found = grid.get_by_category(name)
            if found:
                ps = grid.physical_size(found[0].entity_id)
                if ps:
                    size_ref = f"\n3D perception estimated '{found[0].category}' size: ~{ps*100:.0f}cm (use as rough reference)"
        return f"""You are analyzing a video of an indoor scene. Estimate the size of the specified object.

=== REFERENCE SIZES ===
Door: ~200cm tall. Chair seat: ~45cm high. Table: ~75cm high. Bed: ~200cm long. Sofa: ~85cm high.{size_ref}

=== QUESTION ===
{q}

Look at the object in the video and compare with reference objects for scale.
Respond with ONLY a single integer (centimeters).

Answer:"""

    elif qt == 'room_size_estimation':
        scene_span = grid.meters_per_grid * 64
        return f"""You are analyzing a video of an indoor scene. Estimate the room floor area.

=== SCENE PERCEPTION ===
Detected {len(grid.entities)} objects. Estimated scene span: ~{scene_span:.1f}m.

=== QUESTION ===
{q}

Use furniture (beds~2m, sofas~2m, doors~0.9m wide) as scale references. Most rooms are 5-50 sq meters.
Respond with ONLY a single number (square meters).

Answer:"""

    elif qt == 'object_abs_distance':
        return f"""You are analyzing a video of an indoor scene. Estimate the distance between two objects.

=== REFERENCE SIZES FOR SCALE ===
Door: ~2.0m tall. Chair: ~0.8m tall. Bed: ~2.0m long. Table: ~0.75m high.
{grid_ref}

=== QUESTION ===
{q}

Locate both objects in the video. Use reference objects for depth estimation. Indoor distances typically 0.3-15m.
Respond with ONLY a decimal number (meters, e.g. "2.5").

Answer:"""

    else:
        # === 空间选择题: 给Grid摘要 + Grid参考 + 审查意见 ===
        options_text = "\n".join(task.options) if task.options else ""
        grid_summary = _build_grid_summary(grid, task)

        task_instruction = {
            'object_rel_direction_easy': "Determine the relative direction (left/right) of the target object.",
            'object_rel_direction_medium': "Determine the relative direction (left/right/behind) of the target object.",
            'object_rel_direction_hard': "Determine the relative direction quadrant (front-left/front-right/back-left/back-right).",
            'object_rel_distance': "Determine which object is closest or farthest to the reference.",
            'obj_appearance_order': "Determine the order in which objects first appear in the video.",
            'route_planning': "Determine the correct sequence of turns for the described route.",
        }.get(qt, "Answer the spatial reasoning question.")

        return f"""You are analyzing a video of an indoor scene.
{task_instruction}

=== 3D SPATIAL PERCEPTION DATA (for reference, may contain errors) ===
{grid_summary}
{grid_ref}
{review_note}

=== QUESTION ===
{q}

=== OPTIONS ===
{options_text}

Watch the video carefully. The perception data is for reference — trust your visual observation when they conflict.
Answer with ONLY the option letter (A, B, C, or D).

Answer:"""


def _build_grid_summary(grid: Grid64, task: DecomposedTask) -> str:
    """构建Grid摘要"""
    lines = []
    lines.append(f"Scene: {len(grid.entities)} objects, meters_per_grid={grid.meters_per_grid:.4f}m, "
                 f"scene_span≈{grid.meters_per_grid * 64:.1f}m")

    relevant_eids = set()
    for name in task.key_entities:
        found = grid.get_by_category(name)
        if found:
            for e in found:
                relevant_eids.add(e.entity_id)
        else:
            lines.append(f"  [NOT FOUND] '{name}' not detected in grid")

    if relevant_eids:
        lines.append("\nKey objects:")
        for eid in sorted(relevant_eids):
            e = grid.entities[eid]
            phys = grid.grid_to_physical(e.grid_position)
            size_str = ""
            ps = grid.physical_size(eid)
            if ps:
                size_str = f", size≈{ps:.2f}m"
            lines.append(f"  {eid}: ({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m{size_str}, conf={e.confidence:.2f}")

    # 全局列表
    if task.question_type in ('object_counting', 'room_size_estimation',
                               'obj_appearance_order', 'route_planning',
                               'object_size_estimation', 'object_abs_distance'):
        lines.append(f"\nAll objects ({len(grid.entities)}):")
        for eid, e in sorted(grid.entities.items()):
            phys = grid.grid_to_physical(e.grid_position)
            extra = ""
            if task.question_type == 'object_counting':
                extra = f", count={e.count_in_frame}"
            elif task.question_type == 'obj_appearance_order':
                extra = f", first_frame={e.first_seen_frame}"
            elif task.question_type in ('object_size_estimation', 'object_abs_distance'):
                ps = grid.physical_size(e.entity_id)
                if ps:
                    extra = f", size≈{ps:.2f}m"
            lines.append(f"  {eid}: ({phys[0]:.1f},{phys[1]:.1f},{phys[2]:.1f})m{extra}")

    return "\n".join(lines)


def _parse_numerical(response: str, qt: str) -> Tuple[str, str]:
    m = re.search(r'[\d.]+', response)
    if m:
        return m.group(), response[:100]
    defaults = {
        'object_counting': '1', 'object_size_estimation': '100',
        'room_size_estimation': '25', 'object_abs_distance': '2.0'
    }
    return defaults.get(qt, '0'), f"[parse_fail] {response[:100]}"


def _parse_choice(response: str, options: List[str]) -> Tuple[str, str]:
    response_clean = response.split('[')[0].strip()
    m = re.search(r'^([A-D])', response_clean.upper())
    if m:
        return m.group(1), response[:100]
    for line in response.split('\n')[::-1]:
        line = line.strip()
        if line and line[0].upper() in 'ABCD':
            return line[0].upper(), response[:100]
    response_lower = response.lower()
    for i, opt in enumerate(options):
        opt_content = opt.lower()
        if len(opt) >= 3 and opt[1] in '.、':
            opt_content = opt[3:].strip().lower()
        if opt_content in response_lower:
            return chr(65 + i), response[:100]
    return "A", f"[fallback] {response[:100]}"


# ============================================================================
# 完整Pipeline
# ============================================================================

class AgenticPipelineV3:
    """Critic-Manager-Evolutor Architecture (V3)"""

    def __init__(self, device='cuda:0', vl_model_path=None):
        self.device = device
        self.vl_model_path = vl_model_path or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        self.builder = Grid64Builder(device=device, num_frames=16)
        self.vl = VLModel(device=device)

    def load_models(self):
        self.builder.load_models()
        self.vl.load(self.vl_model_path)

    def unload(self):
        self.builder.unload()
        self.vl.unload()

    def process_scene(self, video_path: str, samples: List[Dict]) -> List[Dict]:
        t0 = time.time()
        grid = self.builder.build_grid(video_path)
        build_time = time.time() - t0
        logger.info(f"  Grid built: {len(grid.entities)} entities, mpg={grid.meters_per_grid:.4f}m ({build_time:.1f}s)")

        results = []
        for sample in samples:
            grid_copy = copy.deepcopy(grid)
            result = self.process_sample(grid_copy, sample, video_path)
            results.append(result)
        return results

    def process_sample(self, grid: Grid64, sample: Dict, video_path: str) -> Dict:
        """5-Phase处理"""
        qt = sample['question_type']
        question = sample['question']
        options = sample.get('options') or []
        gt = sample['ground_truth']

        # Phase 1: Manager — 问题解耦
        task = phase1_manager(question, qt, options, grid)
        task.ground_truth = gt

        # Phase 2: Critic — VL(16帧)审查Grid, 只输出诊断
        critic = phase2_critic(self.vl, grid, task, video_path)

        # Phase 3: Manager审核 — 决定是否Evolution
        decision = phase3_manager_review(task, critic, grid)

        # Phase 4: Evolutor — 执行Grid修正 (传入builder供ADD操作使用)
        evolution = phase4_evolutor(grid, decision, builder=self.builder)

        # Phase 5: Reasoner — VL(16帧)最终回答
        pred, reasoning = phase5_reasoner(self.vl, grid, task, critic, evolution, video_path)

        score = evaluate_sample(qt, pred, gt)

        return {
            'scene_name': sample.get('scene_name', ''),
            'question_type': qt,
            'question': question,
            'ground_truth': gt,
            'options': options,
            'prediction': pred,
            'reasoning': reasoning,
            'score': score,
            # Critic诊断
            'critic_issues_count': len(critic.issues),
            'critic_has_issues': bool(critic.issues),
            'critic_summary': critic.summary[:200],
            'critic_response': critic.raw_response[:300],
            # Manager决策
            'manager_should_evolve': decision.should_evolve,
            'manager_reasoning': decision.reasoning[:200],
            # Evolution结果
            'grid_modified': evolution.grid_modified,
            'evolution_actions': evolution.actions_executed,
            # Grid参考
            'grid_ref_answer': task.grid_tool_answer,
            'grid_ref_reasoning': task.grid_tool_reasoning,
            # 兼容旧字段
            'critic_rounds': 1,
            'self_qa_checks': len(critic.issues),
            'self_qa_pass': len([i for i in critic.issues if i.get('confidence') != 'high']),
            'self_qa_fail': len([i for i in critic.issues if i.get('confidence') == 'high']),
            'evolution_instructions': evolution.actions_executed,
            # V7基准
            'v7_vl_score': sample.get('vl_score', 0),
            'v7_rule_score': sample.get('rule_score', 0),
        }


# ============================================================================
# Main
# ============================================================================

def select_test_samples(results: List[Dict], n_per_type: int = 10) -> List[Dict]:
    by_type = defaultdict(list)
    for r in results:
        by_type[r['question_type']].append(r)

    selected = []
    for qt, samples in sorted(by_type.items()):
        available = [s for s in samples if find_video_path(s['scene_name'])]
        n = min(n_per_type, len(available))
        if n > 0:
            indices = np.linspace(0, len(available) - 1, n, dtype=int)
            for idx in indices:
                selected.append(available[idx])
    return selected


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_per_type', type=int, default=10, help='小样本每类数量')
    parser.add_argument('--full', action='store_true', help='全量测试')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None, help='多卡并行GPU编号')
    parser.add_argument('--num_gpus', type=int, default=8, help='总GPU数')
    parser.add_argument('--vl-model', type=str,
                        default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    if args.gpu_id is not None:
        args.device = f'cuda:{args.gpu_id}'

    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    logger.info(f"Loading V7 baseline: {v7_path}")
    with open(v7_path) as f:
        v7_results = json.load(f)
    logger.info(f"V7: {len(v7_results)} samples")

    if args.full:
        test_samples = [s for s in v7_results if find_video_path(s['scene_name'])]
        logger.info(f"Full test: {len(test_samples)} samples")
    else:
        test_samples = select_test_samples(v7_results, n_per_type=args.n_per_type)
        logger.info(f"Selected {len(test_samples)} test samples")

    by_scene = defaultdict(list)
    for s in test_samples:
        by_scene[s['scene_name']].append(s)

    scene_list = sorted(by_scene.keys())

    if args.gpu_id is not None:
        total = len(scene_list)
        chunk_size = (total + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * chunk_size
        end = min(start + chunk_size, total)
        my_scenes = scene_list[start:end]
        my_samples = sum(len(by_scene[s]) for s in my_scenes)
        logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: scenes {start}-{end-1} ({len(my_scenes)} scenes, {my_samples} samples)")
    else:
        my_scenes = scene_list

    logger.info(f"Processing {len(my_scenes)} scenes on {args.device}")

    pipeline = AgenticPipelineV3(device=args.device, vl_model_path=args.vl_model)
    pipeline.load_models()

    all_results = []
    total_scenes = len(my_scenes)

    for si, scene_name in enumerate(my_scenes):
        samples = by_scene[scene_name]
        video_path = find_video_path(scene_name)
        if not video_path:
            logger.warning(f"[{si+1}/{total_scenes}] No video: {scene_name}")
            for s in samples:
                all_results.append({
                    'scene_name': scene_name, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []),
                    'prediction': '0', 'reasoning': 'no video', 'score': 0.0,
                    'critic_has_issues': False, 'critic_issues_count': 0,
                    'v7_vl_score': s.get('vl_score', 0),
                    'v7_rule_score': s.get('rule_score', 0),
                })
            continue

        logger.info(f"[{si+1}/{total_scenes}] Processing {scene_name} ({len(samples)} questions)")

        try:
            results = pipeline.process_scene(video_path, samples)
            for r in results:
                all_results.append(r)
                delta = r['score'] - r['v7_vl_score']
                marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
                evo_mark = "E!" if r.get('grid_modified') else ""
                n_issues = r.get('critic_issues_count', 0)
                logger.info(
                    f"  {r['question_type'][:25]:25s} [C:{n_issues} {evo_mark:3s}] "
                    f"Score={r['score']:.3f} V7={r['v7_vl_score']:.3f} {marker} "
                    f"| pred={str(r['prediction'])[:20]} gt={str(r['ground_truth'])[:15]}"
                )
        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            for s in samples:
                all_results.append({
                    'scene_name': scene_name, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []),
                    'prediction': '0', 'reasoning': f'error: {str(e)[:100]}', 'score': 0.0,
                    'critic_has_issues': False, 'critic_issues_count': 0,
                    'v7_vl_score': s.get('vl_score', 0),
                    'v7_rule_score': s.get('rule_score', 0),
                })

    pipeline.unload()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.gpu_id is not None:
        output_dir = PROJECT_ROOT / "outputs" / "agentic_pipeline_v4_full" / f"gpu{args.gpu_id}"
    else:
        output_dir = PROJECT_ROOT / "outputs" / f"agentic_pipeline_v4_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_results = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                cr[k] = float(v)
            elif isinstance(v, np.ndarray):
                cr[k] = v.tolist()
            else:
                cr[k] = v
        clean_results.append(cr)

    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_dir} ({len(all_results)} samples)")

    if args.gpu_id is None:
        _print_summary(all_results, output_dir, timestamp)


def _print_summary(all_results, output_dir, timestamp):
    print("\n" + "=" * 130)
    print("Agentic Pipeline V4 (ADD Evolution + Position3D Precision)")
    print(f"基准: V7 VL Overall = 63.61%  |  VL: {VL_DEFAULT_NFRAMES}帧, {VL_DEFAULT_MAX_PIXELS}px")
    print(f"架构: Manager → Critic(诊断) → Manager审核(DELETE/ADD) → Evolutor(DELETE+ADD) → Reasoner")
    print(f"测试样本: {len(all_results)}")
    print("=" * 130)

    task_types = sorted(set(r['question_type'] for r in all_results))

    print(f"\n{'Task':<35} {'N':>4} {'V7_VL':>7} {'V4':>7} {'Delta':>7} {'Issues':>7} {'Evolve%':>8}")
    print(f"{'-'*35} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    all_v7, all_pipe = [], []

    for qt in task_types:
        qt_r = [r for r in all_results if r['question_type'] == qt]
        v7 = np.mean([r['v7_vl_score'] for r in qt_r])
        pipe = np.mean([r['score'] for r in qt_r])
        delta = pipe - v7
        avg_issues = np.mean([r.get('critic_issues_count', 0) for r in qt_r])
        evolve_pct = np.mean([1 if r.get('grid_modified') else 0 for r in qt_r]) * 100
        marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
        print(f"  {qt:<35} {len(qt_r):>4} {v7:>6.3f} {pipe:>6.3f} {delta:>+6.3f} {marker} {avg_issues:>6.1f} {evolve_pct:>7.1f}%")
        all_v7.extend([r['v7_vl_score'] for r in qt_r])
        all_pipe.extend([r['score'] for r in qt_r])

    print(f"{'-'*35} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    ov7, op = np.mean(all_v7), np.mean(all_pipe)
    print(f"  {'Overall':<35} {len(all_results):>4} {ov7:>6.3f} {op:>6.3f} {op-ov7:>+6.3f}")

    total_vl_calls = len(all_results) * 2  # critic + reasoner
    avg_vl = total_vl_calls / len(all_results)
    print(f"\n  VL calls: total={total_vl_calls}, avg={avg_vl:.1f} per sample (critic + reasoner)")

    numerical = [r for r in all_results if r['question_type'] in NUMERICAL_TASKS]
    spatial = [r for r in all_results if r['question_type'] not in NUMERICAL_TASKS]
    if numerical:
        num_score = np.mean([r['score'] for r in numerical])
        num_v7 = np.mean([r['v7_vl_score'] for r in numerical])
        num_evo = np.mean([1 if r.get('grid_modified') else 0 for r in numerical]) * 100
        print(f"  数值型: n={len(numerical)}, V4={num_score:.3f}, V7={num_v7:.3f}, Evolve={num_evo:.1f}%")
    if spatial:
        spa_score = np.mean([r['score'] for r in spatial])
        spa_v7 = np.mean([r['v7_vl_score'] for r in spatial])
        spa_evo = np.mean([1 if r.get('grid_modified') else 0 for r in spatial]) * 100
        print(f"  空间型: n={len(spatial)}, V4={spa_score:.3f}, V7={spa_v7:.3f}, Evolve={spa_evo:.1f}%")

    print(f"\nResults saved to: {output_dir}")
    print(f"\n总结: V4 Overall={op:.4f} vs V7 VL={ov7:.4f} (Delta={op-ov7:+.4f})")

    summary = {
        'timestamp': timestamp,
        'version': 'v4_add_evolution_pos3d',
        'vl_nframes': VL_DEFAULT_NFRAMES,
        'vl_max_pixels': VL_DEFAULT_MAX_PIXELS,
        'architecture': 'Manager → Critic(diagnose) → Manager(review DELETE/ADD) → Evolutor(DELETE+ADD) → Reasoner',
        'n_samples': len(all_results),
        'overall': {'v7_vl': float(ov7), 'pipeline': float(op), 'delta': float(op - ov7)},
        'by_task': {qt: {
            'n': len([r for r in all_results if r['question_type'] == qt]),
            'v7_vl': float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type'] == qt])),
            'pipeline': float(np.mean([r['score'] for r in all_results if r['question_type'] == qt])),
            'avg_issues': float(np.mean([r.get('critic_issues_count', 0) for r in all_results if r['question_type'] == qt])),
            'evolve_rate': float(np.mean([1 if r.get('grid_modified') else 0
                                          for r in all_results if r['question_type'] == qt])),
        } for qt in task_types},
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
