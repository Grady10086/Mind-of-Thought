#!/usr/bin/env python3
"""
64³ Grid Mind Map — Agentic Pipeline V8 (VL-Evolving + Skills + High-Confidence Critic)

核心改进 (V7→V8):
  1. VL-based Evolving: REFINE/ADD用VL直接估算entity相对位置, 替代GroundingDINO重检测
  2. High-confidence Critic: 只对confidence=high的issues触发Evolve
  3. Skills系统: CODER分解为独立Skills, 含VL辅助版本(direction_vl_assist/route_vl_assist)

架构:
  Manager (VL Agent) ── Gather Step ──┬── skill_xxx()        : 独立Skills (Grid计算+VL辅助)
                                       ├── critic_tool()      : VL审查 (high-conf → evolve)
                                       ├── vl_evolve_tool()   : VL-based Grid修正
                                       ├── grid_query_tool()  : 查询Grid数据
                                       └── final_answer()     : 提交最终答案
                       ── Verify Step ── VL Pairwise Verification
                       ── Decide Step ── VL视觉推理

对比基准:
  V7 VL Overall = 63.61%
  V7p Overall = 65.15%
  V6 Overall = 65.48%
"""

import os
import sys
import json
import re
import gc
import copy
import time
import logging
import traceback
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
# 从 grid64_real_test.py 导入 Grid 核心组件
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
# Note: NUMERICAL_TASKS is used ONLY in _print_summary for display grouping, not in pipeline logic

# ============================================================================
# VL Model Wrapper
# ============================================================================

VL_DEFAULT_NFRAMES = 16
VL_DEFAULT_MAX_PIXELS = 480 * 560


class VLModel:
    """VL模型封装"""

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
                    model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
                logger.info("VL model loaded (Qwen3-VL)")
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
                logger.info("VL model loaded (Qwen2.5-VL)")
        except Exception as e:
            logger.error(f"Failed to load VL: {e}")
            traceback.print_exc()

    def unload(self):
        if self.model is not None:
            del self.model; self.model = None
        if self.processor is not None:
            del self.processor; self.processor = None
        gc.collect(); torch.cuda.empty_cache()

    def call(self, prompt: str, video_path: str, max_tokens: int = 512,
             nframes: int = VL_DEFAULT_NFRAMES, max_pixels: int = VL_DEFAULT_MAX_PIXELS) -> str:
        if self.model is None:
            return ""
        try:
            from qwen_vl_utils import process_vision_info
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video_path, "max_pixels": max_pixels, "nframes": nframes},
                {"type": "text", "text": prompt}
            ]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                    padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            response = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            return response.strip()
        except Exception as e:
            logger.warning(f"VL call failed: {e}")
            return ""


# ============================================================================
# 5 个平级 Tool 实现
# ============================================================================

class ToolExecutionContext:
    """一个样本处理过程中所有 Tool 共享的上下文"""
    def __init__(self, grid: Grid64, vl: VLModel, video_path: str,
                 builder: Grid64Builder, question: str, options: List[str]):
        self.grid = grid
        self.vl = vl
        self.video_path = video_path
        self.builder = builder
        self.question = question
        self.options = options
        self.tool_trace: List[Dict] = []
        self.vl_calls = 0
        self._final_answer = None
        # V8 tracking
        self.skills_used: List[str] = []
        self.high_conf_issues: List[Dict] = []
        self.vl_evolve_actions: List[Dict] = []


def _grid_to_text(grid: Grid64) -> str:
    """把 Grid 转为 Manager 可读的文本 — 统一格式, 不按任务类型过滤"""
    lines = [f"Scene: {len(grid.entities)} objects, meters_per_grid={grid.meters_per_grid:.4f}m, "
             f"scene_span≈{grid.meters_per_grid * 64:.1f}m"]
    for eid, e in sorted(grid.entities.items()):
        phys = grid.grid_to_physical(e.grid_position)
        ps = grid.physical_size(eid)
        sz = f", size≈{ps:.2f}m" if ps else ""
        nf = len(set(d['frame_order'] for d in e.detections)) if e.detections else 0
        total_frames = len(grid.camera_positions) if grid.camera_positions else 32
        lines.append(
            f"  {eid}: pos=({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m{sz}, "
            f"conf={e.confidence:.2f}, seen_in={nf}/{total_frames} frames, "
            f"count_per_frame={e.count_in_frame}, first_frame={e.first_seen_frame}")
    return "\n".join(lines)


# ── Tool 1: critic_tool ──────────────────────────────────────────────────────

def critic_tool(ctx: ToolExecutionContext, focus_entities: str = "", checkpoints: str = "") -> str:
    """
    让 VL 模型审查 Grid 数据与视频是否一致。
    Args:
        focus_entities: 逗号分隔的关注 entity 名字 (可空=全部审查)
        checkpoints: 需要验证的方面 (可空=自动判断)
    Returns:
        审查结果文本, 包含发现的 issues
    """
    grid_text = _grid_to_text(ctx.grid)
    focus_part = f"\nFocus on these entities: {focus_entities}" if focus_entities.strip() else ""
    check_part = f"\nVerify: {checkpoints}" if checkpoints.strip() else ""

    prompt = f"""You are a quality reviewer for a 3D perception system. Find errors ONLY — do NOT suggest fixes.

=== 3D PERCEPTION DATA ===
{grid_text}

=== QUESTION TO ANSWER ===
{ctx.question}
{focus_part}{check_part}

Watch the video carefully (16 frames). Report ONLY concrete errors:
ISSUE: entity=<name> | problem=<description> | confidence=<high/medium/low>
If no issues: ISSUE: none
SUMMARY: <one sentence>"""

    response = ctx.vl.call(prompt, ctx.video_path, max_tokens=400)
    ctx.vl_calls += 1

    issues = []
    summary = ""
    for line in response.split('\n'):
        ls = line.strip()
        if ls.upper().startswith('ISSUE:'):
            body = ls.split(':', 1)[1].strip()
            if body.lower() in ('none', 'no issues', 'n/a', ''):
                continue
            iss = {"raw": body}
            for part in body.split('|'):
                p = part.strip()
                if p.startswith('entity='):   iss['entity'] = p.split('=', 1)[1].strip()
                elif p.startswith('problem='): iss['problem'] = p.split('=', 1)[1].strip()
                elif p.startswith('confidence='): iss['confidence'] = p.split('=', 1)[1].strip().lower()
            issues.append(iss)
        elif ls.upper().startswith('SUMMARY:'):
            summary = ls.split(':', 1)[1].strip()

    ctx.tool_trace.append({'tool': 'critic', 'n_issues': len(issues),
                           'n_high_conf': len([i for i in issues if i.get('confidence') == 'high']),
                           'issues': issues, 'summary': summary})
    # V8: 存储high-confidence issues到ctx供后续evolve使用
    ctx.high_conf_issues = [iss for iss in issues if iss.get('confidence') == 'high']

    if not issues:
        return f"No issues found. {summary}"
    n_high = len(ctx.high_conf_issues)
    parts = [f"Found {len(issues)} issue(s) ({n_high} high-confidence):"]
    for iss in issues:
        conf = iss.get('confidence', '?')
        marker = " ★" if conf == 'high' else ""
        parts.append(f"  - {iss.get('entity','?')}: {iss.get('problem','?')} [confidence={conf}]{marker}")
    parts.append(f"Summary: {summary}")
    return "\n".join(parts)


# ── Tool 2: vl_evolve_tool (V8: VL-based Grid修正) ───────────────────────────

def _vl_estimate_entity_position(ctx: ToolExecutionContext, entity_name: str,
                                  reason: str = "") -> Optional[np.ndarray]:
    """用VL估算一个新entity的绝对位置坐标。"""
    grid = ctx.grid
    ref_entities = sorted(grid.entities.values(), key=lambda e: -e.confidence)[:3]
    if not ref_entities:
        return None
    ref_text = []
    for e in ref_entities:
        phys = grid.grid_to_physical(e.grid_position)
        ref_text.append(f"  {e.category}: at ({phys[0]:.2f}, {phys[1]:.2f}, {phys[2]:.2f})m")
    prompt = f"""Look at the video. I need to locate the "{entity_name}" in this scene.

Known object positions (meters):
{chr(10).join(ref_text)}

Scene span ≈ {grid.meters_per_grid * 64:.1f}m.

Estimate where the {entity_name} is located in the same coordinate system.

Answer in this EXACT format:
ESTIMATED_XYZ: <x>, <y>, <z>
If you cannot see it: NOT_VISIBLE"""
    response = ctx.vl.call(prompt, ctx.video_path, max_tokens=150)
    ctx.vl_calls += 1
    if 'NOT_VISIBLE' in response.upper():
        return None
    m_xyz = re.search(r'ESTIMATED_XYZ:\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)', response)
    if m_xyz:
        try:
            return np.array([float(m_xyz.group(1)), float(m_xyz.group(2)), float(m_xyz.group(3))])
        except ValueError:
            pass
    return None


def _vl_refine_entity_position(ctx: ToolExecutionContext, entity_name: str,
                                old_entity: GridEntity, reason: str = "") -> Optional[np.ndarray]:
    """用VL修正已存在entity的位置。V7 REFINE用GroundingDINO重检测结果不变, V8用VL直接修正。"""
    grid = ctx.grid
    old_phys = grid.grid_to_physical(old_entity.grid_position)
    ref_entities = [e for e in grid.entities.values() if e.entity_id != old_entity.entity_id]
    ref_entities.sort(key=lambda e: -e.confidence)
    ref_text = []
    for e in ref_entities[:3]:
        phys = grid.grid_to_physical(e.grid_position)
        ref_text.append(f"  {e.category}: at ({phys[0]:.2f}, {phys[1]:.2f}, {phys[2]:.2f})m")
    issue_text = f"\nReported issue: {reason}" if reason else ""
    prompt = f"""Look at the video. The system detected "{entity_name}" at ({old_phys[0]:.2f}, {old_phys[1]:.2f}, {old_phys[2]:.2f})m, but this may be wrong.
{issue_text}

Reference objects:
{chr(10).join(ref_text)}

Scene span ≈ {grid.meters_per_grid * 64:.1f}m

Is this position approximately correct (within ~1m)?

If correct: POSITION_OK
If wrong: CORRECTED_XYZ: <x>, <y>, <z>"""
    response = ctx.vl.call(prompt, ctx.video_path, max_tokens=150)
    ctx.vl_calls += 1
    if 'POSITION_OK' in response.upper():
        return None
    m_xyz = re.search(r'CORRECTED_XYZ:\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)', response)
    if m_xyz:
        try:
            corrected = np.array([float(m_xyz.group(1)), float(m_xyz.group(2)), float(m_xyz.group(3))])
            scene_span = grid.meters_per_grid * 64
            if np.linalg.norm(corrected - old_phys) > scene_span * 0.5:
                return None
            return corrected
        except ValueError:
            pass
    return None


def vl_evolve_tool(ctx: ToolExecutionContext, action: str, target: str, reason: str = "") -> str:
    """V8 VL-based Grid修正。REFINE/ADD用VL估算位置, 替代GroundingDINO。"""
    grid = ctx.grid
    action = action.strip().upper()

    if action == 'DELETE':
        eid = target.strip().replace(' ', '_')
        if eid not in grid.entities:
            cands = grid.get_by_category(target.strip())
            if cands:
                eid = cands[0].entity_id
            else:
                ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'DELETE', 'target': target, 'ok': False})
                return f"DELETE failed: '{target}' not found in grid."
        del grid.entities[eid]
        ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'DELETE', 'target': eid, 'ok': True})
        ctx.vl_evolve_actions.append({'action': 'DELETE', 'target': eid, 'ok': True})
        return f"DELETE '{eid}' done. Grid now has {len(grid.entities)} entities."

    elif action == 'ADD':
        name = target.strip().lower()
        if grid.get_by_category(name):
            ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'ADD', 'target': name, 'ok': False, 'reason': 'exists'})
            return f"ADD skipped: '{name}' already in grid."
        # V8: 先尝试VL估算位置
        estimated_xyz = _vl_estimate_entity_position(ctx, name, reason)
        if estimated_xyz is not None:
            mpg = grid.meters_per_grid
            grid_pos = tuple(np.clip((estimated_xyz / max(mpg, 1e-8)).astype(int), 0, 63))
            base_id = name.replace(' ', '_').lower()
            eid = f"{base_id}_0"
            cnt = 1
            while eid in grid.entities:
                eid = f"{base_id}_{cnt}"; cnt += 1
            entity = GridEntity(entity_id=eid, category=name.lower(), grid_position=grid_pos,
                                position_3d=estimated_xyz, size_3d=None, confidence=0.60,
                                first_seen_frame=0, count_in_frame=1, detections=[])
            grid.entities[eid] = entity
            phys = grid.grid_to_physical(grid_pos)
            ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'ADD', 'target': name, 'ok': True, 'eid': eid, 'method': 'vl'})
            ctx.vl_evolve_actions.append({'action': 'ADD', 'target': name, 'ok': True, 'method': 'vl'})
            return f"ADD '{name}' as '{eid}' at ({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m (VL-estimated)."
        # VL失败 → fallback to GroundingDINO
        if ctx.builder is not None:
            added = ctx.builder.search_and_add_entity(grid, name)
            if added:
                phys = grid.grid_to_physical(added.grid_position)
                ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'ADD', 'target': name, 'ok': True, 'eid': added.entity_id, 'method': 'gd'})
                ctx.vl_evolve_actions.append({'action': 'ADD', 'target': name, 'ok': True, 'method': 'gd'})
                return f"ADD '{name}' as '{added.entity_id}' at ({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m (GD fallback)."
        ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'ADD', 'target': name, 'ok': False})
        return f"ADD failed: '{name}' not found."

    elif action == 'REFINE':
        name = target.strip().lower()
        old = grid.get_by_category(name)
        if not old:
            ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'REFINE', 'target': name, 'ok': False})
            return f"REFINE failed: '{name}' not found in grid."
        old_entity = old[0]
        old_phys = grid.grid_to_physical(old_entity.grid_position)
        corrected_xyz = _vl_refine_entity_position(ctx, name, old_entity, reason)
        if corrected_xyz is not None:
            old_entity.position_3d = corrected_xyz
            mpg = grid.meters_per_grid
            old_entity.grid_position = tuple(np.clip((corrected_xyz / max(mpg, 1e-8)).astype(int), 0, 63))
            new_phys = grid.grid_to_physical(old_entity.grid_position)
            ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'REFINE', 'target': name, 'ok': True, 'method': 'vl'})
            ctx.vl_evolve_actions.append({'action': 'REFINE', 'target': name, 'ok': True, 'method': 'vl'})
            return f"REFINE '{name}': ({old_phys[0]:.2f},{old_phys[1]:.2f},{old_phys[2]:.2f})m → ({new_phys[0]:.2f},{new_phys[1]:.2f},{new_phys[2]:.2f})m (VL)."
        ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'REFINE', 'target': name, 'ok': False, 'reason': 'vl_no_correction'})
        return f"REFINE '{name}': VL found no correction needed. Kept unchanged."

    elif action == 'SCALE_ADJUST':
        target_val = target.strip().lower()
        old_mpg = grid.meters_per_grid
        if target_val == 'auto':
            grid._meters_per_grid = None
            grid.calibrate_scale()
            new_mpg = grid.meters_per_grid
            ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'SCALE_ADJUST', 'target': 'auto', 'ok': True})
            ctx.vl_evolve_actions.append({'action': 'SCALE_ADJUST', 'target': 'auto', 'ok': True})
            return f"SCALE_ADJUST auto: mpg {old_mpg:.4f} → {new_mpg:.4f}m"
        try:
            factor = float(target_val)
            if factor <= 0 or factor > 100:
                ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'SCALE_ADJUST', 'target': target, 'ok': False})
                return f"SCALE_ADJUST failed: factor {factor} out of range."
            new_mpg = old_mpg * factor
            grid.meters_per_grid = new_mpg
            ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'SCALE_ADJUST', 'target': target, 'ok': True})
            ctx.vl_evolve_actions.append({'action': 'SCALE_ADJUST', 'target': target, 'ok': True})
            return f"SCALE_ADJUST: mpg {old_mpg:.4f} × {factor} → {new_mpg:.4f}m"
        except ValueError:
            ctx.tool_trace.append({'tool': 'vl_evolve', 'action': 'SCALE_ADJUST', 'target': target, 'ok': False})
            return f"SCALE_ADJUST failed: '{target}' invalid."

    return f"Unknown action '{action}'. Use DELETE / ADD / REFINE / SCALE_ADJUST."


# ── Tool 3: Skills系统 (V8: 替代CODER) ────────────────────────────────────────

SKILL_REGISTRY = {}

def _register_skill(name, description):
    def decorator(func):
        SKILL_REGISTRY[name] = {'func': func, 'description': description}
        return func
    return decorator

@_register_skill('direction_compute', 'Compute direction (left/right/front/back) from 3D Grid positions')
def skill_direction_compute(ctx):
    pred, reason = grid_answer_direction(ctx.grid, ctx.question, ctx.options)
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'direction_compute', 'result': pred})
    ctx.skills_used.append('direction_compute')
    return f"Direction computed: answer={pred}, detail={reason}"

@_register_skill('direction_vl_assist', 'Direction with VL cross-check: computes Grid direction and gets VL opinion')
def skill_direction_vl_assist(ctx):
    pred, reason = grid_answer_direction(ctx.grid, ctx.question, ctx.options)
    # 始终获取VL意见作为交叉参考，让Decide自主权衡
    if ctx.options:
        options_text = "\n".join(ctx.options)
        vl_resp = ctx.vl.call(f"""Look at the video. {ctx.question}\n\n{options_text}\n\nAnswer with ONLY the letter (A/B/C/D):""",
                              ctx.video_path, max_tokens=80)
        ctx.vl_calls += 1
        m = re.search(r'^([A-D])', vl_resp.strip())
        if m:
            vl_letter = m.group(1)
            if vl_letter == pred:
                ctx.tool_trace.append({'tool': 'skill', 'skill': 'direction_vl_assist', 'grid': pred, 'vl': vl_letter, 'used': 'agree'})
                ctx.skills_used.append('direction_vl_assist')
                return f"Direction: Grid={pred} VL={vl_letter} (agree), answer={pred}, detail={reason}"
            else:
                ctx.tool_trace.append({'tool': 'skill', 'skill': 'direction_vl_assist', 'grid': pred, 'vl': vl_letter, 'used': 'disagree'})
                ctx.skills_used.append('direction_vl_assist')
                return f"Direction: Grid={pred} VL={vl_letter} (disagree), detail={reason}"
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'direction_vl_assist', 'result': pred, 'used': 'grid_only'})
    ctx.skills_used.append('direction_vl_assist')
    return f"Direction computed: answer={pred}, detail={reason}"

@_register_skill('rel_distance_compute', 'Compare distances: closest/farthest to reference object')
def skill_rel_distance_compute(ctx):
    pred, reason = grid_answer_rel_distance(ctx.grid, ctx.question, ctx.options)
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'rel_distance_compute', 'result': pred})
    ctx.skills_used.append('rel_distance_compute')
    return f"Relative distance computed: answer={pred}, detail={reason}"

@_register_skill('abs_distance_compute', 'Compute absolute distance between two objects in meters')
def skill_abs_distance_compute(ctx):
    pred, reason = grid_answer_abs_distance(ctx.grid, ctx.question)
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'abs_distance_compute', 'result': pred})
    ctx.skills_used.append('abs_distance_compute')
    return f"Distance computed: answer={pred}m, detail={reason}"

@_register_skill('counting', 'Count objects of a category in the scene')
def skill_counting(ctx):
    pred, reason = grid_answer_counting(ctx.grid, ctx.question)
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'counting', 'result': pred})
    ctx.skills_used.append('counting')
    return f"Count computed: answer={pred}, detail={reason}"

@_register_skill('size_estimation', 'Estimate physical size of an object (cm)')
def skill_size_estimation(ctx):
    pred, reason = grid_answer_size(ctx.grid, ctx.question)
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'size_estimation', 'result': pred})
    ctx.skills_used.append('size_estimation')
    return f"Size computed: answer={pred}cm, detail={reason}"

@_register_skill('room_size_estimation', 'Estimate room floor area (sq meters)')
def skill_room_size_estimation(ctx):
    pred, reason = grid_answer_room_size(ctx.grid, ctx.question)
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'room_size_estimation', 'result': pred})
    ctx.skills_used.append('room_size_estimation')
    return f"Room size computed: answer={pred} sq meters, detail={reason}"

@_register_skill('appearance_order', 'Determine which object appears first in the video')
def skill_appearance_order(ctx):
    pred, reason = grid_answer_appearance_order(ctx.grid, ctx.question, ctx.options)
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'appearance_order', 'result': pred})
    ctx.skills_used.append('appearance_order')
    return f"Appearance order computed: answer={pred}, detail={reason}"

@_register_skill('route_planning', 'Simulate robot navigation with turn commands')
def skill_route_planning(ctx):
    pred, reason = grid_answer_route(ctx.grid, ctx.question, ctx.options)
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'route_planning', 'result': pred})
    ctx.skills_used.append('route_planning')
    return f"Route computed: answer={pred}, detail={reason}"

@_register_skill('route_vl_assist', 'Route planning with VL cross-check')
def skill_route_vl_assist(ctx):
    pred, reason = grid_answer_route(ctx.grid, ctx.question, ctx.options)
    # 始终获取VL意见作为交叉参考
    if ctx.options:
        options_text = "\n".join(ctx.options)
        vl_resp = ctx.vl.call(f"""Watch the video.\n\n{ctx.question}\n\n{options_text}\n\nImagine yourself as the robot. Answer ONLY the letter (A/B/C/D):""",
                              ctx.video_path, max_tokens=80)
        ctx.vl_calls += 1
        m = re.search(r'^([A-D])', vl_resp.strip())
        if m:
            vl_letter = m.group(1)
            if vl_letter == pred:
                ctx.tool_trace.append({'tool': 'skill', 'skill': 'route_vl_assist', 'grid': pred, 'vl': vl_letter, 'used': 'agree'})
                ctx.skills_used.append('route_vl_assist')
                return f"Route: Grid={pred} VL={vl_letter} (agree), answer={pred}, detail={reason}"
            else:
                ctx.tool_trace.append({'tool': 'skill', 'skill': 'route_vl_assist', 'grid': pred, 'vl': vl_letter, 'used': 'disagree'})
                ctx.skills_used.append('route_vl_assist')
                return f"Route: Grid={pred} VL={vl_letter} (disagree), detail={reason}"
    ctx.tool_trace.append({'tool': 'skill', 'skill': 'route_vl_assist', 'result': pred})
    ctx.skills_used.append('route_vl_assist')
    return f"Route computed: answer={pred}, detail={reason}"

def execute_skill(ctx, skill_name):
    skill_name = skill_name.strip().lower()
    if skill_name not in SKILL_REGISTRY:
        return f"Unknown skill '{skill_name}'. Available: {', '.join(SKILL_REGISTRY.keys())}"
    try:
        return SKILL_REGISTRY[skill_name]['func'](ctx)
    except Exception as e:
        return f"Skill error ({skill_name}): {e}"


# ── Tool 4: grid_query_tool ──────────────────────────────────────────────────

def grid_query_tool(ctx: ToolExecutionContext, query: str, **kwargs) -> str:
    """
    查询 Grid 数据 (无VL调用, 纯数据查询)。
    Args:
        query: 查询类型 — "entity_info", "all_entities", "find",
               "distances_from", "grid_summary"
        **kwargs: 查询参数
    Returns:
        查询结果
    """
    grid = ctx.grid
    q = query.strip().lower()

    if q == 'entity_info':
        names = kwargs.get('names', '').split(',')
        lines = []
        for name in names:
            name = name.strip()
            found = grid.get_by_category(name)
            if found:
                for e in found:
                    phys = grid.grid_to_physical(e.grid_position)
                    ps = grid.physical_size(e.entity_id)
                    sz = f", size≈{ps:.2f}m" if ps else ""
                    nf = len(set(d['frame_order'] for d in e.detections)) if e.detections else 0
                    lines.append(f"  {e.entity_id}: pos=({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m{sz}, "
                                 f"conf={e.confidence:.2f}, frames={nf}, count={e.count_in_frame}, "
                                 f"first_frame={e.first_seen_frame}")
            else:
                lines.append(f"  '{name}': NOT FOUND")
        return "\n".join(lines) if lines else "No entities specified."

    elif q == 'all_entities':
        return _grid_to_text(grid)

    elif q == 'find':
        name = kwargs.get('name', '').strip()
        found = grid.get_by_category(name)
        if found:
            parts = []
            for e in found:
                phys = grid.grid_to_physical(e.grid_position)
                parts.append(f"{e.entity_id} at ({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m, conf={e.confidence:.2f}")
            return f"Found {len(found)}: " + "; ".join(parts)
        return f"'{name}' not found in grid."

    elif q == 'distances_from':
        ref = kwargs.get('ref', '').strip()
        e_ref = grid.get_by_category(ref)
        if not e_ref:
            return f"Reference '{ref}' not found."
        lines = [f"Distances from '{ref}' ({e_ref[0].entity_id}):"]
        targets = kwargs.get('targets', '').split(',')
        if targets and targets[0]:
            for t in targets:
                t = t.strip()
                e_t = grid.get_by_category(t)
                if e_t:
                    d = grid.physical_distance(e_ref[0].entity_id, e_t[0].entity_id)
                    lines.append(f"  → {t}: {d:.2f}m" if d is not None else f"  → {t}: N/A")
                else:
                    lines.append(f"  → {t}: not in grid")
        else:
            for eid, e in sorted(grid.entities.items()):
                if eid != e_ref[0].entity_id:
                    d = grid.physical_distance(e_ref[0].entity_id, eid)
                    if d is not None:
                        lines.append(f"  → {eid}: {d:.2f}m")
        return "\n".join(lines)

    elif q == 'grid_summary':
        return _grid_to_text(grid)

    return f"Unknown query '{q}'. Available: entity_info, all_entities, find, distances_from, grid_summary"


# ── Tool 5: final_answer ─────────────────────────────────────────────────────

def final_answer(ctx: ToolExecutionContext, answer) -> str:
    """
    提交最终答案。调用后 loop 立即结束。
    Args:
        answer: 最终答案值（数字或选项字母）
    Returns:
        确认信息
    """
    ctx._final_answer = str(answer).strip()
    ctx.tool_trace.append({'tool': 'final_answer', 'answer': ctx._final_answer})
    return f"Answer submitted: {ctx._final_answer}"


# ============================================================================
# Manager — CodeAgent Style VL ReAct Loop
# ============================================================================

def _extract_question_entities(question: str, options: List[str]) -> List[str]:
    """从问题和选项中提取可能相关的实体名 (纯文本解析, 无task-type依赖)"""
    entities = []
    q = question.lower()

    # 常见模式: "standing by X and facing Y, is Z ...", "distance between X and Y",
    # "how many X", "how big is the X", "which object appears first"
    patterns = [
        r'standing by (?:the )?(.+?)\s+and\s+facing (?:the )?(.+?),.*?(?:is|are)\s+(?:the )?(.+?)\s+(?:to|on)\s',
        r'distance between (?:the )?(.+?)\s+and\s+(?:the )?(.+?)[\s\?\.]',
        r'from (?:the )?(\w+(?:\s+\w+)*?) to (?:the )?(\w+(?:\s+\w+)*)',
        r'how (?:big|large|tall|long|wide) (?:is|are) (?:the )?(.+?)[\?\.]',
        r'how many (.+?)(?:\s+are|\s+in|\s+do|\?)',
        r'size of (?:the )?(.+?)[\?\s]',
        r'closest to (?:the )?(.+?)[\?\s]',
        r'farthest from (?:the )?(.+?)[\?\s]',
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            for g in m.groups():
                if g:
                    entities.append(g.strip())

    # 从选项中提取实体名（选项通常是 "A. behind the sofa" 或 "A. left" 等）
    for opt in options:
        opt_clean = opt.strip()
        if len(opt_clean) >= 3 and opt_clean[1] in '.、':
            opt_content = opt_clean[3:].strip().lower()
            # 如果选项是实体名（非方向词），加入
            direction_words = {'left', 'right', 'behind', 'front', 'back', 'front-left',
                             'front-right', 'back-left', 'back-right', 'yes', 'no'}
            if opt_content and opt_content not in direction_words and not opt_content.replace('.','').isdigit():
                entities.append(opt_content)

    return entities


def _auto_select_coder_type(question: str, options: List[str]) -> Optional[str]:
    """V8: 自动推断最合适的Skill名 (保留原函数名兼容调用)"""
    q = question.lower()
    if any(kw in q for kw in ['distance between', 'how far', 'meters apart']):
        return 'rel_distance_compute' if options else 'abs_distance_compute'
    if any(kw in q for kw in ['standing by', 'facing', 'to the left', 'to the right', 'to my']):
        return 'direction_vl_assist'
    if any(kw in q for kw in ['how many', 'count', 'number of']):
        return 'counting'
    if any(kw in q for kw in ['how big', 'how large', 'how tall', 'how long', 'how wide',
                                'size of', 'height of', 'length of', 'width of']):
        return 'size_estimation'
    if any(kw in q for kw in ['room size', 'floor area', 'square meter', 'sq m', 'area of the room',
                                'how big is the room', 'how large is the room']):
        return 'room_size_estimation'
    if any(kw in q for kw in ['appear first', 'appears first', 'appearance order',
                                'which object first', 'seen first']):
        return 'appearance_order'
    if any(kw in q for kw in ['route', 'path', 'navigate', 'walk from', 'go from', 'turn']):
        return 'route_vl_assist'
    return None


def _coder_result_confidence(computation: str, result_str: str, grid: Grid64 = None) -> str:
    """评估skill计算结果的可信度 — 仅基于结果中的客观信号（缺失/解析失败），不人为设定阈值"""
    r = result_str.lower()

    # 明确的失败/不可靠信号 → low (客观: 计算过程本身报错或数据缺失)
    low_signals = ['not found', 'cannot parse', 'same 3d position',
                   'no options', 'error', 'failed', 'n/a', 'no waypoint to infer',
                   'same position', 'insufficient data']
    for sig in low_signals:
        if sig in r:
            return 'low'

    return 'normal'


# ============================================================================
# Self-Verification: VL Pairwise Verification + Auto-ADD
# ============================================================================

def _extract_not_found_entities(coder_result: str) -> List[str]:
    """从 CODER 结果中提取 not found 的实体名"""
    missing = []
    for m in re.finditer(r"(?:ref|observer|facing|target|start|waypoint)[= ]'([^']+)' not found", coder_result):
        missing.append(m.group(1))
    for m in re.finditer(r"not found: \w+='([^']+)'", coder_result):
        missing.append(m.group(1))
    # 也匹配 "ref 'X' not found" 简单格式
    for m in re.finditer(r"'([^']+)' not found", coder_result):
        name = m.group(1)
        if name not in missing:
            missing.append(name)
    return missing


def _auto_add_missing_entities(ctx: ToolExecutionContext, coder_result: str) -> Tuple[bool, str]:
    """CODER 返回 not found 时，自动尝试 ADD 缺失实体。返回 (grid_modified, log)"""
    missing = _extract_not_found_entities(coder_result)
    if not missing:
        return False, ""

    logs = []
    modified = False
    for name in missing[:3]:  # 最多补3个
        name_clean = name.strip().lower()
        # 跳过非实体名（如 "into the room"）
        if len(name_clean) < 2 or any(w in name_clean for w in ['the room', 'into', 'toward']):
            continue
        if ctx.grid.get_by_category(name_clean):
            continue  # 已在Grid中
        if ctx.builder is None:
            continue
        result = vl_evolve_tool(ctx, 'ADD', name_clean, reason=f"auto-add: not found '{name_clean}'")
        logs.append(f"[AUTO_ADD] {result}")
        if 'ADD failed' not in result and 'skipped' not in result:
            modified = True

    return modified, "\n".join(logs)


def _vl_pairwise_verify_rel_distance(ctx: ToolExecutionContext, coder_result: str) -> Tuple[str, str]:
    """对 rel_distance 的 CODER 排序结果，用 VL 做 pairwise verification
    
    策略: 提取 CODER 给出的 top-2 候选，让 VL 判断哪个更近/更远
    如果 VL 判断与 CODER 不一致，返回 VL 的选择
    """
    # 解析 CODER 结果: "ref=microwave, trash can=0.43m, pillow=1.06m, tv=1.67m → trash can"
    m_detail = re.search(r'detail=ref=([^,]+),\s*(.+?)→', coder_result)
    if not m_detail:
        return '', ''
    
    ref_name = m_detail.group(1).strip()
    dist_part = m_detail.group(2).strip()
    
    # 解析各候选距离
    candidates = []
    for m in re.finditer(r'(\w[\w\s]*?)=([\d.]+)m', dist_part):
        candidates.append((m.group(1).strip(), float(m.group(2))))
    
    if len(candidates) < 2:
        return '', ''
    
    # 排序，取 top2
    q = ctx.question.lower()
    is_farthest = 'farthest' in q or 'furthest' in q
    candidates.sort(key=lambda x: -x[1] if is_farthest else x[1])
    
    top1_name, top1_dist = candidates[0]
    top2_name, top2_dist = candidates[1]
    
    # 始终执行VL校验，不设人为margin阈值
    margin = abs(top1_dist - top2_dist)
    
    # 构造 pairwise 问题
    comparison = "closer to" if not is_farthest else "farther from"
    prompt = f"""Look at the video carefully. I need you to compare two objects' distances to the {ref_name}.

Question: Which object is {comparison} the {ref_name}: the {top1_name} or the {top2_name}?

Think about where each object is relative to the {ref_name} in the scene.
Answer with ONLY the object name (either "{top1_name}" or "{top2_name}"):"""

    response = ctx.vl.call(prompt, ctx.video_path, max_tokens=50)
    ctx.vl_calls += 1
    
    response_lower = response.lower().strip()
    
    # 判断 VL 选了谁
    vl_choice = ''
    if top1_name.lower() in response_lower and top2_name.lower() not in response_lower:
        vl_choice = top1_name
    elif top2_name.lower() in response_lower and top1_name.lower() not in response_lower:
        vl_choice = top2_name
    elif top1_name.lower() in response_lower and top2_name.lower() in response_lower:
        # 两个都提到，看谁先出现
        idx1 = response_lower.index(top1_name.lower())
        idx2 = response_lower.index(top2_name.lower())
        vl_choice = top1_name if idx1 < idx2 else top2_name
    
    if not vl_choice:
        return '', f"VL pairwise inconclusive: '{response[:80]}'"
    
    # VL 是否与 CODER 一致
    coder_top = top1_name
    if vl_choice.lower() == coder_top.lower():
        return 'agree', f"VL pairwise confirms: {vl_choice} is {comparison} {ref_name} (margin={margin:.2f}m)"
    else:
        # VL 不同意 → 找到 VL 选的候选对应的选项字母
        vl_letter = ''
        for opt in ctx.options:
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt).strip().lower()
            if vl_choice.lower() in opt_content or opt_content in vl_choice.lower():
                vl_letter = opt[0]
                break
        return f'override:{vl_letter}' if vl_letter else 'disagree', \
               f"VL pairwise disagrees: VL={vl_choice} vs CODER={coder_top} (margin={margin:.2f}m)"


def _vl_pairwise_verify_direction(ctx: ToolExecutionContext, coder_result: str) -> Tuple[str, str]:
    """对 direction 的 CODER 结果，用 VL 做视觉方位确认
    
    策略: 提取 CODER 判断的方向，让 VL 直接判断方向，看是否一致
    """
    # 解析 CODER 结果中的方向和实体
    m_detail = re.search(r'obs=(\S+)\s+fac=(\S+)\s+tgt=(\S+)\s*\|\s*fwd=([-\d.]+)\s+right=([-\d.]+)\s*→\s*(\S+)', coder_result)
    if not m_detail:
        return '', ''
    
    obs_name = m_detail.group(1)
    fac_name = m_detail.group(2)
    tgt_name = m_detail.group(3)
    fwd_val = float(m_detail.group(4))
    right_val = float(m_detail.group(5))
    coder_direction = m_detail.group(6)
    
    # 如果 fwd/right 的绝对值都很大(清晰方向)，方向很明确
    # 不设人为阈值跳过验证 — 始终执行VL校验
    
    # 构造验证问题
    options_text = "\n".join(ctx.options) if ctx.options else ""
    prompt = f"""Look at the video carefully. Imagine you are standing at the {obs_name} and facing toward the {fac_name}.

From this viewpoint, which direction is the {tgt_name}?

{options_text}

Think step by step about the spatial layout, then answer with ONLY the option letter (A, B, C, or D):"""

    response = ctx.vl.call(prompt, ctx.video_path, max_tokens=80)
    ctx.vl_calls += 1
    
    # 提取 VL 的选项
    vl_letter = ''
    m_letter = re.search(r'^([A-D])', response.strip())
    if m_letter:
        vl_letter = m_letter.group(1)
    else:
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].upper() in 'ABCD' and (len(line) == 1 or line[1] in '.、) ,'):
                vl_letter = line[0].upper()
                break
    
    if not vl_letter:
        return '', f"VL direction verify inconclusive: '{response[:80]}'"
    
    # CODER 的答案字母
    m_coder_ans = re.search(r'answer=([A-D])', coder_result)
    coder_letter = m_coder_ans.group(1) if m_coder_ans else ''
    
    if vl_letter == coder_letter:
        return 'agree', f"VL direction confirms: {vl_letter} ({coder_direction})"
    else:
        return f'override:{vl_letter}', f"VL direction disagrees: VL={vl_letter} vs CODER={coder_letter} ({coder_direction})"


def _vl_pairwise_verify_appearance(ctx: ToolExecutionContext, coder_result: str) -> Tuple[str, str]:
    """对 appearance_order 的 CODER 结果，用 VL 做视觉确认
    
    策略: 提取排序中的第一个物体，让 VL 确认它是否确实最先出现
    """
    # 解析 CODER 排序
    m_order = re.search(r"order by first_frame: \[([^\]]+)\]", coder_result)
    if not m_order:
        return '', ''
    
    items_str = m_order.group(1)
    items = [s.strip().strip("'\"") for s in items_str.split(',')]
    if len(items) < 2:
        return '', ''
    
    first_obj = items[0]
    second_obj = items[1]
    
    prompt = f"""Watch the video from the beginning. Which object appears first in the video: the {first_obj} or the {second_obj}?

Answer with ONLY the object name (either "{first_obj}" or "{second_obj}"):"""

    response = ctx.vl.call(prompt, ctx.video_path, max_tokens=50)
    ctx.vl_calls += 1
    
    response_lower = response.lower().strip()
    
    if first_obj.lower() in response_lower and second_obj.lower() not in response_lower:
        return 'agree', f"VL confirms {first_obj} appears before {second_obj}"
    elif second_obj.lower() in response_lower and first_obj.lower() not in response_lower:
        # VL 认为第二个先出现 → CODER 排序可能错
        return 'disagree', f"VL disagrees: {second_obj} appears before {first_obj}"
    
    return '', f"VL appearance verify inconclusive: '{response[:80]}'"


def _vl_pairwise_verify_route(ctx: ToolExecutionContext, coder_result: str) -> Tuple[str, str]:
    """对 route 的 CODER 结果，用 VL 做视觉路线确认"""
    # route 的 CODER 结果中有各选项 score
    m_coder_ans = re.search(r'answer=([A-D])', coder_result)
    if not m_coder_ans:
        return '', ''
    
    coder_letter = m_coder_ans.group(1)
    
    # 直接让 VL 做 route 判断
    options_text = "\n".join(ctx.options) if ctx.options else ""
    prompt = f"""Read the question carefully and watch the video.

{ctx.question}

{options_text}

Think about the spatial layout of the objects in the scene. Imagine yourself as the robot and reason about which turns would lead you to each waypoint.

Answer with ONLY the option letter (A, B, C, or D):"""

    response = ctx.vl.call(prompt, ctx.video_path, max_tokens=80)
    ctx.vl_calls += 1
    
    vl_letter = ''
    m_letter = re.search(r'^([A-D])', response.strip())
    if m_letter:
        vl_letter = m_letter.group(1)
    else:
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].upper() in 'ABCD' and (len(line) == 1 or line[1] in '.、) ,'):
                vl_letter = line[0].upper()
                break
    
    if not vl_letter:
        return '', f"VL route verify inconclusive: '{response[:80]}'"
    
    if vl_letter == coder_letter:
        return 'agree', f"VL route confirms: {vl_letter}"
    else:
        return f'override:{vl_letter}', f"VL route disagrees: VL={vl_letter} vs CODER={coder_letter}"


def _self_verify(ctx: ToolExecutionContext, coder_type: str, coder_result: str,
                 coder_confidence: str) -> Tuple[str, str, str]:
    """V8 Self-Verification: VL pairwise校验 (适配skill name)"""
    if not ctx.options:
        return coder_confidence, '', ''
    m_ans = re.search(r'answer=([A-D])', coder_result)
    if not m_ans:
        return coder_confidence, '', ''
    verify_result = ''
    verify_log = ''
    # 映射skill name到verify函数
    ct = coder_type.lower()
    if 'rel_distance' in ct:
        verify_result, verify_log = _vl_pairwise_verify_rel_distance(ctx, coder_result)
    elif 'direction' in ct:
        verify_result, verify_log = _vl_pairwise_verify_direction(ctx, coder_result)
    elif 'appearance' in ct:
        verify_result, verify_log = _vl_pairwise_verify_appearance(ctx, coder_result)
    elif 'route' in ct:
        verify_result, verify_log = _vl_pairwise_verify_route(ctx, coder_result)
    if not verify_result:
        return coder_confidence, '', verify_log
    if verify_result == 'agree':
        return 'verified', verify_result, verify_log
    else:
        clean_result = 'disagree' if verify_result.startswith('override:') else verify_result
        return coder_confidence, clean_result, verify_log


def _build_gather_prompt(ctx: ToolExecutionContext) -> str:
    """V8 Gather prompt: Manager从Skills列表中选择"""
    grid_text = _grid_to_text(ctx.grid)
    options_text = "\n".join(ctx.options) if ctx.options else "(numerical answer expected)"
    skills_text = "\n".join(f"  - {name}: {info['description']}" for name, info in SKILL_REGISTRY.items())

    return f"""You are a spatial intelligence agent with a 3D Grid and video access.

=== QUESTION ===
{ctx.question}

=== OPTIONS ===
{options_text}

=== 3D GRID DATA ===
{grid_text}

=== AVAILABLE SKILLS ===
{skills_text}

=== OTHER TOOLS ===
- CRITIC: Check Grid data for errors (specify entities to focus on)

=== YOUR TASK ===
Select 1-3 skills/tools. Output a JSON list:
ACTIONS: [{{"tool":"SKILL","name":"<skill_name>"}}, {{"tool":"CRITIC","focus":"<entity_names>"}}]

Output ONLY the ACTIONS line:"""


def _build_decide_prompt(ctx: ToolExecutionContext, gathered_info: str,
                         coder_confidence: str, coder_result: str,
                         verify_result: str = '') -> str:
    """统一的 Decide prompt — 无人为bias，客观呈现所有信息，让VL自主决策
    
    信心级别 (仅客观描述来源，不暗示应该信谁):
      - 'verified': 3D计算 + 独立视觉校验一致
      - 'normal': 3D计算结果
      - 'low': 3D计算可能不完整（如entity缺失）
    """

    # 通用比例参考 (通用常识)
    scale_ref = ("Reference sizes: Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. "
                 "Table ~75cm high. Bed ~200cm long. Sofa ~85cm high.")

    # 客观呈现3D计算结果，不加主观可靠性暗示
    if coder_confidence == 'verified' and coder_result:
        ref_section = f"""
=== 3D COMPUTATION RESULT (cross-verified) ===
{coder_result}
This result was independently confirmed by a separate visual check."""
    elif coder_confidence == 'low' or not coder_result:
        ref_section = """
=== 3D COMPUTATION NOTE ===
The 3D computation could not produce a complete result for this question (some entities may be missing)."""
    else:
        ref_section = f"""
=== 3D COMPUTATION RESULT ===
{coder_result}"""

    # V9: 不注入Critic原文到Decide — Critic仅用于触发Evolve修正Grid
    # Decide只看修正后的Grid计算结果，不看Critic的主观意见

    # 选项区域
    if ctx.options:
        options_section = f"""
=== OPTIONS ===
{chr(10).join(ctx.options)}"""
        answer_instruction = "Answer with ONLY the option letter (A, B, C, or D)."
    else:
        options_section = ""
        answer_instruction = "Respond with ONLY a single number (no units, no explanation)."

    return f"""You are analyzing a video of an indoor scene.

=== SCALE REFERENCES ===
{scale_ref}
Scene has {len(ctx.grid.entities)} detected objects.
{ref_section}

=== QUESTION ===
{ctx.question}
{options_section}

Watch the video carefully. Consider both the 3D computation result and what you observe in the video.
{answer_instruction}

Answer:"""


def _build_evolve_prompt(ctx: ToolExecutionContext, critic_result: str, gathered_info: str) -> str:
    """V8 Evolve prompt: 只针对high-confidence issues"""
    high_conf = ctx.high_conf_issues
    if not high_conf:
        return ""
    issues_text = "\n".join(f"  - {iss.get('entity','?')}: {iss.get('problem','?')}" for iss in high_conf)
    options_text = "\n".join(ctx.options) if ctx.options else ""
    return f"""The quality review found {len(high_conf)} HIGH-CONFIDENCE issue(s):

{issues_text}

Question: {ctx.question}
Options: {options_text}

Fix the Grid. For each issue:
- DELETE: {{"tool":"EVOLUTOR","action":"DELETE","target":"entity_id"}}
- ADD: {{"tool":"EVOLUTOR","action":"ADD","target":"entity_name"}}
- REFINE: {{"tool":"EVOLUTOR","action":"REFINE","target":"entity_name","reason":"<issue>"}}
- SCALE_ADJUST: {{"tool":"EVOLUTOR","action":"SCALE_ADJUST","target":"0.5"}} or "auto"

If issues don't affect the answer: ACTIONS: []

ACTIONS:"""


def _parse_actions(response: str) -> List[Dict]:
    """从 Manager 的响应中解析 ACTIONS JSON 列表"""
    # 找 ACTIONS: [...] 行
    m = re.search(r'ACTIONS:\s*\[(.+?)\]', response, re.DOTALL)
    if not m:
        # 尝试找任意 JSON 数组
        m = re.search(r'\[(\s*\{.+?\}\s*(?:,\s*\{.+?\}\s*)*)\]', response, re.DOTALL)
    if not m:
        return []

    try:
        arr_text = '[' + m.group(1) + ']'
        # 修复常见 JSON 问题
        arr_text = arr_text.replace("'", '"')
        actions = json.loads(arr_text)
        return actions if isinstance(actions, list) else []
    except json.JSONDecodeError:
        # 逐个解析
        actions = []
        for obj_m in re.finditer(r'\{[^}]+\}', m.group(1)):
            try:
                obj_text = obj_m.group().replace("'", '"')
                actions.append(json.loads(obj_text))
            except:
                pass
        return actions


def _execute_gather_actions(ctx: ToolExecutionContext, actions: List[Dict]) -> str:
    """V8: 执行gather actions (Skills + CRITIC + EVOLUTOR)"""
    gathered = []
    # V8 Skill↔CODER映射表
    _skill_map = {
        'direction': 'direction_vl_assist', 'distance': 'abs_distance_compute',
        'rel_distance': 'rel_distance_compute', 'count': 'counting',
        'size': 'size_estimation', 'room_size': 'room_size_estimation',
        'appearance_order': 'appearance_order', 'route': 'route_vl_assist',
    }
    for act in actions[:3]:
        tool = act.get('tool', '').upper()
        if tool == 'SKILL':
            skill_name = act.get('name', '').lower()
            if skill_name:
                result = execute_skill(ctx, skill_name)
                gathered.append(f"[SKILL {skill_name}] {result}")
        elif tool == 'CODER':
            # 兼容V7格式 → 映射到Skill
            comp_type = act.get('type', '').lower()
            mapped = _skill_map.get(comp_type, comp_type)
            if mapped:
                result = execute_skill(ctx, mapped)
                gathered.append(f"[SKILL {mapped}] {result}")
        elif tool == 'CRITIC':
            focus = act.get('focus', '')
            checks = act.get('checkpoints', '')
            result = critic_tool(ctx, focus, checks)
            gathered.append(f"[CRITIC] {result}")
        elif tool == 'EVOLUTOR':
            action_type = act.get('action', '')
            target = act.get('target', '')
            reason = act.get('reason', '')
            if action_type and target:
                result = vl_evolve_tool(ctx, action_type, target, reason)
                gathered.append(f"[EVOLUTOR {action_type}] {result}")
    return "\n".join(gathered) if gathered else "(no actions executed)"


def manager_code_agent_loop(ctx: ToolExecutionContext, max_steps: int = 4) -> Tuple[str, str]:
    """
    V8 Manager Loop — VL-Evolving + Skills + High-Confidence Critic
    
    Architecture:
      Step 1:  Gather — Manager从Skills列表选择 + 自动补充
      Step 1a: Auto-ADD — Skill返回not-found时自动VL-ADD → 重算
      Step 1b: Critic → High-Conf Filter → VL-Evolve → 重算
      Step 1c: Self-Verify — VL pairwise 二次校验
      Step 2:  Decide — VL看视频做视觉推理
    """
    reasoning_parts = []

    # ── Step 1: Gather ──
    gather_prompt = _build_gather_prompt(ctx)
    gather_response = ctx.vl.call(gather_prompt, ctx.video_path, max_tokens=256)
    ctx.vl_calls += 1

    actions = _parse_actions(gather_response)
    logger.info(f"  Manager gather: {len(actions)} actions from: {gather_response[:100]}")
    reasoning_parts.append(f"[gather] actions={json.dumps(actions, default=str)[:150]}")

    # 自动补充Skill
    has_skill = any(a.get('tool', '').upper() in ('SKILL', 'CODER') for a in actions)
    if not has_skill:
        auto_type = _auto_select_coder_type(ctx.question, ctx.options)
        if auto_type:
            actions.append({'tool': 'SKILL', 'name': auto_type})
            reasoning_parts.append(f"[auto_skill] name={auto_type}")

    # 自动补充CRITIC
    has_critic = any(a.get('tool', '').upper() == 'CRITIC' for a in actions)
    if not has_critic:
        entities = _extract_question_entities(ctx.question, ctx.options)
        if entities:
            found_names = [n for n in entities[:3] if ctx.grid.get_by_category(n)]
            if found_names:
                actions.append({'tool': 'CRITIC', 'focus': ', '.join(found_names)})
                reasoning_parts.append(f"[auto_critic] focus={', '.join(found_names)}")

    # 执行
    gathered_info = _execute_gather_actions(ctx, actions)
    reasoning_parts.append(f"[gather_result] {gathered_info[:200]}")

    # 提取skill结果和信心
    skill_result = ""
    skill_confidence = "normal"
    skill_name = ""
    _skill_map = {
        'direction': 'direction_vl_assist', 'distance': 'abs_distance_compute',
        'rel_distance': 'rel_distance_compute', 'count': 'counting',
        'size': 'size_estimation', 'room_size': 'room_size_estimation',
        'appearance_order': 'appearance_order', 'route': 'route_vl_assist',
    }
    for act in actions:
        tool = act.get('tool', '').upper()
        if tool == 'SKILL':
            skill_name = act.get('name', '')
        elif tool == 'CODER':
            skill_name = _skill_map.get(act.get('type', '').lower(), '')
    for line in gathered_info.split('\n'):
        if line.startswith('[SKILL'):
            skill_result = line.split('] ', 1)[1] if '] ' in line else line
            skill_confidence = _coder_result_confidence(skill_name, skill_result, ctx.grid)
            break
    reasoning_parts.append(f"[skill_conf] name={skill_name}, conf={skill_confidence}")

    # ── Step 1a: Auto-ADD ──
    auto_add_modified = False
    if 'not found' in skill_result.lower() or 'not_found' in skill_result.lower():
        auto_add_modified, add_log = _auto_add_missing_entities(ctx, skill_result)
        if add_log:
            gathered_info += f"\n{add_log}"
            reasoning_parts.append(f"[auto_add] modified={auto_add_modified} {add_log[:80]}")
        if auto_add_modified:
            skill_actions = [a for a in actions if a.get('tool', '').upper() in ('SKILL', 'CODER')]
            if skill_actions:
                recompute = _execute_gather_actions(ctx, skill_actions)
                gathered_info += f"\n[RECOMPUTED_AFTER_ADD]\n{recompute}"
                reasoning_parts.append(f"[recompute_add] {recompute[:100]}")
                for line in recompute.split('\n'):
                    if line.startswith('[SKILL'):
                        skill_result = line.split('] ', 1)[1] if '] ' in line else line
                        skill_confidence = _coder_result_confidence(skill_name, skill_result, ctx.grid)
                        break

    # ── Step 1b: Critic → VL-Evolve → Re-Critic 闭环 ──
    # 循环: Critic发现high-conf issues → Evolve修正 → 重新Critic检查 → 直到通过或达到上限
    MAX_EVOLVE_ITERS = 3
    for evolve_iter in range(MAX_EVOLVE_ITERS):
        high_conf_issues = ctx.high_conf_issues
        if not high_conf_issues:
            if evolve_iter == 0:
                critic_results = [g for g in gathered_info.split('\n') if g.startswith('[CRITIC]')]
                if critic_results:
                    reasoning_parts.append(f"[critic_filter] no high-conf issues, skip evolve")
            break

        critic_text = "\n".join(f"  - {iss.get('entity','?')}: {iss.get('problem','?')}" for iss in high_conf_issues)
        evolve_prompt = _build_evolve_prompt(ctx, critic_text, gathered_info)
        evolve_response = ctx.vl.call(evolve_prompt, ctx.video_path, max_tokens=256)
        ctx.vl_calls += 1
        evolve_actions = _parse_actions(evolve_response)
        if not evolve_actions:
            reasoning_parts.append(f"[evolve_iter{evolve_iter}] no actions proposed, break")
            break

        logger.info(f"  VL-evolve iter {evolve_iter}: {len(evolve_actions)} actions (high-conf)")
        # 给EVOLUTOR actions添加reason
        for ea in evolve_actions:
            if ea.get('tool', '').upper() == 'EVOLUTOR' and not ea.get('reason'):
                tgt = ea.get('target', '').lower()
                for iss in high_conf_issues:
                    if tgt in iss.get('entity', '').lower() or iss.get('entity', '').lower() in tgt:
                        ea['reason'] = iss.get('problem', '')
                        break
        evolve_result = _execute_gather_actions(ctx, evolve_actions)
        gathered_info += f"\n{evolve_result}"
        reasoning_parts.append(f"[vl_evolve_iter{evolve_iter}] n_high={len(high_conf_issues)} {evolve_result[:100]}")

        # Grid修改后重算Skill
        grid_actually_modified = any(e.get('ok') for e in ctx.tool_trace if e.get('tool') == 'vl_evolve')
        if grid_actually_modified:
            skill_actions = [a for a in actions if a.get('tool', '').upper() in ('SKILL', 'CODER')]
            if skill_actions:
                recompute = _execute_gather_actions(ctx, skill_actions)
                gathered_info += f"\n[RECOMPUTED]\n{recompute}"
                reasoning_parts.append(f"[recompute] {recompute[:100]}")
                for line in recompute.split('\n'):
                    if line.startswith('[SKILL'):
                        skill_result = line.split('] ', 1)[1] if '] ' in line else line
                        skill_confidence = _coder_result_confidence(skill_name, skill_result, ctx.grid)
                        break

            # 闭环: 重新Critic检查修正后的Grid
            if evolve_iter < MAX_EVOLVE_ITERS - 1:
                entities = _extract_question_entities(ctx.question, ctx.options)
                focus = ', '.join([n for n in entities[:3] if ctx.grid.get_by_category(n)]) if entities else ''
                re_critic_result = critic_tool(ctx, focus)
                gathered_info += f"\n[RE-CRITIC iter{evolve_iter}] {re_critic_result}"
                reasoning_parts.append(f"[re_critic_iter{evolve_iter}] {re_critic_result[:80]}")
                # ctx.high_conf_issues已被critic_tool更新，下一轮循环会检查
                if not ctx.high_conf_issues:
                    reasoning_parts.append(f"[evolve_pass] Grid passed re-critic at iter {evolve_iter}")
                    break
        else:
            # Evolve没有实际修改Grid，跳出
            break

    # ── Step 1c: Self-Verify ──
    verify_result = ''
    verify_log = ''
    if skill_result and skill_name:
        # 如果Skill本身已返回Grid+VL的交叉对比结果，直接从中提取confidence
        if '(agree)' in skill_result:
            skill_confidence = 'verified'
            verify_result = 'agree'
            verify_log = 'Skill internal: Grid+VL agree'
        elif '(disagree)' in skill_result:
            # Grid和VL不一致，保持normal让Decide自主判断
            skill_confidence = 'normal'
            verify_result = 'disagree'
            verify_log = 'Skill internal: Grid+VL disagree'
        else:
            # 没有内置VL交叉对比的Skill，用独立的pairwise verify
            skill_confidence, verify_result, verify_log = _self_verify(
                ctx, skill_name, skill_result, skill_confidence)
        if verify_log:
            reasoning_parts.append(f"[verify] {verify_result} | {verify_log[:100]}")

    # ── Step 2: Decide ──
    decide_prompt = _build_decide_prompt(ctx, gathered_info, skill_confidence, skill_result,
                                         verify_result=verify_result)

    decide_response = ctx.vl.call(decide_prompt, ctx.video_path, max_tokens=128)
    ctx.vl_calls += 1
    reasoning_parts.append(f"[decide] {decide_response[:100]}")

    # ── Step 3: 清理预测结果（无任何后处理融合） ──
    vl_answer = decide_response.strip()
    prediction = _clean_prediction(vl_answer, ctx)

    ctx._final_answer = prediction
    reasoning = " | ".join(reasoning_parts)
    return prediction, reasoning


def _try_extract_answer_from_text(text: str, ctx: ToolExecutionContext) -> Optional[str]:
    """尝试从自由文本中提取答案"""
    # 检查 final_answer 调用
    m = re.search(r'final_answer\(\s*["\']?([^"\')\s]+)', text)
    if m:
        return m.group(1).strip()

    # 检查 "Answer: X"
    m = re.search(r'(?:answer|ans)\s*(?:is|:)\s*([A-Da-d]|\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 选择题: 找第一个独立的 A/B/C/D
    if ctx.options:
        m = re.search(r'\b([A-D])\b', text[:100])
        if m:
            return m.group(1)

    return None


def _clean_prediction(raw: str, ctx: ToolExecutionContext) -> str:
    """把 raw answer 清理为标准格式"""
    raw = str(raw).strip()

    # 数值型
    if ctx.options is None or len(ctx.options) == 0:
        m = re.search(r'[\d.]+', raw)
        return m.group() if m else '0'

    # 数值型任务但有 options (VSIBench的numerical任务没有options)
    # 判断: 如果 question_type 类似 counting/size 等, 也做数值解析
    # 但我们不知道 question_type (Manager 不依赖它), 所以用 options 格式判断
    # 如果 options 里全是数字, 那是选择题格式的数值
    all_numeric = all(bool(re.match(r'^[A-D]\.\s*[\d.]+', opt)) for opt in ctx.options) if ctx.options else False

    # 选择题
    raw_clean = raw.split('\n')[0].strip()
    # 去掉 "Answer submitted:" 等前缀
    for prefix in ['Answer submitted:', 'answer:', 'Answer:']:
        if raw_clean.lower().startswith(prefix.lower()):
            raw_clean = raw_clean[len(prefix):].strip()

    m = re.search(r'^([A-Da-d])', raw_clean)
    if m:
        return m.group(1).upper()

    # 搜索独立字母
    for line in raw.split('\n'):
        line = line.strip()
        if line and line[0].upper() in 'ABCD' and (len(line) == 1 or line[1] in '.、) ,'):
            return line[0].upper()

    # 内容匹配
    raw_lower = raw.lower()
    for i, opt in enumerate(ctx.options):
        opt_content = opt.lower()
        if len(opt) >= 3 and opt[1] in '.、':
            opt_content = opt[3:].strip().lower()
        if opt_content and opt_content in raw_lower:
            return chr(65 + i)

    # 数值 fallback (可能是数值题没有 options)
    m = re.search(r'[\d.]+', raw)
    if m:
        return m.group()

    return "A"


# ============================================================================
# Pipeline 入口
# ============================================================================

class AgenticPipelineV5:
    """V8: VL-Evolving + Skills + High-Confidence Critic"""

    def __init__(self, device='cuda:0', vl_model_path=None, max_steps=4):
        self.device = device
        self.vl_model_path = vl_model_path or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        self.builder = Grid64Builder(device=device, num_frames=32)
        self.vl = VLModel(device=device)
        self.max_steps = max_steps

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
        qt = sample['question_type']
        question = sample['question']
        options = sample.get('options') or []
        gt = sample['ground_truth']

        ctx = ToolExecutionContext(
            grid=grid, vl=self.vl, video_path=video_path,
            builder=self.builder, question=question, options=options)

        t0 = time.time()
        pred, reasoning = manager_code_agent_loop(ctx, max_steps=self.max_steps)
        elapsed = time.time() - t0

        score = evaluate_sample(qt, pred, gt)

        # 从 tool_trace 提取统计
        evo_actions = []
        critic_issues = 0
        critic_high_conf = 0
        grid_modified = False
        skills_used = ctx.skills_used[:]
        for entry in ctx.tool_trace:
            t = entry.get('tool', '')
            if t == 'critic':
                critic_issues += entry.get('n_issues', 0)
                critic_high_conf += entry.get('n_high_conf', 0)
            elif t == 'vl_evolve' and entry.get('ok'):
                evo_actions.append(f"{entry['action']} {entry['target']} ({entry.get('method','')})")
                grid_modified = True
            elif t == 'skill':
                pass  # tracked via ctx.skills_used

        return {
            'scene_name': sample.get('scene_name', ''),
            'question_type': qt,
            'question': question,
            'ground_truth': gt,
            'options': options,
            'prediction': pred,
            'reasoning': reasoning[:500],
            'score': score,
            'critic_issues_count': critic_issues,
            'critic_high_conf_count': critic_high_conf,
            'critic_has_issues': critic_issues > 0,
            'grid_modified': grid_modified,
            'evolution_actions': evo_actions,
            'vl_evolve_actions': [{'action': a['action'], 'target': a['target'],
                                    'ok': a.get('ok', False), 'method': a.get('method', '')}
                                   for a in ctx.vl_evolve_actions],
            'skills_used': skills_used,
            'auto_add_triggered': '[auto_add]' in reasoning.lower(),
            'verify_triggered': '[verify]' in reasoning.lower(),
            'vl_calls': ctx.vl_calls,
            'elapsed_s': round(elapsed, 1),
            'tool_trace': [{'tool': e.get('tool'), 'action': e.get('action', ''),
                            'ok': e.get('ok', ''), 'n_issues': e.get('n_issues', '')}
                           for e in ctx.tool_trace],
            'v7_vl_score': sample.get('vl_score', 0),
            'v7_rule_score': sample.get('rule_score', 0),
        }


# ============================================================================
# Main + Summary
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
    parser = argparse.ArgumentParser(description="V8 Agentic Pipeline — VL-Evolving + Skills + High-Conf Critic")
    parser.add_argument('--n_per_type', type=int, default=10)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--task_types', type=str, default=None,
                        help='Comma-separated task types to test, e.g. "object_rel_direction_easy,route_planning"')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=4)
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

    # V8: 按task type过滤
    if args.task_types:
        allowed_types = [t.strip() for t in args.task_types.split(',')]
        test_samples = [s for s in test_samples if s['question_type'] in allowed_types]
        logger.info(f"Filtered to {len(test_samples)} samples for task types: {allowed_types}")

    by_scene = defaultdict(list)
    for s in test_samples:
        by_scene[s['scene_name']].append(s)
    scene_list = sorted(by_scene.keys())

    if args.gpu_id is not None:
        total = len(scene_list)
        chunk = (total + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * chunk
        end = min(start + chunk, total)
        my_scenes = scene_list[start:end]
        logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: {len(my_scenes)} scenes, "
                     f"{sum(len(by_scene[s]) for s in my_scenes)} samples")
    else:
        my_scenes = scene_list

    vl_model = getattr(args, 'vl_model', None) or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
    pipeline = AgenticPipelineV5(device=args.device, vl_model_path=vl_model, max_steps=args.max_steps)
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
                    'critic_has_issues': False, 'critic_issues_count': 0, 'vl_calls': 0,
                    'v7_vl_score': s.get('vl_score', 0), 'v7_rule_score': s.get('rule_score', 0),
                })
            continue

        logger.info(f"[{si+1}/{total_scenes}] {scene_name} ({len(samples)} q)")
        try:
            results = pipeline.process_scene(video_path, samples)
            for r in results:
                all_results.append(r)
                delta = r['score'] - r['v7_vl_score']
                marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
                evo = "E!" if r.get('grid_modified') else "  "
                cod = "C" if r.get('coder_used') else " "
                logger.info(
                    f"  {r['question_type'][:25]:25s} [VL:{r['vl_calls']} {evo}{cod}] "
                    f"Score={r['score']:.3f} V7={r['v7_vl_score']:.3f} {marker} "
                    f"| pred={str(r['prediction'])[:15]} gt={str(r['ground_truth'])[:12]} "
                    f"({r['elapsed_s']:.0f}s)")
        except Exception as e:
            logger.error(f"  Error: {e}")
            traceback.print_exc()
            for s in samples:
                all_results.append({
                    'scene_name': scene_name, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []),
                    'prediction': '0', 'reasoning': f'error: {str(e)[:100]}', 'score': 0.0,
                    'critic_has_issues': False, 'critic_issues_count': 0, 'vl_calls': 0,
                    'v7_vl_score': s.get('vl_score', 0), 'v7_rule_score': s.get('rule_score', 0),
                })

    pipeline.unload()

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.gpu_id is not None:
        dir_name = "agentic_pipeline_v8_partial" if args.task_types else "agentic_pipeline_v8_full"
        output_dir = PROJECT_ROOT / "outputs" / dir_name / f"gpu{args.gpu_id}"
    else:
        output_dir = PROJECT_ROOT / "outputs" / f"agentic_pipeline_v8_{timestamp}"
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

    logger.info(f"Results saved: {output_dir} ({len(all_results)} samples)")

    if args.gpu_id is None:
        _print_summary(all_results, output_dir, timestamp)


def _print_summary(all_results, output_dir, timestamp):
    print("\n" + "=" * 140)
    print("Agentic Pipeline V8 — VL-Evolving + Skills + High-Confidence Critic")
    print(f"Architecture: Manager(VL) ─ Skills → Auto-ADD(VL) → Critic(high-conf) → VL-Evolve → Self-Verify → Decide")
    print(f"New: Skills replace CODER | VL-based REFINE/ADD | Only high-conf issues trigger Evolve")
    print(f"Baselines: V7 VL=63.61%, V7p=65.15%, V6=65.48%  |  Samples: {len(all_results)}")
    print("=" * 140)

    task_types = sorted(set(r['question_type'] for r in all_results))

    print(f"\n{'Task':<35} {'N':>4} {'V7':>6} {'V8':>6} {'Δ':>6}  {'VL#':>4} {'Evo%':>5} {'Skl#':>5} {'HiC%':>5} {'Add%':>5} {'Vfy%':>5} {'t/s':>5}")
    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}  {'-'*4} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")

    all_v7, all_v8 = [], []
    for qt in task_types:
        qr = [r for r in all_results if r['question_type'] == qt]
        v7 = np.mean([r['v7_vl_score'] for r in qr])
        v8 = np.mean([r['score'] for r in qr])
        d = v8 - v7
        vl = np.mean([r.get('vl_calls', 0) for r in qr])
        evo = np.mean([1 if r.get('grid_modified') else 0 for r in qr]) * 100
        skl = np.mean([len(r.get('skills_used', [])) for r in qr])
        hic = np.mean([1 if r.get('critic_high_conf_count', 0) > 0 else 0 for r in qr]) * 100
        add = np.mean([1 if r.get('auto_add_triggered') else 0 for r in qr]) * 100
        vfy = np.mean([1 if r.get('verify_triggered') else 0 for r in qr]) * 100
        t_avg = np.mean([r.get('elapsed_s', 0) for r in qr])
        mk = "+" if d > 0.01 else ("-" if d < -0.01 else "=")
        print(f"  {qt:<35} {len(qr):>4} {v7:>5.3f} {v8:>5.3f} {d:>+5.3f}{mk} {vl:>4.1f} {evo:>4.0f}% {skl:>4.1f} {hic:>4.0f}% {add:>4.0f}% {vfy:>4.0f}% {t_avg:>4.0f}s")
        all_v7.extend([r['v7_vl_score'] for r in qr])
        all_v8.extend([r['score'] for r in qr])

    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}")
    ov7, ov8 = np.mean(all_v7), np.mean(all_v8)
    print(f"  {'Overall':<35} {len(all_results):>4} {ov7:>5.3f} {ov8:>5.3f} {ov8-ov7:>+5.3f}")

    total_vl = sum(r.get('vl_calls', 0) for r in all_results)
    avg_vl = total_vl / len(all_results) if all_results else 0
    avg_t = np.mean([r.get('elapsed_s', 0) for r in all_results])
    print(f"\n  VL calls: total={total_vl}, avg={avg_vl:.1f}/sample | Avg time: {avg_t:.0f}s/sample")

    n_add = sum(1 for r in all_results if r.get('auto_add_triggered'))
    n_vfy = sum(1 for r in all_results if r.get('verify_triggered'))
    n_evo = sum(1 for r in all_results if r.get('grid_modified'))
    n_hic = sum(1 for r in all_results if r.get('critic_high_conf_count', 0) > 0)
    print(f"  V8 features: Auto-ADD={n_add} ({100*n_add/len(all_results):.1f}%), "
          f"Verify={n_vfy} ({100*n_vfy/len(all_results):.1f}%), "
          f"VL-Evolve={n_evo} ({100*n_evo/len(all_results):.1f}%), "
          f"High-Conf-Critic={n_hic} ({100*n_hic/len(all_results):.1f}%)")

    # Skills usage
    skill_counter = Counter()
    for r in all_results:
        for s in r.get('skills_used', []):
            skill_counter[s] += 1
    print(f"  Skills usage: {dict(skill_counter)}")

    # Tool usage
    tc = Counter()
    for r in all_results:
        for e in r.get('tool_trace', []):
            tc[e.get('tool', '?')] += 1
    print(f"  Tool usage: {dict(tc)}")

    # VL-Evolve methods
    evo_methods = Counter()
    for r in all_results:
        for ea in r.get('vl_evolve_actions', []):
            if ea.get('ok'):
                evo_methods[f"{ea['action']}({ea.get('method','')})"] += 1
    if evo_methods:
        print(f"  VL-Evolve methods: {dict(evo_methods)}")

    num = [r for r in all_results if r['question_type'] in NUMERICAL_TASKS]
    spa = [r for r in all_results if r['question_type'] not in NUMERICAL_TASKS]
    if num:
        print(f"  Numerical: n={len(num)}, V8={np.mean([r['score'] for r in num]):.3f}, V7={np.mean([r['v7_vl_score'] for r in num]):.3f}")
    if spa:
        print(f"  Spatial:   n={len(spa)}, V8={np.mean([r['score'] for r in spa]):.3f}, V7={np.mean([r['v7_vl_score'] for r in spa]):.3f}")

    print(f"\nResults: {output_dir}")
    print(f"\n{'='*60}")
    print(f"  V8 Overall = {ov8:.4f}  vs  V7 VL = {ov7:.4f}  (Δ = {ov8-ov7:+.4f})")
    print(f"{'='*60}")

    summary = {
        'timestamp': timestamp,
        'version': 'v8_vl_evolving_skills_highconf_critic',
        'architecture': 'Manager(VL) → Skills → Auto-ADD(VL) → Critic(high-conf) → VL-Evolve → Verify → Decide',
        'improvements': 'V7→V8: VL-based REFINE/ADD, High-conf Critic filter, Skills system',
        'n_samples': len(all_results),
        'overall': {'v7_vl': float(ov7), 'v8': float(ov8), 'delta': float(ov8 - ov7)},
        'avg_vl_calls': float(avg_vl),
        'avg_time_s': float(avg_t),
        'v8_features': {
            'auto_add_count': n_add,
            'verify_count': n_vfy,
            'vl_evolve_count': n_evo,
            'high_conf_critic_count': n_hic,
        },
        'skills_usage': dict(skill_counter),
        'tool_usage': dict(tc),
        'by_task': {qt: {
            'n': len([r for r in all_results if r['question_type'] == qt]),
            'v7': float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type'] == qt])),
            'v8': float(np.mean([r['score'] for r in all_results if r['question_type'] == qt])),
            'avg_vl': float(np.mean([r.get('vl_calls', 0) for r in all_results if r['question_type'] == qt])),
            'evo_rate': float(np.mean([1 if r.get('grid_modified') else 0 for r in all_results if r['question_type'] == qt])),
            'skills_avg': float(np.mean([len(r.get('skills_used', [])) for r in all_results if r['question_type'] == qt])),
            'auto_add_rate': float(np.mean([1 if r.get('auto_add_triggered') else 0 for r in all_results if r['question_type'] == qt])),
            'verify_rate': float(np.mean([1 if r.get('verify_triggered') else 0 for r in all_results if r['question_type'] == qt])),
        } for qt in task_types},
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
