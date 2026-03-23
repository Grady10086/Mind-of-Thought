#!/usr/bin/env python3
"""
128³ Grid Mind Map — Agentic Pipeline V10 Combined (Scheme 1 + Scheme 2)

核心改进:
  - Scheme 1: grid_answer_route_v2 (路径可行性验证)
  - Scheme 2: 大距离自动触发 SCALE_ADJUST (>5m)

目标: Overall > 70%
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
    grid_answer_appearance_order,
)

# V10: 导入RouteSkill
from scripts.route_skill import RouteSkill

# V9: 创建Grid128别名 (继承Grid64但修改GRID_SIZE)
class Grid128(Grid64):
    GRID_SIZE = 128

class Grid128Builder(Grid64Builder):
    """构建128³ Grid的Builder"""
    pass

NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 'room_size_estimation', 'object_abs_distance']

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
# Scheme 1: 优化后的 grid_answer_route_v2
# ============================================================================

def grid_answer_route_v3(grid: Grid64, question: str, options: List[str]) -> Tuple[str, str]:
    """v10_route_skill: 使用专业化的RouteSkill"""
    skill = RouteSkill(grid)
    return skill.solve(question, options)


# 使用RouteSkill
grid_answer_route = grid_answer_route_v3


# ============================================================================
# 5 个平级 Tool 实现
# ============================================================================

class ToolExecutionContext:
    """一个样本处理过程中所有 Tool 共享的上下文 (V9: Grid128)"""
    def __init__(self, grid: Grid128, vl: VLModel, video_path: str,
                 builder: Grid128Builder, question: str, options: List[str]):
        self.grid = grid
        self.vl = vl
        self.video_path = video_path
        self.builder = builder
        self.question = question
        self.options = options
        self.tool_trace: List[Dict] = []
        self.vl_calls = 0
        self._final_answer = None


def _extract_question_entities_v9(question: str, options: List[str]) -> List[str]:
    """V9: 从问题和选项中提取可能相关的实体名"""
    entities = []
    q = question.lower()
    
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
    
    for opt in options:
        opt_clean = opt.strip()
        if len(opt_clean) >= 3 and opt_clean[1] in '.、':
            opt_content = opt_clean[3:].strip().lower()
            direction_words = {'left', 'right', 'behind', 'front', 'back', 'front-left',
                             'front-right', 'back-left', 'back-right', 'yes', 'no'}
            if opt_content and opt_content not in direction_words and not opt_content.replace('.','').isdigit():
                entities.append(opt_content)
    
    return list(set(entities))


def _grid_to_text(grid: Grid128) -> str:
    """V9: 把 Grid 转为 Manager 可读的文本"""
    lines = [f"Scene: {len(grid.entities)} objects, meters_per_grid={grid.meters_per_grid:.4f}m, "
             f"scene_span≈{grid.meters_per_grid * grid.GRID_SIZE:.1f}m (Grid128)"]
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


def _grid_to_text_focused(grid: Grid128, question: str, options: List[str]) -> str:
    """V9: 只展示与问题相关的entity"""
    relevant_entities = _extract_question_entities_v9(question, options)
    
    lines = [f"Scene: {len(grid.entities)} objects, meters_per_grid={grid.meters_per_grid:.4f}m, "
             f"scene_span≈{grid.meters_per_grid * grid.GRID_SIZE:.1f}m (Grid128)"]
    
    shown_entities = set()
    related_entities = []
    other_entities = []
    
    for eid, e in sorted(grid.entities.items()):
        is_related = any(_match_name(rel, e.category) for rel in relevant_entities)
        if is_related:
            related_entities.append((eid, e))
            shown_entities.add(eid)
        else:
            other_entities.append((eid, e))
    
    if related_entities:
        lines.append(f"\n[Relevant to question: {len(related_entities)} entities]")
        for eid, e in related_entities:
            phys = grid.grid_to_physical(e.grid_position)
            ps = grid.physical_size(eid)
            sz = f", size≈{ps:.2f}m" if ps else ""
            nf = len(set(d['frame_order'] for d in e.detections)) if e.detections else 0
            total_frames = len(grid.camera_positions) if grid.camera_positions else 32
            lines.append(
                f"  {eid}: pos=({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m{sz}, "
                f"conf={e.confidence:.2f}, seen_in={nf}/{total_frames} frames, "
                f"count_per_frame={e.count_in_frame}, first_frame={e.first_seen_frame}")
    
    if other_entities:
        lines.append(f"\n[Other entities: {len(other_entities)}]")
        other_summary = []
        for eid, e in other_entities[:10]:
            phys = grid.grid_to_physical(e.grid_position)
            other_summary.append(f"{eid}@({phys[0]:.1f},{phys[1]:.1f},{phys[2]:.1f})")
        if len(other_entities) > 10:
            other_summary.append(f"... and {len(other_entities)-10} more")
        lines.append("  " + ", ".join(other_summary))
    
    return "\n".join(lines)


# ── Tool 0.5: route_verify_tool (Scheme 3) ───────────────────────────────────

def route_verify_tool(ctx: ToolExecutionContext, option_letter: str) -> str:
    """
    V10_verify: 专门用于验证route planning选项的工具
    让VL详细分析特定路线的可行性
    
    Args:
        option_letter: 要验证的选项字母 (A/B/C/D)
    Returns:
        验证结果，包含可行性评分和理由
    """
    if not ctx.options:
        return "No options available for verification"
    
    # 找到对应选项的内容
    target_option = None
    for opt in ctx.options:
        if opt.startswith(option_letter):
            target_option = opt
            break
    
    if not target_option:
        return f"Option {option_letter} not found in options list"
    
    # 解析路线问题中的关键元素
    q = ctx.question.lower()
    
    # 提取起点、朝向、终点
    m_start = re.search(r'beginning at (?:the )?(.+?)\s+(?:and\s+)?facing (?:the )?(.+?)\.', q)
    if m_start:
        start_name = m_start.group(1).strip()
        facing_name = m_start.group(2).strip()
    else:
        start_name = "starting point"
        facing_name = "facing direction"
    
    # 提取所有waypoints
    waypoints = []
    wp_matches = re.findall(r'go forward\s+until\s+(?:the\s+)?(.+?)(?:\s+is\s+on|\s*$)', q)
    for wp in wp_matches:
        waypoints.append(wp.strip().rstrip('.'))
    
    # 构造验证prompt
    prompt = f"""You are analyzing a robot navigation task. Watch the video and verify if this route is feasible.

=== ROBOT NAVIGATION TASK ===
Starting position: At the {start_name}, facing the {facing_name}
Waypoints to visit: {', then '.join(waypoints) if waypoints else 'as specified in the question'}

=== ROUTE TO VERIFY (Option {option_letter}) ===
{target_option}

=== VERIFICATION TASK ===
Imagine yourself as the robot at the starting position. Walk through each step of this route mentally:
1. Are the turn directions (left/right) physically possible from the starting orientation?
2. After each turn, would you be facing toward the next waypoint?
3. Is the sequence of actions logically consistent?

Rate this route:
- FEASIBLE: The route would successfully navigate from start to end
- INFEASIBLE: There's a logical error (wrong turn, can't reach waypoint, etc.)

Respond in this format:
VERDICT: FEASIBLE or INFEASIBLE
CONFIDENCE: HIGH / MEDIUM / LOW
REASON: Brief explanation of why this route works or fails"""

    response = ctx.vl.call(prompt, ctx.video_path, max_tokens=200)
    ctx.vl_calls += 1
    
    # 解析响应
    verdict = "UNKNOWN"
    confidence = "LOW"
    reason = ""
    
    for line in response.split('\n'):
        line = line.strip()
        if line.upper().startswith('VERDICT:'):
            verdict = line.split(':', 1)[1].strip().upper()
        elif line.upper().startswith('CONFIDENCE:'):
            confidence = line.split(':', 1)[1].strip().upper()
        elif line.upper().startswith('REASON:'):
            reason = line.split(':', 1)[1].strip()
    
    ctx.tool_trace.append({
        'tool': 'route_verify', 
        'option': option_letter, 
        'verdict': verdict,
        'confidence': confidence
    })
    
    result = f"RouteVerify[{option_letter}]: {verdict} (confidence={confidence})"
    if reason:
        result += f" | {reason[:50]}"
    
    return result


# ── Tool 1: critic_tool ──────────────────────────────────────────────────────

def critic_tool(ctx: ToolExecutionContext, focus_entities: str = "", checkpoints: str = "") -> str:
    """让 VL 模型审查 Grid 数据与视频是否一致。"""
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

    ctx.tool_trace.append({'tool': 'critic', 'n_issues': len(issues), 'issues': issues, 'summary': summary})

    if not issues:
        return f"No issues found. {summary}"
    parts = [f"Found {len(issues)} issue(s):"]
    for iss in issues:
        parts.append(f"  - {iss.get('entity','?')}: {iss.get('problem','?')} [confidence={iss.get('confidence','?')}]")
    parts.append(f"Summary: {summary}")
    return "\n".join(parts)


# ── Tool 2: evolutor_tool ────────────────────────────────────────────────────

def evolutor_tool(ctx: ToolExecutionContext, action: str, target: str, reason: str = "") -> str:
    """V9: 修改 Grid，新增FILTER_FRAMES时间过滤动作。"""
    grid = ctx.grid
    action = action.strip().upper()

    if action == 'DELETE':
        eid = target.strip().replace(' ', '_')
        if eid not in grid.entities:
            cands = grid.get_by_category(target.strip())
            if cands:
                eid = cands[0].entity_id
            else:
                ctx.tool_trace.append({'tool': 'evolutor', 'action': 'DELETE', 'target': target, 'ok': False})
                return f"DELETE failed: '{target}' not found in grid."
        del grid.entities[eid]
        ctx.tool_trace.append({'tool': 'evolutor', 'action': 'DELETE', 'target': eid, 'ok': True})
        return f"DELETE '{eid}' done. Grid now has {len(grid.entities)} entities."

    elif action == 'ADD':
        name = target.strip().lower()
        if grid.get_by_category(name):
            ctx.tool_trace.append({'tool': 'evolutor', 'action': 'ADD', 'target': name, 'ok': False, 'reason': 'exists'})
            return f"ADD skipped: '{name}' already in grid."
        if ctx.builder is None:
            return f"ADD failed: builder unavailable."
        added = ctx.builder.search_and_add_entity(grid, name)
        if added:
            phys = grid.grid_to_physical(added.grid_position)
            ctx.tool_trace.append({'tool': 'evolutor', 'action': 'ADD', 'target': name, 'ok': True, 'eid': added.entity_id})
            return (f"ADD '{name}' as '{added.entity_id}' at ({phys[0]:.2f},{phys[1]:.2f},{phys[2]:.2f})m, "
                    f"conf={added.confidence:.2f}. Grid now has {len(grid.entities)} entities.")
        ctx.tool_trace.append({'tool': 'evolutor', 'action': 'ADD', 'target': name, 'ok': False})
        return f"ADD failed: '{name}' not found in video frames."

    elif action == 'FILTER_FRAMES':
        try:
            parts = target.split(':', 1)
            if len(parts) != 2:
                ctx.tool_trace.append({'tool': 'evolutor', 'action': 'FILTER_FRAMES', 'target': target, 'ok': False})
                return f"FILTER_FRAMES failed: format should be 'entity_name:frame_indices'"
            
            entity_name = parts[0].strip().lower()
            frame_spec = parts[1].strip()
            
            bad_frames = set()
            for seg in frame_spec.split(','):
                seg = seg.strip()
                if '-' in seg:
                    try:
                        start, end = map(int, seg.split('-', 1))
                        bad_frames.update(range(start, end + 1))
                    except:
                        pass
                else:
                    try:
                        bad_frames.add(int(seg))
                    except:
                        pass
            
            entities = grid.get_by_category(entity_name)
            if not entities:
                ctx.tool_trace.append({'tool': 'evolutor', 'action': 'FILTER_FRAMES', 'target': target, 'ok': False})
                return f"FILTER_FRAMES failed: '{entity_name}' not found in grid."
            
            entity = entities[0]
            original_det_count = len(entity.detections)
            
            filtered_dets = [d for d in entity.detections if d.get('frame_order') not in bad_frames]
            
            if len(filtered_dets) < original_det_count:
                if filtered_dets:
                    positions = np.array([d['position_3d'] for d in filtered_dets])
                    entity.position_3d = np.median(positions, axis=0)
                    entity.grid_position = grid.world_to_grid(entity.position_3d)
                    entity.confidence = np.mean([d['confidence'] for d in filtered_dets])
                    entity.detections = filtered_dets
                    
                    ctx.tool_trace.append({'tool': 'evolutor', 'action': 'FILTER_FRAMES', 
                                          'target': target, 'ok': True,
                                          'frames_removed': original_det_count - len(filtered_dets)})
                    return (f"FILTER_FRAMES '{entity_name}': removed {original_det_count - len(filtered_dets)} "
                            f"detections, new pos=({entity.position_3d[0]:.2f},{entity.position_3d[1]:.2f},{entity.position_3d[2]:.2f})")
                else:
                    del grid.entities[entity.entity_id]
                    ctx.tool_trace.append({'tool': 'evolutor', 'action': 'FILTER_FRAMES', 
                                          'target': target, 'ok': True, 'result': 'entity_deleted'})
                    return f"FILTER_FRAMES '{entity_name}': all frames filtered, entity removed."
            else:
                ctx.tool_trace.append({'tool': 'evolutor', 'action': 'FILTER_FRAMES', 
                                      'target': target, 'ok': True, 'result': 'no_change'})
                return f"FILTER_FRAMES '{entity_name}': no matching frames to filter."
        except Exception as e:
            ctx.tool_trace.append({'tool': 'evolutor', 'action': 'FILTER_FRAMES', 'target': target, 'ok': False})
            return f"FILTER_FRAMES error: {e}"

    elif action == 'SCALE_ADJUST':
        target_val = target.strip().lower()
        old_mpg = grid.meters_per_grid
        
        if target_val == 'auto':
            grid._meters_per_grid = None
            grid.calibrate_scale()
            new_mpg = grid.meters_per_grid
            ctx.tool_trace.append({'tool': 'evolutor', 'action': 'SCALE_ADJUST', 'target': 'auto', 'ok': True})
            return (f"SCALE_ADJUST auto: mpg {old_mpg:.4f} → {new_mpg:.4f}m "
                    f"(scene_span {old_mpg*64:.1f} → {new_mpg*64:.1f}m)")
        
        try:
            factor = float(target_val)
            if factor <= 0 or factor > 100:
                ctx.tool_trace.append({'tool': 'evolutor', 'action': 'SCALE_ADJUST', 'target': target, 'ok': False})
                return f"SCALE_ADJUST failed: factor {factor} out of range (0, 100]."
            new_mpg = old_mpg * factor
            grid.meters_per_grid = new_mpg
            ctx.tool_trace.append({'tool': 'evolutor', 'action': 'SCALE_ADJUST', 'target': target, 'ok': True})
            return (f"SCALE_ADJUST: mpg {old_mpg:.4f} × {factor} → {new_mpg:.4f}m "
                    f"(scene_span {old_mpg*64:.1f} → {new_mpg*64:.1f}m)")
        except ValueError:
            ctx.tool_trace.append({'tool': 'evolutor', 'action': 'SCALE_ADJUST', 'target': target, 'ok': False})
            return f"SCALE_ADJUST failed: '{target}' is not a valid number or 'auto'."

    return f"Unknown action '{action}'. Use DELETE / ADD / FILTER_FRAMES / SCALE_ADJUST."


# ── Tool 3: coder_tool ───────────────────────────────────────────────────────

def coder_tool(ctx: ToolExecutionContext, computation: str, **kwargs) -> str:
    """确定性空间计算。"""
    grid = ctx.grid
    c = computation.strip().lower()

    try:
        if c == 'direction':
            pred, reason = grid_answer_direction(grid, ctx.question, ctx.options)
            ctx.tool_trace.append({'tool': 'coder', 'comp': c, 'result': pred})
            return f"Direction computed: answer={pred}, detail={reason}"

        elif c == 'distance':
            obj1 = kwargs.get('obj1', '').strip()
            obj2 = kwargs.get('obj2', '').strip()
            if obj1 and obj2:
                e1 = grid.get_by_category(obj1)
                e2 = grid.get_by_category(obj2)
                if e1 and e2:
                    d = grid.physical_distance(e1[0].entity_id, e2[0].entity_id)
                    if d is not None:
                        ctx.tool_trace.append({'tool': 'coder', 'comp': c, 'result': f"{d:.2f}m"})
                        return f"Distance({obj1}, {obj2}) = {d:.2f}m"
                    return f"Distance computation failed for {obj1}-{obj2}."
                miss = [n for n, e in [(obj1, e1), (obj2, e2)] if not e]
                return f"Not found in grid: {miss}"
            pred, reason = grid_answer_abs_distance(grid, ctx.question)
            ctx.tool_trace.append({'tool': 'coder', 'comp': c, 'result': pred})
            return f"Distance computed: answer={pred}m, detail={reason}"

        elif c == 'rel_distance':
            pred, reason = grid_answer_rel_distance(grid, ctx.question, ctx.options)
            ctx.tool_trace.append({'tool': 'coder', 'comp': c, 'result': pred})
            return f"Relative distance computed: answer={pred}, detail={reason}"

        elif c == 'count':
            pred, reason = grid_answer_counting(grid, ctx.question)
            ctx.tool_trace.append({'tool': 'coder', 'comp': c, 'result': pred})
            return f"Count computed: answer={pred}, detail={reason}"

        elif c == 'size':
            pred, reason = grid_answer_size(grid, ctx.question)
            ctx.tool_trace.append({'tool': 'coder', 'comp': c, 'result': pred})
            return f"Size computed: answer={pred}cm, detail={reason}"

        elif c == 'room_size':
            pred, reason = grid_answer_room_size(grid, ctx.question)
            ctx.tool_trace.append({'tool': 'coder', 'comp': c, 'result': pred})
            return f"Room size computed: answer={pred} sq meters, detail={reason}"

        elif c == 'appearance_order':
            pred, reason = grid_answer_appearance_order(grid, ctx.question, ctx.options)
            ctx.tool_trace.append({'tool': 'coder', 'comp': c, 'result': pred})
            return f"Appearance order computed: answer={pred}, detail={reason}"

        elif c == 'route':
            pred, reason = grid_answer_route(grid, ctx.question, ctx.options)
            ctx.tool_trace.append({'tool': 'coder', 'comp': c, 'result': pred})
            return f"Route computed: answer={pred}, detail={reason}"

        else:
            return (f"Unknown computation '{c}'. Available: direction, distance, rel_distance, "
                    f"count, size, room_size, appearance_order, route")
    except Exception as e:
        return f"Coder error ({c}): {e}"


# ── Tool 4: grid_query_tool ──────────────────────────────────────────────────

def grid_query_tool(ctx: ToolExecutionContext, query: str, **kwargs) -> str:
    """查询 Grid 数据 (无VL调用, 纯数据查询)。"""
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
    """提交最终答案。调用后 loop 立即结束。"""
    ctx._final_answer = str(answer).strip()
    ctx.tool_trace.append({'tool': 'final_answer', 'answer': ctx._final_answer})
    return f"Answer submitted: {ctx._final_answer}"


# ============================================================================
# Manager — CodeAgent Style VL ReAct Loop
# ============================================================================

def _extract_question_entities(question: str, options: List[str]) -> List[str]:
    """从问题和选项中提取可能相关的实体名"""
    entities = []
    q = question.lower()

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

    for opt in options:
        opt_clean = opt.strip()
        if len(opt_clean) >= 3 and opt_clean[1] in '.、':
            opt_content = opt_clean[3:].strip().lower()
            direction_words = {'left', 'right', 'behind', 'front', 'back', 'front-left',
                             'front-right', 'back-left', 'back-right', 'yes', 'no'}
            if opt_content and opt_content not in direction_words and not opt_content.replace('.','').isdigit():
                entities.append(opt_content)

    return entities


def _auto_select_coder_type(question: str, options: List[str]) -> Optional[str]:
    """自动推断最合适的coder计算类型"""
    q = question.lower()

    if any(kw in q for kw in ['distance between', 'how far', 'meters apart']):
        if options:
            return 'rel_distance'
        return 'distance'

    if any(kw in q for kw in ['standing by', 'facing', 'to the left', 'to the right', 'to my']):
        return 'direction'

    if any(kw in q for kw in ['how many', 'count', 'number of']):
        return 'count'

    if any(kw in q for kw in ['how big', 'how large', 'how tall', 'how long', 'how wide',
                                'size of', 'height of', 'length of', 'width of']):
        return 'size'

    if any(kw in q for kw in ['room size', 'floor area', 'square meter', 'sq m', 'area of the room',
                                'how big is the room', 'how large is the room']):
        return 'room_size'

    if any(kw in q for kw in ['appear first', 'appears first', 'appearance order',
                                'which object first', 'seen first']):
        return 'appearance_order'

    if any(kw in q for kw in ['route', 'path', 'navigate', 'walk from', 'go from', 'turn']):
        return 'route'

    return None


def _coder_result_confidence(computation: str, result_str: str, grid: Grid128 = None) -> str:
    """
    V10_combined: 评估coder计算结果的可信度 (Scheme 1 + Scheme 2)
    
    Scheme 2: 当distance > 5m 且 mpg > 0.1 时，自动触发SCALE_ADJUST
    """
    r = result_str.lower()
    comp = computation.strip().lower()
    
    numerical_comps = {'distance', 'size', 'room_size'}
    if comp in numerical_comps:
        base_confidence = 'low'
    else:
        base_confidence = 'normal'
    
    low_signals = ['not found', 'cannot parse', 'fallback', 'same 3d position',
                   'no options', 'error', 'failed', 'n/a', 'no waypoint to infer',
                   'same position', 'insufficient data']
    for sig in low_signals:
        if sig in r:
            return 'low'

    if 'route_v2' in r:
        import re as _re
        scores = [float(m) for m in _re.findall(r'score=([-\d.]+)', r)]
        if len(scores) >= 2:
            scores_sorted = sorted(scores, reverse=True)
            margin = scores_sorted[0] - scores_sorted[1]
            if margin > 0.3 and scores_sorted[0] > 0:
                return 'normal'
        return 'low'

    clamp_signals = [
        'answer=20.00m', 'answer=0.10m',
        'answer=80', 'answer=5.0',
        'answer=200.0cm', 'answer=5.0cm',
    ]
    for sig in clamp_signals:
        if sig in r:
            return 'low'

    if grid is not None:
        mpg = grid.meters_per_grid
        has_physical_number = bool(re.search(r'answer=[\d.]+\s*(?:m|cm|sq)', r))
        if has_physical_number and mpg > 0.15:
            return 'low'
        
        # 【Scheme 2】大距离检测：当distance > 5m 且 mpg > 0.1 时触发SCALE_ADJUST
        # 修复：支持多种格式 answer=5m, answer=5.5m, Distance=5m 等
        if comp == 'distance' and mpg > 0.1:
            # 尝试多种匹配模式
            dist_match = None
            # Pattern 1: answer=5m 或 answer=5.5m
            m1 = re.search(r'answer=([\d.]+)\s*m', r)
            if m1:
                dist_match = m1
            # Pattern 2: Distance = 5.5m
            if not dist_match:
                m2 = re.search(r'[Dd]istance[=\s]+([\d.]+)\s*m', r)
                if m2:
                    dist_match = m2
            # Pattern 3: = 5.5m (any number followed by m)
            if not dist_match:
                m3 = re.search(r'=\s*([\d.]+)\s*m', r)
                if m3:
                    dist_match = m3
            
            if dist_match:
                computed_dist = float(dist_match.group(1))
                if computed_dist > 5.0:
                    return 'low_distance_large'

    return base_confidence


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
    for m in re.finditer(r"'([^']+)' not found", coder_result):
        name = m.group(1)
        if name not in missing:
            missing.append(name)
    return missing


def _auto_add_missing_entities(ctx: ToolExecutionContext, coder_result: str) -> Tuple[bool, str]:
    """CODER 返回 not found 时，自动尝试 ADD 缺失实体。"""
    missing = _extract_not_found_entities(coder_result)
    if not missing:
        return False, ""

    logs = []
    modified = False
    for name in missing[:3]:
        name_clean = name.strip().lower()
        if len(name_clean) < 2 or any(w in name_clean for w in ['the room', 'into', 'toward']):
            continue
        if ctx.grid.get_by_category(name_clean):
            continue
        if ctx.builder is None:
            continue
        result = evolutor_tool(ctx, 'ADD', name_clean, reason=f"auto-add: CODER not found '{name_clean}'")
        logs.append(f"[AUTO_ADD] {result}")
        if 'ADD failed' not in result and 'skipped' not in result:
            modified = True

    return modified, "\n".join(logs)


def _vl_pairwise_verify_rel_distance(ctx: ToolExecutionContext, coder_result: str) -> Tuple[str, str]:
    """对 rel_distance 的 CODER 排序结果，用 VL 做 pairwise verification"""
    m_detail = re.search(r'detail=ref=([^,]+),\s*(.+?)→', coder_result)
    if not m_detail:
        return '', ''
    
    ref_name = m_detail.group(1).strip()
    dist_part = m_detail.group(2).strip()
    
    candidates = []
    for m in re.finditer(r'(\w[\w\s]*?)=([\d.]+)m', dist_part):
        candidates.append((m.group(1).strip(), float(m.group(2))))
    
    if len(candidates) < 2:
        return '', ''
    
    q = ctx.question.lower()
    is_farthest = 'farthest' in q or 'furthest' in q
    candidates.sort(key=lambda x: -x[1] if is_farthest else x[1])
    
    top1_name, top1_dist = candidates[0]
    top2_name, top2_dist = candidates[1]
    
    margin = abs(top1_dist - top2_dist)
    if margin > 1.0:
        return '', ''
    
    comparison = "closer to" if not is_farthest else "farther from"
    prompt = f"""Look at the video carefully. I need you to compare two objects' distances to the {ref_name}.

Question: Which object is {comparison} the {ref_name}: the {top1_name} or the {top2_name}?

Think about where each object is relative to the {ref_name} in the scene.
Answer with ONLY the object name (either "{top1_name}" or "{top2_name}"):"""

    response = ctx.vl.call(prompt, ctx.video_path, max_tokens=50)
    ctx.vl_calls += 1
    
    response_lower = response.lower().strip()
    
    vl_choice = ''
    if top1_name.lower() in response_lower and top2_name.lower() not in response_lower:
        vl_choice = top1_name
    elif top2_name.lower() in response_lower and top1_name.lower() not in response_lower:
        vl_choice = top2_name
    elif top1_name.lower() in response_lower and top2_name.lower() in response_lower:
        idx1 = response_lower.index(top1_name.lower())
        idx2 = response_lower.index(top2_name.lower())
        vl_choice = top1_name if idx1 < idx2 else top2_name
    
    if not vl_choice:
        return '', f"VL pairwise inconclusive: '{response[:80]}'"
    
    coder_top = top1_name
    if vl_choice.lower() == coder_top.lower():
        return 'agree', f"VL pairwise confirms: {vl_choice} is {comparison} {ref_name}"
    else:
        vl_letter = ''
        for opt in ctx.options:
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt).strip().lower()
            if vl_choice.lower() in opt_content or opt_content in vl_choice.lower():
                vl_letter = opt[0]
                break
        return f'override:{vl_letter}' if vl_letter else 'disagree', \
               f"VL pairwise disagrees: VL={vl_choice} vs CODER={coder_top}"


def _vl_pairwise_verify_direction(ctx: ToolExecutionContext, coder_result: str) -> Tuple[str, str]:
    """对 direction 的 CODER 结果，用 VL 做视觉方位确认"""
    m_detail = re.search(r'obs=(\S+)\s+fac=(\S+)\s+tgt=(\S+)\s*\|\s*fwd=([-\d.]+)\s+right=([-\d.]+)\s*→\s*(\S+)', coder_result)
    if not m_detail:
        return '', ''
    
    obs_name = m_detail.group(1)
    fac_name = m_detail.group(2)
    tgt_name = m_detail.group(3)
    fwd_val = float(m_detail.group(4))
    right_val = float(m_detail.group(5))
    coder_direction = m_detail.group(6)
    
    min_component = min(abs(fwd_val), abs(right_val))
    max_component = max(abs(fwd_val), abs(right_val))
    if max_component > 0 and min_component / max_component < 0.3:
        return '', ''
    
    options_text = "\n".join(ctx.options) if ctx.options else ""
    prompt = f"""Look at the video carefully. Imagine you are standing at the {obs_name} and facing toward the {fac_name}.

From this viewpoint, which direction is the {tgt_name}?

{options_text}

Think step by step about the spatial layout, then answer with ONLY the option letter (A, B, C, or D):"""

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
        return '', f"VL direction verify inconclusive: '{response[:80]}'"
    
    m_coder_ans = re.search(r'answer=([A-D])', coder_result)
    coder_letter = m_coder_ans.group(1) if m_coder_ans else ''
    
    if vl_letter == coder_letter:
        return 'agree', f"VL direction confirms: {vl_letter} ({coder_direction})"
    else:
        return f'override:{vl_letter}', f"VL direction disagrees: VL={vl_letter} vs CODER={coder_letter}"


def _vl_pairwise_verify_appearance(ctx: ToolExecutionContext, coder_result: str) -> Tuple[str, str]:
    """对 appearance_order 的 CODER 结果，用 VL 做视觉确认"""
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
        return 'disagree', f"VL disagrees: {second_obj} appears before {first_obj}"
    
    return '', f"VL appearance verify inconclusive: '{response[:80]}'"


def _vl_pairwise_verify_route(ctx: ToolExecutionContext, coder_result: str) -> Tuple[str, str]:
    """对 route 的 CODER 结果，用 VL 做视觉路线确认"""
    m_coder_ans = re.search(r'answer=([A-D])', coder_result)
    if not m_coder_ans:
        return '', ''
    
    coder_letter = m_coder_ans.group(1)
    
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
    """Self-Verification: 对 CODER 结果做 VL pairwise 二次校验"""
    if not ctx.options:
        return coder_confidence, '', ''
    
    m_ans = re.search(r'answer=([A-D])', coder_result)
    if not m_ans:
        return coder_confidence, '', ''
    
    verify_result = ''
    verify_log = ''
    
    if coder_type == 'rel_distance':
        verify_result, verify_log = _vl_pairwise_verify_rel_distance(ctx, coder_result)
    elif coder_type == 'direction':
        verify_result, verify_log = _vl_pairwise_verify_direction(ctx, coder_result)
    elif coder_type == 'appearance_order':
        verify_result, verify_log = _vl_pairwise_verify_appearance(ctx, coder_result)
    elif coder_type == 'route':
        verify_result, verify_log = _vl_pairwise_verify_route(ctx, coder_result)
    
    if not verify_result:
        return coder_confidence, '', verify_log
    
    if verify_result == 'agree':
        return 'verified', verify_result, verify_log
    else:
        clean_result = 'disagree' if verify_result.startswith('override:') else verify_result
        return coder_confidence, clean_result, verify_log


def _build_gather_prompt(ctx: ToolExecutionContext) -> str:
    """V9: Step 1 prompt"""
    grid_text = _grid_to_text_focused(ctx.grid, ctx.question, ctx.options)
    options_text = "\n".join(ctx.options) if ctx.options else "(numerical answer expected)"

    return f"""You are a spatial intelligence agent. You have a 3D perception system (Grid) and can watch the video.

=== QUESTION ===
{ctx.question}

=== OPTIONS ===
{options_text}

=== 3D GRID DATA ===
{grid_text}

=== AVAILABLE ACTIONS ===
You can request these actions (pick 1-3 that are most useful):

1. CODER: Run spatial computation on Grid data
   Types: direction, distance, rel_distance, count, size, room_size, appearance_order, route

2. CRITIC: Have the video reviewer check if Grid data has errors
   Specify which entities to focus on

3. EVOLUTOR: Fix Grid errors (only after CRITIC finds issues)
   Actions: 
   - DELETE: {{"tool":"EVOLUTOR","action":"DELETE","target":"entity_id"}}
   - ADD: {{"tool":"EVOLUTOR","action":"ADD","target":"entity_name"}}
   - FILTER_FRAMES: {{"tool":"EVOLUTOR","action":"FILTER_FRAMES","target":"entity_name:frame_indices"}}
   - SCALE_ADJUST: {{"tool":"EVOLUTOR","action":"SCALE_ADJUST","target":"0.5"}} or "auto"

=== YOUR TASK ===
Analyze the question and the Grid data. Decide which actions would help answer this question.
Output a JSON list of actions:

ACTIONS: [{{"tool":"CODER","type":"<computation_type>"}}]

You can also add CRITIC if some entities look suspicious:
ACTIONS: [{{"tool":"CODER","type":"<type>"}}, {{"tool":"CRITIC","focus":"<entity_names>"}}]

Output ONLY the ACTIONS line:"""


def _build_decide_prompt(ctx: ToolExecutionContext, gathered_info: str,
                         coder_confidence: str, coder_result: str,
                         verify_result: str = '') -> str:
    """统一的 Decide prompt"""

    scale_ref = ("Reference sizes: Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. "
                 "Table ~75cm high. Bed ~200cm long. Sofa ~85cm high.")

    if coder_confidence == 'verified' and coder_result:
        ref_section = f"""
=== 3D PERCEPTION REFERENCE (VERIFIED) ===
{coder_result}
NOTE: This result has been cross-verified by both the 3D computation system AND an independent visual check. They agree.
Trust this result with high confidence unless you see a clear contradiction in the video."""
    elif coder_confidence == 'low' or not coder_result:
        ref_section = """
=== 3D PERCEPTION NOTE ===
The 3D system computation may be unreliable for this question. Rely primarily on your visual judgment."""
    else:
        ref_section = f"""
=== 3D PERCEPTION REFERENCE ===
{coder_result}
IMPORTANT: The 3D system has limited precision (meters_per_grid can cause 2-5x errors in distances and sizes).
Use this as a rough hint only. Your visual estimate from the video is more reliable."""

    review_section = ""
    critic_results = [g for g in gathered_info.split('\n') if g.startswith('[CRITIC]')]
    if critic_results:
        review_section = f"""
=== QUALITY REVIEW ===
{chr(10).join(critic_results)}"""

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
{review_section}

=== QUESTION ===
{ctx.question}
{options_section}

Watch the video carefully. Locate the relevant objects and reason about spatial relationships.
{answer_instruction}

Answer:"""


def _parse_actions(response: str) -> List[Dict]:
    """从 Manager 的响应中解析 ACTIONS JSON 列表"""
    m = re.search(r'ACTIONS:\s*\[(.+?)\]', response, re.DOTALL)
    if not m:
        m = re.search(r'\[(\s*\{.+?\}\s*(?:,\s*\{.+?\}\s*)*)\]', response, re.DOTALL)
    if not m:
        return []

    try:
        arr_text = '[' + m.group(1) + ']'
        arr_text = arr_text.replace("'", '"')
        actions = json.loads(arr_text)
        return actions if isinstance(actions, list) else []
    except json.JSONDecodeError:
        actions = []
        for obj_m in re.finditer(r'\{[^}]+\}', m.group(1)):
            try:
                obj_text = obj_m.group().replace("'", '"')
                actions.append(json.loads(obj_text))
            except:
                pass
        return actions


def _execute_gather_actions(ctx: ToolExecutionContext, actions: List[Dict]) -> str:
    """执行 Manager 选择的 gather actions"""
    gathered = []

    for act in actions[:3]:
        tool = act.get('tool', '').upper()

        if tool == 'CODER':
            comp_type = act.get('type', '').lower()
            obj1 = act.get('obj1', '')
            obj2 = act.get('obj2', '')
            if comp_type:
                if comp_type == 'distance' and obj1 and obj2:
                    result = coder_tool(ctx, comp_type, obj1=obj1, obj2=obj2)
                else:
                    result = coder_tool(ctx, comp_type)
                gathered.append(f"[CODER {comp_type}] {result}")

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
                result = evolutor_tool(ctx, action_type, target, reason)
                gathered.append(f"[EVOLUTOR {action_type}] {result}")

    return "\n".join(gathered) if gathered else "(no actions executed)"


def _compute_cognitive_dissonance(ctx: ToolExecutionContext, coder_result: str, 
                                   coder_confidence: str, verify_result: str) -> Tuple[str, float]:
    """V10: 计算认知失调度 D"""
    if not coder_result or 'error' in coder_result.lower():
        return 'error', 1.0
    
    if verify_result == 'agree':
        return 'verified', 0.0
    
    if coder_confidence == 'verified':
        return 'verified', 0.1
    elif coder_confidence == 'low_distance_large':
        return 'low_distance_large', 0.75
    elif coder_confidence == 'low':
        r = coder_result.lower()
        if 'not found' in r:
            return 'low', 0.8
        elif 'clamp' in r or 'fallback' in r:
            return 'low', 0.7
        else:
            return 'low', 0.6
    
    return 'normal', 0.3


def _diagnose_and_select_action(ctx: ToolExecutionContext, D_level: str, D_score: float,
                                 coder_result: str, coder_type: str) -> Optional[Dict]:
    """V10_combined: 根据认知失调度诊断问题并选择信念更新动作"""
    if D_level == 'verified':
        return None
    
    # 【Scheme 2】大距离检测 - 优先触发SCALE_ADJUST
    if D_level == 'low_distance_large':
        return {'tool': 'EVOLUTOR', 'action': 'SCALE_ADJUST', 'target': 'auto',
                'reason': 'Large distance (>5m) with high mpg, recalibrating scale'}
    
    if D_level == 'error':
        return {'tool': 'EVOLUTOR', 'action': 'SCALE_ADJUST', 'target': 'auto', 
                'reason': 'CODER execution error, recalibrating scale'}
    
    if 'not found' in coder_result.lower():
        missing = _extract_not_found_entities(coder_result)
        if missing:
            return {'tool': 'EVOLUTOR', 'action': 'ADD', 'target': missing[0],
                    'reason': f'Entity {missing[0]} not found in grid, need to add'}
    
    if D_score > 0.5 and ctx.grid.entities:
        worst_entity = min(ctx.grid.entities.values(), key=lambda e: e.confidence)
        if worst_entity.confidence < 0.5:
            return {'tool': 'EVOLUTOR', 'action': 'FILTER_FRAMES', 
                    'target': f"{worst_entity.category}:0-15",
                    'reason': f'Low confidence entity {worst_entity.category}, filtering first half frames'}
    
    if D_score > 0.6:
        return {'tool': 'EVOLUTOR', 'action': 'SCALE_ADJUST', 'target': 'auto',
                'reason': 'High cognitive dissonance, recalibrating scale'}
    
    return None


def manager_code_agent_loop(ctx: ToolExecutionContext, max_steps: int = 4) -> Tuple[str, str]:
    """V10_combined: 简化的MoT闭环架构"""
    reasoning_parts = []
    MAX_ITER = 2
    
    # ── Step 1: Gather ──
    gather_prompt = _build_gather_prompt(ctx)
    gather_response = ctx.vl.call(gather_prompt, ctx.video_path, max_tokens=256)
    ctx.vl_calls += 1
    
    actions = _parse_actions(gather_response)
    logger.info(f"  Manager gather: {len(actions)} actions")
    reasoning_parts.append(f"[gather] actions={len(actions)}")
    
    # 自动补充CODER
    has_coder = any(a.get('tool', '').upper() == 'CODER' for a in actions)
    if not has_coder:
        auto_type = _auto_select_coder_type(ctx.question, ctx.options)
        if auto_type:
            actions.append({'tool': 'CODER', 'type': auto_type})
            reasoning_parts.append(f"[auto_coder] {auto_type}")
    
    # 提取coder类型
    coder_type = ""
    for act in actions:
        if act.get('tool', '').upper() == 'CODER':
            coder_type = act.get('type', '')
            break
    
    # ── 简化MoT循环: 最多2轮 ──
    final_coder_result = ""
    final_coder_confidence = "normal"
    final_verify_result = ""
    original_coder_choice = None  # 【P0 Fix 3.0】保存CODER原始选择
    
    for iteration in range(MAX_ITER):
        logger.info(f"  MoT iter={iteration+1}/{MAX_ITER}")
        
        # Step 2: CODER计算
        coder_actions = [a for a in actions if a.get('tool', '').upper() == 'CODER']
        if coder_actions:
            coder_result_str = _execute_gather_actions(ctx, coder_actions)
        else:
            coder_result_str = "[CODER none] No computation"
        
        # 提取coder结果
        extracted_result = ""
        for line in coder_result_str.split('\n'):
            if line.startswith('[CODER'):
                extracted_result = line.split('] ', 1)[1] if '] ' in line else line
                break
        if not extracted_result and coder_result_str:
            extracted_result = coder_result_str
        
        final_coder_result = extracted_result
        
        # 【P0 Fix 3.0】保存第一次CODER的原始选择
        if iteration == 0 and not original_coder_choice:
            m = re.search(r'answer=([A-D])', final_coder_result)
            if m:
                original_coder_choice = m.group(1)
                reasoning_parts.append(f"[debug] original_coder_choice={original_coder_choice} from '{final_coder_result[:50]}'")
        
        # Step 3: Auto-ADD
        if 'not found' in final_coder_result.lower():
            auto_add_modified, add_log = _auto_add_missing_entities(ctx, final_coder_result)
            if add_log:
                reasoning_parts.append(f"[auto_add] {add_log[:80]}")
            if auto_add_modified and iteration < MAX_ITER - 1:
                continue
        
        # Step 4: Critic
        has_critic = any(a.get('tool', '').upper() == 'CRITIC' for a in actions)
        if not has_critic:
            entities = _extract_question_entities_v9(ctx.question, ctx.options)
            if entities:
                found_names = [n for n in entities[:3] if ctx.grid.get_by_category(n)]
                if found_names:
                    actions.append({'tool': 'CRITIC', 'focus': ', '.join(found_names)})
        
        critic_actions = [a for a in actions if a.get('tool', '').upper() == 'CRITIC']
        has_issues = False
        if critic_actions:
            critic_result = _execute_gather_actions(ctx, critic_actions)
            has_issues = any('issue' in c.lower() and 'no issue' not in c.lower() 
                           for c in critic_result.split('\n') if c.startswith('[CRITIC]'))
            
            if has_issues and iteration < MAX_ITER - 1:
                evolve_actions = [a for a in actions if a.get('tool', '').upper() == 'EVOLUTOR']
                if not evolve_actions:
                    action = _diagnose_and_select_action(ctx, 'low', 0.7, final_coder_result, coder_type)
                    if action:
                        result = evolutor_tool(ctx, action['action'], action['target'], action.get('reason', ''))
                        reasoning_parts.append(f"[evolve] {action['action']} {action['target']}")
                        if 'failed' not in result.lower():
                            continue
        
        # Step 5: Self-Verify
        verify_result = ''
        if final_coder_result and coder_type and ctx.options:
            _, verify_result, verify_log = _self_verify(ctx, coder_type, final_coder_result, 'normal')
            if verify_log:
                reasoning_parts.append(f"[verify] {verify_result}")
            final_verify_result = verify_result
        
        # 【Scheme 3 P0 Fix 3.0】Route保守覆盖策略 — 全选项验证+评分体系
        # P0 Fix 3.0: 使用original_coder_choice对比验证结果
        if coder_type == 'route' and ctx.options:
            reasoning_parts.append(f"[debug] route check: coder_type={coder_type}, original_coder_choice={original_coder_choice}")
        if coder_type == 'route' and ctx.options and original_coder_choice:
                # 【P0 Fix】全选项验证 + 评分体系
                all_option_scores = {}
                all_option_results = {}
                
                # 验证所有选项
                for opt in ctx.options:
                    opt_letter = opt[0]
                    verify_res = route_verify_tool(ctx, opt_letter)
                    all_option_results[opt_letter] = verify_res
                    
                    # 评分体系
                    score = 0
                    if opt_letter == original_coder_choice:
                        score += 2  # CODER原始推荐 +2
                    
                    if 'FEASIBLE' in verify_res:
                        if 'HIGH' in verify_res:
                            score += 3
                        elif 'MEDIUM' in verify_res:
                            score += 2
                        elif 'LOW' in verify_res:
                            score += 1
                    else:  # INFEASIBLE
                        score -= 2
                    
                    all_option_scores[opt_letter] = score
                
                # 找出最高分选项
                best_letter = max(all_option_scores, key=all_option_scores.get)
                best_score = all_option_scores[best_letter]
                coder_score = all_option_scores.get(original_coder_choice, 0)
                
                reasoning_parts.append(f"[route_scores] {all_option_scores}")
                
                # 【P0 Fix 3.0】策略调整：总是选route_verify最高分
                # 更新final_coder_result为route_verify最高分选项
                if best_letter != original_coder_choice:
                    reasoning_parts.append(f"[route_override] CODER={original_coder_choice}(score={coder_score}) → VL={best_letter}(score={best_score})")
                    final_coder_result = f"Route verified: answer={best_letter} (was {original_coder_choice}), scores={all_option_scores}"
                    verify_result = 'route_override'
                else:
                    # 即使CODER选择和route_verify一致，也更新result格式
                    final_coder_result = f"Route verified: answer={best_letter}, scores={all_option_scores}"
        
        # 【Scheme 2 Fix】Step 5c: 大距离检测和自动SCALE_ADJUST
        # 强制检查distance任务的scale问题 (包括 distance 和 rel_distance)
        if coder_type in ('distance', 'rel_distance') and not verify_result:
            # 提取计算出的距离值
            dist_match = re.search(r'(?:distance|answer)=([\d.]+)\s*m', final_coder_result.lower())
            if dist_match:
                computed_dist = float(dist_match.group(1))
                mpg = ctx.grid.meters_per_grid
                
                # 触发条件：距离 > 5m 且 mpg > 0.1
                if computed_dist > 5.0 and mpg > 0.1:
                    reasoning_parts.append(f"[scale_trigger] dist={computed_dist:.1f}m, mpg={mpg:.3f}")
                    
                    # 【修复】移除迭代限制，只要有迭代空间就执行
                    # 如果已经是最后一轮，也要尝试执行（虽然无法重算）
                    result = evolutor_tool(ctx, 'SCALE_ADJUST', 'auto', 
                                           f'Large distance {computed_dist:.1f}m detected, auto recalibrating')
                    reasoning_parts.append(f"[evolve] SCALE_ADJUST auto")
                    
                    if 'failed' not in result.lower():
                        ctx.tool_trace.append({'tool': 'evolutor', 'action': 'SCALE_ADJUST', 
                                               'target': 'auto', 'ok': True})
                        # 如果有迭代空间，重算CODER
                        if iteration < MAX_ITER - 1:
                            reasoning_parts.append(f"[scale_retry] iter={iteration}")
                            continue
                        else:
                            # 最后一轮，标记但无法重算
                            reasoning_parts.append(f"[scale_adjusted_no_retry] last_iter")
                    else:
                        reasoning_parts.append(f"[scale_failed]")
        
        # 计算confidence
        final_coder_confidence = _coder_result_confidence(coder_type, final_coder_result, ctx.grid)
        
        # 关键：如果Verify通过，跳出循环
        if verify_result == 'agree':
            logger.info(f"  Converged at iter={iteration+1}, verify=agree")
            reasoning_parts.append(f"[converged] iter={iteration+1} verify=agree")
            final_coder_confidence = 'verified'
            break
        
        if iteration < MAX_ITER - 1:
            reasoning_parts.append(f"[retry] iter={iteration+1} verify={verify_result}")
        else:
            reasoning_parts.append(f"[max_iter] verify={verify_result}")
    
    # ── Step 7: Decide ──
    if not final_coder_result:
        final_coder_result = "[CODER failed]"
        final_coder_confidence = 'low'
    
    gathered_info = f"[CODER {coder_type}] {final_coder_result}"
    
    decide_prompt = _build_decide_prompt(ctx, gathered_info, final_coder_confidence,
                                         final_coder_result, verify_result=final_verify_result)
    
    decide_response = ctx.vl.call(decide_prompt, ctx.video_path, max_tokens=128)
    ctx.vl_calls += 1
    reasoning_parts.append(f"[decide] {decide_response[:80]}")
    
    vl_answer = decide_response.strip()
    prediction = _clean_prediction(vl_answer, ctx)
    
    ctx._final_answer = prediction
    reasoning = " | ".join(reasoning_parts)
    return prediction, reasoning


def _try_extract_answer_from_text(text: str, ctx: ToolExecutionContext) -> Optional[str]:
    """尝试从自由文本中提取答案"""
    m = re.search(r'final_answer\(\s*["\']?([^"\')\s]+)', text)
    if m:
        return m.group(1).strip()

    m = re.search(r'(?:answer|ans)\s*(?:is|:)\s*([A-Da-d]|\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    if ctx.options:
        m = re.search(r'\b([A-D])\b', text[:100])
        if m:
            return m.group(1)

    return None


def _clean_prediction(raw: str, ctx: ToolExecutionContext) -> str:
    """把 raw answer 清理为标准格式"""
    raw = str(raw).strip()

    if ctx.options is None or len(ctx.options) == 0:
        m = re.search(r'[\d.]+', raw)
        return m.group() if m else '0'

    raw_clean = raw.split('\n')[0].strip()
    for prefix in ['Answer submitted:', 'answer:', 'Answer:']:
        if raw_clean.lower().startswith(prefix.lower()):
            raw_clean = raw_clean[len(prefix):].strip()

    m = re.search(r'^([A-Da-d])', raw_clean)
    if m:
        return m.group(1).upper()

    for line in raw.split('\n'):
        line = line.strip()
        if line and line[0].upper() in 'ABCD' and (len(line) == 1 or line[1] in '.、) ,'):
            return line[0].upper()

    raw_lower = raw.lower()
    for i, opt in enumerate(ctx.options):
        opt_content = opt.lower()
        if len(opt) >= 3 and opt[1] in '.、':
            opt_content = opt[3:].strip().lower()
        if opt_content and opt_content in raw_lower:
            return chr(65 + i)

    m = re.search(r'[\d.]+', raw)
    if m:
        return m.group()

    return "A"


# ============================================================================
# Pipeline 入口
# ============================================================================

class AgenticPipelineV10Combined:
    """V10 Combined: Scheme 1 + Scheme 2"""

    def __init__(self, device='cuda:0', vl_model_path=None, max_steps=4):
        self.device = device
        self.vl_model_path = vl_model_path or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        self.builder = Grid128Builder(device=device, num_frames=32)
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
        logger.info(f"  Grid128 built: {len(grid.entities)} entities, mpg={grid.meters_per_grid:.4f}m ({build_time:.1f}s)")

        results = []
        for sample in samples:
            grid_copy = copy.deepcopy(grid)
            result = self.process_sample(grid_copy, sample, video_path)
            results.append(result)
        return results

    def process_sample(self, grid: Grid128, sample: Dict, video_path: str) -> Dict:
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
        filter_frames_count = 0
        critic_issues = 0
        grid_modified = False
        coder_used = False
        scale_adjust_triggered = False
        for entry in ctx.tool_trace:
            t = entry.get('tool', '')
            if t == 'critic':
                critic_issues += entry.get('n_issues', 0)
            elif t == 'evolutor' and entry.get('ok'):
                evo_actions.append(f"{entry['action']} {entry['target']}")
                grid_modified = True
                if entry.get('action') == 'FILTER_FRAMES':
                    filter_frames_count += entry.get('frames_removed', 0)
                if entry.get('action') == 'SCALE_ADJUST':
                    scale_adjust_triggered = True
            elif t == 'coder':
                coder_used = True

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
            'critic_has_issues': critic_issues > 0,
            'grid_modified': grid_modified,
            'evolution_actions': evo_actions,
            'filter_frames_count': filter_frames_count,
            'scale_adjust_triggered': scale_adjust_triggered,
            'coder_used': coder_used,
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
    parser = argparse.ArgumentParser(description="V10 Combined (Scheme 1+2) — Full Test")
    parser.add_argument('--n_per_type', type=int, default=10)
    parser.add_argument('--full', action='store_true', help='Run full 5130 sample test')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=4)
    parser.add_argument('--vl-model', type=str,
                        default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    # Fix: When using CUDA_VISIBLE_DEVICES, the visible GPU is always cuda:0 in the process
    # So we should NOT use cuda:$gpu_id, just use cuda:0
    if args.gpu_id is not None:
        # Check if we're using CUDA_VISIBLE_DEVICES restriction
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible_devices:
            # When CUDA_VISIBLE_DEVICES is set, only device 0 is visible in this process
            args.device = 'cuda:0'
            logger.info(f"GPU {args.gpu_id}: CUDA_VISIBLE_DEVICES={visible_devices}, using cuda:0")
        else:
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
        chunk = (total + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * chunk
        end = min(start + chunk, total)
        my_scenes = scene_list[start:end]
        logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: {len(my_scenes)} scenes, "
                     f"{sum(len(by_scene[s]) for s in my_scenes)} samples")
    else:
        my_scenes = scene_list

    vl_model = getattr(args, 'vl_model', None) or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
    pipeline = AgenticPipelineV10Combined(device=args.device, vl_model_path=vl_model, max_steps=args.max_steps)
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
                sa = "S" if r.get('scale_adjust_triggered') else " "
                ff = f"F{r['filter_frames_count']}" if r.get('filter_frames_count', 0) > 0 else "  "
                logger.info(
                    f"  {r['question_type'][:25]:25s} [VL:{r['vl_calls']} {evo}{cod}{sa}{ff}] "
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
        output_dir = PROJECT_ROOT / "outputs" / "agentic_pipeline_v10_combined_full" / f"gpu{args.gpu_id}"
    else:
        output_dir = PROJECT_ROOT / "outputs" / f"agentic_pipeline_v10_combined_{timestamp}"
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
    print("Agentic Pipeline V10 Combined — Scheme 1 (Route Opt) + Scheme 2 (Scale Adj)")
    print(f"Architecture: MoT Loop + Route_v2 + Auto Scale Adjust")
    print(f"Baselines: V7 VL=63.61%, V5=64.09%, V10=62.80%  |  Samples: {len(all_results)}")
    print("=" * 140)

    task_types = sorted(set(r['question_type'] for r in all_results))

    print(f"\n{'Task':<35} {'N':>4} {'V7':>6} {'V10+':>6} {'Δ':>6}  {'VL#':>4} {'Evo%':>5} {'SAdj%':>5} {'Cod%':>5} {'Vfy%':>5} {'t/s':>5}")
    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}  {'-'*4} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")

    all_v7, all_v10 = [], []
    for qt in task_types:
        qr = [r for r in all_results if r['question_type'] == qt]
        v7 = np.mean([r['v7_vl_score'] for r in qr])
        v10 = np.mean([r['score'] for r in qr])
        d = v10 - v7
        vl = np.mean([r.get('vl_calls', 0) for r in qr])
        evo = np.mean([1 if r.get('grid_modified') else 0 for r in qr]) * 100
        sa = np.mean([1 if r.get('scale_adjust_triggered') else 0 for r in qr]) * 100
        cod = np.mean([1 if r.get('coder_used') else 0 for r in qr]) * 100
        vfy = np.mean([1 if r.get('verify_triggered') else 0 for r in qr]) * 100
        t_avg = np.mean([r.get('elapsed_s', 0) for r in qr])
        mk = "+" if d > 0.01 else ("-" if d < -0.01 else "=")
        print(f"  {qt:<35} {len(qr):>4} {v7:>5.3f} {v10:>5.3f} {d:>+5.3f}{mk} {vl:>4.1f} {evo:>4.0f}% {sa:>4.0f}% {cod:>4.0f}% {vfy:>4.0f}% {t_avg:>4.0f}s")
        all_v7.extend([r['v7_vl_score'] for r in qr])
        all_v10.extend([r['score'] for r in qr])

    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}")
    ov7, ov10 = np.mean(all_v7), np.mean(all_v10)
    print(f"  {'Overall':<35} {len(all_results):>4} {ov7:>5.3f} {ov10:>5.3f} {ov10-ov7:>+5.3f}")

    total_vl = sum(r.get('vl_calls', 0) for r in all_results)
    avg_vl = total_vl / len(all_results) if all_results else 0
    avg_t = np.mean([r.get('elapsed_s', 0) for r in all_results])
    print(f"\n  VL calls: total={total_vl}, avg={avg_vl:.1f}/sample | Avg time: {avg_t:.0f}s/sample")

    n_add = sum(1 for r in all_results if r.get('auto_add_triggered'))
    n_vfy = sum(1 for r in all_results if r.get('verify_triggered'))
    n_sa = sum(1 for r in all_results if r.get('scale_adjust_triggered'))
    print(f"  Features: Auto-ADD={n_add} ({100*n_add/len(all_results):.1f}%), "
          f"Verify={n_vfy} ({100*n_vfy/len(all_results):.1f}%), "
          f"ScaleAdj={n_sa} ({100*n_sa/len(all_results):.1f}%)")

    tc = Counter()
    for r in all_results:
        for e in r.get('tool_trace', []):
            tc[e.get('tool', '?')] += 1
    print(f"  Tool usage: {dict(tc)}")

    print(f"\nResults: {output_dir}")
    print(f"\n{'='*60}")
    print(f"  V10+ Overall = {ov10:.4f}  vs  V7 VL = {ov7:.4f}  (Δ = {ov10-ov7:+.4f})")
    print(f"{'='*60}")

    summary = {
        'timestamp': timestamp,
        'version': 'v10_combined_scheme1_2',
        'architecture': 'MoT Loop + Route_v2 + Auto Scale Adjust',
        'improvements': 'Scheme 1: grid_answer_route_v2 | Scheme 2: Auto SCALE_ADJUST for distance > 5m',
        'n_samples': len(all_results),
        'overall': {'v7_vl': float(ov7), 'v10_combined': float(ov10), 'delta': float(ov10 - ov7)},
        'avg_vl_calls': float(avg_vl),
        'avg_time_s': float(avg_t),
        'features': {
            'auto_add_count': n_add,
            'verify_count': n_vfy,
            'scale_adjust_count': n_sa,
        },
        'tool_usage': dict(tc),
        'by_task': {qt: {
            'n': len([r for r in all_results if r['question_type'] == qt]),
            'v7': float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type'] == qt])),
            'v10_combined': float(np.mean([r['score'] for r in all_results if r['question_type'] == qt])),
            'avg_vl': float(np.mean([r.get('vl_calls', 0) for r in all_results if r['question_type'] == qt])),
            'evo_rate': float(np.mean([1 if r.get('grid_modified') else 0 for r in all_results if r['question_type'] == qt])),
            'scale_adj_rate': float(np.mean([1 if r.get('scale_adjust_triggered') else 0 for r in all_results if r['question_type'] == qt])),
            'coder_rate': float(np.mean([1 if r.get('coder_used') else 0 for r in all_results if r['question_type'] == qt])),
        } for qt in task_types},
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
