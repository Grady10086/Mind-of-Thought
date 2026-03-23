#!/usr/bin/env python3
"""
256³ Grid Mind Map — Agentic Pipeline V14.1 (Resolution Upgrade)

核心改进 (V14→V14.1): 全局VL分辨率升级 360×420 → 480×560

  V14发现:
    - Overall=66.9%, 但Choice仅41.7% (预期50%+未达标)
    - 根因: V10_g256起为适配32帧将分辨率从480×560降到360×420
    - Choice bypass不需要Grid, 完全可以用更高分辨率
    - Numerical pipeline也可受益于更高分辨率
  
  V14.1变更:
    - VL_DEFAULT_MAX_PIXELS: 360×420 (151,200) → 480×560 (268,800)
    - VL_DEFAULT_NFRAMES: 保持32帧不变
    - 其他逻辑完全继承V14 (choice bypass + numerical pipeline)

  V14策略(继承):
    选择题 (appear/direction/rel_dist/route):
      - 完全跳过Grid/CODER pipeline
      - 使用QwenVL官方VLMEvalKit简洁prompt
      - 5-vote Self-Consistency投票 (temperature=0.7)
      - VL调用次数: 5次/样本
    
    数值题 (abs_dist/counting/obj_size/room_size):
      - 保持V13完整pipeline: Grid → CODER → MoT → Decide
      - 数值题的Grid/CODER信息是正面帮助

  预期:
    选择题: 从40% → 50%+ (对齐QwenVL benchmark水平)
    数值题: 保持87%+ (远超QwenVL的62%)
    Overall: 65%+ → 70%+

对比基准:
  V13 = 64.2% (选择40.0% + 数值87.3%)
  V12 = 66.70%
  QwenVL = 62.4% (选择53.4% + 数值62.4%)
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

# V9: 创建Grid128别名 (继承Grid64但修改GRID_SIZE)
class Grid128(Grid64):
    GRID_SIZE = 128

class Grid128Builder(Grid64Builder):
    """构建128³ Grid的Builder"""
    GRID_CLASS = Grid128

# V10 G256: Grid256
class Grid256(Grid64):
    GRID_SIZE = 256

class Grid256Builder(Grid64Builder):
    """构建256³ Grid的Builder"""
    GRID_CLASS = Grid256

NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 'room_size_estimation', 'object_abs_distance']

# V14: 选择题类型 — 使用VL-only bypass, 不过Grid/CODER pipeline
CHOICE_TASKS = [
    'obj_appearance_order',
    'object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_medi',
    'object_rel_direction_hard',
    'object_rel_distance',
    'route_planning',
]

def is_choice_task(question_type: str) -> bool:
    """V14: 判断是否为选择题类型 (应该用VL-only bypass)"""
    qt = question_type.lower().strip()
    for ct in CHOICE_TASKS:
        if ct.lower() in qt or qt in ct.lower():
            return True
    if 'appearance' in qt or 'direction' in qt or 'rel_dist' in qt or 'route' in qt:
        return True
    return False

# ============================================================================
# VL Model Wrapper
# ============================================================================

VL_DEFAULT_NFRAMES = 32
VL_DEFAULT_MAX_PIXELS = 480 * 560  # V14.1: 从360*420升级到480*560 (+78%像素)


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

    def call_sampled(self, prompt: str, video_path: str, max_tokens: int = 128,
                     n_samples: int = 3, temperature: float = 0.7, top_p: float = 0.9,
                     nframes: int = VL_DEFAULT_NFRAMES, max_pixels: int = VL_DEFAULT_MAX_PIXELS) -> List[str]:
        """V12: 采样多次用于Self-Consistency投票
        
        优化: 先尝试 num_return_sequences (一次prefill, 多次decode)
        如果失败则fallback到逐次采样
        """
        if self.model is None:
            return [""]
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

            # 尝试 num_return_sequences: 一次prefill, 多次decode
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=max_tokens,
                        do_sample=True, temperature=temperature, top_p=top_p,
                        num_return_sequences=n_samples)
                    input_len = inputs.input_ids.shape[1]
                    responses = self.processor.batch_decode(
                        outputs[:, input_len:], skip_special_tokens=True)
                    return [r.strip() for r in responses]
            except Exception:
                # fallback: 逐次采样 (兼容不支持 num_return_sequences 的情况)
                responses = []
                with torch.no_grad():
                    for _ in range(n_samples):
                        outputs = self.model.generate(
                            **inputs, max_new_tokens=max_tokens,
                            do_sample=True, temperature=temperature, top_p=top_p)
                        resp = self.processor.batch_decode(
                            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                        responses.append(resp.strip())
                return responses
        except Exception as e:
            logger.warning(f"VL call_sampled failed: {e}")
            return [""]


# ============================================================================
# 5 个平级 Tool 实现
# ============================================================================

class ToolExecutionContext:
    """一个样本处理过程中所有 Tool 共享的上下文 (V10: Grid256)"""
    def __init__(self, grid: Grid256, vl: VLModel, video_path: str,
                 builder: Grid256Builder, question: str, options: List[str]):
        self.grid = grid
        self.vl = vl
        self.video_path = video_path
        self.builder = builder
        self.question = question
        self.options = options
        self.tool_trace: List[Dict] = []
        self.vl_calls = 0
        self._final_answer = None  # 一旦设置, loop结束


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


def _grid_to_text(grid: Grid256) -> str:
    """V9: 把 Grid 转为 Manager 可读的文本 — 精简格式，显示Grid128"""
    lines = [f"Scene: {len(grid.entities)} objects, meters_per_grid={grid.meters_per_grid:.4f}m, "
             f"scene_span≈{grid.meters_per_grid * grid.GRID_SIZE:.1f}m (Grid256)"]
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


def _grid_to_text_focused(grid: Grid256, question: str, options: List[str]) -> str:
    """V9: 只展示与问题相关的entity，减少Manager注意力稀释"""
    relevant_entities = _extract_question_entities_v9(question, options)
    
    lines = [f"Scene: {len(grid.entities)} objects, meters_per_grid={grid.meters_per_grid:.4f}m, "
             f"scene_span≈{grid.meters_per_grid * grid.GRID_SIZE:.1f}m (Grid256)"]
    
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
    
    # 详细展示相关entity
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
    
    # 简略展示其他entity
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
    """
    V9: 修改 Grid，新增FILTER_FRAMES时间过滤动作。
    Args:
        action: "DELETE" / "ADD" / "FILTER_FRAMES" / "SCALE_ADJUST"
        target: entity_id(DELETE) / entity_name(ADD) / "entity_name:frame_indices"(FILTER_FRAMES) / scale_factor(SCALE_ADJUST)
    """
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
        # V9: 时间维度过滤
        try:
            parts = target.split(':', 1)
            if len(parts) != 2:
                ctx.tool_trace.append({'tool': 'evolutor', 'action': 'FILTER_FRAMES', 'target': target, 'ok': False})
                return f"FILTER_FRAMES failed: format should be 'entity_name:frame_indices'"
            
            entity_name = parts[0].strip().lower()
            frame_spec = parts[1].strip()
            
            # 解析frame_indices (支持 "3,5,7" 或 "3-7" 或混合)
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
            
            # 过滤detections
            filtered_dets = [d for d in entity.detections if d.get('frame_order') not in bad_frames]
            
            if len(filtered_dets) < original_det_count:
                if filtered_dets:
                    # 重新计算position_3d (使用剩余帧的中位数)
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
                    # 所有帧都被过滤，删除entity
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
        # target 可以是:
        #   - 一个浮点数 (scale factor, e.g. "0.5" 表示场景缩小一半)
        #   - "auto" — 根据已知标定物重新校准
        target_val = target.strip().lower()
        old_mpg = grid.meters_per_grid
        
        if target_val == 'auto':
            # 重新校准
            grid._meters_per_grid = None  # 重置校准
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

    return f"Unknown action '{action}'. Use DELETE / ADD / REFINE / SCALE_ADJUST."


# ── Tool 3: coder_tool ───────────────────────────────────────────────────────

def coder_tool(ctx: ToolExecutionContext, computation: str, **kwargs) -> str:
    """
    确定性空间计算。
    Args:
        computation: 计算类型 — "direction", "distance", "rel_distance",
                     "count", "size", "room_size", "appearance_order", "route"
        **kwargs: 计算所需参数
    Returns:
        计算结果
    """
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
    """自动推断最合适的coder计算类型 (纯文本分析, 无task-type依赖)"""
    q = question.lower()

    # 距离类
    if any(kw in q for kw in ['distance between', 'how far', 'meters apart']):
        if options:
            return 'rel_distance'
        return 'distance'

    # 方向类
    if any(kw in q for kw in ['standing by', 'facing', 'to the left', 'to the right', 'to my']):
        return 'direction'

    # 计数类
    if any(kw in q for kw in ['how many', 'count', 'number of']):
        return 'count'

    # 尺寸类
    if any(kw in q for kw in ['how big', 'how large', 'how tall', 'how long', 'how wide',
                                'size of', 'height of', 'length of', 'width of']):
        return 'size'

    # 房间面积
    if any(kw in q for kw in ['room size', 'floor area', 'square meter', 'sq m', 'area of the room',
                                'how big is the room', 'how large is the room']):
        return 'room_size'

    # 出现顺序
    if any(kw in q for kw in ['appear first', 'appears first', 'appearance order',
                                'which object first', 'seen first']):
        return 'appearance_order'

    # 路线
    if any(kw in q for kw in ['route', 'path', 'navigate', 'walk from', 'go from', 'turn']):
        return 'route'

    return None


def _coder_result_confidence(computation: str, result_str: str, grid: Grid256 = None) -> str:
    """
    V12: 评估coder计算结果的可信度
    
    核心策略:
    - 数值型任务(distance/size/room_size): 默认标low，让VL自主估算
    - rel_distance: 强制low — CODER准确率~36%(接近随机), verify无区分力
    - 空间拓扑型(direction/count/appearance): 保持normal
    - route: route_sim有margin检查
    """
    r = result_str.lower()
    comp = computation.strip().lower()
    
    # V12: 数值型任务 + rel_distance默认low信心
    # rel_distance: CODER准确率~36%(4选1接近随机), verify agree/disagree无区分力
    low_comps = {'distance', 'size', 'room_size', 'rel_distance'}
    if comp in low_comps:
        base_confidence = 'low'
    else:
        base_confidence = 'normal'
    
    # 明确的失败/不可靠信号 → low
    low_signals = ['not found', 'cannot parse', 'fallback', 'same 3d position',
                   'no options', 'error', 'failed', 'n/a', 'no waypoint to infer',
                   'same position', 'insufficient data']
    for sig in low_signals:
        if sig in r:
            return 'low'

    # route_sim: 检查score margin
    if 'route_sim' in r:
        import re as _re
        scores = [float(m) for m in _re.findall(r'score=([-\d.]+)', r)]
        if len(scores) >= 2:
            scores_sorted = sorted(scores, reverse=True)
            margin = scores_sorted[0] - scores_sorted[1]
            if margin > 0.3 and scores_sorted[0] > 0:
                return 'normal'
        return 'low'

    # 检查是否触碰了clamp边界
    clamp_signals = [
        'answer=20.00m', 'answer=0.10m',
        'answer=80', 'answer=5.0',
        'answer=200.0cm', 'answer=5.0cm',
    ]
    for sig in clamp_signals:
        if sig in r:
            return 'low'

    # Grid质量指标检查
    if grid is not None:
        mpg = grid.meters_per_grid
        has_physical_number = bool(re.search(r'answer=[\d.]+\s*(?:m|cm|sq)', r))
        if has_physical_number and mpg > 0.15:
            return 'low'

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
        result = evolutor_tool(ctx, 'ADD', name_clean, reason=f"auto-add: CODER not found '{name_clean}'")
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
    
    # margin 检查: 如果 margin 很大(>1m)，不需要 VL 验证
    margin = abs(top1_dist - top2_dist)
    if margin > 1.0:
        return '', ''
    
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
    
    # 如果 fwd/right 的绝对值都很大(清晰方向)，不需要验证
    min_component = min(abs(fwd_val), abs(right_val))
    max_component = max(abs(fwd_val), abs(right_val))
    if max_component > 0 and min_component / max_component < 0.3:
        # 方向很明确（一个分量远大于另一个），不验证
        return '', ''
    
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
    """Self-Verification: 对 CODER 结果做 VL pairwise 二次校验
    
    V12策略:
      - agree → 提升为 'verified' (仅对可靠类型: appearance_order, direction_easy相关)
      - agree → 保持normal (对不可靠类型: rel_distance, route)
        数据显示: dir_hard/dir_medi/route的verify=agree准确率比非verified更低
      - override:X → 转为disagree (VL override不可靠)
      - disagree → 保持原有confidence
    
    Returns: (updated_confidence, verification_result, verify_log)
    """
    # 只对选择题且 coder 有答案时做验证
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
        # V12: 只对可靠类型升级为verified
        # 数据显示 direction(dir_easy) 和 appearance_order 的 verify=agree 有正效果
        # 但 route(-7.8%), dir_hard(-7.9%), dir_medi(-8.2%), rel_dist(-0.1%) 的 verify=agree 无效或有害
        reliable_verify_types = {'appearance_order'}
        # direction 也可能可靠(dir_easy +22.4%), 但dir_hard/medi有害
        # 由于我们无法区分easy/hard，对direction也升级（dir_easy受益更多）
        reliable_verify_types.add('direction')
        
        if coder_type in reliable_verify_types:
            return 'verified', verify_result, verify_log
        else:
            # rel_distance, route: agree不升级，保持原confidence
            return coder_confidence, verify_result, verify_log
    else:
        clean_result = 'disagree' if verify_result.startswith('override:') else verify_result
        return coder_confidence, clean_result, verify_log


def _build_gather_prompt(ctx: ToolExecutionContext) -> str:
    """V9: Step 1 prompt — 使用精简的Grid展示"""
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
   Available computation types and when to use them:
   - "count": When the question asks "how many" objects or the number of something
   - "distance": When asked about absolute distance in meters between objects
   - "rel_distance": When comparing distances (closest/farthest object to a reference)
   - "direction": When asked about relative direction (left/right/behind) from a viewpoint
   - "size": When asked about the size/height/length of an object
   - "room_size": When asked about floor area or room size
   - "appearance_order": When asked which object appears first in the video
   - "route": When asked about navigation/turns/path between locations

2. CRITIC: Have the video reviewer check if Grid data has errors
   Specify which entities to focus on

3. EVOLUTOR: Fix Grid errors (only after CRITIC finds issues)
   Actions: 
   - DELETE (remove wrong entity): {{"tool":"EVOLUTOR","action":"DELETE","target":"entity_id"}}
   - ADD (search for missing entity): {{"tool":"EVOLUTOR","action":"ADD","target":"entity_name"}}
   - FILTER_FRAMES (remove bad frame detections): {{"tool":"EVOLUTOR","action":"FILTER_FRAMES","target":"entity_name:frame_indices"}}
     Example: "chair:3,5,7" removes detections from frames 3, 5, 7
   - SCALE_ADJUST (adjust Grid scale): {{"tool":"EVOLUTOR","action":"SCALE_ADJUST","target":"0.5"}} or "auto"

=== YOUR TASK ===
Analyze the question and the Grid data. Decide which actions would help answer this question.
Output a JSON list of actions:

ACTIONS: [{{"tool":"CODER","type":"<computation_type>"}}]

You can also add CRITIC if some entities look suspicious:
ACTIONS: [{{"tool":"CODER","type":"<type>"}}, {{"tool":"CRITIC","focus":"<entity_names>"}}]

Output ONLY the ACTIONS line:"""


def _build_decide_prompt_phase1(ctx: ToolExecutionContext) -> str:
    """V11 Phase 1: VL独立判断 — 不展示任何Grid/CODER信息
    
    让VL仅基于视频和问题给出独立判断
    """
    scale_ref = ("Reference sizes: Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. "
                 "Table ~75cm high. Bed ~200cm long. Sofa ~85cm high.")
    
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

=== QUESTION ===
{ctx.question}
{options_section}

Watch the video carefully. Locate the relevant objects and reason about spatial relationships.
{answer_instruction}

Answer:"""


def _build_decide_prompt_phase2(ctx: ToolExecutionContext, vl_independent_answer: str,
                                 coder_result: str, verify_result: str = '',
                                 coder_confidence: str = 'normal') -> str:
    """V11 Phase 2: VL看到Grid建议后做最终决策
    
    V11修正设计 — 根据CODER验证状态分级呈现:
    - verified (CODER+VL verify一致): 强信任 — "Cross-verified, trust with high confidence"
    - normal (有CODER但未验证/不一致): 中性 — "3D system suggests Option X"  
    - low (CODER不可靠): 弱参考 — "May be unreliable"
    
    对于选择题只展示选项字母，不展示具体数值/坐标
    """
    # 提取CODER建议的选项字母 (选择题)
    m_ans = re.search(r'answer=([A-D])', coder_result) if coder_result else None
    coder_letter = m_ans.group(1) if m_ans else None
    
    # 提取VL verify的override答案
    vl_override_letter = None
    if verify_result and verify_result.startswith('override:'):
        vl_override_letter = verify_result.split(':')[1].strip()
    
    # 根据confidence级别构建不同强度的参考信息
    if coder_confidence == 'verified' and coder_letter:
        # CODER + VL pairwise 验证一致 → 强信任
        ref_section = f"""
=== CROSS-VERIFIED REFERENCE ===
The 3D spatial computation system recommends: Option {coder_letter}
This result has been independently verified by both the 3D computation AND a visual consistency check. They agree.
Trust this result with high confidence unless you see a clear, obvious contradiction in the video."""
    elif coder_confidence == 'low' or not coder_letter:
        # CODER不可靠
        if coder_letter:
            ref_section = f"""
=== 3D PERCEPTION NOTE ===
The 3D system tentatively suggests Option {coder_letter}, but this computation may be unreliable.
Rely primarily on your own visual judgment from the video."""
        else:
            ref_section = """
=== 3D PERCEPTION NOTE ===
The 3D computation could not produce a reliable result. Rely on your visual judgment."""
    else:
        # normal — 中性呈现
        ref_parts = []
        if coder_letter:
            ref_parts.append(f"The 3D spatial computation system suggests: Option {coder_letter}")
        if vl_override_letter and vl_override_letter != coder_letter:
            ref_parts.append(f"However, an independent visual check suggests: Option {vl_override_letter}")
        if verify_result == 'disagree' and coder_letter:
            ref_parts.append("Note: The visual verification did not fully agree with this suggestion.")
        
        if ref_parts:
            ref_section = f"""
=== ADDITIONAL REFERENCE ===
{chr(10).join(ref_parts)}
Use this as a reference, but your visual judgment from the video is more reliable."""
        else:
            ref_section = ""
    
    if ctx.options:
        options_section = f"""
=== OPTIONS ===
{chr(10).join(ctx.options)}"""
        answer_instruction = "Answer with ONLY the option letter (A, B, C, or D)."
    else:
        options_section = ""
        answer_instruction = "Respond with ONLY a single number (no units, no explanation)."
    
    scale_ref = ("Reference sizes: Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. "
                 "Table ~75cm high. Bed ~200cm long. Sofa ~85cm high.")
    
    return f"""You are analyzing a video of an indoor scene. You have already given an initial answer, but now you have additional information to consider.

=== YOUR INITIAL JUDGMENT ===
You previously answered: {vl_independent_answer}

=== SCALE REFERENCES ===
{scale_ref}
{ref_section}

=== QUESTION ===
{ctx.question}
{options_section}

Consider all the information above carefully.
{answer_instruction}

Final Answer:"""


def _build_decide_prompt(ctx: ToolExecutionContext, gathered_info: str,
                         coder_confidence: str, coder_result: str,
                         verify_result: str = '') -> str:
    """统一的 Decide prompt — 选择题和数值题共用，VL视觉推理为主
    
    信心级别 (与V10一致):
      - 'verified': CODER + VL pairwise 一致 → 强信任 CODER
      - 'normal': 默认，VL 作为主要参考
      - 'low': 不可靠，仅视觉判断
    """
    scale_ref = ("Reference sizes: Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. "
                 "Table ~75cm high. Bed ~200cm long. Sofa ~85cm high.")

    # 根据coder信心级别决定怎么呈现Grid参考（与V10完全一致）
    if coder_confidence == 'verified' and coder_result:
        # CODER + VL pairwise 验证一致 → 高度信任
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

    # Critic 审查结果（如果有）
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


def _build_evolve_prompt(ctx: ToolExecutionContext, critic_result: str, gathered_info: str) -> str:
    """V9: Step 1b prompt — 包含FILTER_FRAMES选项"""
    options_text = "\n".join(ctx.options) if ctx.options else ""
    return f"""The quality review found these issues in the 3D Grid:

{critic_result}

Question: {ctx.question}
Options: {options_text}

Decide which actions to take to fix the Grid. Output a JSON list.
For each issue, you can:
- DELETE a wrong entity: {{"tool":"EVOLUTOR","action":"DELETE","target":"entity_id"}}
- ADD a missing entity: {{"tool":"EVOLUTOR","action":"ADD","target":"entity_name"}}
- FILTER_FRAMES (if entity exists but some frame detections are wrong): {{"tool":"EVOLUTOR","action":"FILTER_FRAMES","target":"entity_name:frame_indices"}}
  Example: "chair:3,5" filters frames 3 and 5 for chair
- SCALE_ADJUST the Grid: {{"tool":"EVOLUTOR","action":"SCALE_ADJUST","target":"0.5"}} (factor) or {{"tool":"EVOLUTOR","action":"SCALE_ADJUST","target":"auto"}}

If the issues don't affect the answer, output: ACTIONS: []

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
    """执行 Manager 选择的 gather actions，返回收集到的信息"""
    gathered = []

    for act in actions[:3]:  # 最多3个action
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
    """
    V10: 计算认知失调度 D
    
    Returns:
        D_level: 'verified' | 'normal' | 'low' | 'error'
        D_score: 0.0-1.0 (越高表示失调越严重)
    """
    # Error状态
    if not coder_result or 'error' in coder_result.lower():
        return 'error', 1.0
    
    # 如果VL pairwise验证通过
    if verify_result == 'agree':
        return 'verified', 0.0
    
    # 基于coder_confidence
    if coder_confidence == 'verified':
        return 'verified', 0.1
    elif coder_confidence == 'low':
        # 检查具体原因
        r = coder_result.lower()
        if 'not found' in r:
            return 'low', 0.8  # 实体缺失，需要ADD
        elif 'clamp' in r or 'fallback' in r:
            return 'low', 0.7  # 计算触碰边界，需要SCALE_ADJUST
        else:
            return 'low', 0.6  # 一般性不可靠
    
    # normal状态
    return 'normal', 0.3


def _diagnose_and_select_action(ctx: ToolExecutionContext, D_level: str, D_score: float,
                                 coder_result: str, coder_type: str) -> Optional[Dict]:
    """
    V10: 根据认知失调度诊断问题并选择信念更新动作
    
    Returns:
        action: {'tool': 'EVOLUTOR', 'action': 'ADD'|'FILTER_FRAMES'|'DELETE'|'SCALE_ADJUST', 
                'target': '...', 'reason': '...'} or None
    """
    if D_level == 'verified':
        return None  # 无需更新
    
    if D_level == 'error':
        # CODER执行错误，尝试SCALE_ADJUST
        return {'tool': 'EVOLUTOR', 'action': 'SCALE_ADJUST', 'target': 'auto', 
                'reason': 'CODER execution error, recalibrating scale'}
    
    # 解析缺失实体
    if 'not found' in coder_result.lower():
        missing = _extract_not_found_entities(coder_result)
        if missing:
            return {'tool': 'EVOLUTOR', 'action': 'ADD', 'target': missing[0],
                    'reason': f'Entity {missing[0]} not found in grid, need to add'}
    
    # 检查是否需要FILTER_FRAMES (基于检测质量)
    # 这里简化处理：如果confidence是low且有实体，尝试FILTER
    if D_score > 0.5 and ctx.grid.entities:
        # 找到confidence最低的entity
        worst_entity = min(ctx.grid.entities.values(), key=lambda e: e.confidence)
        if worst_entity.confidence < 0.5:
            # 假设低confidence是因为某些帧检测不好，过滤后50%的帧
            return {'tool': 'EVOLUTOR', 'action': 'FILTER_FRAMES', 
                    'target': f"{worst_entity.category}:0-15",
                    'reason': f'Low confidence entity {worst_entity.category}, filtering first half frames'}
    
    # 默认：SCALE_ADJUST
    if D_score > 0.6:
        return {'tool': 'EVOLUTOR', 'action': 'SCALE_ADJUST', 'target': 'auto',
                'reason': 'High cognitive dissonance, recalibrating scale'}
    
    return None


def _pairwise_condorcet_appear(ctx: ToolExecutionContext) -> Tuple[str, str]:
    """V13: 对appearance_order做全量pairwise比较 + Condorcet投票
    
    策略: 从选项中提取物体名, 对每对问VL "Which appears first?",
    根据pairwise胜负确定最终排序。
    
    二选一比四选一容易得多 → VL pairwise精度更高 → 整体准确率提升
    """
    if not ctx.options or len(ctx.options) < 2:
        return '', ''
    
    # 提取选项中的物体名
    option_objects = []
    for opt in ctx.options:
        m = re.match(r'^([A-D])\.\s*(.+)', opt.strip())
        if m:
            option_objects.append((m.group(1), m.group(2).strip()))
    
    if len(option_objects) < 2:
        return '', ''
    
    # 全量pairwise比较
    from itertools import combinations
    wins = Counter()  # letter → win count
    comparisons = []
    
    for (l1, name1), (l2, name2) in combinations(option_objects, 2):
        prompt = f"""Watch the video carefully from the beginning. 
Which object appears FIRST in the video: the {name1} or the {name2}?

Think about when each object first becomes visible as the camera moves through the scene.
Answer with ONLY the object name (either "{name1}" or "{name2}"):"""
        
        response = ctx.vl.call(prompt, ctx.video_path, max_tokens=50)
        ctx.vl_calls += 1
        
        resp_lower = response.lower().strip()
        n1_low = name1.lower()
        n2_low = name2.lower()
        
        winner = ''
        if n1_low in resp_lower and n2_low not in resp_lower:
            winner = l1
        elif n2_low in resp_lower and n1_low not in resp_lower:
            winner = l2
        elif n1_low in resp_lower and n2_low in resp_lower:
            idx1 = resp_lower.index(n1_low)
            idx2 = resp_lower.index(n2_low)
            winner = l1 if idx1 < idx2 else l2
        
        if winner:
            wins[winner] += 1
            comparisons.append(f"{winner}>{l1 if winner==l2 else l2}")
    
    if not wins:
        return '', 'pairwise inconclusive'
    
    # Condorcet: 最多胜的选项
    best_letter = wins.most_common(1)[0][0]
    detail = f"pairwise_wins={dict(wins)} comps={comparisons}"
    return best_letter, detail


def _pairwise_condorcet_rel_distance(ctx: ToolExecutionContext, coder_result: str) -> Tuple[str, str]:
    """V13: 对rel_distance做全量pairwise比较 + Condorcet投票
    
    策略: 提取reference物体和所有候选, 对每对候选问VL
    "Which is closer/farther to the reference?", 根据胜负确定排序。
    
    同时融合CODER的距离信息: CODER margin大(>1m)的pair直接信任CODER
    """
    if not ctx.options or len(ctx.options) < 2:
        return '', ''
    
    q = ctx.question.lower()
    is_farthest = 'farthest' in q or 'furthest' in q
    comparison_word = "farther from" if is_farthest else "closer to"
    
    # 提取reference物体
    ref_name = ''
    m_ref = re.search(r'(?:closest|nearest|farthest|furthest) to (?:the )?(.+?)[\?\.]', q)
    if m_ref:
        ref_name = m_ref.group(1).strip()
    if not ref_name:
        m_ref = re.search(r'distance .+? (?:the )?(.+?)[\?\.]', q)
        if m_ref:
            ref_name = m_ref.group(1).strip()
    
    # 也从CODER结果中提取ref
    if not ref_name:
        m_coder_ref = re.search(r'ref=([^,]+)', coder_result)
        if m_coder_ref:
            ref_name = m_coder_ref.group(1).strip()
    
    if not ref_name:
        return '', 'no ref found'
    
    # 提取选项中的物体名
    option_objects = []
    for opt in ctx.options:
        m = re.match(r'^([A-D])\.\s*(.+)', opt.strip())
        if m:
            option_objects.append((m.group(1), m.group(2).strip()))
    
    if len(option_objects) < 2:
        return '', ''
    
    # 解析CODER的距离数据 (用于large margin skip)
    coder_dists = {}
    if coder_result:
        for m_d in re.finditer(r'(\w[\w\s]*?)=([\d.]+)m', coder_result):
            coder_dists[m_d.group(1).strip().lower()] = float(m_d.group(2))
    
    # 全量pairwise比较
    from itertools import combinations
    wins = Counter()
    comparisons = []
    
    for (l1, name1), (l2, name2) in combinations(option_objects, 2):
        # 检查CODER margin: 如果距离差>1.5m, 信任CODER不做VL调用
        d1 = coder_dists.get(name1.lower(), None)
        d2 = coder_dists.get(name2.lower(), None)
        
        if d1 is not None and d2 is not None and abs(d1 - d2) > 1.5:
            # CODER大margin, 直接信任
            if is_farthest:
                winner = l1 if d1 > d2 else l2
            else:
                winner = l1 if d1 < d2 else l2
            wins[winner] += 1
            comparisons.append(f"{winner}>{l1 if winner==l2 else l2}(coder)")
            continue
        
        prompt = f"""Look at the video carefully. Compare the distances of two objects to the {ref_name}.

Which object is {comparison_word} the {ref_name}: the {name1} or the {name2}?

Think about where each object is relative to the {ref_name} in the scene.
Answer with ONLY the object name (either "{name1}" or "{name2}"):"""
        
        response = ctx.vl.call(prompt, ctx.video_path, max_tokens=50)
        ctx.vl_calls += 1
        
        resp_lower = response.lower().strip()
        n1_low = name1.lower()
        n2_low = name2.lower()
        
        winner = ''
        if n1_low in resp_lower and n2_low not in resp_lower:
            winner = l1
        elif n2_low in resp_lower and n1_low not in resp_lower:
            winner = l2
        elif n1_low in resp_lower and n2_low in resp_lower:
            idx1 = resp_lower.index(n1_low)
            idx2 = resp_lower.index(n2_low)
            winner = l1 if idx1 < idx2 else l2
        
        if winner:
            wins[winner] += 1
            comparisons.append(f"{winner}>{l1 if winner==l2 else l2}")
    
    if not wins:
        return '', 'pairwise inconclusive'
    
    best_letter = wins.most_common(1)[0][0]
    detail = f"pairwise_wins={dict(wins)} comps={comparisons}"
    return best_letter, detail


def _extract_coder_involved_entities(coder_result: str, question: str, options: List[str]) -> List[str]:
    """V13: 从CODER结果中提取涉及的question-relevant entities
    
    用于targeted FILTER_FRAMES — 只修复与问题相关的entity，不碰无关背景物
    """
    entities = []
    r = coder_result.lower()
    
    # 从CODER detail中提取entity名
    # direction: obs=X fac=Y tgt=Z
    for m in re.finditer(r'(?:obs|fac|tgt|ref|start|waypoint\d*)=(\S+)', r):
        name = m.group(1).strip("',\"")
        if name and name not in ('not', 'found', 'none'):
            entities.append(name)
    
    # rel_distance: ref=X, cand1=dist, cand2=dist
    for m in re.finditer(r'ref=([^,]+)', r):
        entities.append(m.group(1).strip())
    for m in re.finditer(r'(\w[\w\s]*?)=[\d.]+m', r):
        entities.append(m.group(1).strip())
    
    # 也从question中提取
    q_entities = _extract_question_entities_v9(question, options)
    entities.extend(q_entities)
    
    # 去重、去空
    seen = set()
    result = []
    for e in entities:
        e_clean = e.strip().lower()
        if e_clean and len(e_clean) >= 2 and e_clean not in seen:
            seen.add(e_clean)
            result.append(e_clean)
    return result[:5]  # 最多5个


def _targeted_filter_frames(ctx: ToolExecutionContext, entity_names: List[str]) -> Tuple[bool, str]:
    """V13: 对question-relevant entities做targeted FILTER_FRAMES
    
    策略: 对每个entity, 过滤掉confidence最低的那半帧detections
    这样position会基于更可靠的帧重新计算
    """
    modified = False
    logs = []
    
    for name in entity_names[:3]:  # 最多处理3个entity
        found = ctx.grid.get_by_category(name)
        if not found:
            continue
        entity = found[0]
        
        if not entity.detections or len(entity.detections) < 4:
            continue  # 太少帧不值得过滤
        
        # 按confidence排序，过滤最差的1/3帧
        dets_sorted = sorted(entity.detections, key=lambda d: d.get('confidence', 0))
        n_remove = max(1, len(dets_sorted) // 3)
        bad_frames = [d.get('frame_order', -1) for d in dets_sorted[:n_remove]]
        bad_frames = [f for f in bad_frames if f >= 0]
        
        if not bad_frames:
            continue
        
        frame_spec = ','.join(str(f) for f in bad_frames)
        result = evolutor_tool(ctx, 'FILTER_FRAMES', f"{name}:{frame_spec}",
                              reason=f"V13 targeted filter: improve {name} position")
        logs.append(f"[TARGETED_FILTER] {name}: {result[:80]}")
        if 'failed' not in result.lower() and 'no change' not in result.lower():
            modified = True
    
    return modified, "\n".join(logs)


def _build_official_mca_prompt(question: str, options: List[str]) -> str:
    """V14: QwenVL官方VLMEvalKit MCA prompt — 简洁干净, 不带任何额外信息
    
    来源: VLMEvalKit/vlmeval/dataset/vsibench.py
    Official prompt: "These are frames of a video.\n{question}\nOptions:\n{options}\n
                      Answer with the option's letter from the given choices directly."
    """
    opts_str = "\n".join(options)
    return f"""These are frames of a video.
{question}
Options:
{opts_str}
Answer with the option's letter from the given choices directly."""


def choice_task_bypass(vl: 'VLModel', video_path: str, question: str, 
                       options: List[str], n_votes: int = 5) -> Tuple[str, str, int]:
    """V14: 选择题VL-only bypass — 使用官方简洁prompt + Self-Consistency投票
    
    核心思路: Grid/CODER pipeline 对选择题有害 (V13选择题40% vs QwenVL 53.4%)
    直接用QwenVL官方prompt + 多次投票, 绕过整个pipeline
    
    Args:
        vl: VL model wrapper
        video_path: path to video
        question: question text
        options: list of options like ["A. xxx", "B. xxx", ...]
        n_votes: number of votes for self-consistency
    
    Returns:
        (prediction, reasoning, vl_calls)
    """
    prompt = _build_official_mca_prompt(question, options)
    
    # 使用 call_sampled 做多次采样 (一次prefill, 多次decode)
    responses = vl.call_sampled(prompt, video_path, max_tokens=32,
                                n_samples=n_votes, temperature=0.7, top_p=0.9)
    vl_calls = len(responses)
    
    # 提取每个response的答案字母
    votes = []
    for r in responses:
        r_clean = r.strip()
        # 尝试提取字母
        m = re.search(r'^([A-Da-d])', r_clean)
        if m:
            votes.append(m.group(1).upper())
        else:
            # 搜索独立字母
            m = re.search(r'\b([A-D])\b', r_clean[:50])
            if m:
                votes.append(m.group(1))
            else:
                # 内容匹配
                r_lower = r_clean.lower()
                for i, opt in enumerate(options):
                    opt_content = opt.lower()
                    if len(opt) >= 3 and opt[1] in '.、':
                        opt_content = opt[3:].strip().lower()
                    if opt_content and opt_content in r_lower:
                        votes.append(chr(65 + i))
                        break
    
    if not votes:
        # 全部解析失败, fallback到单次deterministic call
        fallback = vl.call(prompt, video_path, max_tokens=32)
        vl_calls += 1
        m = re.search(r'([A-Da-d])', fallback.strip())
        prediction = m.group(1).upper() if m else 'A'
        reasoning = f"[v14_bypass_fallback] {fallback.strip()[:80]}"
        return prediction, reasoning, vl_calls
    
    vote_counts = Counter(votes)
    prediction = vote_counts.most_common(1)[0][0]
    reasoning = f"[v14_bypass_{n_votes}vote] {dict(vote_counts)} → {prediction}"
    
    return prediction, reasoning, vl_calls


def manager_code_agent_loop(ctx: ToolExecutionContext, max_steps: int = 4) -> Tuple[str, str]:
    """
    V13: True Closed-Loop MoT (Mechanism of Thought)
    
    核心改进 (V12→V13): 真正的闭环信息流
      V12问题: Critic发现被丢弃 → hardcoded diagnose → FILTER无关背景物 → MoT空转
      V13修复:
        1. 去掉Critic→Evolve空转 (V12中60%的case白做)
        2. Verify disagree → 提取question-relevant entities → targeted FILTER → re-CODER
        3. VL独立判断(Phase 1): 不看Grid，纯视觉回答
        4. Decide: 根据CODER可靠度选择信任CODER还是VL-only
        5. MAX_ITER=3: 真正的闭环有意义了
    
    关键数据:
      - Verify disagree precision: rel_dist=0.600, route=0.662, dir_hard=0.604 (有效信号)
      - Verify agree precision: rel_dist=0.346, route=0.269 (不可靠，不升级confidence)
      - V12-wrong-V7-right: 424 samples → VL独立判断可以rescue
    
    Architecture:
      Phase A: VL独立判断 (no Grid, no CODER — pure visual reasoning)
      Phase B: Gather → CODER → [Auto-ADD → re-CODER] → Verify
                → if disagree: targeted FILTER → re-CODER (MAX_ITER=3)
      Phase C: Decide — 根据CODER confidence选择答案来源
                verified → trust CODER
                normal → Phase 2 (VL看Grid参考)
                low → trust VL-only (Phase A answer)
    """
    reasoning_parts = []
    MAX_ITER = 3
    
    # ══════════════════════════════════════════════════════════
    # Phase A: VL独立判断 — 不看Grid, 不看CODER, 纯视觉推理
    # 这是V13的关键: 当CODER不可靠时, VL不受Grid误导
    # ══════════════════════════════════════════════════════════
    vl_independent_prompt = _build_decide_prompt_phase1(ctx)
    vl_independent_response = ctx.vl.call(vl_independent_prompt, ctx.video_path, max_tokens=128)
    ctx.vl_calls += 1
    vl_independent_answer = _clean_prediction(vl_independent_response.strip(), ctx)
    reasoning_parts.append(f"[vl_only] {vl_independent_answer}")
    logger.info(f"  Phase A (VL-only): {vl_independent_answer}")
    
    # ══════════════════════════════════════════════════════════
    # Phase B: Gather → CODER → MoT Loop
    # ══════════════════════════════════════════════════════════
    
    # ── Step 1: Gather ──
    gather_prompt = _build_gather_prompt(ctx)
    gather_response = ctx.vl.call(gather_prompt, ctx.video_path, max_tokens=256)
    ctx.vl_calls += 1
    
    actions = _parse_actions(gather_response)
    logger.info(f"  Gather: {len(actions)} actions")
    reasoning_parts.append(f"[gather] actions={len(actions)}")
    
    # 提取coder类型 + soft fallback
    coder_type = ""
    for act in actions:
        if act.get('tool', '').upper() == 'CODER':
            coder_type = act.get('type', '')
            if coder_type == 'distance' and ctx.options:
                coder_type = 'rel_distance'
                act['type'] = 'rel_distance'
                reasoning_parts.append("[fix] distance→rel_distance")
            break
    
    # 如果Manager没选CODER, 尝试自动推断
    if not coder_type:
        auto_type = _auto_select_coder_type(ctx.question, ctx.options)
        if auto_type:
            coder_type = auto_type
            actions.insert(0, {'tool': 'CODER', 'type': auto_type})
            reasoning_parts.append(f"[auto_coder] {auto_type}")
    
    logger.info(f"  CODER type: {coder_type or 'none'}")
    
    # ── MoT循环: 最多3轮 ──
    final_coder_result = ""
    final_coder_confidence = "normal"
    final_verify_result = ""
    has_coder_result = False
    
    for iteration in range(MAX_ITER):
        logger.info(f"  MoT iter={iteration+1}/{MAX_ITER}")
        
        # Step 2: CODER计算
        coder_actions = [a for a in actions if a.get('tool', '').upper() == 'CODER']
        if coder_actions:
            coder_result_str = _execute_gather_actions(ctx, coder_actions)
            has_coder_result = True
        else:
            coder_result_str = "[CODER none] No computation"
            has_coder_result = False
        
        # 提取coder结果
        extracted_result = ""
        for line in coder_result_str.split('\n'):
            if line.startswith('[CODER'):
                extracted_result = line.split('] ', 1)[1] if '] ' in line else line
                break
        if not extracted_result and coder_result_str:
            extracted_result = coder_result_str
        
        final_coder_result = extracted_result
        
        # Step 3: Auto-ADD — 如果not found
        if 'not found' in final_coder_result.lower():
            auto_add_modified, add_log = _auto_add_missing_entities(ctx, final_coder_result)
            if add_log:
                reasoning_parts.append(f"[auto_add] {add_log[:80]}")
            if auto_add_modified and iteration < MAX_ITER - 1:
                continue  # 重算CODER
        
        # Step 4: Self-Verify (V13: 去掉Critic→Evolve空转)
        final_coder_confidence = _coder_result_confidence(coder_type, final_coder_result, ctx.grid)
        
        verify_result = ''
        if has_coder_result and final_coder_result and coder_type and ctx.options:
            updated_conf, verify_result, verify_log = _self_verify(
                ctx, coder_type, final_coder_result, final_coder_confidence)
            final_coder_confidence = updated_conf
            if verify_log:
                reasoning_parts.append(f"[verify] {verify_result} conf={final_coder_confidence}")
            final_verify_result = verify_result
        
        # V13: Verify结果驱动的闭环
        if verify_result == 'agree':
            # 只有appear类型的agree可靠, 其他类型不提前终止
            if coder_type == 'appearance_order':
                logger.info(f"  Converged at iter={iteration+1}, verify=agree (appear)")
                reasoning_parts.append(f"[converged] iter={iteration+1} verify=agree")
                break
            else:
                # 其他类型: agree不可靠, 不提前终止, 但也不retry
                reasoning_parts.append(f"[agree_untrusted] iter={iteration+1}")
                break
        
        # V13: Verify disagree → 针对性修复 question-relevant entities
        if verify_result and verify_result != 'agree' and iteration < MAX_ITER - 1:
            # 提取CODER涉及的entities
            involved_entities = _extract_coder_involved_entities(
                final_coder_result, ctx.question, ctx.options)
            
            if involved_entities:
                filter_modified, filter_log = _targeted_filter_frames(ctx, involved_entities)
                if filter_log:
                    reasoning_parts.append(f"[targeted_filter] {filter_log[:100]}")
                if filter_modified:
                    reasoning_parts.append(f"[retry] iter={iteration+1} verify={verify_result}")
                    continue  # Grid modified → re-CODER
            
            # 没有可修复的entity, 停止
            reasoning_parts.append(f"[no_fix] iter={iteration+1} verify={verify_result}")
            break
        
        if iteration >= MAX_ITER - 1:
            reasoning_parts.append(f"[max_iter] verify={verify_result}")
    
    # ══════════════════════════════════════════════════════════
    # Phase C: Decide — 类型自适应决策
    # ══════════════════════════════════════════════════════════
    #
    # V13核心策略:
    #   appearance_order → Pairwise Condorcet投票 (二选一比四选一容易)
    #   rel_distance → Pairwise Condorcet投票 (融合CODER距离+VL pairwise)
    #   route → VL-only 3-vote (CODER不可靠)
    #   direction (verified) → 信任CODER+Verify一致结果
    #   direction (normal) → V12风格单次Decide (Grid参考)
    #   数值题 → V12风格单次Decide (CODER+Grid参考)
    
    if ctx.options:
        # ── 选择题 ──
        if coder_type == 'appearance_order':
            # Appear: Pairwise Condorcet — 每对物体问VL "谁先出现"
            pw_answer, pw_detail = _pairwise_condorcet_appear(ctx)
            if pw_answer:
                # 融合: Condorcet结果 + CODER结果 + VL-only
                # Condorcet权重最高(pairwise更准), CODER次之(first_frame排序)
                coder_letter = ''
                m_ans = re.search(r'answer=([A-D])', final_coder_result)
                if m_ans:
                    coder_letter = m_ans.group(1)
                
                # 投票: pairwise给2票, CODER给1票, VL-only给1票
                votes = [pw_answer, pw_answer]  # pairwise 2票
                if coder_letter:
                    votes.append(coder_letter)  # CODER 1票
                votes.append(vl_independent_answer)  # VL-only 1票
                
                vote_counts = Counter(votes)
                final_answer_str = vote_counts.most_common(1)[0][0]
                reasoning_parts.append(f"[appear_condorcet] pw={pw_answer} coder={coder_letter} vl={vl_independent_answer} → {final_answer_str} ({pw_detail[:80]})")
            else:
                # Pairwise失败, fallback到V12风格
                gathered_info = f"[CODER {coder_type}] {final_coder_result}" if final_coder_result else ""
                decide_prompt = _build_decide_prompt(ctx, gathered_info, final_coder_confidence,
                                                     final_coder_result, verify_result=final_verify_result)
                responses = ctx.vl.call_sampled(decide_prompt, ctx.video_path,
                                                max_tokens=128, n_samples=3,
                                                temperature=0.7, top_p=0.9)
                ctx.vl_calls += len(responses)
                votes = []
                for r in responses:
                    cleaned = _clean_prediction(r.strip(), ctx)
                    if cleaned and cleaned in 'ABCD':
                        votes.append(cleaned)
                if votes:
                    vote_counts = Counter(votes)
                    final_answer_str = vote_counts.most_common(1)[0][0]
                else:
                    final_answer_str = vl_independent_answer
                reasoning_parts.append(f"[appear_fallback] {final_answer_str}")
        
        elif coder_type == 'rel_distance':
            # Rel_dist: Pairwise Condorcet — 每对候选问VL "谁更近/远"
            pw_answer, pw_detail = _pairwise_condorcet_rel_distance(ctx, final_coder_result)
            if pw_answer:
                # 融合: Condorcet(2票) + VL-only(1票)
                # 不给CODER投票权 — CODER rel_dist准确率仅~36%
                votes = [pw_answer, pw_answer, vl_independent_answer]
                vote_counts = Counter(votes)
                final_answer_str = vote_counts.most_common(1)[0][0]
                reasoning_parts.append(f"[reldist_condorcet] pw={pw_answer} vl={vl_independent_answer} → {final_answer_str} ({pw_detail[:80]})")
            else:
                # Pairwise失败, fallback到VL-only投票
                vote_prompt = vl_independent_prompt
                responses = ctx.vl.call_sampled(vote_prompt, ctx.video_path,
                                                max_tokens=128, n_samples=3,
                                                temperature=0.7, top_p=0.9)
                ctx.vl_calls += len(responses)
                votes = [vl_independent_answer]
                for r in responses:
                    cleaned = _clean_prediction(r.strip(), ctx)
                    if cleaned and cleaned in 'ABCD':
                        votes.append(cleaned)
                vote_counts = Counter(votes)
                final_answer_str = vote_counts.most_common(1)[0][0]
                reasoning_parts.append(f"[reldist_vl_vote:{len(votes)}] {dict(vote_counts)} → {final_answer_str}")
        
        elif coder_type == 'route':
            # Route: VL-only 3-vote (CODER不可靠)
            vote_prompt = vl_independent_prompt
            responses = ctx.vl.call_sampled(vote_prompt, ctx.video_path,
                                            max_tokens=128, n_samples=3,
                                            temperature=0.7, top_p=0.9)
            ctx.vl_calls += len(responses)
            votes = [vl_independent_answer]
            for r in responses:
                cleaned = _clean_prediction(r.strip(), ctx)
                if cleaned and cleaned in 'ABCD':
                    votes.append(cleaned)
            vote_counts = Counter(votes)
            final_answer_str = vote_counts.most_common(1)[0][0]
            reasoning_parts.append(f"[route_vl_vote:{len(votes)}] {dict(vote_counts)} → {final_answer_str}")
        
        elif final_coder_confidence == 'verified':
            # Direction verified → V12风格, 强信任CODER
            gathered_info = f"[CODER {coder_type}] {final_coder_result}" if final_coder_result else ""
            decide_prompt = _build_decide_prompt(ctx, gathered_info, 'verified',
                                                 final_coder_result, verify_result=final_verify_result)
            responses = ctx.vl.call_sampled(decide_prompt, ctx.video_path,
                                            max_tokens=128, n_samples=3,
                                            temperature=0.7, top_p=0.9)
            ctx.vl_calls += len(responses)
            votes = []
            for r in responses:
                cleaned = _clean_prediction(r.strip(), ctx)
                if cleaned and cleaned in 'ABCD':
                    votes.append(cleaned)
            if votes:
                vote_counts = Counter(votes)
                final_answer_str = vote_counts.most_common(1)[0][0]
                reasoning_parts.append(f"[verified_vote:{len(votes)}] {dict(vote_counts)} → {final_answer_str}")
            else:
                final_answer_str = vl_independent_answer
                reasoning_parts.append(f"[verified_fallback] {final_answer_str}")
        
        else:
            # 其他类型 normal confidence → V12风格单次Decide + 3-vote
            gathered_info = f"[CODER {coder_type}] {final_coder_result}" if final_coder_result else ""
            decide_prompt = _build_decide_prompt(ctx, gathered_info, final_coder_confidence,
                                                 final_coder_result, verify_result=final_verify_result)
            responses = ctx.vl.call_sampled(decide_prompt, ctx.video_path,
                                            max_tokens=128, n_samples=3,
                                            temperature=0.7, top_p=0.9)
            ctx.vl_calls += len(responses)
            votes = []
            for r in responses:
                cleaned = _clean_prediction(r.strip(), ctx)
                if cleaned and cleaned in 'ABCD':
                    votes.append(cleaned)
            if votes:
                vote_counts = Counter(votes)
                final_answer_str = vote_counts.most_common(1)[0][0]
                reasoning_parts.append(f"[decide_vote:{len(votes)}] {dict(vote_counts)} → {final_answer_str}")
            else:
                decide_response = ctx.vl.call(decide_prompt, ctx.video_path, max_tokens=128)
                ctx.vl_calls += 1
                final_answer_str = decide_response.strip()
                reasoning_parts.append(f"[decide:{final_coder_confidence}] {final_answer_str[:80]}")
    else:
        # 数值题: V12风格 (Grid参考很重要)
        gathered_info = f"[CODER {coder_type}] {final_coder_result}" if final_coder_result else ""
        decide_prompt = _build_decide_prompt(ctx, gathered_info, final_coder_confidence,
                                             final_coder_result, verify_result=final_verify_result)
        decide_response = ctx.vl.call(decide_prompt, ctx.video_path, max_tokens=128)
        ctx.vl_calls += 1
        final_answer_str = decide_response.strip()
        reasoning_parts.append(f"[decide_num:{final_coder_confidence}] {final_answer_str[:80]}")
    
    prediction = _clean_prediction(final_answer_str, ctx)
    
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

class AgenticPipelineV14:
    """V14: Type-Selective Bypass — 选择题VL-only + 数值题Full Pipeline"""

    def __init__(self, device='cuda:0', vl_model_path=None, max_steps=4):
        self.device = device
        self.vl_model_path = vl_model_path or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        self.builder = Grid256Builder(device=device, num_frames=32)
        self.vl = VLModel(device=device)
        self.max_steps = max_steps

    def load_models(self):
        self.builder.load_models()
        self.vl.load(self.vl_model_path)

    def unload(self):
        self.builder.unload()
        self.vl.unload()

    def process_scene(self, video_path: str, samples: List[Dict]) -> List[Dict]:
        # V14: 分离选择题和数值题
        choice_samples = [s for s in samples if is_choice_task(s['question_type'])]
        numerical_samples = [s for s in samples if not is_choice_task(s['question_type'])]
        
        results = []
        
        # V14: 选择题 — VL-only bypass, 不需要Grid
        for sample in choice_samples:
            result = self._process_choice_sample(sample, video_path)
            results.append(result)
        
        # 数值题 — 需要Grid/CODER pipeline
        if numerical_samples:
            t0 = time.time()
            grid = self.builder.build_grid(video_path)
            build_time = time.time() - t0
            logger.info(f"  Grid256 built: {len(grid.entities)} entities, mpg={grid.meters_per_grid:.4f}m ({build_time:.1f}s)")
            
            for sample in numerical_samples:
                grid_copy = copy.deepcopy(grid)
                result = self._process_numerical_sample(grid_copy, sample, video_path)
                results.append(result)
        
        return results

    def _process_choice_sample(self, sample: Dict, video_path: str) -> Dict:
        """V14: 选择题 — VL-only bypass, 使用官方简洁prompt + 5-vote"""
        qt = sample['question_type']
        question = sample['question']
        options = sample.get('options') or []
        gt = sample['ground_truth']

        t0 = time.time()
        pred, reasoning, vl_calls = choice_task_bypass(
            self.vl, video_path, question, options, n_votes=5)
        elapsed = time.time() - t0

        score = evaluate_sample(qt, pred, gt)

        return {
            'scene_name': sample.get('scene_name', ''),
            'question_type': qt,
            'question': question,
            'ground_truth': gt,
            'options': options,
            'prediction': pred,
            'reasoning': reasoning[:500],
            'score': score,
            'critic_issues_count': 0,
            'critic_has_issues': False,
            'grid_modified': False,
            'evolution_actions': [],
            'filter_frames_count': 0,
            'coder_used': False,
            'auto_add_triggered': False,
            'verify_triggered': False,
            'vl_calls': vl_calls,
            'elapsed_s': round(elapsed, 1),
            'tool_trace': [],
            'v7_vl_score': sample.get('vl_score', 0),
            'v7_rule_score': sample.get('rule_score', 0),
            'v14_bypass': True,
        }

    def _process_numerical_sample(self, grid: Grid256, sample: Dict, video_path: str) -> Dict:
        """数值题 — 保持V13完整pipeline"""
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

        evo_actions = []
        filter_frames_count = 0
        critic_issues = 0
        grid_modified = False
        coder_used = False
        for entry in ctx.tool_trace:
            t = entry.get('tool', '')
            if t == 'critic':
                critic_issues += entry.get('n_issues', 0)
            elif t == 'evolutor' and entry.get('ok'):
                evo_actions.append(f"{entry['action']} {entry['target']}")
                grid_modified = True
                if entry.get('action') == 'FILTER_FRAMES':
                    filter_frames_count += entry.get('frames_removed', 0)
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
            'v14_bypass': False,
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
    parser = argparse.ArgumentParser(description="V14 Agentic Pipeline — Type-Selective Bypass")
    parser.add_argument('--n_per_type', type=int, default=10)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=4)
    parser.add_argument('--vl-model', type=str,
                        default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    if args.gpu_id is not None:
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible_devices:
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
    pipeline = AgenticPipelineV14(device=args.device, vl_model_path=vl_model, max_steps=args.max_steps)
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
                bp = "B" if r.get('v14_bypass') else " "
                ff = f"F{r['filter_frames_count']}" if r.get('filter_frames_count', 0) > 0 else "  "
                logger.info(
                    f"  {r['question_type'][:25]:25s} [VL:{r['vl_calls']} {bp}{evo}{cod}{ff}] "
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
        output_dir = PROJECT_ROOT / "outputs" / "agentic_pipeline_v14_1_full" / f"gpu{args.gpu_id}"
    else:
        output_dir = PROJECT_ROOT / "outputs" / f"agentic_pipeline_v14_{timestamp}"
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
    print("Agentic Pipeline V14 — Type-Selective Bypass")
    print(f"Architecture: Choice→VL-only(official prompt + 5-vote) | Numerical→Grid/CODER/MoT pipeline")
    print(f"V14 Strategy: Bypass Grid/CODER for choice tasks, keep full pipeline for numerical tasks")
    print(f"Baselines: V13=64.2%, V12=66.70%, QwenVL=62.4%  |  Samples: {len(all_results)}")
    print("=" * 140)

    task_types = sorted(set(r['question_type'] for r in all_results))

    print(f"\n{'Task':<35} {'N':>4} {'V7':>6} {'V14':>6} {'Δ':>6}  {'VL#':>4} {'Byp%':>5} {'Cod%':>5} {'t/s':>5}")
    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}  {'-'*4} {'-'*5} {'-'*5} {'-'*5}")

    all_v7, all_v14 = [], []
    for qt in task_types:
        qr = [r for r in all_results if r['question_type'] == qt]
        v7 = np.mean([r['v7_vl_score'] for r in qr])
        v14 = np.mean([r['score'] for r in qr])
        d = v14 - v7
        vl = np.mean([r.get('vl_calls', 0) for r in qr])
        byp = np.mean([1 if r.get('v14_bypass') else 0 for r in qr]) * 100
        cod = np.mean([1 if r.get('coder_used') else 0 for r in qr]) * 100
        t_avg = np.mean([r.get('elapsed_s', 0) for r in qr])
        mk = "+" if d > 0.01 else ("-" if d < -0.01 else "=")
        print(f"  {qt:<35} {len(qr):>4} {v7:>5.3f} {v14:>5.3f} {d:>+5.3f}{mk} {vl:>4.1f} {byp:>4.0f}% {cod:>4.0f}% {t_avg:>4.0f}s")
        all_v7.extend([r['v7_vl_score'] for r in qr])
        all_v14.extend([r['score'] for r in qr])

    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}")
    ov7, ov14 = np.mean(all_v7), np.mean(all_v14)
    print(f"  {'Overall':<35} {len(all_results):>4} {ov7:>5.3f} {ov14:>5.3f} {ov14-ov7:>+5.3f}")

    total_vl = sum(r.get('vl_calls', 0) for r in all_results)
    avg_vl = total_vl / len(all_results) if all_results else 0
    avg_t = np.mean([r.get('elapsed_s', 0) for r in all_results])
    print(f"\n  VL calls: total={total_vl}, avg={avg_vl:.1f}/sample | Avg time: {avg_t:.0f}s/sample")

    # V14 bypass stats
    n_bypass = sum(1 for r in all_results if r.get('v14_bypass'))
    n_pipeline = len(all_results) - n_bypass
    print(f"  V14 bypass: {n_bypass} choice ({100*n_bypass/len(all_results):.1f}%), "
          f"{n_pipeline} pipeline ({100*n_pipeline/len(all_results):.1f}%)")

    # Tool usage
    tc = Counter()
    for r in all_results:
        for e in r.get('tool_trace', []):
            tc[e.get('tool', '?')] += 1
    print(f"  Tool usage: {dict(tc)}")

    num = [r for r in all_results if r['question_type'] in NUMERICAL_TASKS]
    spa = [r for r in all_results if r['question_type'] not in NUMERICAL_TASKS]
    if num:
        print(f"  Numerical: n={len(num)}, V14={np.mean([r['score'] for r in num]):.3f}, V7={np.mean([r['v7_vl_score'] for r in num]):.3f}")
    if spa:
        print(f"  Choice:    n={len(spa)}, V14={np.mean([r['score'] for r in spa]):.3f}, V7={np.mean([r['v7_vl_score'] for r in spa]):.3f}")

    print(f"\nResults: {output_dir}")
    print(f"\n{'='*60}")
    print(f"  V14 Overall = {ov14:.4f}  vs  V7 VL = {ov7:.4f}  (Δ = {ov14-ov7:+.4f})")
    print(f"{'='*60}")

    summary = {
        'timestamp': timestamp,
        'version': 'v14_type_selective_bypass',
        'architecture': 'Choice→VL-only(official prompt + 5-vote) | Numerical→Grid/CODER/MoT pipeline',
        'design': 'Type-Selective Bypass: bypass Grid/CODER for choice, keep full pipeline for numerical',
        'improvements': 'V13→V14: Official QwenVL prompt for choice tasks, 5-vote SC, skip Grid/CODER for choice',
        'n_samples': len(all_results),
        'overall': {'v7_vl': float(ov7), 'v14': float(ov14), 'delta': float(ov14 - ov7)},
        'avg_vl_calls': float(avg_vl),
        'avg_time_s': float(avg_t),
        'v14_stats': {
            'bypass_count': n_bypass,
            'pipeline_count': n_pipeline,
        },
        'tool_usage': dict(tc),
        'by_task': {qt: {
            'n': len([r for r in all_results if r['question_type'] == qt]),
            'v7': float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type'] == qt])),
            'v14': float(np.mean([r['score'] for r in all_results if r['question_type'] == qt])),
            'avg_vl': float(np.mean([r.get('vl_calls', 0) for r in all_results if r['question_type'] == qt])),
            'bypass_rate': float(np.mean([1 if r.get('v14_bypass') else 0 for r in all_results if r['question_type'] == qt])),
            'coder_rate': float(np.mean([1 if r.get('coder_used') else 0 for r in all_results if r['question_type'] == qt])),
        } for qt in task_types},
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
