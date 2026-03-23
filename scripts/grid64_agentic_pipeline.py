#!/usr/bin/env python3
"""
64³ Grid Mind Map — Agentic Pipeline (4-Phase)

Phase 1: Manager    — 任务拆解与路由 (确定需要关注的Entity和Grid数据)
Phase 2: Retriever  — 定向数据检索 (从Grid中提取与问题相关的空间信息)
Phase 3: Evolver    — VL找错与进化 (VL对比Grid和视频，输出Evolution指令修正Grid)
Phase 4: Reasoner   — 最终回答 (Grid工具确定性计算 + VL辅助推理)

对比基准: V7 VL Overall = 63.61%
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

# 数值型任务 vs 选择型任务
NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 'room_size_estimation', 'object_abs_distance']
CHOICE_TASKS = ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard',
                'object_rel_distance', 'obj_appearance_order', 'route_planning']

# ============================================================================
# Phase 1: Manager — 任务拆解与路由
# ============================================================================

@dataclass
class TaskPlan:
    """Manager输出的任务计划"""
    question_type: str
    question: str
    options: List[str]
    ground_truth: str
    
    # Manager决定的路由策略
    needs_vl_evolution: bool      # 是否需要VL Evolution来修正Grid
    grid_can_answer: bool         # Grid工具能否直接确定性回答
    key_entities: List[str]       # 与问题相关的关键Entity名称
    focus_aspects: List[str]      # VL审视时需要关注的方面
    reasoning_mode: str           # "grid_only", "vl_only", "grid_then_vl", "vl_verify_grid"


def phase1_manager(question: str, question_type: str, options: List[str], grid: Grid64) -> TaskPlan:
    """Phase 1: 分析问题，确定路由策略"""
    
    key_entities = []
    focus_aspects = []
    
    q = question.lower()
    
    # 提取问题中提到的Entity
    if 'direction' in question_type:
        m = re.search(r'standing by (?:the )?(.+?)\s+and\s+facing (?:the )?(.+?),\s*is (?:the )?(.+?)\s+to\s', q)
        if m:
            key_entities = [m.group(1).strip(), m.group(2).strip(), m.group(3).strip()]
            focus_aspects = ["relative positions of these 3 objects", "are they correctly located in the grid"]
    
    elif question_type == 'object_counting':
        m = re.search(r'how many (\w+)', q)
        if m:
            key_entities = [m.group(1).strip()]
            focus_aspects = ["actual count in video", "possible missed or duplicate detections"]
    
    elif question_type == 'object_size_estimation':
        m = re.search(r'(?:size|height|length|width).*?(?:of|for)\s+(?:the\s+)?(.+?)[\?\.]', q)
        if m:
            key_entities = [m.group(1).strip()]
            focus_aspects = ["physical size estimation", "calibration accuracy"]
    
    elif question_type == 'room_size_estimation':
        focus_aspects = ["room dimensions from video", "furniture spread", "room type"]
    
    elif question_type == 'object_abs_distance':
        patterns = [
            r'distance between (?:the )?(.+?)\s+and\s+(?:the )?(.+?)[\s\(\?\.]',
            r'from (?:the )?(\w+(?:\s+\w+)*) to (?:the )?(\w+(?:\s+\w+)*)',
        ]
        for p in patterns:
            m = re.search(p, q)
            if m:
                key_entities = [m.group(1).strip(), m.group(2).strip()]
                focus_aspects = ["distance between objects", "depth accuracy"]
                break
    
    elif question_type == 'object_rel_distance':
        m = re.search(r'(?:closest|nearest|farthest|furthest)\s+(?:to|from)\s+(?:the\s+)?(.+?)[\?\.]', q)
        if m:
            key_entities = [m.group(1).strip()]
            for opt in options:
                cand = re.sub(r'^[A-D]\.\s*', '', opt).strip().lower()
                key_entities.append(cand)
            focus_aspects = ["relative distances between objects"]
    
    elif question_type == 'obj_appearance_order':
        focus_aspects = ["temporal order of object appearances"]
    
    elif question_type == 'route_planning':
        focus_aspects = ["spatial layout", "efficient path"]
    
    # 检查Grid中有多少key entities能找到
    found_count = 0
    for name in key_entities:
        if grid.get_by_category(name):
            found_count += 1
    entity_coverage = found_count / max(len(key_entities), 1)
    
    # 路由决策
    if 'direction' in question_type:
        # Direction: Grid的核心优势，如果entity都找到就用Grid
        if entity_coverage >= 0.9:
            reasoning_mode = "grid_then_vl"   # Grid计算 → VL验证
            needs_vl_evolution = True          # VL检查Grid位置是否正确
            grid_can_answer = True
        else:
            reasoning_mode = "vl_verify_grid"  # VL补充缺失的空间信息
            needs_vl_evolution = True
            grid_can_answer = False
    
    elif question_type in ('object_counting',):
        # Counting: VL直接看视频更准
        reasoning_mode = "vl_only"
        needs_vl_evolution = False
        grid_can_answer = False
    
    elif question_type in ('room_size_estimation',):
        # Room size: DA3尺度不稳定，VL更可靠
        reasoning_mode = "vl_only"
        needs_vl_evolution = False
        grid_can_answer = False
    
    elif question_type in ('object_size_estimation',):
        # Size: VL直接估计更好（Grid prompt过于复杂反而干扰VL判断）
        reasoning_mode = "vl_only"
        needs_vl_evolution = False
        grid_can_answer = False
    
    elif question_type in ('object_abs_distance',):
        # Abs distance: VL直接估计（DA3尺度不稳定，Grid距离不可靠）
        reasoning_mode = "vl_only"
        needs_vl_evolution = False
        grid_can_answer = False
    
    elif question_type in ('object_rel_distance',):
        # Relative distance: VL直接判断（Grid rel_distance accuracy低）
        reasoning_mode = "vl_only"
        needs_vl_evolution = False
        grid_can_answer = False
    
    elif question_type in ('obj_appearance_order',):
        # Appearance order: Grid有first_seen_frame
        reasoning_mode = "grid_then_vl"
        needs_vl_evolution = False
        grid_can_answer = True
    
    elif question_type in ('route_planning',):
        # Route: Grid空间距离 + VL场景理解
        reasoning_mode = "grid_then_vl"
        needs_vl_evolution = False
        grid_can_answer = True
    
    else:
        reasoning_mode = "vl_only"
        needs_vl_evolution = False
        grid_can_answer = False
    
    return TaskPlan(
        question_type=question_type,
        question=question,
        options=options,
        ground_truth="",
        needs_vl_evolution=needs_vl_evolution,
        grid_can_answer=grid_can_answer,
        key_entities=key_entities,
        focus_aspects=focus_aspects,
        reasoning_mode=reasoning_mode,
    )


# ============================================================================
# Phase 2: Retriever — 定向数据检索
# ============================================================================

@dataclass
class RetrievedContext:
    """Retriever从Grid中检索到的与问题相关的上下文"""
    grid_text: str                    # 完整Grid文本
    relevant_entities: Dict[str, GridEntity]  # 相关Entity子集
    grid_answer: Optional[str]        # Grid工具的确定性回答
    grid_reasoning: Optional[str]     # Grid工具的推理过程
    spatial_context: str              # 为VL准备的空间上下文描述
    missing_entities: List[str]       # 在Grid中找不到的Entity


def phase2_retriever(grid: Grid64, plan: TaskPlan) -> RetrievedContext:
    """Phase 2: 从Grid中提取与问题相关的信息"""
    
    # 1. 提取相关Entity
    relevant = {}
    missing = []
    for name in plan.key_entities:
        found = grid.get_by_category(name)
        if found:
            for e in found:
                relevant[e.entity_id] = e
        else:
            missing.append(name)
    
    # 2. Grid工具确定性回答
    grid_pred, grid_reason = None, None
    if plan.grid_can_answer:
        qt = plan.question_type
        q = plan.question
        opts = plan.options
        
        if qt == 'object_counting':
            grid_pred, grid_reason = grid_answer_counting(grid, q)
        elif qt == 'object_size_estimation':
            grid_pred, grid_reason = grid_answer_size(grid, q)
        elif qt == 'room_size_estimation':
            grid_pred, grid_reason = grid_answer_room_size(grid, q)
        elif qt == 'object_abs_distance':
            grid_pred, grid_reason = grid_answer_abs_distance(grid, q)
        elif 'direction' in qt:
            grid_pred, grid_reason = grid_answer_direction(grid, q, opts)
        elif qt == 'object_rel_distance':
            grid_pred, grid_reason = grid_answer_rel_distance(grid, q, opts)
        elif qt == 'obj_appearance_order':
            grid_pred, grid_reason = grid_answer_appearance_order(grid, q, opts)
        elif qt == 'route_planning':
            grid_pred, grid_reason = grid_answer_route(grid, q, opts)
    
    # 3. 构建空间上下文 (给VL看的精简版)
    spatial_lines = []
    spatial_lines.append(f"Scene: {len(grid.entities)} objects detected, "
                        f"meters_per_grid={grid.meters_per_grid:.3f}m, "
                        f"scene_span≈{grid.meters_per_grid*64:.1f}m")
    
    if relevant:
        spatial_lines.append("Key objects for this question:")
        for eid, e in relevant.items():
            phys = grid.grid_to_physical(e.grid_position)
            size_str = ""
            if e.size_3d is not None:
                ps = grid.physical_size(eid)
                if ps:
                    size_str = f", size≈{ps:.2f}m"
            spatial_lines.append(
                f"  - {eid}: grid=({e.grid_position[0]},{e.grid_position[1]},{e.grid_position[2]}), "
                f"phys=({phys[0]:.1f},{phys[1]:.1f},{phys[2]:.1f})m{size_str}, "
                f"confidence={e.confidence:.2f}, first_frame={e.first_seen_frame}"
            )
    
    if missing:
        spatial_lines.append(f"NOT FOUND in grid: {', '.join(missing)}")
    
    if grid_pred is not None:
        spatial_lines.append(f"Grid tool answer: {grid_pred} ({grid_reason})")
    
    return RetrievedContext(
        grid_text=grid.to_text(),
        relevant_entities=relevant,
        grid_answer=grid_pred,
        grid_reasoning=grid_reason,
        spatial_context="\n".join(spatial_lines),
        missing_entities=missing,
    )


# ============================================================================
# Phase 3: Evolver — VL找错与Evolution (核心创新)
# ============================================================================

@dataclass
class EvolutionResult:
    """Evolution的输出"""
    instructions: List[str]           # 结构化指令列表
    review_text: str                  # VL的审查报告
    grid_modified: bool               # Grid是否被修改
    confidence_boost: float           # Evolution后的置信度提升


class VLEvolver:
    """VL-Driven Grid Evolution: 对比视频和Grid，发现冲突并修正"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.vl_model = None
        self.vl_processor = None
    
    def load_model(self, model_path: str):
        if self.vl_model is not None:
            return
        
        logger.info(f"Loading VL model: {model_path}")
        try:
            from transformers import AutoProcessor
            self.vl_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            
            # Qwen3-VL需要用Qwen3VLForConditionalGeneration
            if 'qwen3' in model_path.lower() or 'Qwen3' in model_path:
                from transformers import Qwen3VLForConditionalGeneration
                self.vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True,
                )
                logger.info("VL model loaded (Qwen3-VL)")
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True,
                )
                logger.info("VL model loaded (Qwen2.5-VL)")
        except Exception as e:
            logger.error(f"Failed to load VL: {e}")
            import traceback
            traceback.print_exc()
    
    def unload(self):
        if self.vl_model is not None:
            del self.vl_model
            self.vl_model = None
        if self.vl_processor is not None:
            del self.vl_processor
            self.vl_processor = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def _call_vl(self, prompt: str, video_path: str, max_tokens: int = 512) -> str:
        """调用VL模型"""
        if self.vl_model is None:
            return ""
        try:
            from qwen_vl_utils import process_vision_info
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_pixels": 360 * 420, "nframes": 8},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = self.vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.vl_model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                )
            
            response = self.vl_processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0]
            return response.strip()
        except Exception as e:
            logger.warning(f"VL call failed: {e}")
            return ""
    
    def evolve(
        self,
        grid: Grid64,
        context: RetrievedContext,
        plan: TaskPlan,
        video_path: str,
    ) -> EvolutionResult:
        """Phase 3: VL审视Grid + 视频，输出Evolution指令修正Grid"""
        
        if not plan.needs_vl_evolution or self.vl_model is None:
            return EvolutionResult(
                instructions=[], review_text="", grid_modified=False, confidence_boost=0.0
            )
        
        # 构建Evolution审视prompt
        focus = "\n".join(f"- {a}" for a in plan.focus_aspects) if plan.focus_aspects else "- General spatial accuracy"
        
        missing_section = ""
        if context.missing_entities:
            missing_section = f"\n\nMISSING FROM GRID (mentioned in question but not detected):\n" + \
                             "\n".join(f"  - {m}" for m in context.missing_entities)
        
        grid_answer_section = ""
        if context.grid_answer is not None:
            grid_answer_section = f"\n\nGRID TOOL ANSWER: {context.grid_answer}\nGrid Reasoning: {context.grid_reasoning}"
        
        evolution_prompt = f"""You are a spatial perception reviewer. Compare the 3D Grid data below with what you see in the video. Find conflicts and output correction instructions.

=== 64x64x64 GRID DATA ===
{context.spatial_context}

=== FULL GRID ===
{context.grid_text}
{missing_section}
{grid_answer_section}

=== QUESTION CONTEXT ===
{plan.question}

=== WHAT TO CHECK ===
{focus}

=== YOUR TASK ===
Compare the grid data with the video. For each issue found, output a structured instruction:

Available instructions:
- DELETE <entity_id> : remove false detection
- MOVE <entity_id> <new_x> <new_y> <new_z> : fix position (grid coords 0-63)
- ADD <category> <x> <y> <z> : add missed object
- SET_SCALE <meters_per_grid> : fix scale if sizes look wrong
- NOOP : no changes needed

Output format (one per line):
INSTRUCTION: <instruction>
REASON: <why>

If the grid looks correct, output:
INSTRUCTION: NOOP
REASON: Grid is consistent with video

Review:"""
        
        review_text = self._call_vl(evolution_prompt, video_path, max_tokens=512)
        
        # 解析Evolution指令
        instructions = []
        for line in review_text.split('\n'):
            line = line.strip()
            if line.upper().startswith('INSTRUCTION:'):
                instr = line.split(':', 1)[1].strip()
                if instr:
                    instructions.append(instr)
        
        # 执行Evolution指令
        grid_modified = False
        for instr in instructions:
            parts = instr.strip().split()
            if not parts:
                continue
            cmd = parts[0].upper()
            
            if cmd == 'DELETE' and len(parts) >= 2:
                eid = parts[1]
                if eid in grid.entities:
                    del grid.entities[eid]
                    grid_modified = True
                    logger.info(f"  Evolution DELETE: {eid}")
            
            elif cmd == 'MOVE' and len(parts) >= 5:
                eid = parts[1]
                try:
                    new_pos = (int(parts[2]), int(parts[3]), int(parts[4]))
                    new_pos = tuple(max(0, min(63, c)) for c in new_pos)
                    if eid in grid.entities:
                        grid.entities[eid].grid_position = new_pos
                        grid_modified = True
                        logger.info(f"  Evolution MOVE: {eid} → {new_pos}")
                except (ValueError, IndexError):
                    pass
            
            elif cmd == 'ADD' and len(parts) >= 5:
                try:
                    category = parts[1]
                    pos = (int(parts[2]), int(parts[3]), int(parts[4]))
                    pos = tuple(max(0, min(63, c)) for c in pos)
                    eid = category.replace(' ', '_')
                    if eid not in grid.entities:
                        grid.entities[eid] = GridEntity(
                            entity_id=eid, category=category,
                            grid_position=pos,
                            position_3d=np.array(pos, dtype=float) * grid.meters_per_grid,
                            confidence=0.3, first_seen_frame=0,
                        )
                        grid_modified = True
                        logger.info(f"  Evolution ADD: {eid} at {pos}")
                except (ValueError, IndexError):
                    pass
            
            elif cmd == 'SET_SCALE' and len(parts) >= 2:
                try:
                    new_mpg = float(parts[1])
                    if 0.01 < new_mpg < 2.0:
                        grid.meters_per_grid = new_mpg
                        grid_modified = True
                        logger.info(f"  Evolution SET_SCALE: mpg={new_mpg}")
                except ValueError:
                    pass
            
            elif cmd == 'NOOP':
                pass
        
        return EvolutionResult(
            instructions=instructions,
            review_text=review_text,
            grid_modified=grid_modified,
            confidence_boost=0.1 if grid_modified else 0.0,
        )


# ============================================================================
# Phase 4: Reasoner — 最终回答
# ============================================================================

class Reasoner:
    """Phase 4: 基于修正后的Grid + VL做最终回答"""
    
    def __init__(self, vl_evolver: VLEvolver):
        self.vl = vl_evolver
    
    def answer(
        self,
        grid: Grid64,
        plan: TaskPlan,
        context: RetrievedContext,
        evolution: EvolutionResult,
        video_path: str,
    ) -> Tuple[str, str]:
        """最终回答: 根据routing策略选择Grid/VL/融合"""
        
        qt = plan.question_type
        mode = plan.reasoning_mode
        
        # 如果Evolution修改了Grid，重新计算Grid answer
        if evolution.grid_modified and plan.grid_can_answer:
            context = phase2_retriever(grid, plan)
        
        # === grid_only: 直接用Grid确定性回答 ===
        if mode == "grid_only":
            if context.grid_answer is not None:
                return context.grid_answer, f"[grid_only] {context.grid_reasoning}"
            return self._vl_answer(grid, plan, context, evolution, video_path)
        
        # === vl_only: VL直接看视频回答 ===
        elif mode == "vl_only":
            return self._vl_answer(grid, plan, context, evolution, video_path)
        
        # === grid_then_vl: Grid先算，VL验证/补充 ===
        elif mode == "grid_then_vl":
            grid_pred = context.grid_answer
            grid_reason = context.grid_reasoning
            
            if grid_pred is None or 'not found' in (grid_reason or ''):
                # Grid无法回答，fallback到VL
                return self._vl_answer(grid, plan, context, evolution, video_path)
            
            # Grid有答案，让VL验证
            vl_pred, vl_reason = self._vl_answer(grid, plan, context, evolution, video_path)
            
            # 选择策略: direction用Grid (Grid强项), 其他看情况
            if 'direction' in qt:
                # Direction: 信任Grid的空间计算
                return grid_pred, f"[grid_then_vl] Grid={grid_pred}({grid_reason}) | VL={vl_pred}"
            else:
                # 其他: 如果VL和Grid一致，增加信心；不一致，用VL
                if grid_pred == vl_pred:
                    return grid_pred, f"[agree] Grid=VL={grid_pred}"
                else:
                    # 数值任务取平均，选择任务用VL
                    if qt in NUMERICAL_TASKS:
                        try:
                            g = float(re.search(r'[\d.]+', str(grid_pred)).group())
                            v = float(re.search(r'[\d.]+', str(vl_pred)).group())
                            avg = (g + v) / 2
                            return f"{avg:.2f}" if '.' in str(grid_pred) else str(int(round(avg))), \
                                   f"[avg] Grid={grid_pred} VL={vl_pred} → {avg:.1f}"
                        except:
                            return vl_pred, f"[vl_fallback] Grid={grid_pred} VL={vl_pred}"
                    else:
                        return vl_pred, f"[vl_preferred] Grid={grid_pred} VL={vl_pred}"
        
        # === vl_verify_grid: VL用Grid信息辅助回答 ===
        elif mode == "vl_verify_grid":
            return self._vl_answer(grid, plan, context, evolution, video_path)
        
        # fallback
        return self._vl_answer(grid, plan, context, evolution, video_path)
    
    def _vl_answer(
        self,
        grid: Grid64,
        plan: TaskPlan,
        context: RetrievedContext,
        evolution: EvolutionResult,
        video_path: str,
    ) -> Tuple[str, str]:
        """VL模型回答 — 分任务定制prompt (仿V7)"""
        
        if self.vl.vl_model is None:
            if context.grid_answer is not None:
                return context.grid_answer, f"[no_vl_fallback] {context.grid_reasoning}"
            return ("0" if plan.question_type in NUMERICAL_TASKS else "A"), "[no_vl] default"
        
        qt = plan.question_type
        is_numerical = qt in NUMERICAL_TASKS
        mode = plan.reasoning_mode
        
        # 对于vl_only模式，使用简洁的分任务prompt（不加过多Grid噪音）
        if mode == "vl_only":
            prompt = self._build_vl_only_prompt(plan, context)
        else:
            prompt = self._build_grid_aware_prompt(plan, context, evolution)
        
        response = self.vl._call_vl(prompt, video_path, max_tokens=256)
        
        if is_numerical:
            m = re.search(r'[\d.]+', response)
            if m:
                return m.group(), f"[vl] {response[:100]}"
            # Fallback: V7对数值任务有合理默认值
            defaults = {'object_counting': '1', 'object_size_estimation': '100',
                        'room_size_estimation': '25', 'object_abs_distance': '2.0'}
            return defaults.get(qt, '0'), f"[vl_fail] {response[:100]}"
        else:
            answer = self._extract_choice(response, plan.options)
            return answer, f"[vl] {response[:100]}"
    
    def _build_vl_only_prompt(self, plan: TaskPlan, context: RetrievedContext) -> str:
        """构建简洁的VL-only prompt (仿V7分任务定制)"""
        qt = plan.question_type
        q = plan.question
        
        # Grid文本（类似V7的mind_map文本）
        grid_text = context.grid_text if context.grid_text else ""
        spatial_hint = context.spatial_context if context.spatial_context else ""
        
        if qt == 'object_counting':
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
            return f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to estimate the size of a specific object.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{grid_text}

=== QUESTION ===
{q}

=== INSTRUCTIONS ===
1. Look at the target object in the video frames.
2. Compare it with known reference objects (doors ~2m, chairs ~0.8m) for scale.
3. Consider standard sizes of common indoor objects.

Please respond with ONLY a single integer number representing the size in centimeters.
Do not include any explanation or units, just the number.

Answer:"""
        
        elif qt == 'room_size_estimation':
            return f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to estimate the room floor area.

=== SPATIAL PERCEPTION DATA ===
{spatial_hint}

=== QUESTION ===
{q}

=== INSTRUCTIONS ===
1. Observe the room in the video frames.
2. Look for reference objects (doors, beds, sofas) to estimate scale.
3. Estimate the room's length and width, then calculate area.
4. Most indoor rooms are between 5 and 50 square meters.

Please respond with ONLY a single number representing the room area in square meters.
Do not include any explanation or units, just the number.

Answer:"""
        
        elif qt == 'object_abs_distance':
            return f"""You are a spatial intelligence assistant analyzing a video of an indoor scene. Your task is to estimate the distance between two objects.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{grid_text}

=== REFERENCE SIZES FOR SCALE ===
Standard door: ~2.0m tall. Chair: ~0.8m tall. Bed: ~2.0m long. Refrigerator: ~1.75m tall.

=== QUESTION ===
{q}

=== INSTRUCTIONS ===
1. Locate both objects in the video frames.
2. Use the reference sizes above to calibrate your distance estimate.
3. Consider the 3D positions from the detected objects list.
4. Account for depth (objects farther from camera may appear closer together).
5. Distances in indoor scenes typically range from 0.5m to 15m.

Please respond with ONLY a decimal number representing the distance in meters.
Do not include any explanation or units, just the number (e.g., "2.5" or "1.8").

Answer:"""
        
        else:
            # 选择题 (direction, rel_distance, appearance_order, route)
            options_text = "\n".join(plan.options) if plan.options else ""
            
            task_hint = ""
            if 'direction' in qt:
                task_hint = "Determine the relative direction of objects in the scene."
            elif qt == 'object_rel_distance':
                task_hint = "Determine which object is closest/farthest."
            elif qt == 'obj_appearance_order':
                task_hint = "Determine the order in which objects first appear in the video."
            elif qt == 'route_planning':
                task_hint = "Determine the most efficient route between objects."
            
            return f"""You are a spatial intelligence assistant analyzing a video of an indoor scene.
{task_hint}

=== SPATIAL PERCEPTION DATA ===
{spatial_hint}

=== QUESTION ===
{q}

=== OPTIONS ===
{options_text}

=== INSTRUCTIONS ===
1. Watch the video carefully.
2. Use the spatial data as reference.
3. Trust your visual observation when uncertain.

Answer with ONLY the option letter (A, B, C, or D).

Answer:"""
    
    def _build_grid_aware_prompt(self, plan: TaskPlan, context: RetrievedContext,
                                  evolution: EvolutionResult) -> str:
        """构建带Grid信息的prompt (用于grid_then_vl和vl_verify_grid模式)"""
        qt = plan.question_type
        is_numerical = qt in NUMERICAL_TASKS
        
        evolution_section = ""
        if evolution.review_text:
            evolution_section = f"\n=== SPATIAL REVIEW ===\n{evolution.review_text[:300]}"
        
        grid_section = ""
        if context.grid_answer is not None:
            grid_section = f"\n=== GRID COMPUTATION ===\nGrid answer: {context.grid_answer} ({context.grid_reasoning})"
        
        options_section = ""
        if plan.options and not is_numerical:
            options_section = "\n=== OPTIONS ===\n" + "\n".join(plan.options)
        
        if is_numerical:
            answer_instruction = "Please respond with ONLY a single number. Do not include any explanation or units."
        else:
            answer_instruction = "Answer with ONLY the option letter (A, B, C, or D)."
        
        return f"""You are a spatial intelligence assistant analyzing a video of an indoor scene.

=== SPATIAL DATA ===
{context.spatial_context}
{evolution_section}
{grid_section}

=== QUESTION ===
{plan.question}
{options_section}

=== INSTRUCTIONS ===
1. Watch the video carefully.
2. Use the spatial data as reference, but trust your visual observation when there are conflicts.

{answer_instruction}

Answer:"""
    
    def _extract_choice(self, response: str, options: List[str]) -> str:
        """从VL响应提取选项字母"""
        response_clean = response.split('[')[0].strip()
        
        m = re.search(r'^([A-D])', response_clean.upper())
        if m:
            return m.group(1)
        
        for line in response.split('\n')[::-1]:
            line = line.strip()
            if line and line[0].upper() in 'ABCD':
                return line[0].upper()
        
        response_lower = response.lower()
        for i, opt in enumerate(options):
            opt_content = opt.lower()
            if len(opt) >= 3 and opt[1] in '.、':
                opt_content = opt[3:].strip().lower()
            if opt_content in response_lower:
                return chr(65 + i)
        
        return "A"


# ============================================================================
# 完整Pipeline
# ============================================================================

class AgenticPipeline:
    """4-Phase Agentic Pipeline"""
    
    def __init__(self, device='cuda:0', vl_model_path=None):
        self.device = device
        self.vl_model_path = vl_model_path or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        
        self.builder = Grid64Builder(device=device, num_frames=16)
        self.evolver = VLEvolver(device=device)
        self.reasoner = Reasoner(self.evolver)
    
    def load_models(self):
        self.builder.load_models()
        self.evolver.load_model(self.vl_model_path)
    
    def unload(self):
        self.builder.unload()
        self.evolver.unload()
    
    def process_scene(self, video_path: str, samples: List[Dict]) -> List[Dict]:
        """处理一个场景的所有问题"""
        
        # Build Grid (共享)
        t0 = time.time()
        grid = self.builder.build_grid(video_path)
        build_time = time.time() - t0
        logger.info(f"  Grid built: {len(grid.entities)} entities, mpg={grid.meters_per_grid:.4f}m ({build_time:.1f}s)")
        
        results = []
        for sample in samples:
            result = self.process_sample(grid, sample, video_path)
            results.append(result)
        
        return results
    
    def process_sample(self, grid: Grid64, sample: Dict, video_path: str) -> Dict:
        """4-Phase处理单个样本"""
        qt = sample['question_type']
        question = sample['question']
        options = sample.get('options') or []
        gt = sample['ground_truth']
        
        # Phase 1: Manager
        plan = phase1_manager(question, qt, options, grid)
        plan.ground_truth = gt
        
        # Phase 2: Retriever
        context = phase2_retriever(grid, plan)
        
        # Phase 3: Evolver (VL审视+修正Grid)
        evolution = self.evolver.evolve(grid, context, plan, video_path)
        
        # Phase 4: Reasoner (最终回答)
        pred, reasoning = self.reasoner.answer(grid, plan, context, evolution, video_path)
        
        # 评估
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
            'routing_mode': plan.reasoning_mode,
            'evolution_instructions': evolution.instructions,
            'evolution_review': evolution.review_text[:200] if evolution.review_text else '',
            'grid_modified': evolution.grid_modified,
            'grid_answer': context.grid_answer,
            'grid_reasoning': context.grid_reasoning,
            'missing_entities': context.missing_entities,
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
    parser.add_argument('--n_per_type', type=int, default=10, help='小样本测试每类数量 (--full时忽略)')
    parser.add_argument('--full', action='store_true', help='全量测试 (5130样本)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None, help='多卡并行: 当前GPU编号 (0-7)')
    parser.add_argument('--num_gpus', type=int, default=8, help='多卡并行: 总GPU数')
    parser.add_argument('--vl-model', type=str, 
                       default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()
    
    # 多卡模式下自动设置device
    if args.gpu_id is not None:
        args.device = f'cuda:{args.gpu_id}'
    
    # 加载V7基准
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    logger.info(f"Loading V7 baseline: {v7_path}")
    with open(v7_path) as f:
        v7_results = json.load(f)
    logger.info(f"V7: {len(v7_results)} samples")
    
    # 选样本
    if args.full:
        test_samples = [s for s in v7_results if find_video_path(s['scene_name'])]
        logger.info(f"Full test: {len(test_samples)} samples")
    else:
        test_samples = select_test_samples(v7_results, n_per_type=args.n_per_type)
        logger.info(f"Selected {len(test_samples)} test samples")
    
    # 按scene分组
    by_scene = defaultdict(list)
    for s in test_samples:
        by_scene[s['scene_name']].append(s)
    
    # 按scene名排序确保每卡分配一致
    scene_list = sorted(by_scene.keys())
    
    # 多卡分片
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
    
    # Pipeline
    pipeline = AgenticPipeline(device=args.device, vl_model_path=args.vl_model)
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
                    'prediction': '0', 'reasoning': 'no video', 'score': 0.0,
                    'routing_mode': 'none', 'v7_vl_score': s.get('vl_score', 0),
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
                logger.info(
                    f"  {r['question_type'][:25]:25s} [{r['routing_mode']:15s}] "
                    f"Score={r['score']:.3f} VL={r['v7_vl_score']:.3f} {marker} "
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
                    'prediction': '0', 'reasoning': f'error: {str(e)[:100]}', 'score': 0.0,
                    'routing_mode': 'error', 'v7_vl_score': s.get('vl_score', 0),
                    'v7_rule_score': s.get('rule_score', 0),
                })
    
    pipeline.unload()
    
    # ========================================================================
    # 保存分片结果 (多卡模式下每卡保存自己的结果)
    # ========================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.gpu_id is not None:
        output_dir = PROJECT_ROOT / "outputs" / f"agentic_pipeline_full" / f"gpu{args.gpu_id}"
    else:
        output_dir = PROJECT_ROOT / "outputs" / f"agentic_pipeline_{timestamp}"
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
    
    logger.info(f"Shard results saved to: {output_dir} ({len(all_results)} samples)")
    
    # 非分片模式或者全量汇总时打印统计
    if args.gpu_id is None:
        _print_summary(all_results, output_dir, timestamp)


def _print_summary(all_results, output_dir, timestamp):
    """打印和保存汇总统计"""
    print("\n" + "=" * 120)
    print("Agentic Pipeline (Manager → Retriever → Evolver → Reasoner)")
    print(f"基准: V7 VL Overall = 63.61%")
    print(f"测试样本: {len(all_results)}")
    print("=" * 120)
    
    task_types = sorted(set(r['question_type'] for r in all_results))
    
    print(f"\n{'Task':<35} {'N':>4} {'V7_VL':>7} {'Pipe':>7} {'Delta':>7} {'Mode':>18}")
    print(f"{'-'*35} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*18}")
    
    all_v7, all_pipe = [], []
    
    for qt in task_types:
        qt_r = [r for r in all_results if r['question_type'] == qt]
        v7 = np.mean([r['v7_vl_score'] for r in qt_r])
        pipe = np.mean([r['score'] for r in qt_r])
        delta = pipe - v7
        modes = Counter(r.get('routing_mode', '?') for r in qt_r)
        mode_str = modes.most_common(1)[0][0] if modes else '?'
        marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
        print(f"  {qt:<35} {len(qt_r):>4} {v7:>6.3f} {pipe:>6.3f} {delta:>+6.3f} {marker} {mode_str:>18}")
        all_v7.extend([r['v7_vl_score'] for r in qt_r])
        all_pipe.extend([r['score'] for r in qt_r])
    
    print(f"{'-'*35} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*18}")
    ov7, op = np.mean(all_v7), np.mean(all_pipe)
    print(f"  {'Overall':<35} {len(all_results):>4} {ov7:>6.3f} {op:>6.3f} {op-ov7:>+6.3f}")
    
    # Routing分析
    print(f"\n{'='*60}")
    print("Routing Mode Analysis:")
    for mode in sorted(set(r.get('routing_mode', '?') for r in all_results)):
        mr = [r for r in all_results if r.get('routing_mode', '?') == mode]
        mv7 = np.mean([r['v7_vl_score'] for r in mr])
        mp = np.mean([r['score'] for r in mr])
        print(f"  {mode:<18} n={len(mr):>4} V7_VL={mv7:.3f} Pipe={mp:.3f} Delta={mp-mv7:>+.3f}")
    
    # Evolution分析
    evo_count = sum(1 for r in all_results if r.get('grid_modified', False))
    print(f"\nEvolution: {evo_count}/{len(all_results)} samples had grid modifications")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\n总结: Pipeline Overall={op:.4f} vs V7 VL={ov7:.4f} (Delta={op-ov7:+.4f})")
    
    # 保存summary
    summary = {
        'timestamp': timestamp,
        'n_samples': len(all_results),
        'overall': {'v7_vl': float(ov7), 'pipeline': float(op), 'delta': float(op - ov7)},
        'by_task': {qt: {
            'n': len([r for r in all_results if r['question_type'] == qt]),
            'v7_vl': float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type'] == qt])),
            'pipeline': float(np.mean([r['score'] for r in all_results if r['question_type'] == qt])),
        } for qt in task_types},
        'routing_stats': dict(Counter(r.get('routing_mode', '?') for r in all_results)),
        'evolution_count': evo_count,
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
