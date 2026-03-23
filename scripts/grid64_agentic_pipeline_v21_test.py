#!/usr/bin/env python3
"""
V21 Ablation Test: 4 variants to improve choice-question performance.

Variant 1: CODER Soft Vote — CODER answer participates in final weighted vote
Variant 3: Disagreement-Driven Evolution — extra VL call on alternative frames when CODER disagrees
Variant 4: Frame Quality Scoring — select best co-occurrence frames by detection confidence
Variant 5: Multi-View Retry — on disagreement, retry VL on alternative (non-overlapping) frames

Run: python3 scripts/grid64_agentic_pipeline_v21_test.py --variant {1,3,4,5,all} [--n_samples 100]
"""

import os, sys, json, re, gc, copy, time, math, logging, traceback
import base64, io
import numpy as np, cv2, torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

# Import everything from V20's base
from scripts.grid64_agentic_pipeline_v20 import (
    Grid256, Grid256Builder, Grid64Builder,
    VLModel, ToolExecutionContext,
    generate_grid_slice,
    _extract_question_entities, _match_name,
    coder_tool, evolutor_tool, _extract_not_found, _auto_add, _targeted_filter,
    _auto_coder_type, _is_temporal_question,
    _get_cooccurrence_frames, _get_entity_union_frames,
    _build_vl_independent_prompt, _build_vl_focused_prompt, _build_temporal_vl_prompt,
    _build_numerical_vl_prompt, _numerical_path,
    _clean, _evolve_belief, _select_frames, _vl_on_frames,
    extract_focused_frames, _get_video_fps_nframes,
    UNIFIED_FPS, GRID_MAX_FRAMES, VL_DEFAULT_MAX_PIXELS,
    AgenticPipelineV20,
)
from scripts.grid64_real_test import (
    find_video_path, evaluate_sample, mean_relative_accuracy,
)


# ============================================================================
# Variant 1: CODER Soft Vote
# ============================================================================

# Per-type CODER reliability weights (based on V7 Rule accuracy analysis)
CODER_VOTE_WEIGHTS = {
    'object_rel_direction_easy': 1,
    'object_rel_direction_medium': 1,
    'object_rel_direction_hard': 1,
    'object_rel_distance': 2,
    'obj_appearance_order': 2,
    'route_planning': 1,
}

def v21_variant1_coder_soft_vote(ctx, max_rounds=3):
    """V20 + CODER answer participates in final weighted vote."""
    rp = []
    ct = _auto_coder_type(ctx.question, ctx.options) or ''

    if not ctx.options:
        return _numerical_path(ctx, ct, rp)

    is_temporal = _is_temporal_question(ctx.question)
    rel_names = _extract_question_entities(ctx.question, ctx.options)

    # P0: Build Belief
    rp.append("[P0:belief]")
    cr = coder_tool(ctx, ct) if ct else ''
    if cr and 'not found' in cr.lower():
        mod, log = _auto_add(ctx, cr)
        if mod: cr = coder_tool(ctx, ct); rp.append(f"[belief_fix] {log[:60]}")
    m_coder_ans = re.search(r'answer=([A-D])', cr)
    coder_ans = m_coder_ans.group(1) if m_coder_ans else ''
    rp.append(f"[coder] {coder_ans}")

    # P1: Global VL
    rp.append("[P1:vl_global]")
    vl_prompt = _build_vl_independent_prompt(ctx)
    vl_global = _clean(ctx.vl.call(vl_prompt, ctx.video_path, max_tokens=128), ctx)
    ctx.vl_calls += 1
    rp.append(f"[vl_full] {vl_global}")

    # Iterative Loop (same as V20)
    std_thresholds = [0.4, 0.3, 0.2, 0.15, 0.1]
    vl_history = [('global', vl_global)]
    prev_frames = set()
    prev_focused_answer = None
    converge_type = None
    n_rounds = 0

    for round_id in range(1, max_rounds + 1):
        rp.append(f"[R{round_id}]")
        frames, ents, ftype = _select_frames(ctx, is_temporal)
        cur_frames = set(frames) if frames else set()

        if round_id > 1 and cur_frames == prev_frames:
            rp.append(f"[R{round_id}:no_new_frames]"); break

        vl_ans, _ = _vl_on_frames(ctx, frames, ents, ftype, is_temporal, rp, round_id)
        vl_history.append((f'R{round_id}', vl_ans))
        n_rounds = round_id

        if vl_ans == vl_global and vl_ans in 'ABCD':
            converge_type = 'global_consensus'
            rp.append(f"[R{round_id}:global_consensus] {vl_ans}"); break

        if prev_focused_answer is not None and vl_ans == prev_focused_answer and vl_ans in 'ABCD':
            converge_type = 'evolution_stable'
            rp.append(f"[R{round_id}:evolution_stable] {vl_ans}"); break

        threshold = std_thresholds[min(round_id - 1, len(std_thresholds) - 1)]
        _evolve_belief(ctx, rel_names, threshold, rp)
        prev_frames = cur_frames
        prev_focused_answer = vl_ans

    # === KEY CHANGE: Final Decision with CODER vote ===
    all_valid = [(ans, i) for i, (src, ans) in enumerate(vl_history) if ans in 'ABCD']

    if converge_type == 'global_consensus':
        ans = vl_global
        rp.append(f"[final:global_consensus] {ans}")
    elif converge_type == 'evolution_stable':
        ans = prev_focused_answer
        rp.append(f"[final:evolution_stable] {ans}")
    elif not all_valid:
        ans = coder_ans if coder_ans else 'A'
        rp.append(f"[fallback_coder] {ans}")
    else:
        # Weighted vote: VL sources + CODER
        weighted_counts = Counter()
        for src_ans, idx in all_valid:
            src = vl_history[idx][0]
            if src == 'global':
                w = max_rounds
            else:
                w = int(src[1:]) + 1
            weighted_counts[src_ans] += w

        # CODER participates with type-specific weight
        if coder_ans and coder_ans in 'ABCD':
            coder_w = CODER_VOTE_WEIGHTS.get(ctx.question_type, 1)
            weighted_counts[coder_ans] += coder_w
            rp.append(f"[coder_vote] {coder_ans} w={coder_w}")

        ans = weighted_counts.most_common(1)[0][0]
        rp.append(f"[final:weighted_vote+coder] {dict(weighted_counts)}→{ans}")

    rp.append(f"[rounds={n_rounds}]")
    ctx._final_answer = ans
    return ans, " | ".join(rp)


# ============================================================================
# Variant 3: Disagreement-Driven Evolution
# ============================================================================

def v21_variant3_disagreement_evolution(ctx, max_rounds=3):
    """V20 + when CODER disagrees with VL, try alternative frames."""
    rp = []
    ct = _auto_coder_type(ctx.question, ctx.options) or ''

    if not ctx.options:
        return _numerical_path(ctx, ct, rp)

    is_temporal = _is_temporal_question(ctx.question)
    rel_names = _extract_question_entities(ctx.question, ctx.options)

    # P0: Build Belief
    rp.append("[P0:belief]")
    cr = coder_tool(ctx, ct) if ct else ''
    if cr and 'not found' in cr.lower():
        mod, log = _auto_add(ctx, cr)
        if mod: cr = coder_tool(ctx, ct); rp.append(f"[belief_fix] {log[:60]}")
    m_coder_ans = re.search(r'answer=([A-D])', cr)
    coder_ans = m_coder_ans.group(1) if m_coder_ans else ''
    rp.append(f"[coder] {coder_ans}")

    # P1: Global VL
    rp.append("[P1:vl_global]")
    vl_prompt = _build_vl_independent_prompt(ctx)
    vl_global = _clean(ctx.vl.call(vl_prompt, ctx.video_path, max_tokens=128), ctx)
    ctx.vl_calls += 1
    rp.append(f"[vl_full] {vl_global}")

    # Iterative Loop (same as V20)
    std_thresholds = [0.4, 0.3, 0.2, 0.15, 0.1]
    vl_history = [('global', vl_global)]
    prev_frames = set()
    prev_focused_answer = None
    converge_type = None
    n_rounds = 0
    all_used_frames = set()

    for round_id in range(1, max_rounds + 1):
        rp.append(f"[R{round_id}]")
        frames, ents, ftype = _select_frames(ctx, is_temporal)
        cur_frames = set(frames) if frames else set()
        all_used_frames.update(cur_frames)

        if round_id > 1 and cur_frames == prev_frames:
            rp.append(f"[R{round_id}:no_new_frames]"); break

        vl_ans, _ = _vl_on_frames(ctx, frames, ents, ftype, is_temporal, rp, round_id)
        vl_history.append((f'R{round_id}', vl_ans))
        n_rounds = round_id

        if vl_ans == vl_global and vl_ans in 'ABCD':
            converge_type = 'global_consensus'
            rp.append(f"[R{round_id}:global_consensus] {vl_ans}"); break

        if prev_focused_answer is not None and vl_ans == prev_focused_answer and vl_ans in 'ABCD':
            converge_type = 'evolution_stable'
            rp.append(f"[R{round_id}:evolution_stable] {vl_ans}"); break

        # === KEY CHANGE: If CODER disagrees with current VL, try alternative frames ===
        if coder_ans and coder_ans in 'ABCD' and coder_ans != vl_ans and round_id == 1:
            # Get all entity frames, exclude already-used frames
            all_entity_frames = set()
            for name in rel_names:
                ents_list = ctx.grid.get_by_category(name)
                if not ents_list: continue
                for e in ents_list[:2]:
                    for d in e.detections:
                        fi = d.get('frame_idx', -1)
                        if fi >= 0: all_entity_frames.add(fi)
            alt_frames = sorted(all_entity_frames - all_used_frames)

            if len(alt_frames) >= 2:
                rp.append(f"[R{round_id}:disagree_alt] coder={coder_ans}!=vl={vl_ans}, trying {len(alt_frames)} alt frames")
                vl_alt, _ = _vl_on_frames(ctx, alt_frames, ents if ents else rel_names, ftype, is_temporal, rp, round_id)
                vl_history.append((f'R{round_id}_alt', vl_alt))
                all_used_frames.update(alt_frames)

                # If alt agrees with either global or focused → stronger signal
                if vl_alt == vl_global and vl_alt in 'ABCD':
                    converge_type = 'global_consensus'
                    rp.append(f"[R{round_id}:alt_confirms_global] {vl_alt}"); break
                if vl_alt == vl_ans and vl_alt in 'ABCD':
                    converge_type = 'evolution_stable'
                    rp.append(f"[R{round_id}:alt_confirms_focused] {vl_alt}"); break

        threshold = std_thresholds[min(round_id - 1, len(std_thresholds) - 1)]
        _evolve_belief(ctx, rel_names, threshold, rp)
        prev_frames = cur_frames
        prev_focused_answer = vl_ans

    # Final Decision (same as V20 + CODER tiebreaker)
    all_valid = [(ans, i) for i, (src, ans) in enumerate(vl_history) if ans in 'ABCD']

    if converge_type == 'global_consensus':
        ans = vl_global
    elif converge_type == 'evolution_stable':
        ans = prev_focused_answer
    elif not all_valid:
        ans = coder_ans if coder_ans else 'A'
    else:
        weighted_counts = Counter()
        for src_ans, idx in all_valid:
            src = vl_history[idx][0]
            if src == 'global':
                w = max_rounds
            elif '_alt' in src:
                w = 2  # alt frames get moderate weight
            else:
                w = int(src[1]) + 1
            weighted_counts[src_ans] += w
        # CODER as tiebreaker
        if coder_ans and coder_ans in 'ABCD':
            weighted_counts[coder_ans] += 1
        ans = weighted_counts.most_common(1)[0][0]
        rp.append(f"[final:weighted_vote] {dict(weighted_counts)}→{ans}")

    rp.append(f"[rounds={n_rounds}]")
    ctx._final_answer = ans
    return ans, " | ".join(rp)


# ============================================================================
# Variant 4: Frame Quality Scoring
# ============================================================================

def _select_frames_quality(ctx, is_temporal):
    """Select best frames by detection confidence and spatial clarity."""
    if is_temporal:
        frames, ents = _get_entity_union_frames(ctx.grid, ctx.question, ctx.options)
        return frames, ents, 'temporal'

    # For spatial questions: score co-occurrence frames by quality
    rel = _extract_question_entities(ctx.question, ctx.options)
    if not rel:
        return [], [], 'cooccur'

    entity_frames = {}
    entity_frame_details = {}  # (entity_name, frame_idx) -> {confidence, position}

    for name in rel:
        ents = ctx.grid.get_by_category(name)
        if not ents: continue
        frames = set()
        for e in ents[:2]:
            for d in e.detections:
                fi = d.get('frame_idx', -1)
                if fi >= 0:
                    frames.add(fi)
                    key = (name, fi)
                    entity_frame_details[key] = {
                        'confidence': d.get('confidence', 0.5),
                        'position_3d': d.get('position_3d', [0,0,0]),
                    }
        if frames:
            entity_frames[name] = frames

    if len(entity_frames) < 2:
        all_frames = set()
        for fs in entity_frames.values(): all_frames.update(fs)
        return sorted(all_frames), list(entity_frames.keys()), 'cooccur'

    from itertools import combinations
    cooccur = set()
    for n1, n2 in combinations(entity_frames.keys(), 2):
        cooccur.update(entity_frames[n1] & entity_frames[n2])

    if not cooccur:
        all_frames = set()
        for fs in entity_frames.values(): all_frames.update(fs)
        return sorted(all_frames), list(entity_frames.keys()), 'cooccur'

    # Score each co-occurrence frame
    ent_names = list(entity_frames.keys())
    scored = []
    for fi in cooccur:
        # Average confidence of all entities visible in this frame
        confs = []
        positions = []
        for name in ent_names:
            key = (name, fi)
            if key in entity_frame_details:
                confs.append(entity_frame_details[key]['confidence'])
                positions.append(entity_frame_details[key]['position_3d'])

        if len(confs) < 2:
            scored.append((fi, 0.0))
            continue

        avg_conf = np.mean(confs)

        # Prefer frames where entities are at moderate distance (not too close, not too far)
        if len(positions) >= 2:
            p1, p2 = np.array(positions[0]), np.array(positions[1])
            dist = float(np.linalg.norm(p1 - p2))
            # Bell curve: best at ~2m, reasonable from 0.5-5m
            dist_score = np.exp(-0.5 * ((dist - 2.0) / 2.0) ** 2)
        else:
            dist_score = 0.5

        score = avg_conf * 0.7 + dist_score * 0.3
        scored.append((fi, score))

    # Sort by score descending, take top 8
    scored.sort(key=lambda x: -x[1])
    best_frames = [fi for fi, _ in scored[:8]]
    best_frames.sort()  # Restore temporal order

    return best_frames, ent_names, 'cooccur_quality'


def v21_variant4_frame_quality(ctx, max_rounds=3):
    """V20 + frame quality scoring for co-occurrence frame selection."""
    rp = []
    ct = _auto_coder_type(ctx.question, ctx.options) or ''

    if not ctx.options:
        return _numerical_path(ctx, ct, rp)

    is_temporal = _is_temporal_question(ctx.question)
    rel_names = _extract_question_entities(ctx.question, ctx.options)

    # P0: Build Belief
    rp.append("[P0:belief]")
    cr = coder_tool(ctx, ct) if ct else ''
    if cr and 'not found' in cr.lower():
        mod, log = _auto_add(ctx, cr)
        if mod: cr = coder_tool(ctx, ct)
    m_coder_ans = re.search(r'answer=([A-D])', cr)
    coder_ans = m_coder_ans.group(1) if m_coder_ans else ''
    rp.append(f"[coder] {coder_ans}")

    # P1: Global VL
    rp.append("[P1:vl_global]")
    vl_prompt = _build_vl_independent_prompt(ctx)
    vl_global = _clean(ctx.vl.call(vl_prompt, ctx.video_path, max_tokens=128), ctx)
    ctx.vl_calls += 1
    rp.append(f"[vl_full] {vl_global}")

    # Iterative Loop — use quality-scored frames
    std_thresholds = [0.4, 0.3, 0.2, 0.15, 0.1]
    vl_history = [('global', vl_global)]
    prev_frames = set()
    prev_focused_answer = None
    converge_type = None
    n_rounds = 0

    for round_id in range(1, max_rounds + 1):
        rp.append(f"[R{round_id}]")
        # === KEY CHANGE: quality-scored frame selection ===
        frames, ents, ftype = _select_frames_quality(ctx, is_temporal)
        cur_frames = set(frames) if frames else set()

        if round_id > 1 and cur_frames == prev_frames:
            rp.append(f"[R{round_id}:no_new_frames]"); break

        vl_ans, _ = _vl_on_frames(ctx, frames, ents, ftype, is_temporal, rp, round_id)
        vl_history.append((f'R{round_id}', vl_ans))
        n_rounds = round_id

        if vl_ans == vl_global and vl_ans in 'ABCD':
            converge_type = 'global_consensus'
            rp.append(f"[R{round_id}:global_consensus] {vl_ans}"); break

        if prev_focused_answer is not None and vl_ans == prev_focused_answer and vl_ans in 'ABCD':
            converge_type = 'evolution_stable'
            rp.append(f"[R{round_id}:evolution_stable] {vl_ans}"); break

        threshold = std_thresholds[min(round_id - 1, len(std_thresholds) - 1)]
        _evolve_belief(ctx, rel_names, threshold, rp)
        prev_frames = cur_frames
        prev_focused_answer = vl_ans

    # Final Decision (same as V20)
    all_valid = [(ans, i) for i, (src, ans) in enumerate(vl_history) if ans in 'ABCD']

    if converge_type == 'global_consensus':
        ans = vl_global
    elif converge_type == 'evolution_stable':
        ans = prev_focused_answer
    elif not all_valid:
        ans = 'A'
    else:
        weighted_counts = Counter()
        for src_ans, idx in all_valid:
            src = vl_history[idx][0]
            w = max_rounds if src == 'global' else int(src[1:]) + 1
            weighted_counts[src_ans] += w
        ans = weighted_counts.most_common(1)[0][0]
        rp.append(f"[final:weighted_vote] {dict(weighted_counts)}→{ans}")

    rp.append(f"[rounds={n_rounds}]")
    ctx._final_answer = ans
    return ans, " | ".join(rp)


# ============================================================================
# Variant 5: Multi-View Retry on Disagreement
# ============================================================================

def v21_variant5_multiview_retry(ctx, max_rounds=3):
    """V20 + when global and focused disagree, try a third set of frames (non-overlapping)."""
    rp = []
    ct = _auto_coder_type(ctx.question, ctx.options) or ''

    if not ctx.options:
        return _numerical_path(ctx, ct, rp)

    is_temporal = _is_temporal_question(ctx.question)
    rel_names = _extract_question_entities(ctx.question, ctx.options)

    # P0: Build Belief
    rp.append("[P0:belief]")
    cr = coder_tool(ctx, ct) if ct else ''
    if cr and 'not found' in cr.lower():
        mod, log = _auto_add(ctx, cr)
        if mod: cr = coder_tool(ctx, ct)
    m_coder_ans = re.search(r'answer=([A-D])', cr)
    coder_ans = m_coder_ans.group(1) if m_coder_ans else ''
    rp.append(f"[coder] {coder_ans}")

    # P1: Global VL
    rp.append("[P1:vl_global]")
    vl_prompt = _build_vl_independent_prompt(ctx)
    vl_global = _clean(ctx.vl.call(vl_prompt, ctx.video_path, max_tokens=128), ctx)
    ctx.vl_calls += 1
    rp.append(f"[vl_full] {vl_global}")

    # Iterative Loop
    std_thresholds = [0.4, 0.3, 0.2, 0.15, 0.1]
    vl_history = [('global', vl_global)]
    prev_frames = set()
    prev_focused_answer = None
    converge_type = None
    n_rounds = 0
    all_used_frames = set()

    for round_id in range(1, max_rounds + 1):
        rp.append(f"[R{round_id}]")
        frames, ents, ftype = _select_frames(ctx, is_temporal)
        cur_frames = set(frames) if frames else set()
        all_used_frames.update(cur_frames)

        if round_id > 1 and cur_frames == prev_frames:
            rp.append(f"[R{round_id}:no_new_frames]"); break

        vl_ans, _ = _vl_on_frames(ctx, frames, ents, ftype, is_temporal, rp, round_id)
        vl_history.append((f'R{round_id}', vl_ans))
        n_rounds = round_id

        if vl_ans == vl_global and vl_ans in 'ABCD':
            converge_type = 'global_consensus'
            rp.append(f"[R{round_id}:global_consensus] {vl_ans}"); break

        if prev_focused_answer is not None and vl_ans == prev_focused_answer and vl_ans in 'ABCD':
            converge_type = 'evolution_stable'
            rp.append(f"[R{round_id}:evolution_stable] {vl_ans}"); break

        # === KEY CHANGE: Multi-view retry on first disagreement ===
        if round_id == 1 and vl_ans != vl_global:
            # Try to get frames from a different "view" — use entity union frames
            # excluding already-used co-occurrence frames
            if not is_temporal:
                union_frames, _ = _get_entity_union_frames(ctx.grid, ctx.question, ctx.options)
                alt_frames = sorted(set(union_frames) - all_used_frames)
                if len(alt_frames) >= 2:
                    rp.append(f"[R{round_id}:multiview] {len(alt_frames)} alt frames")
                    vl_mv, _ = _vl_on_frames(ctx, alt_frames, ents, ftype, is_temporal, rp, round_id)
                    vl_history.append((f'R{round_id}_mv', vl_mv))
                    all_used_frames.update(alt_frames)

                    # Check for consensus with multi-view
                    if vl_mv == vl_global and vl_mv in 'ABCD':
                        converge_type = 'global_consensus'
                        rp.append(f"[R{round_id}:mv_confirms_global] {vl_mv}"); break
                    if vl_mv == vl_ans and vl_mv in 'ABCD':
                        converge_type = 'evolution_stable'
                        rp.append(f"[R{round_id}:mv_confirms_focused] {vl_mv}"); break

        threshold = std_thresholds[min(round_id - 1, len(std_thresholds) - 1)]
        _evolve_belief(ctx, rel_names, threshold, rp)
        prev_frames = cur_frames
        prev_focused_answer = vl_ans

    # Final Decision
    all_valid = [(ans, i) for i, (src, ans) in enumerate(vl_history) if ans in 'ABCD']

    if converge_type == 'global_consensus':
        ans = vl_global
    elif converge_type == 'evolution_stable':
        ans = prev_focused_answer
    elif not all_valid:
        ans = 'A'
    else:
        weighted_counts = Counter()
        for src_ans, idx in all_valid:
            src = vl_history[idx][0]
            if src == 'global':
                w = max_rounds
            elif '_mv' in src:
                w = 2  # multi-view gets moderate weight
            else:
                w = int(src[1]) + 1
            weighted_counts[src_ans] += w
        ans = weighted_counts.most_common(1)[0][0]
        rp.append(f"[final:weighted_vote] {dict(weighted_counts)}→{ans}")

    rp.append(f"[rounds={n_rounds}]")
    ctx._final_answer = ans
    return ans, " | ".join(rp)


# ============================================================================
# V20 Baseline (for comparison)
# ============================================================================

from scripts.grid64_agentic_pipeline_v20 import v20_loop as v20_baseline


# ============================================================================
# Test Runner
# ============================================================================

VARIANT_MAP = {
    'v20': ('V20 Baseline', v20_baseline),
    '1': ('V21-1: CODER Soft Vote', v21_variant1_coder_soft_vote),
    '3': ('V21-3: Disagreement-Driven Evolution', v21_variant3_disagreement_evolution),
    '4': ('V21-4: Frame Quality Scoring', v21_variant4_frame_quality),
    '5': ('V21-5: Multi-View Retry', v21_variant5_multiview_retry),
}

CHOICE_TYPES = {
    'object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard',
    'object_rel_distance', 'obj_appearance_order', 'route_planning',
}


def run_variant(variant_name, variant_fn, samples_by_scene, pipe, max_rounds=3):
    """Run a variant on all samples, return results list."""
    all_results = []
    scene_list = sorted(samples_by_scene.keys())

    for si, sn in enumerate(scene_list):
        samples = samples_by_scene[sn]
        vp = find_video_path(sn)
        if not vp:
            for s in samples:
                all_results.append({
                    'scene_name': sn, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []), 'prediction': '0', 'score': 0.0,
                })
            continue

        logger.info(f"  [{si+1}/{len(scene_list)}] {sn} ({len(samples)} q)")

        # Build grid once per scene
        grid = pipe.builder.build_grid_fps(vp, fps=UNIFIED_FPS, max_frames=pipe.grid_max_frames)

        for sample in samples:
            q = sample['question']; opts = sample.get('options') or []
            gt = sample['ground_truth']; qt = sample['question_type']

            gc_ = copy.deepcopy(grid)
            ctx = ToolExecutionContext(gc_, pipe.vl, vp, pipe.builder, q, opts, qt)
            t0 = time.time()

            try:
                ans, reasoning = variant_fn(ctx, max_rounds=max_rounds)
            except Exception as e:
                logger.error(f"    Error: {e}"); traceback.print_exc()
                ans = 'A'; reasoning = f"[error] {e}"

            elapsed = time.time() - t0
            score = evaluate_sample(qt, ans, gt)

            all_results.append({
                'scene_name': sn, 'question_type': qt,
                'question': q, 'ground_truth': gt, 'options': opts,
                'prediction': ans, 'score': score,
                'reasoning': reasoning, 'vl_calls': ctx.vl_calls,
                'elapsed_s': round(elapsed, 1),
            })

            logger.info(f"    [{qt[:20]:20s}] pred={ans} gt={gt} score={score:.3f} vl={ctx.vl_calls} t={elapsed:.0f}s")

    return all_results


def print_comparison(results_by_variant):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 140)
    print("V21 Variant Comparison — Choice Questions Only")
    print("=" * 140)

    variant_names = list(results_by_variant.keys())

    # Header
    header = f"  {'Task':<30}"
    for vn in variant_names:
        header += f" {vn:>12}"
    print(header)
    print("-" * 140)

    # Get all question types
    all_types = sorted(CHOICE_TYPES)

    for qt in all_types:
        row = f"  {qt:<30}"
        for vn in variant_names:
            results = results_by_variant[vn]
            qr = [r for r in results if r['question_type'] == qt]
            if qr:
                score = np.mean([r['score'] for r in qr])
                row += f" {score:>11.3f}({len(qr)})"
            else:
                row += f" {'—':>12}"
        print(row)

    # Overall choice
    print("-" * 140)
    row = f"  {'CHOICE OVERALL':<30}"
    for vn in variant_names:
        results = results_by_variant[vn]
        qr = [r for r in results if r['question_type'] in CHOICE_TYPES]
        if qr:
            score = np.mean([r['score'] for r in qr])
            row += f" {score:>11.3f}({len(qr)})"
        else:
            row += f" {'—':>12}"
    print(row)

    # All overall
    row = f"  {'ALL OVERALL':<30}"
    for vn in variant_names:
        results = results_by_variant[vn]
        if results:
            score = np.mean([r['score'] for r in results])
            row += f" {score:>11.3f}({len(results)})"
        else:
            row += f" {'—':>12}"
    print(row)

    # VL calls
    row = f"  {'Avg VL Calls':<30}"
    for vn in variant_names:
        results = results_by_variant[vn]
        if results:
            avg_vl = np.mean([r.get('vl_calls', 0) for r in results])
            row += f" {avg_vl:>12.1f}"
        else:
            row += f" {'—':>12}"
    print(row)

    # Avg time
    row = f"  {'Avg Time (s)':<30}"
    for vn in variant_names:
        results = results_by_variant[vn]
        if results:
            avg_t = np.mean([r.get('elapsed_s', 0) for r in results])
            row += f" {avg_t:>12.1f}"
        else:
            row += f" {'—':>12}"
    print(row)

    print("=" * 140)


def get_selected_samples(n_samples=100):
    """Select balanced choice samples."""
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    with open(v7_path) as f: v7_results = json.load(f)

    choice_samples = [s for s in v7_results if s['question_type'] in CHOICE_TYPES and find_video_path(s['scene_name'])]

    by_type = defaultdict(list)
    for s in choice_samples: by_type[s['question_type']].append(s)

    selected = []
    per_type = max(1, n_samples // len(CHOICE_TYPES))
    for qt in sorted(CHOICE_TYPES):
        avail = by_type[qt]
        n = min(per_type, len(avail))
        if n > 0:
            indices = np.linspace(0, len(avail)-1, n, dtype=int)
            for idx in indices:
                selected.append(avail[idx])
    return selected


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V21 Variant Test (8-GPU parallel)")
    parser.add_argument('--variant', type=str, required=True,
                       help='Variant to test: v20, 1, 3, 4, 5')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--max_rounds', type=int, default=3)
    parser.add_argument('--grid_max_frames', type=int, default=64)
    parser.add_argument('--vl-model', type=str,
                       default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    if args.gpu_id is not None:
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        args.device = 'cuda:0' if vis else f'cuda:{args.gpu_id}'

    if args.variant not in VARIANT_MAP:
        logger.error(f"Unknown variant: {args.variant}. Choose from: {list(VARIANT_MAP.keys())}")
        sys.exit(1)

    vname, vfn = VARIANT_MAP[args.variant]
    logger.info(f"Variant: {vname}")

    # Select samples
    selected = get_selected_samples(args.n_samples)
    logger.info(f"Total selected: {len(selected)} samples")

    # Group by scene and partition for multi-GPU
    by_scene = defaultdict(list)
    for s in selected: by_scene[s['scene_name']].append(s)
    scene_list = sorted(by_scene.keys())

    if args.gpu_id is not None:
        total = len(scene_list)
        chunk = (total + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * chunk
        end = min(start + chunk, total)
        my_scenes = scene_list[start:end]
        my_by_scene = {sn: by_scene[sn] for sn in my_scenes}
        my_n = sum(len(v) for v in my_by_scene.values())
        logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: {len(my_scenes)} scenes, {my_n} samples")
    else:
        my_by_scene = by_scene

    # Load model
    vl_model = getattr(args, 'vl_model', None) or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
    pipe = AgenticPipelineV20(device=args.device, vl_model_path=vl_model,
                               max_rounds=args.max_rounds, grid_max_frames=args.grid_max_frames)
    pipe.load_models()

    # Run
    results = run_variant(vname, vfn, my_by_scene, pipe, max_rounds=args.max_rounds)
    pipe.unload()

    # Save per-GPU results
    od = PROJECT_ROOT / "outputs" / "v21_variant_test" / args.variant
    od.mkdir(parents=True, exist_ok=True)

    if args.gpu_id is not None:
        out_file = od / f"gpu{args.gpu_id}.json"
    else:
        out_file = od / "results.json"

    clean = []
    for r in results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)): cr[k] = float(v)
            elif isinstance(v, np.ndarray): cr[k] = v.tolist()
            else: cr[k] = v
        clean.append(cr)
    with open(out_file, 'w') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    # Print quick summary
    choice_r = [r for r in results if r['question_type'] in CHOICE_TYPES]
    if choice_r:
        score = np.mean([r['score'] for r in choice_r])
        logger.info(f"GPU{args.gpu_id or 0} {vname}: {score:.4f} ({len(choice_r)} samples)")

    # Print per-type
    for qt in sorted(CHOICE_TYPES):
        qr = [r for r in results if r['question_type'] == qt]
        if qr:
            s = np.mean([r['score'] for r in qr])
            logger.info(f"  {qt}: {s:.4f} ({len(qr)})")


if __name__ == '__main__':
    main()
