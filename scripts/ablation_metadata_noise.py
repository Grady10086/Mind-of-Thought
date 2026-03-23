#!/usr/bin/env python3
"""
Exp-M2: Metadata Noise / Hypothesis Anchoring Experiment

Purpose: Demonstrate that (1) blindly injecting 3D metadata misleads VLM,
         (2) V7 VL's gains come from prompt engineering not Grid itself,
         (3) MindCube's Active Focusing is far superior to any form of injection.

Conditions:
  A: Pure VL             — bare Qwen3-VL fps=2, no Grid info          → Already done: 0.509
  B: VL + Raw Grid Dump  — inject raw 3D entity data into VL prompt    → THIS SCRIPT
  C: VL + V7 Engineered  — inject carefully packaged Grid info         → Already done: 0.636 (V7 VL)
  D: VL + CODER Answer   — inject CODER's computed answer hypothesis   → THIS SCRIPT
  E: MindCube (V20)      — Grid guides frames only, no injection       → Already done: 0.693

This script runs conditions B and D (both need Grid construction + full video VL).

Usage:
    # Run condition B (Raw Grid Dump)
    for GPU_ID in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$GPU_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python3 scripts/ablation_metadata_noise.py \
            --condition B --full --gpu_id $GPU_ID --num_gpus 8 \
            > outputs/ablation_metadata_noise_B/gpu${GPU_ID}.log 2>&1 &
        sleep 5
    done; wait

    # Run condition D (CODER Answer Injection)
    for GPU_ID in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$GPU_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python3 scripts/ablation_metadata_noise.py \
            --condition D --full --gpu_id $GPU_ID --num_gpus 8 \
            > outputs/ablation_metadata_noise_D/gpu${GPU_ID}.log 2>&1 &
        sleep 5
    done; wait
"""

import os, sys, json, re, gc, copy, time, math, logging, traceback
import base64, io
import numpy as np, cv2, torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime
from PIL import Image

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

from scripts.grid64_agentic_pipeline_v21 import (
    Grid256Builder, UNIFIED_FPS, VL_DEFAULT_MAX_PIXELS,
    VLModel, ToolExecutionContext,
    _extract_question_entities, _clean, _auto_coder_type, _auto_add,
    _build_vl_independent_prompt, _build_numerical_vl_prompt,
    _generate_spatial_hypothesis,
    coder_tool, find_video_path, evaluate_sample, mean_relative_accuracy,
    _get_video_fps_nframes, _match_name,
)


# ============================================================================
# Condition B: Raw Grid Dump Prompt
# ============================================================================

def _build_raw_grid_dump_prompt(ctx, grid):
    """Condition B: Inject raw Grid 3D data directly into VL prompt.
    No packaging, no instructions, no reference sizes — just raw data."""

    # Serialize Grid entities
    entity_lines = []
    for eid, e in grid.entities.items():
        pos = e.position_3d
        if pos is None:
            continue
        # Raw 3D position
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m"

        # Size if available
        size_str = ""
        if e.size_3d is not None:
            size_str = f", size=({e.size_3d[0]:.2f}, {e.size_3d[1]:.2f}, {e.size_3d[2]:.2f})m"

        # Detection count and confidence
        n_det = len(e.detections) if e.detections else 0
        conf = e.confidence if e.confidence else 0

        # Frame indices
        frame_idxs = sorted(set(d.get('frame_idx', -1) for d in (e.detections or []) if d.get('frame_idx', -1) >= 0))
        frames_str = f", frames={frame_idxs[:10]}" if frame_idxs else ""

        entity_lines.append(
            f"  {e.category}: position={pos_str}{size_str}, count={n_det}, conf={conf:.2f}{frames_str}"
        )

    # Camera trajectory
    cam_str = ""
    if grid.camera_positions:
        cam_points = []
        for c in grid.camera_positions[:10]:
            p = c.get('position')
            if p is not None and len(p) >= 3:
                cam_points.append(f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
        if cam_points:
            cam_str = f"\nCamera trajectory: [{', '.join(cam_points)}]"

    # Scale factor
    scale_str = ""
    if hasattr(grid, 'scale_factor') and grid.scale_factor:
        scale_str = f"\nScale factor: {grid.scale_factor:.3f}"

    grid_dump = f"""=== 3D SPATIAL GRID ({grid.GRID_SIZE}³{scale_str}) ===
Entities detected:
{chr(10).join(entity_lines) if entity_lines else '  (no entities detected)'}
{cam_str}"""

    # Build full prompt
    if ctx.options:
        opts = f"\n=== OPTIONS ===\n{chr(10).join(ctx.options)}"
        inst = "Answer with ONLY the option letter (A, B, C, or D)."
    else:
        opts = ""
        inst = "Respond with ONLY a single number (no units, no explanation)."

    return f"""You are analyzing a video of an indoor scene.

{grid_dump}

=== QUESTION ===
{ctx.question}
{opts}

{inst}

Answer:"""


# ============================================================================
# Condition D: CODER Answer Injection Prompt
# ============================================================================

def _build_coder_injection_prompt(ctx, coder_type, coder_result, hypothesis):
    """Condition D: Inject CODER's computed answer hypothesis into VL prompt.
    No scale_ref — only the CODER hypothesis is injected."""

    if ctx.options:
        opts = f"\n=== OPTIONS ===\n{chr(10).join(ctx.options)}"
        inst = "Answer with ONLY the option letter (A, B, C, or D)."
    else:
        opts = ""
        inst = "Respond with ONLY a single number (no units, no explanation)."

    hyp_section = ""
    if hypothesis:
        hyp_section = f"""
=== 3D SPATIAL ANALYSIS RESULT ===
{hypothesis}
Based on the above 3D analysis, consider this as a strong reference for your answer."""

    return f"""You are analyzing a video of an indoor scene.
{hyp_section}

=== QUESTION ===
{ctx.question}
{opts}

Watch the video carefully and combine your visual observation with the 3D analysis result.
{inst}

Answer:"""


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_condition_B(vl, video_path, grid, sample, builder):
    """Condition B: Raw Grid Dump injection, VL sees full video."""
    question = sample['question']
    options = sample.get('options') or []
    gt = sample['ground_truth']
    qt = sample['question_type']

    ctx = type('Ctx', (), {
        'question': question, 'options': options, 'question_type': qt,
        'grid': grid, 'vl': vl, 'video_path': video_path,
        'builder': builder, 'tool_trace': [], 'vl_calls': 0, '_final_answer': None,
    })()

    prompt = _build_raw_grid_dump_prompt(ctx, grid)
    raw = vl.call(prompt, video_path, max_tokens=128)
    ans = _clean(raw, ctx)
    score = evaluate_sample(qt, ans, gt)

    return {
        'prediction': ans,
        'score': score,
        'raw': raw[:200] if raw else '',
        'prompt_length': len(prompt),
    }


def evaluate_condition_D(vl, video_path, grid, sample, builder):
    """Condition D: CODER Answer injection, VL sees full video."""
    question = sample['question']
    options = sample.get('options') or []
    gt = sample['ground_truth']
    qt = sample['question_type']

    ctx = ToolExecutionContext(grid, vl, video_path, builder, question, options, qt)

    # Run CODER
    ct = _auto_coder_type(question, options) or ''
    if ct:
        cr = coder_tool(ctx, ct)
        if 'not found' in cr.lower():
            mod, log = _auto_add(ctx, cr)
            if mod:
                cr = coder_tool(ctx, ct)
    else:
        cr = ''

    # Generate hypothesis
    hypothesis = _generate_spatial_hypothesis(ctx, ct, cr) if cr else ''

    prompt = _build_coder_injection_prompt(ctx, ct, cr, hypothesis)
    raw = vl.call(prompt, video_path, max_tokens=128)
    ans = _clean(raw, ctx)
    score = evaluate_sample(qt, ans, gt)

    return {
        'prediction': ans,
        'score': score,
        'raw': raw[:200] if raw else '',
        'coder_type': ct,
        'coder_result': cr[:200] if cr else '',
        'hypothesis': hypothesis[:200] if hypothesis else '',
        'prompt_length': len(prompt),
    }


# ============================================================================
# Main Pipeline
# ============================================================================

class MetadataNoiseExperiment:
    def __init__(self, condition, device='cuda:0', vl_model_path=None, grid_max_frames=128):
        assert condition in ('B', 'D'), f"Condition must be B or D, got {condition}"
        self.condition = condition
        self.device = device
        self.vl_model_path = vl_model_path or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        self.builder = Grid256Builder(device=device)
        self.vl = VLModel(device=device)
        self.grid_max_frames = grid_max_frames

    def load_models(self):
        self.builder.load_models()
        self.vl.load(self.vl_model_path)

    def unload(self):
        self.builder.unload()
        self.vl.unload()

    def process_scene(self, video_path, samples):
        """Process all samples for a scene under the given condition."""
        # Build Grid once per scene
        grid = self.builder.build_grid_fps(video_path, fps=UNIFIED_FPS,
                                           max_frames=self.grid_max_frames)
        results = []
        for sample in samples:
            t0 = time.time()
            try:
                grid_copy = copy.deepcopy(grid)
                if self.condition == 'B':
                    result = evaluate_condition_B(self.vl, video_path, grid_copy, sample, self.builder)
                else:  # D
                    result = evaluate_condition_D(self.vl, video_path, grid_copy, sample, self.builder)
            except Exception as e:
                logger.warning(f"  Error: {e}")
                result = {'prediction': 'A' if sample.get('options') else '0',
                          'score': 0.0, 'error': str(e)[:200]}

            elapsed = time.time() - t0
            result.update({
                'scene_name': sample.get('scene_name', ''),
                'question_type': sample['question_type'],
                'question': sample['question'],
                'ground_truth': sample['ground_truth'],
                'options': sample.get('options', []),
                'condition': self.condition,
                'elapsed_s': round(elapsed, 1),
            })
            results.append(result)

            logger.info(f"  [{sample['question_type'][:25]}] score={result['score']:.3f} "
                        f"pred={result['prediction'][:10]} gt={sample['ground_truth'][:10]}")

        return results


def print_summary(all_results, condition, output_dir):
    """Print and save summary."""
    print("\n" + "=" * 100)
    print(f"Exp-M2-{condition}: {'Raw Grid Dump Injection' if condition == 'B' else 'CODER Answer Injection'}")
    print("=" * 100)

    qtypes = sorted(set(r['question_type'] for r in all_results))

    # Separate MCA and NA
    mca_types = {'object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard',
                 'object_rel_distance', 'route_planning', 'obj_appearance_order'}
    na_types = {'object_counting', 'object_abs_distance', 'object_size_estimation', 'room_size_estimation'}

    mca_results = [r for r in all_results if r['question_type'] in mca_types]
    na_results = [r for r in all_results if r['question_type'] in na_types]

    summary = {'condition': condition, 'by_type': {}}

    print(f"\n  {'Task':<40} {'Score':>8} {'N':>5}")
    print("-" * 60)

    for qt in qtypes:
        qr = [r for r in all_results if r['question_type'] == qt]
        avg = np.mean([r['score'] for r in qr])
        tag = 'MCA' if qt in mca_types else 'NA'
        print(f"  [{tag}] {qt:<35} {avg:>7.4f} {len(qr):>5}")
        summary['by_type'][qt] = {'score': round(float(avg), 4), 'n': len(qr)}

    # Overall by category
    if mca_results:
        mca_avg = np.mean([r['score'] for r in mca_results])
        print(f"\n  {'MCA Overall':<40} {mca_avg:>7.4f} {len(mca_results):>5}")
        summary['mca_overall'] = round(float(mca_avg), 4)

    if na_results:
        na_avg = np.mean([r['score'] for r in na_results])
        print(f"  {'NA Overall':<40} {na_avg:>7.4f} {len(na_results):>5}")
        summary['na_overall'] = round(float(na_avg), 4)

    overall = np.mean([r['score'] for r in all_results])
    print(f"\n  {'OVERALL':<40} {overall:>7.4f} {len(all_results):>5}")
    summary['overall'] = round(float(overall), 4)

    # Comparison context
    print(f"\n  Reference (from prior experiments):")
    print(f"    Condition A (Pure VL):          0.5090")
    print(f"    Condition C (V7 Engineered):    0.6361")
    print(f"    Condition E (MindCube V20):     0.6933")

    print("=" * 100)

    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exp-M2: Metadata Noise")
    parser.add_argument('--condition', type=str, required=True, choices=['B', 'D'])
    parser.add_argument('--n_per_type', type=int, default=10)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--grid_max_frames', type=int, default=128)
    parser.add_argument('--vl-model', type=str,
                        default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    if args.gpu_id is not None:
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        args.device = 'cuda:0' if vis else f'cuda:{args.gpu_id}'

    # Load V7 data
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    logger.info(f"Loading V7: {v7_path}")
    with open(v7_path) as f:
        v7_results = json.load(f)
    logger.info(f"V7: {len(v7_results)} samples")

    if args.full:
        test_samples = [s for s in v7_results if find_video_path(s['scene_name'])]
    else:
        by_type = defaultdict(list)
        for r in v7_results:
            by_type[r['question_type']].append(r)
        test_samples = []
        for qt, samps in sorted(by_type.items()):
            avail = [s for s in samps if find_video_path(s['scene_name'])]
            n = min(args.n_per_type, len(avail))
            if n > 0:
                for idx in np.linspace(0, len(avail) - 1, n, dtype=int):
                    test_samples.append(avail[idx])

    logger.info(f"Test samples: {len(test_samples)}")
    type_counts = Counter(s['question_type'] for s in test_samples)
    for qt, n in sorted(type_counts.items()):
        logger.info(f"  {qt}: {n}")

    # Group by scene
    by_scene = defaultdict(list)
    for s in test_samples:
        by_scene[s['scene_name']].append(s)
    scene_list = sorted(by_scene.keys())

    # Multi-GPU split
    if args.gpu_id is not None:
        total = len(scene_list)
        chunk = (total + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * chunk
        end = min(start + chunk, total)
        my_scenes = scene_list[start:end]
        logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: {len(my_scenes)} scenes")
    else:
        my_scenes = scene_list

    # Run experiment
    vl_model = getattr(args, 'vl_model', None) or args.vl_model
    exp = MetadataNoiseExperiment(
        condition=args.condition, device=args.device,
        vl_model_path=vl_model, grid_max_frames=args.grid_max_frames,
    )
    exp.load_models()

    all_results = []
    for si, sn in enumerate(my_scenes):
        samples = by_scene[sn]
        vp = find_video_path(sn)
        if not vp:
            logger.warning(f"  [{si+1}/{len(my_scenes)}] {sn}: no video, skip")
            continue

        logger.info(f"[{si+1}/{len(my_scenes)}] {sn} ({len(samples)} q)")
        try:
            results = exp.process_scene(vp, samples)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"  Scene error: {e}")
            traceback.print_exc()

    exp.unload()

    # Save results
    cond = args.condition
    if args.gpu_id is not None:
        od = PROJECT_ROOT / "outputs" / f"ablation_metadata_noise_{cond}" / f"gpu{args.gpu_id}"
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        od = PROJECT_ROOT / "outputs" / f"ablation_metadata_noise_{cond}_{ts}"
    od.mkdir(parents=True, exist_ok=True)

    # Clean numpy types
    clean = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                cr[k] = float(v)
            elif isinstance(v, np.ndarray):
                cr[k] = v.tolist()
            else:
                cr[k] = v
        clean.append(cr)

    with open(od / "detailed_results.json", 'w') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {od} ({len(all_results)} samples)")

    if args.gpu_id is None:
        print_summary(all_results, cond, str(od))


if __name__ == '__main__':
    main()
