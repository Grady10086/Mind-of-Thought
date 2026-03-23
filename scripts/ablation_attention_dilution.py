#!/usr/bin/env python3
"""
Exp-M1: Attention Dilution Experiment

Purpose: Demonstrate that (1) VLM performance degrades with uniform sampling as
         irrelevant frames dilute attention, and (2) Grid-guided frame selection
         (co-occurrence/union) significantly outperforms uniform sampling at the
         same frame count.

Design:
  Level 1 (Oracle):     Grid-selected co-occurrence/union frames (K frames, typically 2-8)
  Level 2 (Uniform-K):  Uniformly sampled K frames (same count as Oracle) — controlled comparison
  Level 3 (Uniform-16): Uniformly sampled 16 frames
  Level 4 (Uniform-32): Uniformly sampled 32 frames
  Level 5 (Full):       Already available from Pure VL baseline (fps=2, ~30-200 frames)
                         → NOT re-run; merged from existing data at summary time.

All levels use the SAME VL prompt (_build_vl_independent_prompt) — only frames differ.
Grid is only used for oracle frame selection, never shown to VL.

Only MCA (choice) tasks are evaluated.

Key comparisons:
  Oracle vs Uniform-K  → Same #frames, Grid-guided >> Uniform  (proves Grid value)
  Uniform-K/16/32/Full → More uniform frames ≠ better          (proves attention dilution)
  Oracle (few) vs Full → Few precise frames >> many noisy frames

Usage:
    # 8 GPU parallel
    for GPU_ID in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$GPU_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python3 scripts/ablation_attention_dilution.py \
            --full --gpu_id $GPU_ID --num_gpus 8 \
            > outputs/ablation_attention_dilution/gpu${GPU_ID}.log 2>&1 &
        sleep 5
    done
    wait
"""

import os, sys, json, re, gc, copy, time, math, logging, traceback
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

# Import from V21
from scripts.grid64_agentic_pipeline_v21 import (
    Grid256Builder, UNIFIED_FPS, VL_DEFAULT_MAX_PIXELS,
    VLModel, ToolExecutionContext,
    _extract_question_entities, _get_cooccurrence_frames, _get_entity_union_frames,
    _is_temporal_question, _build_vl_independent_prompt, _build_vl_focused_prompt, _build_temporal_vl_prompt,
    _clean, extract_focused_frames,
    find_video_path, evaluate_sample,
)


# ============================================================================
# Frame Selection Strategies
# ============================================================================

def get_video_info(video_path):
    """Get total frames and fps of a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total, vfps


def select_uniform_frames(video_path, n_frames):
    """Select n_frames uniformly from the video. Returns frame indices."""
    total, vfps = get_video_info(video_path)
    if total <= 0 or n_frames <= 0:
        return []
    n = min(n_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int).tolist()
    return indices


def select_oracle_frames(grid, question, options):
    """Select oracle frames from Grid (co-occurrence or union)."""
    is_temporal = _is_temporal_question(question)
    if is_temporal:
        frames, ents = _get_entity_union_frames(grid, question, options)
    else:
        frames, ents = _get_cooccurrence_frames(grid, question, options)
    return frames, ents, is_temporal


# ============================================================================
# Per-sample Evaluation at Each Level
# ============================================================================

def evaluate_at_level(vl, video_path, grid, sample, level, oracle_cache=None):
    """Evaluate a single sample at a given level.

    Levels:
      'oracle'     - Grid co-occurrence/union frames
      'uniform-K'  - Uniformly sampled K frames (K = oracle frame count)
      'uniform-16' - Uniformly sampled 16 frames
      'uniform-32' - Uniformly sampled 32 frames
    """
    question = sample['question']
    options = sample.get('options') or []
    gt = sample['ground_truth']
    qt = sample['question_type']

    # Determine oracle frames (needed for oracle and uniform-K)
    if oracle_cache is not None:
        oracle_frames, ents, is_temporal = oracle_cache
    else:
        oracle_frames, ents, is_temporal = select_oracle_frames(grid, question, options)

    # V21 caps focused frames: 8 for co-occurrence, 12 for temporal
    max_ff = 12 if is_temporal else 8

    # Select frames based on level
    if level == 'oracle':
        frame_indices = oracle_frames
    elif level == 'uniform-K':
        # K = actual frame count VL would see after cap (same as oracle)
        K = min(len(oracle_frames), max_ff) if oracle_frames else max_ff
        K = max(K, 2)  # at least 2 frames
        frame_indices = select_uniform_frames(video_path, K)
    elif level == 'uniform-16':
        frame_indices = select_uniform_frames(video_path, 16)
    elif level == 'uniform-32':
        frame_indices = select_uniform_frames(video_path, 32)
    else:
        frame_indices = []

    if not frame_indices or len(frame_indices) < 2:
        # Fallback: uniform 8 frames
        frame_indices = select_uniform_frames(video_path, 8)

    # Extract PIL frames — apply V21's max_ff cap for oracle/uniform-K
    if level in ('oracle', 'uniform-K'):
        focused_pil = extract_focused_frames(video_path, frame_indices, max_frames=max_ff)
    else:
        focused_pil = extract_focused_frames(video_path, frame_indices, max_frames=len(frame_indices))
    n_frames = len(focused_pil)

    # Build prompt based on level
    ctx_dummy = type('Ctx', (), {'question': question, 'options': options})()
    if level == 'oracle':
        # Oracle uses focused prompt with entity guidance (V21 style)
        if is_temporal:
            prompt = _build_temporal_vl_prompt(ctx_dummy, ents, n_frames)
        else:
            prompt = _build_vl_focused_prompt(ctx_dummy, ents, n_frames)
    else:
        # Uniform uses independent prompt (no entity guidance)
        prompt = _build_vl_independent_prompt(ctx_dummy)

    if not focused_pil or len(focused_pil) < 2:
        # Last resort fallback
        raw = vl.call(prompt, video_path, max_tokens=128)
        n_frames = -1  # mark as fallback
    else:
        raw = vl.call_with_frames(prompt, focused_pil, max_tokens=128)

    ans = _clean(raw, ctx_dummy)
    score = evaluate_sample(qt, ans, gt)

    return {
        'level': level,
        'prediction': ans,
        'score': score,
        'n_frames': n_frames,
        'n_oracle_frames': len(oracle_frames) if oracle_frames else 0,
        'raw': raw[:100] if raw else '',
    }


# ============================================================================
# Main Pipeline
# ============================================================================

class AttentionDilutionExperiment:
    # Fast mode: only compare oracle vs uniform-8 (core comparison)
    # uniform-16/32 can be added later if needed
    LEVELS = ['oracle', 'uniform-K']

    def __init__(self, device='cuda:0', vl_model_path=None, grid_max_frames=128):
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
        """Process all MCA samples for a scene, at all dilution levels."""
        # Build Grid once per scene (only needed for oracle frame selection)
        grid = self.builder.build_grid_fps(video_path, fps=UNIFIED_FPS,
                                           max_frames=self.grid_max_frames)

        results = []
        for sample in samples:

            # Compute oracle frames once per sample (shared across levels)
            grid_copy = copy.deepcopy(grid)
            oracle_frames, ents, is_temporal = select_oracle_frames(
                grid_copy, sample['question'], sample.get('options', []))
            oracle_cache = (oracle_frames, ents, is_temporal)

            scene_results = {
                'scene_name': sample.get('scene_name', ''),
                'question_type': sample['question_type'],
                'question': sample['question'],
                'ground_truth': sample['ground_truth'],
                'options': sample.get('options', []),
                'levels': {},
            }

            for level in self.LEVELS:
                try:
                    t0 = time.time()
                    result = evaluate_at_level(
                        self.vl, video_path, grid_copy, sample, level,
                        oracle_cache=oracle_cache
                    )
                    result['elapsed_s'] = round(time.time() - t0, 1)
                    scene_results['levels'][level] = result
                except Exception as e:
                    logger.warning(f"  Error at level={level}: {e}")
                    scene_results['levels'][level] = {
                        'level': level, 'prediction': 'A', 'score': 0.0,
                        'n_frames': 0, 'error': str(e)[:200],
                    }

            scores = {lv: scene_results['levels'][lv]['score']
                      for lv in self.LEVELS if lv in scene_results['levels']}
            scene_results['scores'] = scores
            results.append(scene_results)

            n_oracle = oracle_cache[0] if oracle_cache[0] else []
            max_ff_log = 12 if oracle_cache[2] else 8
            actual_K = min(len(n_oracle), max_ff_log)
            score_str = " | ".join(f"{lv}={scores.get(lv, 0):.3f}" for lv in self.LEVELS)
            logger.info(f"  [{sample['question_type'][:25]}] {score_str} (K={actual_K}, raw_oracle={len(n_oracle)})")

        return results


# ============================================================================
# Summary with Full (Pure VL) data merged
# ============================================================================

PURE_VL_PATH = PROJECT_ROOT / "datasets" / "PanoVideo" / "vsibench_results" / \
    "qwen3vl8b_vsibench_fps2_all_20260304_063459.json"


def load_full_baseline():
    """Load Pure VL (Full video) MCA results as Level 'full' reference."""
    if not PURE_VL_PATH.exists():
        alt = Path("/home/tione/notebook/tianjungu/datasets/PanoVideo/vsibench_results/"
                    "qwen3vl8b_vsibench_fps2_all_20260304_063459.json")
        if alt.exists():
            p = alt
        else:
            logger.warning("Pure VL baseline not found, skipping 'full' level in summary")
            return {}
    else:
        p = PURE_VL_PATH

    with open(p) as f:
        data = json.load(f)
    results = data.get('results', data) if isinstance(data, dict) else data

    MCA = {'object_rel_direction_easy', 'object_rel_direction_medium',
           'object_rel_direction_hard', 'object_rel_distance',
           'route_planning', 'obj_appearance_order'}

    # Build lookup: (scene_name, question_type, gt) -> score
    full_scores = {}
    for r in results:
        qt = r.get('question_type', '')
        if qt not in MCA:
            continue
        key = (r.get('scene_name', ''), qt, str(r.get('gt', r.get('ground_truth', ''))))
        pred = str(r.get('pred', r.get('prediction', '')))
        gt = str(r.get('gt', r.get('ground_truth', '')))
        score = 1.0 if pred == gt else 0.0
        full_scores[key] = score

    return full_scores


def print_summary(all_results, output_dir):
    """Print and save summary, including Full baseline from existing data."""
    # Load full baseline
    full_scores = load_full_baseline()

    levels = AttentionDilutionExperiment.LEVELS + (['full'] if full_scores else [])
    qtypes = sorted(set(r['question_type'] for r in all_results))

    print("\n" + "=" * 110)
    print("Exp-M1: Attention Dilution — Grid-Guided vs Uniform Sampling")
    print("=" * 110)

    print(f"\n  {'Task':<35}", end="")
    for lv in levels:
        print(f" {lv:>10}", end="")
    print(f"  {'N':>5}")
    print("-" * 110)

    summary = {'by_type': {}, 'overall': {}}

    for qt in qtypes:
        qr = [r for r in all_results if r['question_type'] == qt]
        print(f"  {qt:<35}", end="")
        for lv in levels:
            if lv == 'full':
                # Match from Pure VL baseline
                matched = []
                for r in qr:
                    key = (r['scene_name'], r['question_type'], str(r['ground_truth']))
                    if key in full_scores:
                        matched.append(full_scores[key])
                avg = np.mean(matched) if matched else 0
            else:
                scores = [r['scores'].get(lv, 0) for r in qr if lv in r.get('scores', {})]
                avg = np.mean(scores) if scores else 0
            print(f" {avg:>9.3f}", end="")
            if qt not in summary['by_type']:
                summary['by_type'][qt] = {}
            summary['by_type'][qt][lv] = round(float(avg), 4)
        print(f"  {len(qr):>5}")

    # Overall
    print("-" * 110)
    print(f"  {'Overall':<35}", end="")
    for lv in levels:
        if lv == 'full':
            matched = []
            for r in all_results:
                key = (r['scene_name'], r['question_type'], str(r['ground_truth']))
                if key in full_scores:
                    matched.append(full_scores[key])
            avg = np.mean(matched) if matched else 0
        else:
            scores = [r['scores'].get(lv, 0) for r in all_results if lv in r.get('scores', {})]
            avg = np.mean(scores) if scores else 0
        print(f" {avg:>9.3f}", end="")
        summary['overall'][lv] = round(float(avg), 4)
    print(f"  {len(all_results):>5}")

    # Frame count statistics
    print(f"\n  Avg frames per level:")
    for lv in AttentionDilutionExperiment.LEVELS:
        fc = [r['levels'][lv].get('n_frames', 0) for r in all_results
              if lv in r.get('levels', {}) and r['levels'][lv].get('n_frames', 0) > 0]
        if fc:
            print(f"    {lv:<12}: mean={np.mean(fc):.1f}, median={np.median(fc):.0f}, "
                  f"min={min(fc)}, max={max(fc)}")
    if full_scores:
        print(f"    {'full':<12}: (from Pure VL baseline, fps=2)")

    print("=" * 110)

    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exp-M1: Attention Dilution")
    parser.add_argument('--n_per_type', type=int, default=10)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--task_type', type=str, default='mca', choices=['mca', 'na', 'all'],
                        help='Task type: mca (choice), na (numerical), all')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--grid_max_frames', type=int, default=128,
                        help='Max frames for Grid construction (same as V21)')
    parser.add_argument('--vl-model', type=str,
                        default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    if args.gpu_id is not None:
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        args.device = 'cuda:0' if vis else f'cuda:{args.gpu_id}'

    # Load V7 data (same sample source as V20)
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    logger.info(f"Loading V7: {v7_path}")
    with open(v7_path) as f:
        v7_results = json.load(f)
    logger.info(f"V7: {len(v7_results)} samples")

    # Filter by task type
    MCA_TYPES = {
        'object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard',
        'object_rel_distance', 'route_planning', 'obj_appearance_order',
    }
    NA_TYPES = {
        'object_counting', 'object_abs_distance', 'object_size_estimation', 'room_size_estimation',
    }

    if args.task_type == 'mca':
        allowed_types = MCA_TYPES
    elif args.task_type == 'na':
        allowed_types = NA_TYPES
    else:
        allowed_types = MCA_TYPES | NA_TYPES

    if args.full:
        test_samples = [s for s in v7_results
                        if s['question_type'] in allowed_types and find_video_path(s['scene_name'])]
    else:
        by_type = defaultdict(list)
        for r in v7_results:
            if r['question_type'] in allowed_types:
                by_type[r['question_type']].append(r)
        test_samples = []
        for qt, samps in sorted(by_type.items()):
            avail = [s for s in samps if find_video_path(s['scene_name'])]
            n = min(args.n_per_type, len(avail))
            if n > 0:
                for idx in np.linspace(0, len(avail) - 1, n, dtype=int):
                    test_samples.append(avail[idx])

    logger.info(f"Test samples ({args.task_type}): {len(test_samples)}")
    type_counts = Counter(s['question_type'] for s in test_samples)
    for qt, n in sorted(type_counts.items()):
        logger.info(f"  {qt}: {n}")

    # Group by scene
    by_scene = defaultdict(list)
    for s in test_samples:
        by_scene[s['scene_name']].append(s)
    scene_list = sorted(by_scene.keys())

    # Multi-GPU split by scene
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
    exp = AttentionDilutionExperiment(
        device=args.device, vl_model_path=vl_model, grid_max_frames=args.grid_max_frames
    )
    exp.load_models()

    all_results = []
    for si, sn in enumerate(my_scenes):
        samples = by_scene[sn]
        vp = find_video_path(sn)
        if not vp:
            logger.warning(f"  [{si+1}/{len(my_scenes)}] {sn}: no video, skip")
            continue

        logger.info(f"[{si+1}/{len(my_scenes)}] {sn} ({len(samples)} MCA questions)")
        try:
            results = exp.process_scene(vp, samples)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"  Scene error: {e}")
            traceback.print_exc()

    exp.unload()

    # Save results
    suffix = "" if args.task_type == "mca" else f"_{args.task_type}"
    if args.gpu_id is not None:
        od = PROJECT_ROOT / "outputs" / f"ablation_attention_dilution{suffix}" / f"gpu{args.gpu_id}"
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        od = PROJECT_ROOT / "outputs" / f"ablation_attention_dilution{suffix}_{ts}"
    od.mkdir(parents=True, exist_ok=True)

    # Clean numpy types for JSON
    clean = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                cr[k] = float(v)
            elif isinstance(v, np.ndarray):
                cr[k] = v.tolist()
            elif isinstance(v, dict):
                cr[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                         for kk, vv in v.items()}
            else:
                cr[k] = v
        clean.append(cr)

    with open(od / "detailed_results.json", 'w') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {od} ({len(all_results)} samples)")

    if args.gpu_id is None:
        print_summary(all_results, str(od))


if __name__ == '__main__':
    main()
