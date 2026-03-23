#!/usr/bin/env python3
"""
V21-Confidence: Break False Consensus with Logit Confidence

Key insight from V20 case analysis:
  - 73% of samples reach "global consensus" at R1 (global VL == focused VL)
  - But 27% of those are FALSE consensus (both wrong, happen to agree)
  - Current V20 trusts global_consensus unconditionally → trapped in false consensus

Solution:
  - After detecting global_consensus, check the LOGIT CONFIDENCE of the VL answer
  - If confidence is LOW (prob < threshold), DON'T trust the consensus → continue evolving
  - Only trust consensus when VL is actually confident about its answer

Confidence measurement:
  - Forward pass with the prompt, get logits for first generated token
  - Compare P(answered_letter) vs P(other_letters)
  - Confidence = P(answer) / sum(P(A,B,C,D))

Run: python3 scripts/grid64_agentic_pipeline_v21_confidence.py --variant {conf,v20} --gpu_id X --num_gpus 8
"""

import os, sys, json, re, gc, copy, time, math, logging, traceback
import base64, io
import numpy as np, cv2, torch
import torch.nn.functional as F
from pathlib import Path
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

from scripts.grid64_agentic_pipeline_v20 import (
    Grid256, Grid256Builder,
    VLModel, ToolExecutionContext,
    generate_grid_slice,
    _extract_question_entities, _match_name,
    coder_tool, evolutor_tool, _auto_add,
    _auto_coder_type, _is_temporal_question,
    _get_cooccurrence_frames, _get_entity_union_frames,
    _build_vl_independent_prompt, _build_vl_focused_prompt, _build_temporal_vl_prompt,
    _build_numerical_vl_prompt, _numerical_path,
    _clean, _evolve_belief, _select_frames, _vl_on_frames,
    extract_focused_frames,
    UNIFIED_FPS, GRID_MAX_FRAMES, VL_DEFAULT_MAX_PIXELS,
    AgenticPipelineV20, v20_loop,
)
from scripts.grid64_real_test import find_video_path, evaluate_sample
from qwen_vl_utils import process_vision_info


# ============================================================================
# Confidence-aware VL calls
# ============================================================================

def _get_abcd_token_ids(processor):
    """Get token IDs for A, B, C, D."""
    ids = {}
    for letter in 'ABCD':
        toks = processor.tokenizer.encode(letter, add_special_tokens=False)
        ids[letter] = toks[0] if toks else None
    return ids


def _compute_choice_confidence(model, processor, inputs, abcd_ids):
    """Forward pass to get P(A), P(B), P(C), P(D) for the first generated token.
    Returns dict of {letter: probability} and the top letter."""
    with torch.no_grad():
        outputs = model(**inputs)
        # logits shape: (batch, seq_len, vocab_size)
        # Last position = next token prediction
        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Extract logits for A, B, C, D
        abcd_logits = []
        valid_letters = []
        for letter in 'ABCD':
            tid = abcd_ids.get(letter)
            if tid is not None:
                abcd_logits.append(last_logits[tid].item())
                valid_letters.append(letter)

        if not abcd_logits:
            return {}, '', 0.0

        # Softmax over just ABCD
        abcd_tensor = torch.tensor(abcd_logits)
        probs = F.softmax(abcd_tensor, dim=0).numpy()

        result = {letter: float(p) for letter, p in zip(valid_letters, probs)}
        top_letter = valid_letters[int(np.argmax(probs))]
        top_conf = float(np.max(probs))

        return result, top_letter, top_conf


def vl_call_with_confidence(vl, prompt, video_path, abcd_ids, max_tokens=128):
    """Call VL and return (response_text, confidence_dict, top_letter, top_conf)."""
    if vl.model is None:
        return "", {}, '', 0.0
    try:
        from qwen_vl_utils import process_vision_info
        from scripts.grid64_agentic_pipeline_v20 import _get_video_fps_nframes
        nframes = _get_video_fps_nframes(video_path, UNIFIED_FPS)
        content = [{"type": "video", "video": f"file://{video_path}",
                    "nframes": nframes, "max_pixels": VL_DEFAULT_MAX_PIXELS},
                   {"type": "text", "text": prompt}]
        msgs = [{"role": "user", "content": content}]
        text = vl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(msgs)
        inputs = vl.processor(text=[text], images=img_inputs, videos=vid_inputs,
                              return_tensors="pt", padding=True).to(vl.model.device)

        # 1. Get confidence from forward pass
        conf_dict, conf_letter, conf_val = _compute_choice_confidence(
            vl.model, vl.processor, inputs, abcd_ids)

        # 2. Generate response
        with torch.no_grad():
            outputs = vl.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        resp = vl.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True)[0].strip()

        return resp, conf_dict, conf_letter, conf_val
    except Exception as e:
        logger.warning(f"VL confidence call failed: {e}")
        return "", {}, '', 0.0


def vl_call_frames_with_confidence(vl, prompt, frames_pil, abcd_ids, max_tokens=128):
    """Call VL with frames and return (response_text, confidence_dict, top_letter, top_conf)."""
    if vl.model is None or not frames_pil:
        return "", {}, '', 0.0
    try:
        from qwen_vl_utils import process_vision_info
        content = []
        for f in frames_pil:
            buf = io.BytesIO(); f.save(buf, format='JPEG', quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image", "image": f"data:image/jpeg;base64,{b64}"})
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        text = vl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(msgs)
        inputs = vl.processor(text=[text], images=img_inputs, videos=vid_inputs,
                              return_tensors="pt", padding=True).to(vl.model.device)

        conf_dict, conf_letter, conf_val = _compute_choice_confidence(
            vl.model, vl.processor, inputs, abcd_ids)

        with torch.no_grad():
            outputs = vl.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        resp = vl.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True)[0].strip()

        return resp, conf_dict, conf_letter, conf_val
    except Exception as e:
        logger.warning(f"VL confidence frames call failed: {e}")
        return "", {}, '', 0.0


# ============================================================================
# V21-Confidence Loop
# ============================================================================

CONFIDENCE_THRESHOLD = 0.6  # Below this, don't trust consensus

def _vl_on_frames_conf(ctx, frames, ents, ftype, is_temporal, rp, round_id, abcd_ids):
    """Run VL on focused frames with confidence. Returns (answer, conf, frames_pil)."""
    if not frames or len(frames) < 2:
        si = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
        vl_prompt = _build_vl_independent_prompt(ctx)
        resp, conf_dict, conf_letter, conf_val = vl_call_with_confidence(
            ctx.vl, vl_prompt, ctx.video_path, abcd_ids)
        ans = _clean(resp, ctx)
        ctx.vl_calls += 1
        rp.append(f"[R{round_id}:vl_slice] {ans} conf={conf_val:.2f}")
        return ans, conf_val, None

    max_ff = 12 if is_temporal else 8
    focused_pil = extract_focused_frames(ctx.video_path, frames, max_frames=max_ff)

    if not focused_pil or len(focused_pil) < 2:
        vl_prompt = _build_vl_independent_prompt(ctx)
        resp, conf_dict, conf_letter, conf_val = vl_call_with_confidence(
            ctx.vl, vl_prompt, ctx.video_path, abcd_ids)
        ans = _clean(resp, ctx)
        ctx.vl_calls += 1
        rp.append(f"[R{round_id}:vl_full_fallback] {ans} conf={conf_val:.2f}")
        return ans, conf_val, None

    if is_temporal:
        prompt = _build_temporal_vl_prompt(ctx, ents, len(focused_pil))
    else:
        prompt = _build_vl_focused_prompt(ctx, ents, len(focused_pil))

    resp, conf_dict, conf_letter, conf_val = vl_call_frames_with_confidence(
        ctx.vl, prompt, focused_pil, abcd_ids)
    ans = _clean(resp, ctx)
    ctx.vl_calls += 1
    rp.append(f"[R{round_id}:vl_focused_{ftype}] {ans} conf={conf_val:.2f} ({len(focused_pil)}f)")
    logger.info(f"    R{round_id}: VL(focused)={ans} conf={conf_val:.2f} ({len(focused_pil)} frames)")
    return ans, conf_val, focused_pil


def v21_confidence_loop(ctx, max_rounds=3, abcd_ids=None):
    """V20 + confidence-aware consensus: don't trust low-confidence agreement."""
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

    # P1: Global VL with confidence
    rp.append("[P1:vl_global]")
    vl_prompt = _build_vl_independent_prompt(ctx)
    resp_g, conf_g_dict, _, conf_g = vl_call_with_confidence(
        ctx.vl, vl_prompt, ctx.video_path, abcd_ids)
    vl_global = _clean(resp_g, ctx)
    ctx.vl_calls += 1
    rp.append(f"[vl_full] {vl_global} conf={conf_g:.2f}")
    logger.info(f"  P1: VL(global)={vl_global} conf={conf_g:.2f}")

    # Iterative Loop
    std_thresholds = [0.4, 0.3, 0.2, 0.15, 0.1]
    vl_history = [('global', vl_global, conf_g)]
    prev_frames = set()
    prev_focused_answer = None
    converge_type = None
    n_rounds = 0

    for round_id in range(1, max_rounds + 1):
        rp.append(f"[R{round_id}]")
        logger.info(f"  Round {round_id}/{max_rounds}")

        frames, ents, ftype = _select_frames(ctx, is_temporal)
        cur_frames = set(frames) if frames else set()

        if round_id > 1 and cur_frames == prev_frames:
            rp.append(f"[R{round_id}:no_new_frames]")
            break

        vl_ans, vl_conf, _ = _vl_on_frames_conf(
            ctx, frames, ents, ftype, is_temporal, rp, round_id, abcd_ids)
        vl_history.append((f'R{round_id}', vl_ans, vl_conf))
        n_rounds = round_id

        # === KEY CHANGE: Confidence-aware consensus check ===
        if vl_ans == vl_global and vl_ans in 'ABCD':
            avg_conf = (conf_g + vl_conf) / 2
            if avg_conf >= CONFIDENCE_THRESHOLD:
                converge_type = 'global_consensus_confident'
                rp.append(f"[R{round_id}:confident_consensus] {vl_ans} avg_conf={avg_conf:.2f}")
                logger.info(f"    R{round_id}: CONFIDENT CONSENSUS {vl_ans} (avg_conf={avg_conf:.2f})")
                break
            else:
                # Low confidence consensus — DON'T trust, continue evolving
                rp.append(f"[R{round_id}:weak_consensus_skip] {vl_ans} avg_conf={avg_conf:.2f} < {CONFIDENCE_THRESHOLD}")
                logger.info(f"    R{round_id}: WEAK CONSENSUS {vl_ans} (avg_conf={avg_conf:.2f}) → continue evolving")
                # Still evolve the grid
                threshold = std_thresholds[min(round_id - 1, len(std_thresholds) - 1)]
                _evolve_belief(ctx, rel_names, threshold, rp)
                prev_frames = cur_frames
                prev_focused_answer = vl_ans
                continue

        # Evolution stability check (same as V20)
        if prev_focused_answer is not None and vl_ans == prev_focused_answer and vl_ans in 'ABCD':
            converge_type = 'evolution_stable'
            rp.append(f"[R{round_id}:evolution_stable] {vl_ans} conf={vl_conf:.2f}")
            break

        threshold = std_thresholds[min(round_id - 1, len(std_thresholds) - 1)]
        _evolve_belief(ctx, rel_names, threshold, rp)
        prev_frames = cur_frames
        prev_focused_answer = vl_ans

    # Final Decision — confidence-weighted
    all_valid = [(ans, i, conf) for i, (src, ans, conf) in enumerate(vl_history) if ans in 'ABCD']

    if converge_type == 'global_consensus_confident':
        ans = vl_global
        rp.append(f"[final:confident_consensus] {ans}")

    elif converge_type == 'evolution_stable':
        ans = prev_focused_answer
        rp.append(f"[final:evolution_stable] {ans}")

    elif not all_valid:
        ans = 'A'
        rp.append(f"[fallback] {ans}")

    else:
        # Confidence-weighted vote
        weighted_counts = Counter()
        for src_ans, idx, conf in all_valid:
            src = vl_history[idx][0]
            # Base weight from position
            if src == 'global':
                w_base = max_rounds
            else:
                w_base = int(src[1:]) + 1
            # Multiply by confidence
            w = w_base * conf
            weighted_counts[src_ans] += w

        ans = weighted_counts.most_common(1)[0][0]
        rp.append(f"[final:conf_weighted_vote] {dict((k, f'{v:.2f}') for k,v in weighted_counts.items())}→{ans}")
        logger.info(f"  Final: conf-weighted vote → {ans}")

    rp.append(f"[rounds={n_rounds}]")
    ctx._final_answer = ans
    return ans, " | ".join(rp)


# ============================================================================
# Main — supports multi-GPU
# ============================================================================

CHOICE_TYPES = {
    'obj_appearance_order', 'object_rel_direction_easy', 'object_rel_direction_medium',
    'object_rel_direction_hard', 'object_rel_distance', 'route_planning',
}

VARIANT_MAP = {
    'v20': ('V20 Baseline', None),
    'conf': ('V21-Conf: Break False Consensus', None),
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, help='v20 or conf')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--max_rounds', type=int, default=3)
    parser.add_argument('--grid_max_frames', type=int, default=64)
    parser.add_argument('--confidence_threshold', type=float, default=0.6)
    parser.add_argument('--vl-model', type=str,
                       default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence_threshold

    if args.gpu_id is not None:
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        args.device = 'cuda:0' if vis else f'cuda:{args.gpu_id}'

    # Load samples
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    with open(v7_path) as f: v7_results = json.load(f)

    choice_samples = [s for s in v7_results if s['question_type'] in CHOICE_TYPES and find_video_path(s['scene_name'])]
    by_type = defaultdict(list)
    for s in choice_samples: by_type[s['question_type']].append(s)

    selected = []
    per_type = max(1, args.n_samples // len(CHOICE_TYPES))
    for qt in sorted(CHOICE_TYPES):
        avail = by_type[qt]
        n = min(per_type, len(avail))
        if n > 0:
            indices = np.linspace(0, len(avail)-1, n, dtype=int)
            for idx in indices: selected.append(avail[idx])

    logger.info(f"Selected {len(selected)} samples")

    by_scene = defaultdict(list)
    for s in selected: by_scene[s['scene_name']].append(s)
    scene_list = sorted(by_scene.keys())

    if args.gpu_id is not None:
        total = len(scene_list)
        chunk = (total + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * chunk
        end = min(start + chunk, total)
        my_scenes = scene_list[start:end]
        logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: {len(my_scenes)} scenes")
    else:
        my_scenes = scene_list

    # Load model
    vl_model_path = getattr(args, 'vl_model', None) or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
    pipe = AgenticPipelineV20(device=args.device, vl_model_path=vl_model_path,
                               max_rounds=args.max_rounds, grid_max_frames=args.grid_max_frames)
    pipe.load_models()

    # Get ABCD token IDs
    abcd_ids = _get_abcd_token_ids(pipe.vl.processor)
    logger.info(f"ABCD token IDs: {abcd_ids}")

    all_results = []
    for si, sn in enumerate(my_scenes):
        samples = by_scene[sn]
        vp = find_video_path(sn)
        if not vp:
            for s in samples:
                all_results.append({
                    'scene_name': sn, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []), 'prediction': '0', 'score': 0.0,
                })
            continue

        logger.info(f"[{si+1}/{len(my_scenes)}] {sn} ({len(samples)} q)")
        grid = pipe.builder.build_grid_fps(vp, fps=UNIFIED_FPS, max_frames=pipe.grid_max_frames)

        for sample in samples:
            q = sample['question']; opts = sample.get('options') or []
            gt = sample['ground_truth']; qt = sample['question_type']

            gc_ = copy.deepcopy(grid)
            ctx = ToolExecutionContext(gc_, pipe.vl, vp, pipe.builder, q, opts, qt)
            t0 = time.time()

            try:
                if args.variant == 'conf':
                    ans, reasoning = v21_confidence_loop(ctx, max_rounds=args.max_rounds, abcd_ids=abcd_ids)
                else:
                    ans, reasoning = v20_loop(ctx, max_rounds=args.max_rounds)
            except Exception as e:
                logger.error(f"  Error: {e}"); traceback.print_exc()
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

            logger.info(f"  [{qt[:25]:25s}] pred={ans} gt={gt} score={score:.3f} vl={ctx.vl_calls} t={elapsed:.0f}s")

    pipe.unload()

    # Save
    od = PROJECT_ROOT / "outputs" / "v21_confidence_test" / args.variant
    od.mkdir(parents=True, exist_ok=True)
    out_file = od / (f"gpu{args.gpu_id}.json" if args.gpu_id is not None else "results.json")

    clean = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)): cr[k] = float(v)
            elif isinstance(v, np.ndarray): cr[k] = v.tolist()
            else: cr[k] = v
        clean.append(cr)
    with open(out_file, 'w') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    # Summary
    choice_r = [r for r in all_results if r['question_type'] in CHOICE_TYPES]
    if choice_r:
        s = np.mean([r['score'] for r in choice_r])
        logger.info(f"GPU{args.gpu_id or 0} {args.variant}: {s:.4f} ({len(choice_r)} samples)")
        for qt in sorted(CHOICE_TYPES):
            qr = [r for r in choice_r if r['question_type'] == qt]
            if qr: logger.info(f"  {qt}: {np.mean([r['score'] for r in qr]):.4f} ({len(qr)})")


if __name__ == '__main__':
    main()
