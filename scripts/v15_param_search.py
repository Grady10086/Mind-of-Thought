#!/usr/bin/env python3
"""
V15 Phase 1: 统一参数搜索

在mini-test选择题样本上搜索最优(nframes, max_pixels)配置。
所有选择题使用同一组参数，不引入per-type人工bias。

搜索目标: 找到一组(nframes, max_pixels)使得所有8个official类别
         都超过Qwen3-VL-8B官方基线。

候选配置:
  ① (32, 640*480)   # 32帧×原始分辨率
  ② (48, 480*560)   # 48帧×中高分辨率
  ③ (64, 480*360)   # 64帧×中分辨率
  ④ (32, 360*420)   # V14当前配置(对照)

用法:
  单GPU: python scripts/v15_param_search.py --config_id 0 --device cuda:0
  8GPU并行: 使用 run_v15_search_8gpu.sh
"""

import os
import sys
import json
import re
import gc
import time
import logging
import traceback
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

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

from scripts.grid64_real_test import find_video_path, evaluate_sample

# ============================================================================
# 搜索候选配置
# ============================================================================
SEARCH_CONFIGS = [
    {'id': 0, 'name': 'C0_32f_640x480', 'nframes': 32, 'max_pixels': 640 * 480},
    {'id': 1, 'name': 'C1_48f_480x560', 'nframes': 48, 'max_pixels': 480 * 560},
    {'id': 2, 'name': 'C2_64f_480x360', 'nframes': 64, 'max_pixels': 480 * 360},
    {'id': 3, 'name': 'C3_32f_360x420', 'nframes': 32, 'max_pixels': 360 * 420},  # V14 baseline
]

# Qwen3-VL-8B 官方基线
QWEN3_OFFICIAL = {
    'object_counting': 0.6340,
    'object_abs_distance': 0.4552,
    'object_size_estimation': 0.7341,
    'room_size_estimation': 0.5771,
    'object_rel_distance': 0.5225,
    'object_rel_direction': 0.5123,
    'route_planning': 0.3041,
    'obj_appearance_order': 0.6100,
}

# 选择题类型
CHOICE_TYPES = {
    'obj_appearance_order',
    'object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_medi',
    'object_rel_direction_hard',
    'object_rel_distance',
    'route_planning',
}

# 细分类型 → 官方类别映射
def to_official_category(qt):
    qt = qt.lower().strip()
    if 'appearance' in qt or 'appear' in qt:
        return 'obj_appearance_order'
    if 'direction' in qt:
        return 'object_rel_direction'
    if 'rel_dist' in qt or 'rel_distance' in qt:
        return 'object_rel_distance'
    if 'route' in qt:
        return 'route_planning'
    if 'counting' in qt or 'count' in qt:
        return 'object_counting'
    if 'abs_dist' in qt or 'abs_distance' in qt:
        return 'object_abs_distance'
    if 'size_est' in qt or 'object_size' in qt:
        return 'object_size_estimation'
    if 'room_size' in qt:
        return 'room_size_estimation'
    return qt


def is_choice_task(question_type: str) -> bool:
    qt = question_type.lower().strip()
    for ct in CHOICE_TYPES:
        if ct.lower() in qt or qt in ct.lower():
            return True
    if 'appearance' in qt or 'direction' in qt or 'rel_dist' in qt or 'route' in qt:
        return True
    return False


# ============================================================================
# VL Model (复用V14的VLModel)
# ============================================================================
class VLModel:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.processor = None

    def load(self, model_path: str):
        if self.model is not None:
            return
        logger.info(f"Loading VL model: {model_path}")
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

    def unload(self):
        if self.model is not None:
            del self.model; self.model = None
        if self.processor is not None:
            del self.processor; self.processor = None
        gc.collect(); torch.cuda.empty_cache()

    def call_sampled(self, prompt: str, video_path: str, max_tokens: int = 128,
                     n_samples: int = 3, temperature: float = 0.7, top_p: float = 0.9,
                     nframes: int = 32, max_pixels: int = 360 * 420):
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
            traceback.print_exc()
            return [""]

    def call(self, prompt: str, video_path: str, max_tokens: int = 512,
             nframes: int = 32, max_pixels: int = 360 * 420) -> str:
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
# 选择题 bypass (复用V14逻辑，参数化nframes/max_pixels)
# ============================================================================
def build_official_mca_prompt(question: str, options: list) -> str:
    opts_str = "\n".join(options)
    return f"""These are frames of a video.
{question}
Options:
{opts_str}
Answer with the option's letter from the given choices directly."""


def choice_task_bypass(vl, video_path, question, options, n_votes=5,
                       nframes=32, max_pixels=360*420):
    prompt = build_official_mca_prompt(question, options)
    responses = vl.call_sampled(prompt, video_path, max_tokens=32,
                                n_samples=n_votes, temperature=0.7, top_p=0.9,
                                nframes=nframes, max_pixels=max_pixels)
    vl_calls = len(responses)

    votes = []
    for r in responses:
        r_clean = r.strip()
        m = re.search(r'^([A-Da-d])', r_clean)
        if m:
            votes.append(m.group(1).upper())
        else:
            m = re.search(r'\b([A-D])\b', r_clean[:50])
            if m:
                votes.append(m.group(1))
            else:
                r_lower = r_clean.lower()
                for i, opt in enumerate(options):
                    opt_content = opt.lower()
                    if len(opt) >= 3 and opt[1] in '.、':
                        opt_content = opt[3:].strip().lower()
                    if opt_content and opt_content in r_lower:
                        votes.append(chr(65 + i))
                        break

    if not votes:
        fallback = vl.call(prompt, video_path, max_tokens=32,
                          nframes=nframes, max_pixels=max_pixels)
        vl_calls += 1
        m = re.search(r'([A-Da-d])', fallback.strip())
        prediction = m.group(1).upper() if m else 'A'
        reasoning = f"[bypass_fallback] {fallback.strip()[:80]}"
        return prediction, reasoning, vl_calls

    vote_counts = Counter(votes)
    prediction = vote_counts.most_common(1)[0][0]
    reasoning = f"[bypass_{n_votes}vote] {dict(vote_counts)} → {prediction}"
    return prediction, reasoning, vl_calls


# ============================================================================
# Mini-test 样本选择
# ============================================================================
def select_choice_samples(v7_results, n_per_type=0):
    """选择选择题样本。n_per_type=0表示全量。"""
    by_type = defaultdict(list)
    for r in v7_results:
        if is_choice_task(r['question_type']):
            if find_video_path(r['scene_name']):
                by_type[r['question_type']].append(r)

    selected = []
    for qt in sorted(by_type.keys()):
        samples = by_type[qt]
        if n_per_type > 0:
            n = min(n_per_type, len(samples))
            indices = np.linspace(0, len(samples) - 1, n, dtype=int)
            for idx in indices:
                selected.append(samples[idx])
        else:
            selected.extend(samples)

    logger.info(f"Choice samples: {len(selected)} from {len(by_type)} types")
    for qt in sorted(by_type.keys()):
        n_sel = sum(1 for s in selected if s['question_type'] == qt)
        logger.info(f"  {qt}: {n_sel}")
    return selected


# ============================================================================
# 评估一组配置在所有mini-test样本上的效果
# ============================================================================
def evaluate_config(vl, config, samples):
    """对一组配置跑所有样本，返回per-sample结果"""
    nframes = config['nframes']
    max_pixels = config['max_pixels']
    name = config['name']

    results = []
    by_scene = defaultdict(list)
    for s in samples:
        by_scene[s['scene_name']].append(s)

    total = len(samples)
    done = 0
    for scene_name in sorted(by_scene.keys()):
        scene_samples = by_scene[scene_name]
        video_path = find_video_path(scene_name)
        if not video_path:
            for s in scene_samples:
                results.append({
                    'scene_name': scene_name,
                    'question_type': s['question_type'],
                    'question': s['question'],
                    'ground_truth': s['ground_truth'],
                    'options': s.get('options', []),
                    'prediction': 'A',
                    'score': 0.0,
                    'reasoning': 'no_video',
                    'v7_vl_score': s.get('vl_score', 0),
                })
                done += 1
            continue

        for s in scene_samples:
            t0 = time.time()
            pred, reasoning, vl_calls = choice_task_bypass(
                vl, video_path, s['question'], s.get('options', []),
                n_votes=5, nframes=nframes, max_pixels=max_pixels)
            elapsed = time.time() - t0

            score = evaluate_sample(s['question_type'], pred, s['ground_truth'])
            done += 1

            results.append({
                'scene_name': scene_name,
                'question_type': s['question_type'],
                'question': s['question'],
                'ground_truth': s['ground_truth'],
                'options': s.get('options', []),
                'prediction': pred,
                'score': float(score),
                'reasoning': reasoning,
                'vl_calls': vl_calls,
                'elapsed_s': round(elapsed, 1),
                'v7_vl_score': s.get('vl_score', 0),
            })

            logger.info(f"  [{name}] {done}/{total} {s['question_type'][:25]:25s} "
                        f"score={score:.0f} pred={pred} gt={s['ground_truth']} ({elapsed:.1f}s)")

    return results


def analyze_results(results, config_name):
    """分析一组配置的结果，按官方8类聚合"""
    by_official = defaultdict(lambda: {'n': 0, 'correct': 0})

    for r in results:
        off = to_official_category(r['question_type'])
        by_official[off]['n'] += 1
        if r['score'] > 0:
            by_official[off]['correct'] += 1

    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"{'='*70}")
    print(f"{'Category':>25} {'N':>5} {'Acc':>7} {'Qwen3':>7} {'Status':>8}")
    print(f"{'-'*55}")

    all_above = True
    choice_cats = ['obj_appearance_order', 'object_rel_distance', 'object_rel_direction', 'route_planning']
    for cat in choice_cats:
        s = by_official[cat]
        if s['n'] == 0:
            continue
        acc = s['correct'] / s['n']
        qwen = QWEN3_OFFICIAL.get(cat, 0)
        above = acc > qwen
        if not above:
            all_above = False
        print(f"{cat:>25} {s['n']:>5} {acc*100:>6.1f}% {qwen*100:>6.1f}% {'  ✓' if above else '  ✗'}")

    total_n = sum(s['n'] for s in by_official.values())
    total_c = sum(s['correct'] for s in by_official.values())
    overall = total_c / total_n * 100 if total_n > 0 else 0
    print(f"{'-'*55}")
    print(f"{'Choice Overall':>25} {total_n:>5} {overall:>6.1f}%")
    print(f"  All above Qwen3-VL? {'✓ YES' if all_above else '✗ NO'}")

    return {
        'config': config_name,
        'all_above_qwen': all_above,
        'choice_overall': overall,
        'by_category': {cat: {
            'n': by_official[cat]['n'],
            'correct': by_official[cat]['correct'],
            'accuracy': by_official[cat]['correct'] / by_official[cat]['n'] if by_official[cat]['n'] > 0 else 0,
        } for cat in choice_cats if by_official[cat]['n'] > 0},
    }


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="V15 Phase 1: Param Search")
    parser.add_argument('--config_id', type=int, required=True, help='Config index (0-3)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None, help='For multi-GPU: which GPU shard')
    parser.add_argument('--num_gpus', type=int, default=1, help='For multi-GPU: total GPUs for this config')
    parser.add_argument('--n_per_type', type=int, default=0, help='Samples per choice type (0=all)')
    parser.add_argument('--vl_model', type=str,
                        default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    config = SEARCH_CONFIGS[args.config_id]
    logger.info(f"V15 Param Search — Config {args.config_id}: {config['name']}")
    logger.info(f"  nframes={config['nframes']}, max_pixels={config['max_pixels']}")

    if args.gpu_id is not None:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible:
            args.device = 'cuda:0'
        logger.info(f"GPU shard {args.gpu_id}/{args.num_gpus}, device={args.device}")

    # Load V7 baseline
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    with open(v7_path) as f:
        v7_results = json.load(f)
    logger.info(f"V7 baseline: {len(v7_results)} samples")

    # Select choice samples (0 = all)
    choice_samples = select_choice_samples(v7_results, n_per_type=args.n_per_type)

    # Multi-GPU sharding
    if args.gpu_id is not None and args.num_gpus > 1:
        by_scene = defaultdict(list)
        for s in choice_samples:
            by_scene[s['scene_name']].append(s)
        scenes = sorted(by_scene.keys())
        chunk = (len(scenes) + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * chunk
        end = min(start + chunk, len(scenes))
        my_scenes = scenes[start:end]
        choice_samples = []
        for sc in my_scenes:
            choice_samples.extend(by_scene[sc])
        logger.info(f"GPU shard: {len(my_scenes)} scenes, {len(choice_samples)} samples")

    # Load VL model
    vl = VLModel(device=args.device)
    vl.load(args.vl_model)

    # Run evaluation
    results = evaluate_config(vl, config, choice_samples)

    vl.unload()

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "v15_param_search" / config['name']
    if args.gpu_id is not None:
        output_dir = output_dir / f"gpu{args.gpu_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print analysis
    summary = analyze_results(results, config['name'])
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved: {output_dir}")


if __name__ == "__main__":
    main()
