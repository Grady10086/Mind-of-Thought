#!/usr/bin/env python3
"""Bare Qwen3-VL-8B test: fps=2, official prompt, single call, no pipeline.
Measures raw VL capability at fps=2 for comparison with Official baseline.
"""
import json, os, sys, re, gc, time, argparse, traceback, logging
import numpy as np, torch, cv2
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from grid64_real_test import find_video_path, evaluate_sample


VL_FPS = 2
VL_MAX_PIXELS = 640 * 480  # same as V16


def get_nframes(video_path, target_fps=2.0):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 32
        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vfps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        if vfps <= 0: return 32
        n = max(8, int((tf / vfps) * target_fps))
        return min(n, 300)
    except: return 32


def build_official_prompt(question, options):
    """Official VLMEvalKit MCA prompt for VSI-Bench."""
    opts_str = "\n".join(options)
    return f"""These are frames of a video.
{question}
Options:
{opts_str}
Answer with the option's letter from the given choices directly."""


def build_numerical_prompt(question):
    """Simple numerical prompt."""
    return f"""These are frames of a video.
{question}
Answer with only a number."""


class BareVL:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.processor = None

    def load(self, model_path):
        if self.model is not None: return
        logger.info(f"Loading VL model: {model_path}")
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
        logger.info("VL model loaded")

    def call(self, prompt, video_path, max_tokens=32):
        from qwen_vl_utils import process_vision_info
        nframes = get_nframes(video_path, VL_FPS)
        content = [
            {"type": "video", "video": video_path, "max_pixels": VL_MAX_PIXELS, "nframes": nframes},
            {"type": "text", "text": prompt}
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        resp = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return resp.strip()

    def unload(self):
        if self.model: del self.model; self.model = None
        if self.processor: del self.processor; self.processor = None
        gc.collect(); torch.cuda.empty_cache()


def extract_choice(resp):
    m = re.search(r'^([A-Da-d])', resp.strip())
    if m: return m.group(1).upper()
    m = re.search(r'\b([A-D])\b', resp[:50])
    if m: return m.group(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--n_per_type', type=int, default=50)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--vl_model', type=str, 
                        default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    if args.gpu_id is not None:
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        args.device = 'cuda:0' if vis else f'cuda:{args.gpu_id}'
    else:
        args.device = 'cuda:0'

    # Load test data from V7
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    with open(v7_path) as f: v7_results = json.load(f)
    logger.info(f"V7: {len(v7_results)} samples")

    if args.full:
        test_samples = [s for s in v7_results if find_video_path(s['scene_name'])]
    else:
        by_type = defaultdict(list)
        for r in v7_results: by_type[r['question_type']].append(r)
        test_samples = []
        for qt, samps in sorted(by_type.items()):
            avail = [s for s in samps if find_video_path(s['scene_name'])]
            n = min(args.n_per_type, len(avail))
            if n > 0:
                for idx in np.linspace(0, len(avail)-1, n, dtype=int):
                    test_samples.append(avail[idx])

    logger.info(f"Test: {len(test_samples)} samples")

    # Split by scene for GPU sharding
    by_scene = defaultdict(list)
    for s in test_samples: by_scene[s['scene_name']].append(s)
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

    # Load VL model
    vl = BareVL(device=args.device)
    vl.load(args.vl_model)

    all_results = []
    for si, sn in enumerate(my_scenes):
        samples = by_scene[sn]
        vp = find_video_path(sn)
        if not vp:
            for s in samples:
                all_results.append({
                    'scene_name': sn, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []), 'prediction': '0',
                    'score': 0.0, 'v7_vl_score': s.get('vl_score', 0)
                })
            continue

        logger.info(f"[{si+1}/{len(my_scenes)}] {sn} ({len(samples)} q)")
        for s in samples:
            t0 = time.time()
            try:
                q = s['question']
                opts = s.get('options', [])
                gt = s['ground_truth']

                if opts:
                    prompt = build_official_prompt(q, opts)
                    resp = vl.call(prompt, vp, max_tokens=32)
                    pred = extract_choice(resp) or 'A'
                else:
                    prompt = build_numerical_prompt(q)
                    resp = vl.call(prompt, vp, max_tokens=64)
                    m = re.search(r'[\d.]+', resp)
                    pred = m.group() if m else '0'

                score = evaluate_sample(pred, gt, s['question_type'], opts)
                elapsed = time.time() - t0
                v7s = s.get('vl_score', 0)
                mk = '+' if score > v7s + 0.01 else ('-' if score < v7s - 0.01 else '=')

                logger.info(f"  {s['question_type']:<34} Score={score:.3f} V7={v7s:.3f} {mk} | pred={str(pred)[:15]} gt={str(gt)[:12]} ({elapsed:.0f}s)")

                all_results.append({
                    'scene_name': sn, 'question_type': s['question_type'],
                    'question': q, 'ground_truth': gt, 'options': opts,
                    'prediction': pred, 'score': float(score),
                    'v7_vl_score': v7s, 'elapsed_s': elapsed, 'raw_response': resp
                })
            except Exception as e:
                logger.error(f"  Error on {s['question_type']}: {e}")
                traceback.print_exc()
                all_results.append({
                    'scene_name': sn, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []), 'prediction': '0',
                    'score': 0.0, 'v7_vl_score': s.get('vl_score', 0)
                })

    vl.unload()

    # Save results
    if args.gpu_id is not None:
        od = PROJECT_ROOT / "outputs" / "bare_vl_fps2_full" / f"gpu{args.gpu_id}"
    else:
        od = PROJECT_ROOT / "outputs" / "bare_vl_fps2_test"
    od.mkdir(parents=True, exist_ok=True)

    clean = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)): cr[k] = float(v)
            elif isinstance(v, np.ndarray): cr[k] = v.tolist()
            else: cr[k] = v
        clean.append(cr)

    with open(od / "detailed_results.json", 'w') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {od} ({len(all_results)} samples)")

    # Print summary if single GPU
    if args.gpu_id is None:
        by_type = defaultdict(list)
        for r in all_results: by_type[r['question_type']].append(r)
        print(f"\n{'Task':<38} {'N':>4} {'BareVL':>7} {'V7':>6}")
        print('-' * 60)
        all_s = []
        for qt in sorted(by_type):
            rs = by_type[qt]
            sc = np.mean([r['score'] for r in rs])
            v7 = np.mean([r['v7_vl_score'] for r in rs])
            print(f"  {qt:<36} {len(rs):>4} {sc:>6.3f} {v7:>6.3f}")
            all_s.extend([r['score'] for r in rs])
        print('-' * 60)
        print(f"  {'Overall':<36} {len(all_s):>4} {np.mean(all_s):>6.3f}")


if __name__ == "__main__":
    main()
