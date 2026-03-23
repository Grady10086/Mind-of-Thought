#!/usr/bin/env python3
"""
Benchmark Evaluation using V21 VL Model + Confidence-Aware Voting.

Supports: EgoSchema, VideoMMMU, EscherVerse
Uses same VL model and confidence strategy as V21 pipeline.

Usage:
    python scripts/benchmark_eval_v21.py --benchmark egoschema --gpu_id 0 --num_gpus 8
    python scripts/benchmark_eval_v21.py --benchmark videommmu --gpu_id 0 --num_gpus 8
    python scripts/benchmark_eval_v21.py --benchmark escherverse --gpu_id 0 --num_gpus 8
"""

import os, sys, json, re, gc, time, traceback, argparse, logging
import numpy as np, cv2, torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import io, base64
from PIL import Image

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

UNIFIED_FPS = 2.0
VL_DEFAULT_MAX_PIXELS = 640 * 480
DEFAULT_VL_MODEL = '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'


# ============================================================================
# VL Model (reused from V21)
# ============================================================================

def _get_video_fps_nframes(video_path, target_fps=2.0):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return 32
        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vfps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        if vfps <= 0: return 32
        n = max(8, int((tf / vfps) * target_fps))
        return min(n, 300)
    except:
        return 32


class VLModel:
    def __init__(self, device='cuda'):
        self.device = device; self.model = None; self.processor = None
        self._abcd_ids = None; self._abcde_ids = None

    def load(self, model_path):
        if self.model is not None: return
        logger.info(f"Loading VL model: {model_path}")
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if 'qwen3' in model_path.lower() or 'Qwen3' in model_path:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=self.device, trust_remote_code=True)
        logger.info("VL model loaded")

    def _get_token_ids(self, letters):
        if self.processor is None: return {}
        result = {}
        for letter in letters:
            ids = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            if ids: result[letter] = ids[0]
        return result

    def _ensure_abcd_ids(self, n_options=4):
        letters = ALL_LETTERS[:min(n_options, 26)]
        key = f"_ids_{n_options}"
        cached = getattr(self, key, None)
        if cached is None:
            cached = self._get_token_ids(letters)
            setattr(self, key, cached)
        return cached

    def call(self, prompt, video_path, max_tokens=512, max_pixels=VL_DEFAULT_MAX_PIXELS):
        if self.model is None: return ""
        try:
            from qwen_vl_utils import process_vision_info
            nframes = _get_video_fps_nframes(video_path, UNIFIED_FPS)
            content = [{"type": "video", "video": f"file://{video_path}",
                       "nframes": nframes, "max_pixels": max_pixels},
                      {"type": "text", "text": prompt}]
            msgs = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(msgs)
            inputs = self.processor(text=[text], images=img_inputs, videos=vid_inputs,
                                   return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            resp = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            return resp.strip()
        except Exception as e:
            logger.warning(f"VL call failed: {e}"); return ""

    def call_with_confidence(self, prompt, video_path, n_options=4,
                              max_tokens=128, max_pixels=VL_DEFAULT_MAX_PIXELS):
        """Call VL and return (response, top_letter, top_conf, all_probs)."""
        if self.model is None: return "", '', 0.0, {}
        try:
            from qwen_vl_utils import process_vision_info
            nframes = _get_video_fps_nframes(video_path, UNIFIED_FPS)
            content = [{"type": "video", "video": f"file://{video_path}",
                        "nframes": nframes, "max_pixels": max_pixels},
                       {"type": "text", "text": prompt}]
            msgs = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(msgs)
            inputs = self.processor(text=[text], images=img_inputs, videos=vid_inputs,
                                   return_tensors="pt", padding=True).to(self.model.device)

            token_ids = self._ensure_abcd_ids(n_options)
            with torch.no_grad():
                outputs_fwd = self.model(**inputs)
                last_logits = outputs_fwd.logits[0, -1, :]
                all_letters = ALL_LETTERS
                logit_vals, valid_letters = [], []
                for letter in all_letters[:min(n_options, 26)]:
                    tid = token_ids.get(letter)
                    if tid is not None:
                        logit_vals.append(last_logits[tid].item())
                        valid_letters.append(letter)
                if logit_vals:
                    probs = F.softmax(torch.tensor(logit_vals), dim=0).numpy()
                    top_idx = int(np.argmax(probs))
                    conf_letter = valid_letters[top_idx]
                    conf_val = float(probs[top_idx])
                    all_probs = {l: float(p) for l, p in zip(valid_letters, probs)}
                else:
                    conf_letter, conf_val, all_probs = '', 0.0, {}

                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            resp = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                               skip_special_tokens=True)[0].strip()
            return resp, conf_letter, conf_val, all_probs
        except Exception as e:
            logger.warning(f"VL call_with_confidence failed: {e}"); return "", '', 0.0, {}


# ============================================================================
# Extract answer letter from VL response
# ============================================================================

ALL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def extract_answer_letter(response, n_options=4):
    """Extract single answer letter from VL response."""
    valid = set(ALL_LETTERS[:n_options])
    resp = response.strip()
    if resp and resp[0].upper() in valid:
        return resp[0].upper()
    m = re.search(r'\b([A-Z])\b', resp)
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.search(r'([A-Z])\)', resp)
    if m and m.group(1) in valid:
        return m.group(1)
    first = resp[:1].upper() if resp else ''
    return first if first in valid else ''


# ============================================================================
# Benchmark Data Loaders
# ============================================================================

EGOSCHEMA_VIDEO_DIR = '/home/tione/notebook/tianjungu/datasets/egoschema_videos/videos'

def load_egoschema(gpu_id, num_gpus):
    """Load EgoSchema Subset (500 with answers) from local parquet + videos."""
    import pandas as pd
    logger.info("Loading EgoSchema Subset (500) from local files...")
    pq_path = os.path.join(os.environ['HF_HOME'],
        'hub/datasets--lmms-lab--egoschema/snapshots/58350524ea7eb29c47000121f4f4b65eb6b4acb9/Subset/test-00000-of-00001.parquet')
    df = pd.read_parquet(pq_path)
    samples = []
    for _, row in df.iterrows():
        video_idx = row['video_idx']
        video_path = os.path.join(EGOSCHEMA_VIDEO_DIR, f"{video_idx}.mp4")
        options = list(row['option'])
        question = row['question']
        # Options already have "A. ..." prefix in some cases; clean and rebuild
        clean_opts = []
        for opt in options:
            opt = opt.strip()
            if len(opt) > 3 and opt[0] in 'ABCDE' and opt[1] == '.':
                opt = opt[2:].strip()
            clean_opts.append(opt)
        options_str = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_opts)])
        prompt = f"{question}\n{options_str}\nAnswer with the option's letter from the given choices directly."
        # answer is index string like '3' -> 'D'
        ans_idx = int(row['answer']) if row['answer'] is not None else 0
        answer = chr(65 + ans_idx)
        samples.append({
            'id': video_idx,
            'video_path': video_path,
            'question': question,
            'prompt': prompt,
            'options': clean_opts,
            'n_options': len(clean_opts),
            'answer': answer,
            'benchmark': 'egoschema',
        })
    samples.sort(key=lambda x: x['id'])
    shard = samples[gpu_id::num_gpus]
    logger.info(f"EgoSchema: {len(shard)}/{len(samples)} samples for GPU {gpu_id}")
    return shard


VIDEOMMMU_DIR = '/home/tione/notebook/tianjungu/datasets/VideoMMMU'

def load_videommmu(gpu_id, num_gpus):
    """Load VideoMMMU from local downloaded dataset."""
    import pandas as pd
    logger.info("Loading VideoMMMU dataset from local files...")
    samples = []

    # Subject -> domain folder mapping
    subject_to_domain = {}
    for domain in ['Art', 'Business', 'Engineering', 'Humanities', 'Medicine', 'Science']:
        domain_dir = os.path.join(VIDEOMMMU_DIR, domain)
        if os.path.isdir(domain_dir):
            for fname in os.listdir(domain_dir):
                if fname.endswith('.mp4'):
                    vid_id = fname[:-4]  # remove .mp4
                    subject_to_domain[vid_id] = domain

    for split_name in ['Perception', 'Comprehension', 'Adaptation']:
        pq_path = os.path.join(VIDEOMMMU_DIR, split_name, 'test-00000-of-00001.parquet')
        if not os.path.exists(pq_path):
            logger.warning(f"Parquet not found: {pq_path}")
            continue
        df = pd.read_parquet(pq_path)
        for _, row in df.iterrows():
            qid = row['id']
            question = row['question']
            options = list(row['options']) if hasattr(row['options'], '__iter__') else []
            q_type = row.get('question_type', 'multiple-choice')
            qa_type = row.get('qa_type', '')
            answer = row.get('answer', 'A')

            if q_type != 'multiple-choice':
                continue

            # Find video path
            domain = subject_to_domain.get(qid, '')
            if domain:
                video_path = os.path.join(VIDEOMMMU_DIR, domain, f"{qid}.mp4")
            else:
                video_path = ''
                for d in ['Art', 'Business', 'Engineering', 'Humanities', 'Medicine', 'Science']:
                    p = os.path.join(VIDEOMMMU_DIR, d, f"{qid}.mp4")
                    if os.path.exists(p):
                        video_path = p
                        break

            n_options = len(options)
            options_str = '\n'.join([f"{ALL_LETTERS[i]}. {opt}" for i, opt in enumerate(options)])
            prompt = f"{question}\n{options_str}\nAnswer with the option's letter from the given choices directly."

            samples.append({
                'id': qid,
                'video_path': video_path,
                'question': question,
                'prompt': prompt,
                'options': options,
                'n_options': n_options,
                'answer': answer,
                'split': split_name,
                'qa_type': qa_type,
                'benchmark': 'videommmu',
            })

    samples.sort(key=lambda x: x['id'])
    shard = samples[gpu_id::num_gpus]
    logger.info(f"VideoMMMU: {len(shard)}/{len(samples)} MCQ samples for GPU {gpu_id}")
    return shard


VIDEOMME_DIR = '/home/tione/notebook/tianjungu/datasets/VideoMME'
VIDEOMME_VIDEO_DIR = '/home/tione/notebook/tianjungu/datasets/VideoMME/videos/data'

def load_videomme(gpu_id, num_gpus):
    """Load Video-MME from local parquet + extracted videos."""
    import pandas as pd
    logger.info("Loading Video-MME dataset from local files...")
    pq_path = os.path.join(VIDEOMME_DIR, 'videomme', 'test-00000-of-00001.parquet')
    df = pd.read_parquet(pq_path)

    samples = []
    missing_videos = 0
    for _, row in df.iterrows():
        video_id = row['videoID']
        # Try multiple extensions
        video_path = None
        for ext in ['mp4', 'MP4', 'mkv']:
            p = os.path.join(VIDEOMME_VIDEO_DIR, f"{video_id}.{ext}")
            if os.path.exists(p):
                video_path = p
                break
        if video_path is None:
            missing_videos += 1
            continue

        question = row['question']
        options = list(row['options'])  # Already has "A. ...", "B. ...", etc.
        answer = row['answer']  # e.g., 'C'
        duration = row['duration']  # short/medium/long
        domain = row['domain']
        sub_category = row.get('sub_category', '')
        task_type = row.get('task_type', '')
        question_id = row.get('question_id', '')

        options_str = '\n'.join(options)
        prompt = (f"Select the best answer to the following multiple-choice question based on the video. "
                  f"Respond with only the letter (A, B, C, or D) of the correct option.\n"
                  f"{question}\n{options_str}\nThe best answer is:")

        samples.append({
            'id': question_id or f"{video_id}_{len(samples)}",
            'video_path': video_path,
            'question': question,
            'prompt': prompt,
            'options': options,
            'n_options': len(options),
            'answer': answer,
            'duration': duration,
            'domain': domain,
            'sub_category': sub_category,
            'task_type': task_type,
            'benchmark': 'videomme',
        })

    if missing_videos > 0:
        logger.warning(f"Video-MME: {missing_videos} videos not found (not yet downloaded?)")
    logger.info(f"Video-MME: loaded {len(samples)} samples ({df['videoID'].nunique()} unique videos)")

    samples.sort(key=lambda x: x['id'])
    shard = samples[gpu_id::num_gpus]
    logger.info(f"Video-MME: {len(shard)}/{len(samples)} samples for GPU {gpu_id}")
    return shard


ESCHER_VIDEO_DIR = '/home/tione/notebook/tianjungu/datasets/Spatial_understanding/Spatial_datas'
ESCHER_BENCH_PATH = '/home/tione/notebook/tianjungu/projects/escher_processed/Escher-Bench.json'

def load_escherverse(gpu_id, num_gpus):
    """Load EscherVerse from local Escher-Bench.json."""
    logger.info("Loading EscherVerse dataset...")
    with open(ESCHER_BENCH_PATH) as f:
        data = json.load(f)

    samples = []
    skipped = Counter()
    for item in data:
        video_name = item['P']
        video_path = os.path.join(ESCHER_VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            skipped['no_video'] += 1
            continue

        q_text = item['Q']
        answer = item['A']
        q_type = item.get('question_type', 'Single-Choice')
        category = item.get('C', '')

        # Only support MCQ types (Single-Choice and True/False)
        if q_type == 'Single-Choice':
            opt_match = re.search(r'\[Options\]\s*(.*)', q_text, re.DOTALL)
            if opt_match:
                opts_text = opt_match.group(1).strip()
                question_part = q_text[:opt_match.start()].strip()
                question_part = re.sub(r'^\[Single-Choice\]\s*', '', question_part).strip()
                options = re.findall(r'([A-E])\)\s*(.*?)(?=\s+[A-E]\)|$)', opts_text, re.DOTALL)
                options_list = [f"{l}) {t.strip()}" for l, t in options]
                n_options = len(options_list)
                options_str = '\n'.join(options_list)
            else:
                question_part = re.sub(r'^\[Single-Choice\]\s*', '', q_text).strip()
                options_str = ''
                n_options = 4
        elif q_type in ('true_false', 'True/False'):
            question_part = re.sub(r'^\[True/False\]\s*', '', q_text).strip()
            options_str = "A) True\nB) False"
            n_options = 2
            answer = 'A' if answer.strip().lower() in ['true', 'a'] else 'B'
        else:
            skipped[q_type] += 1
            continue

        prompt = f"{question_part}\n{options_str}\nAnswer with the option's letter from the given choices directly."

        samples.append({
            'id': str(item['index']),
            'video_path': video_path,
            'question': question_part,
            'prompt': prompt,
            'options': options_str,
            'n_options': n_options,
            'answer': answer,
            'category': category,
            'question_type': q_type,
            'benchmark': 'escherverse',
        })

    if skipped:
        logger.info(f"EscherVerse skipped: {dict(skipped)}")

    samples.sort(key=lambda x: int(x['id']))
    shard = samples[gpu_id::num_gpus]
    logger.info(f"EscherVerse: {len(shard)}/{len(samples)} samples for GPU {gpu_id}")
    return shard


# ============================================================================
# V21-style Evaluation: Confidence-Aware 3-Vote
# ============================================================================

def evaluate_sample_v21(vl, sample):
    """Evaluate a single sample using V21's confidence-aware strategy.

    Strategy:
    1. Single greedy call with confidence
    2. If confidence >= 0.7, use that answer directly
    3. Otherwise, do 2 more calls (3 total), confidence-weighted vote
    """
    video_path = sample['video_path']
    prompt = sample['prompt']
    n_options = sample.get('n_options', 4)

    if not os.path.exists(video_path):
        logger.warning(f"Video not found: {video_path}")
        return {'prediction': '', 'correct': False, 'confidence': 0.0, 'n_calls': 0}

    # Call 1: greedy with confidence
    resp1, conf_letter1, conf1, probs1 = vl.call_with_confidence(
        prompt, video_path, n_options=n_options)
    answer1 = extract_answer_letter(resp1, n_options)
    if not answer1 and conf_letter1:
        answer1 = conf_letter1

    # High confidence → use directly
    if conf1 >= 0.7 and answer1:
        pred = answer1
        correct = (pred == sample['answer'])
        return {
            'prediction': pred, 'correct': correct, 'confidence': conf1,
            'n_calls': 1, 'strategy': 'high_confidence',
            'response': resp1, 'probs': probs1,
        }

    # Otherwise: 2 more calls
    votes = [(answer1, conf1)]
    for _ in range(2):
        resp_i, conf_letter_i, conf_i, probs_i = vl.call_with_confidence(
            prompt, video_path, n_options=n_options)
        ans_i = extract_answer_letter(resp_i, n_options)
        if not ans_i and conf_letter_i:
            ans_i = conf_letter_i
        votes.append((ans_i, conf_i))

    # Confidence-weighted vote
    vote_scores = defaultdict(float)
    for ans, conf in votes:
        if ans:
            vote_scores[ans] += conf
    if vote_scores:
        pred = max(vote_scores, key=vote_scores.get)
        total_conf = vote_scores[pred] / sum(vote_scores.values()) if sum(vote_scores.values()) > 0 else 0
    else:
        pred = answer1 or ''
        total_conf = conf1

    correct = (pred == sample['answer'])
    return {
        'prediction': pred, 'correct': correct, 'confidence': total_conf,
        'n_calls': 3, 'strategy': 'confidence_vote',
        'votes': [(a, c) for a, c in votes],
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True, choices=['egoschema', 'videommmu', 'escherverse', 'videomme'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--vl-model', default=DEFAULT_VL_MODEL)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--max_samples', type=int, default=0, help='0 = all')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / 'outputs' / f'benchmark_{args.benchmark}_v21')
    os.makedirs(args.output_dir, exist_ok=True)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Load data
    loader_map = {
        'egoschema': load_egoschema,
        'videommmu': load_videommmu,
        'escherverse': load_escherverse,
        'videomme': load_videomme,
    }
    samples = loader_map[args.benchmark](args.gpu_id, args.num_gpus)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    logger.info(f"Evaluating {len(samples)} samples on {args.benchmark} (GPU {args.gpu_id})")

    # Load VL model
    vl = VLModel(device='cuda')
    vl.load(args.vl_model)

    # Run evaluation
    results = []
    correct_count = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        t0 = time.time()
        result = evaluate_sample_v21(vl, sample)
        elapsed = time.time() - t0

        result['id'] = sample['id']
        result['ground_truth'] = sample['answer']
        result['benchmark'] = sample['benchmark']
        result['elapsed'] = elapsed

        if sample.get('category'):
            result['category'] = sample['category']
        if sample.get('split'):
            result['split'] = sample['split']
        if sample.get('question_type'):
            result['question_type'] = sample['question_type']
        if sample.get('qa_type'):
            result['qa_type'] = sample['qa_type']
        if sample.get('duration'):
            result['duration'] = sample['duration']
        if sample.get('domain'):
            result['domain'] = sample['domain']
        if sample.get('sub_category'):
            result['sub_category'] = sample['sub_category']
        if sample.get('task_type'):
            result['task_type'] = sample['task_type']

        results.append(result)
        if result['correct']:
            correct_count += 1

        acc = correct_count / (i + 1)
        eta_s = (time.time() - start_time) / (i + 1) * (len(samples) - i - 1)
        eta_m = eta_s / 60

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            logger.info(f"[{i+1}/{len(samples)}] acc={acc:.4f} ({correct_count}/{i+1}) "
                       f"elapsed={elapsed:.1f}s ETA={eta_m:.1f}min")

        # Periodic save
        if (i + 1) % 50 == 0:
            _save_results(results, args, correct_count)

    # Final save
    _save_results(results, args, correct_count)

    total_time = time.time() - start_time
    acc = correct_count / len(results) if results else 0
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Total: {len(results)} samples")
    logger.info(f"Accuracy: {acc:.4f} ({correct_count}/{len(results)})")
    logger.info(f"Time: {total_time/60:.1f} min")
    logger.info(f"Results saved to: {args.output_dir}")

    # Per-category breakdown
    by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    cat_key = 'category' if args.benchmark == 'escherverse' else ('duration' if args.benchmark == 'videomme' else 'split')
    for r in results:
        cat = r.get(cat_key, 'all')
        by_cat[cat]['total'] += 1
        if r['correct']:
            by_cat[cat]['correct'] += 1
    for cat in sorted(by_cat):
        d = by_cat[cat]
        logger.info(f"  {cat}: {d['correct']}/{d['total']} = {d['correct']/d['total']:.4f}")


def _save_results(results, args, correct_count):
    out_path = os.path.join(args.output_dir, f'gpu{args.gpu_id}_results.json')
    summary = {
        'benchmark': args.benchmark,
        'gpu_id': args.gpu_id,
        'total': len(results),
        'correct': correct_count,
        'accuracy': correct_count / len(results) if results else 0,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
