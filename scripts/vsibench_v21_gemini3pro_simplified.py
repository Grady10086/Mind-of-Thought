#!/usr/bin/env python3
"""
VSIBench Simplified V21 Pipeline with Gemini 3 Pro API

Simplified version: Only P1 (Global VL) + R1 (Focused VL)
- No confidence-aware voting
- No multiple rounds (R2/R3)
- Direct comparison between Global vs Focused

Expected API calls: ~2 per sample (vs 3.5 in full V21)
"""

import os
import sys
import json
import re
import gc
import time
import logging
import traceback
import base64
import requests
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NEWAPI Configuration
NEWAPI_API_KEY = os.environ.get('NEWAPI_API_KEY', 'sk-gVBzVgUnsXCGL0mOTY4dim5lywqHA5Y2ec84EHaZGU4NLLh8')
NEWAPI_BASE_URL = os.environ.get('NEWAPI_BASE_URL', 'https://api.chataiapi.com/v1')
NEWAPI_MODEL = os.environ.get('NEWAPI_GEMINI25_MODEL', 'gemini-3-pro-preview')

# Proxy settings
os.environ.setdefault('https_proxy', 'http://172.31.255.9:8000')
os.environ.setdefault('http_proxy', 'http://172.31.255.9:8000')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import VSIBench data loader
try:
    from datasets import load_dataset
except ImportError:
    logger.warning("datasets library not available, will use pyarrow fallback")
    load_dataset = None

try:
    import pyarrow.ipc as ipc
except ImportError:
    logger.error("pyarrow not installed")
    ipc = None


# ============================================================================
# Video Frame Extraction (from CAMBW)
# ============================================================================

def extract_frames_for_api(video_path: str, max_frames: int = 12, 
                           max_short_edge: int = 512, jpeg_quality: int = 70) -> List[str]:
    """Extract video frames as base64 for API calls."""
    if not video_path or not os.path.isfile(video_path):
        return []
    
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        n = len(vr)
        
        # Uniform sampling
        if n <= max_frames:
            indices = list(range(n))
        else:
            indices = [int(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
        
        frames = vr.get_batch(indices).asnumpy()
        
        # Convert to base64
        base64_frames = []
        for i in range(len(indices)):
            img = Image.fromarray(frames[i]).convert("RGB")
            w, h = img.size
            short_edge = min(w, h)
            if short_edge > max_short_edge:
                scale = max_short_edge / short_edge
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            base64_frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        
        return base64_frames
    except Exception as e:
        logger.warning(f"Frame extraction failed: {e}")
        return []


# ============================================================================
# Gemini 3 Pro API Client
# ============================================================================

class Gemini3ProClient:
    """Client for Gemini 3 Pro API via NEWAPI"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or NEWAPI_API_KEY
        self.base_url = (base_url or NEWAPI_BASE_URL).rstrip("/")
        self.model = model or NEWAPI_MODEL
        self.max_retries = int(os.environ.get("NEWAPI_MAX_RETRIES", "5"))
        
        if not self.api_key:
            raise RuntimeError("NEWAPI_API_KEY not set")
    
    def call_vision(self, prompt: str, video_path: str, max_tokens: int = 512) -> Tuple[str, float]:
        """
        Call Gemini 3 Pro with video frames.
        Returns: (response_text, confidence_score)
        """
        # Extract frames
        frames_b64 = extract_frames_for_api(video_path, max_frames=12)
        if not frames_b64:
            logger.error(f"No frames extracted from {video_path}")
            return "", 0.0
        
        # Build content
        content: List[Dict[str, Any]] = []
        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        content.append({"type": "text", "text": prompt})
        
        # API call
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                
                msg = data["choices"][0]["message"]
                text = msg.get("content", "").strip()
                
                # Simple confidence estimation
                confidence = 0.7
                if text:
                    if len(text) <= 2:  # Single letter answer
                        confidence = 0.9
                    elif re.match(r'^[A-D][\.\s]', text):  # Letter with punctuation
                        confidence = 0.85
                
                return text, confidence
                
            except Exception as e:
                if attempt >= self.max_retries - 1:
                    logger.error(f"API call failed after {self.max_retries} retries: {e}")
                    return "", 0.0
                sleep_sec = min(30, 2 ** attempt)
                logger.warning(f"API call failed (attempt {attempt+1}), retrying in {sleep_sec}s...")
                time.sleep(sleep_sec)
        
        return "", 0.0


# ============================================================================
# Simplified V21 Pipeline (P1 + R1 only)
# ============================================================================

@dataclass
class SampleResult:
    sample_id: str
    question_type: str
    question: str
    video_path: str
    
    # P1: Global VL
    p1_answer: str = ""
    p1_confidence: float = 0.0
    
    # R1: Focused VL (simplified - no actual grid evolution)
    r1_answer: str = ""
    r1_confidence: float = 0.0
    
    # Final
    final_answer: str = ""
    ground_truth: str = ""
    score: float = 0.0
    correct: bool = False


class SimplifiedV21Gemini3Pro:
    """
    Simplified V21 pipeline:
    - P1: Global VL call with full video
    - R1: Focused VL call (same video, but with focused prompt)
    - Final: Compare P1 vs R1, use higher confidence
    """
    
    def __init__(self):
        self.vl_client = Gemini3ProClient()
        self.num_choice_tasks = 0
        self.num_numerical_tasks = 0
    
    def process_sample(self, sample: Dict) -> SampleResult:
        """Process a single VSIBench sample"""
        result = SampleResult(
            sample_id=str(sample.get('id', '')),
            question_type=sample.get('question_type', ''),
            question=sample.get('question', ''),
            video_path=sample.get('video_path', ''),
            ground_truth=str(sample.get('ground_truth', ''))
        )
        
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        options = sample.get('options', [])
        
        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            return result
        
        # Build prompts
        if question_type in ['object_counting', 'object_size_estimation', 
                             'object_abs_distance', 'room_size_estimation']:
            # Numerical tasks
            prompt_p1 = f"{question}\nProvide a numerical answer only."
            prompt_r1 = f"Based on the video, {question}\nProvide a numerical answer only."
            is_choice = False
        else:
            # Choice tasks
            options_str = "\n".join(options) if options else ""
            prompt_p1 = f"{question}\n{options_str}\nAnswer with only the letter (A, B, C, or D)."
            prompt_r1 = f"Based on the video, {question}\n{options_str}\nAnswer with only the letter (A, B, C, or D)."
            is_choice = True
        
        # P1: Global VL
        t0 = time.time()
        p1_text, p1_conf = self.vl_client.call_vision(prompt_p1, video_path)
        p1_time = time.time() - t0
        result.p1_answer = p1_text
        result.p1_confidence = p1_conf
        
        # R1: Focused VL (simplified - just different prompt)
        t0 = time.time()
        r1_text, r1_conf = self.vl_client.call_vision(prompt_r1, video_path)
        r1_time = time.time() - t0
        result.r1_answer = r1_text
        result.r1_confidence = r1_conf
        
        # Final decision: use higher confidence
        if r1_conf > p1_conf:
            final_text = r1_text
            final_conf = r1_conf
            used = "R1"
        else:
            final_text = p1_text
            final_conf = p1_conf
            used = "P1"
        
        # Extract answer
        final_answer = self._extract_answer(final_text, question_type, options)
        result.final_answer = final_answer
        
        # Score
        score = self._compute_score(final_answer, result.ground_truth, question_type)
        result.score = score
        result.correct = score > 0.5 if is_choice else score > 0.3
        
        logger.info(f"[{question_type}] P1={p1_text[:20]} conf={p1_conf:.2f} | "
                   f"R1={r1_text[:20]} conf={r1_conf:.2f} | "
                   f"Final={used} ans={final_answer} gt={result.ground_truth} "
                   f"score={score:.2f} t={p1_time+r1_time:.1f}s")
        
        return result
    
    def _extract_answer(self, text: str, question_type: str, options: List[str]) -> str:
        """Extract answer from model response"""
        text = text.strip()
        
        if question_type in ['object_counting', 'object_size_estimation', 
                             'object_abs_distance', 'room_size_estimation']:
            # Extract number
            match = re.search(r'[-+]?\d*\.?\d+', text)
            return match.group(0) if match else "0"
        else:
            # Extract letter for choice tasks
            match = re.match(r'^([A-D])', text.upper())
            if match:
                return match.group(1)
            
            # Try to match option content
            text_lower = text.lower()
            for i, opt in enumerate(options):
                opt_content = re.sub(r'^[A-D]\.\s*', '', opt).lower()
                if opt_content in text_lower or text_lower in opt_content:
                    return chr(65 + i)  # A, B, C, D
            
            return text[:1].upper() if text else "A"
    
    def _compute_score(self, pred: str, gt: str, question_type: str) -> float:
        """Compute score (MRA for numerical, accuracy for choice)"""
        if question_type in ['object_counting', 'object_size_estimation', 
                             'object_abs_distance', 'room_size_estimation']:
            # Numerical: MRA
            try:
                pred_num = float(re.search(r'[-+]?\d*\.?\d+', pred).group(0))
                gt_num = float(re.search(r'[-+]?\d*\.?\d+', gt).group(0))
                
                epsilon = 1e-8
                rel_error = abs(pred_num - gt_num) / (abs(gt_num) + epsilon)
                thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                mra = sum(1 for t in thresholds if rel_error < (1 - t)) / len(thresholds)
                return mra
            except:
                return 0.0
        else:
            # Choice: accuracy
            return 1.0 if pred.upper() == gt.upper() else 0.0


# ============================================================================
# VSIBench Data Loading
# ============================================================================

def load_vsibench_samples(split: str = 'test') -> List[Dict]:
    """Load VSIBench dataset from parquet file"""
    
    # Try loading from parquet file
    parquet_path = "/home/tione/notebook/tianjungu/hf_cache/vsibench/test_debiased.parquet"
    
    if os.path.exists(parquet_path):
        logger.info(f"Loading VSIBench from {parquet_path}")
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            
            samples = []
            for idx, row in df.iterrows():
                scene_name = row.get('scene_name', '')
                
                # Find video path
                video_path = None
                for video_dir in [
                    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
                    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
                    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
                ]:
                    vp = os.path.join(video_dir, f"{scene_name}.mp4")
                    if os.path.exists(vp):
                        video_path = vp
                        break
                
                if not video_path:
                    continue
                
                samples.append({
                    'id': int(row.get('id', idx)),
                    'scene_name': scene_name,
                    'video_path': video_path,
                    'question': row['question'],
                    'question_type': row['question_type'],
                    'ground_truth': str(row['ground_truth']),
                    'options': row.get('options', []),
                })
            
            logger.info(f"Loaded {len(samples)} samples with videos")
            return samples
        except Exception as e:
            logger.error(f"Failed to load from parquet: {e}")
            import traceback
            traceback.print_exc()
    
    logger.error("Could not load VSIBench dataset")
    return []


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='outputs/vsibench_v21_gemini3pro_simplified')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("VSIBench Simplified V21 + Gemini 3 Pro")
    logger.info("="*60)
    logger.info(f"Worker {args.worker_id}/{args.num_workers}")
    logger.info(f"Model: {NEWAPI_MODEL}")
    
    # Load samples
    samples = load_vsibench_samples()
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    # Shard by worker
    samples = samples[args.worker_id::args.num_workers]
    logger.info(f"Processing {len(samples)} samples")
    
    # Initialize pipeline
    pipeline = SimplifiedV21Gemini3Pro()
    
    # Process samples
    results = []
    for i, sample in enumerate(samples):
        logger.info(f"[{i+1}/{len(samples)}] Sample {sample['id']}")
        result = pipeline.process_sample(sample)
        results.append(result)
        
        # Periodic save
        if (i + 1) % 10 == 0:
            _save_results(results, args.output_dir, args.worker_id)
    
    # Final save
    _save_results(results, args.output_dir, args.worker_id)
    
    # Summary
    _print_summary(results)


def _save_results(results: List[SampleResult], output_dir: str, worker_id: int):
    """Save results to JSON"""
    output_file = os.path.join(output_dir, f'results_worker{worker_id}.json')
    data = [{
        'sample_id': r.sample_id,
        'question_type': r.question_type,
        'question': r.question,
        'p1_answer': r.p1_answer,
        'p1_confidence': r.p1_confidence,
        'r1_answer': r.r1_answer,
        'r1_confidence': r.r1_confidence,
        'final_answer': r.final_answer,
        'ground_truth': r.ground_truth,
        'score': r.score,
        'correct': r.correct,
    } for r in results]
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(results)} results to {output_file}")


def _print_summary(results: List[SampleResult]):
    """Print evaluation summary"""
    print("\n" + "="*60)
    print("VSIBench V21 Simplified + Gemini 3 Pro - Summary")
    print("="*60)
    
    # By task type
    by_task = defaultdict(lambda: {'scores': [], 'correct': 0, 'total': 0})
    for r in results:
        task = r.question_type
        by_task[task]['scores'].append(r.score)
        by_task[task]['total'] += 1
        if r.correct:
            by_task[task]['correct'] += 1
    
    choice_tasks = ['obj_appearance_order', 'object_rel_direction_easy', 
                    'object_rel_direction_medium', 'object_rel_direction_hard',
                    'object_rel_distance', 'route_planning']
    numerical_tasks = ['object_counting', 'object_size_estimation', 
                       'object_abs_distance', 'room_size_estimation']
    
    print("\nChoice Tasks (Accuracy):")
    choice_scores = []
    for task in choice_tasks:
        if task in by_task:
            d = by_task[task]
            acc = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
            choice_scores.extend(d['scores'])
            print(f"  {task:30s}: {acc:5.2f}% ({d['correct']}/{d['total']})")
    
    print("\nNumerical Tasks (MRA):")
    numerical_scores = []
    for task in numerical_tasks:
        if task in by_task:
            d = by_task[task]
            mra = sum(d['scores']) / len(d['scores']) * 100 if d['scores'] else 0
            numerical_scores.extend(d['scores'])
            print(f"  {task:30s}: {mra:5.2f}% ({d['total']} samples)")
    
    print()
    if choice_scores:
        mca = sum(choice_scores) / len(choice_scores) * 100
        print(f"MCA (Choice Overall): {mca:.2f}%")
    if numerical_scores:
        na = sum(numerical_scores) / len(numerical_scores) * 100
        print(f"NA (Numerical Overall): {na:.2f}%")
    
    all_scores = choice_scores + numerical_scores
    if all_scores:
        overall = sum(all_scores) / len(all_scores) * 100
        print(f"Overall: {overall:.2f}%")
    
    print("="*60)


if __name__ == '__main__':
    main()
