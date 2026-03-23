#!/usr/bin/env python3
"""
VSI-Bench V21 Pipeline with Gemini 2.5 Pro API

Replace local VL model with Gemini 2.5 Pro API.
Keep all V21 pipeline logic unchanged.
"""

import os
import sys
import json
import re
import gc
import time
import logging
import traceback
import copy
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

# NEWAPI configuration
os.environ.setdefault('NEWAPI_API_KEY', 'sk-gVBzVgUnsXCGL0mOTY4dim5lywqHA5Y2ec84EHaZGU4NLLh8')
os.environ.setdefault('NEWAPI_BASE_URL', 'https://api.chataiapi.com/v1')
os.environ.setdefault('NEWAPI_GEMINI25_MODEL', 'gemini-2.5-pro')
os.environ.setdefault('https_proxy', 'http://172.31.255.9:8000')
os.environ.setdefault('http_proxy', 'http://172.31.255.9:8000')
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Gemini 2.5 Pro VL Model (replaces local model)
# ============================================================================

def extract_frames_for_api(video_path: str, max_frames: int = 8, max_short_edge: int = 384, jpeg_quality: int = 60) -> List[str]:
    """Extract video frames as base64 JPEG images."""
    try:
        from decord import VideoReader, cpu
        import numpy as np
        from PIL import Image
        from io import BytesIO
        import base64
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return []
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total = len(vr)
        idx = np.linspace(0, total - 1, min(max_frames, total), dtype=int).tolist()
        frames = vr.get_batch(idx).asnumpy()
        
        out = []
        for f in frames:
            img = Image.fromarray(f).convert("RGB")
            w, h = img.size
            short_edge = min(w, h)
            if short_edge > max_short_edge:
                scale = max_short_edge / short_edge
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return out
    except Exception as e:
        logger.error(f"Failed to extract frames: {e}")
        return []


def call_gemini_vision(prompt: str, video_path: str, max_tokens: int = 512, max_frames: int = 8) -> Tuple[str, str, float]:
    """Call Gemini 2.5 Pro API via NEWAPI.
    
    Returns: (text, answer_letter, confidence)
    """
    import requests
    
    base_url = os.environ.get("NEWAPI_BASE_URL", "https://api.chataiapi.com/v1").rstrip("/")
    api_key = os.environ.get("NEWAPI_API_KEY")
    model = os.environ.get("NEWAPI_GEMINI25_MODEL", "gemini-2.5-pro")
    
    if not api_key:
        raise RuntimeError("NEWAPI_API_KEY not set")
    
    # Extract frames
    frames_b64 = extract_frames_for_api(video_path, max_frames=max_frames)
    if not frames_b64:
        logger.error(f"Failed to extract frames from {video_path}")
        return "", "", 0.0
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # Build content with image frames
    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})
    
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            
            msg = data["choices"][0]["message"]
            text = msg.get("content", "").strip()
            answer_letter = ""
            
            # If content is empty, try reasoning_content
            if not text and msg.get("reasoning_content"):
                reasoning = msg.get("reasoning_content", "").strip()
                # Extract answer from reasoning
                match = re.search(r'(?:answer|选择)[:\s]*([A-D])', reasoning, re.IGNORECASE)
                if match:
                    answer_letter = match.group(1).upper()
                else:
                    match = re.search(r'([A-D])[.\s]*$', reasoning)
                    if match:
                        answer_letter = match.group(1).upper()
            
            # Estimate confidence
            confidence = 0.7
            if text and len(text) < 10:
                confidence = 0.85
            if re.match(r'^[A-D]\b', text):
                confidence = 0.9
            
            return text, answer_letter, confidence
            
        except Exception as e:
            if attempt >= max_retries - 1:
                logger.error(f"API call failed: {e}")
                return "", "", 0.0
            time.sleep(min(30, 2 ** attempt))
    
    return "", "", 0.0


class GeminiVLModel:
    """VL Model wrapper using Gemini 2.5 Pro API"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.processor = True  # Dummy
        self._abcd_ids = {'A': 1, 'B': 2, 'C': 3, 'D': 4}  # Dummy
        
    def load(self, model_path: str = None):
        logger.info("Gemini 2.5 Pro API model ready (via NEWAPI)")
        
    def unload(self):
        pass
        
    def __call__(self, prompt: str, video_path: str, max_tokens: int = 512, **kwargs):
        """Direct call, returns text"""
        text, _, _ = call_gemini_vision(prompt, video_path, max_tokens)
        return text
    
    def call_with_confidence(self, prompt: str, video_path: str, n_options: int = 4,
                            max_tokens: int = 128, **kwargs):
        """Call with confidence estimation"""
        text, answer_letter, conf = call_gemini_vision(prompt, video_path, max_tokens)
        return text, answer_letter, conf, {}


# ============================================================================
# Import V21 Pipeline components
# ============================================================================

from scripts.grid64_agentic_pipeline_v21 import (
    Grid256, Grid256Builder, UNIFIED_FPS, VL_DEFAULT_MAX_PIXELS,
    generate_grid_slice, extract_focused_frames, v21_loop,
    ToolExecutionContext, _get_abcd_token_ids
)
from scripts.grid64_real_test import (
    find_video_path, evaluate_sample
)


# ============================================================================
# V21 Pipeline with Gemini
# ============================================================================

class AgenticPipelineV21:
    """V21 Pipeline with Gemini 2.5 Pro as VL backend"""
    
    def __init__(self, device='cuda:0', vl_model_path=None, max_rounds=3, grid_max_frames=128):
        self.device = device
        self.vl_model_path = vl_model_path
        self.builder = Grid256Builder(device=device)
        self.vl = GeminiVLModel(device=device)
        self.max_rounds = max_rounds
        self.grid_max_frames = grid_max_frames
        self.abcd_ids = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        
    def load_models(self):
        self.builder.load_models()
        self.vl.load()
        logger.info(f"ABCD token IDs: {self.abcd_ids}")
        
    def unload(self):
        self.builder.unload()
        self.vl.unload()
        
    def process_scene(self, video_path, questions, grid=None):
        if grid is None:
            grid = self.builder.build_grid_fps(video_path, fps=UNIFIED_FPS,
                                               max_frames=self.grid_max_frames)
        results = []
        for sample in questions:
            q = sample['question']
            opts = sample.get('options') or []
            gt = sample['ground_truth']
            qt = sample['question_type']
            
            gc_ = copy.deepcopy(grid)
            ctx = ToolExecutionContext(gc_, self.vl, video_path, self.builder, q, opts, qt)
            t0 = time.time()
            
            try:
                ans, reasoning = v21_loop(ctx, max_rounds=self.max_rounds, abcd_ids=self.abcd_ids)
            except Exception as e:
                logger.error(f"  Error: {e}")
                ans = 'A'
                reasoning = f"[error] {e}"
            
            elapsed = time.time() - t0
            score = evaluate_sample(qt, ans, gt)
            
            ffc = sum(1 for x in ctx.tool_trace if x.get('action') == 'FILTER_FRAMES' and x.get('ok'))
            cu = any(x.get('tool') == 'coder' for x in ctx.tool_trace)
            gm = any(x.get('tool') == 'evolutor' for x in ctx.tool_trace)
            
            results.append({
                'scene_name': sample.get('scene_name', ''),
                'question_type': qt, 'question': q, 'ground_truth': gt,
                'options': opts, 'prediction': ans, 'answer': ctx._final_answer,
                'reasoning': reasoning, 'score': score,
                'belief_modified': gm,
                'filter_frames_count': ffc, 'coder_used': cu,
                'vl_focused_used': '[vl_focused_' in reasoning,
                'converged': 'confident_consensus' in reasoning or 'evolution_stable' in reasoning,
                'vl_calls': ctx.vl_calls, 'elapsed_s': round(elapsed, 1),
            })
            
            logger.info(f"  [{qt}] ans={ans} gt={gt} score={score:.3f} "
                        f"vl={ctx.vl_calls} t={elapsed:.0f}s | {reasoning[:80]}")
        return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="V21 + Gemini 2.5 Pro")
    parser.add_argument('--n_per_type', type=int, default=10)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--max_rounds', type=int, default=3)
    parser.add_argument('--grid_max_frames', type=int, default=128)
    parser.add_argument('--vl-model', type=str, default=None)
    args = parser.parse_args()
    
    if args.gpu_id is not None:
        args.device = f'cuda:{args.gpu_id}'
    
    # Load V7 results as test samples
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    logger.info(f"Loading V7: {v7_path}")
    with open(v7_path) as f:
        v7_results = json.load(f)
    logger.info(f"V7: {len(v7_results)} samples")
    
    # Filter samples
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
                for idx in np.linspace(0, len(avail)-1, n, dtype=int):
                    test_samples.append(avail[idx])
    logger.info(f"Test: {len(test_samples)} samples")
    
    # Group by scene
    by_scene = defaultdict(list)
    for s in test_samples:
        by_scene[s['scene_name']].append(s)
    scene_list = sorted(by_scene.keys())
    
    # Shard
    if args.gpu_id is not None:
        total = len(scene_list)
        chunk = (total + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * chunk
        end = min(start + chunk, total)
        my_scenes = scene_list[start:end]
        logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: {len(my_scenes)} scenes")
    else:
        my_scenes = scene_list
    
    # Run pipeline
    pipe = AgenticPipelineV21(device=args.device, vl_model_path=args.vl_model,
                               max_rounds=args.max_rounds, grid_max_frames=args.grid_max_frames)
    pipe.load_models()
    
    all_results = []
    total_scenes = len(my_scenes)
    start_time = time.time()
    
    for si, sn in enumerate(my_scenes):
        samples = by_scene[sn]
        vp = find_video_path(sn)
        
        if not vp:
            for s in samples:
                all_results.append({
                    'scene_name': sn, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []), 'prediction': '0', 'reasoning': 'no video',
                    'score': 0.0, 'vl_calls': 0
                })
            continue
        
        # Progress
        elapsed = time.time() - start_time
        rate = (si + 1) / elapsed if elapsed > 0 else 0
        eta = (total_scenes - si - 1) / rate / 60 if rate > 0 else 0
        logger.info(f"[{si+1}/{total_scenes}] {sn} ({len(samples)} q) ETA: {eta:.0f}min")
        
        try:
            results = pipe.process_scene(vp, samples)
            for r in results:
                all_results.append(r)
        except Exception as e:
            logger.error(f"  Error: {e}")
            traceback.print_exc()
    
    # Save results
    output_dir = PROJECT_ROOT / "outputs" / f"vsibench_v21_gemini25pro_gpu{args.gpu_id or 0}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "results.json"
    
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compute accuracy
    correct = sum(1 for r in all_results if r.get('score', 0) == 1.0)
    total = len(all_results)
    accuracy = correct / total if total > 0 else 0
    
    logger.info(f"GPU {args.gpu_id or 0} done: {correct}/{total} = {accuracy:.4f}")
    logger.info(f"Total time: {(time.time() - start_time)/60:.1f} min")


if __name__ == '__main__':
    main()
