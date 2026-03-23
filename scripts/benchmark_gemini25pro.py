#!/usr/bin/env python3
"""
Video-MME Benchmark Evaluation using Gemini 2.5 Pro API (via NEWAPI).

Usage:
    python scripts/benchmark_gemini25pro.py --gpu_id 0 --num_gpus 8 --output_dir outputs/benchmark_videomme_gemini25pro
"""

import os
import sys
import json
import re
import gc
import time
import traceback
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# NEWAPI configuration
os.environ.setdefault('NEWAPI_API_KEY', 'sk-gVBzVgUnsXCGL0mOTY4dim5lywqHA5Y2ec84EHaZGU4NLLh8')
os.environ.setdefault('NEWAPI_BASE_URL', 'https://api.chataiapi.com/v1')
os.environ.setdefault('NEWAPI_GEMINI25_MODEL', 'gemini-2.5-pro')
os.environ.setdefault('https_proxy', 'http://172.31.255.9:8000')
os.environ.setdefault('http_proxy', 'http://172.31.255.9:8000')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = '/home/tione/notebook/tianjungu/datasets/VideoMME'
VIDEO_DIR = os.path.join(DATA_ROOT, 'videos/data')


def extract_frames_for_api(video_path: str, max_frames: int = 12, max_short_edge: int = 384, jpeg_quality: int = 60) -> List[str]:
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
        logger.error(f"Failed to extract frames from {video_path}: {e}")
        return []


def call_gemini25pro_vision(prompt: str, video_path: str, max_tokens: int = 512, max_frames: int = 12) -> Tuple[str, float]:
    """Call Gemini 2.5 Pro API via NEWAPI for video understanding using frame extraction."""
    import requests
    
    base_url = os.environ.get("NEWAPI_BASE_URL", "https://api.chataiapi.com/v1").rstrip("/")
    api_key = os.environ.get("NEWAPI_API_KEY")
    model = os.environ.get("NEWAPI_GEMINI25_MODEL", "gemini-2.5-pro")
    
    if not api_key:
        raise RuntimeError("NEWAPI_API_KEY not set")
    
    # Extract frames as images
    frames_b64 = extract_frames_for_api(video_path, max_frames=max_frames)
    if not frames_b64:
        logger.error(f"Failed to extract frames from {video_path}")
        return "", 0.0
    
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
    
    max_retries = int(os.environ.get("NEWAPI_MAX_RETRIES", "3"))
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            
            # Debug logging (only first few calls)
            # logger.info(f"API response keys: {data.keys()}")
            # logger.info(f"API response choices: {data.get('choices', [])}")
            # logger.info(f"API usage: {data.get('usage', {})}")
            
            msg = data["choices"][0]["message"]
            text = msg.get("content", "").strip()
            
            # If content is empty, check reasoning_content
            if not text and msg.get("reasoning_content"):
                reasoning = msg.get("reasoning_content", "").strip()
                logger.info(f"Using reasoning_content: {reasoning[:200]}...")
                
                # Try to extract answer from reasoning text
                # Look for patterns like "The answer is C" or "C." in reasoning
                match = re.search(r'(?:answer|answer is|选择)[:\s]*([A-D])', reasoning, re.IGNORECASE)
                if match:
                    text = match.group(1)
                else:
                    # Try to find letter at end
                    match = re.search(r'([A-D])[.\s]*$', reasoning)
                    if match:
                        text = match.group(1)
            
            logger.info(f"API text: {text[:200] if text else '(empty)'}")
            
            # Estimate confidence
            confidence = 0.7
            if text and len(text) < 10:
                confidence = 0.85
            if re.match(r'^[A-D]\b', text):
                confidence = 0.9
                
            return text, confidence
            
        except Exception as e:
            logger.error(f"API call exception: {e}")
            if attempt >= max_retries - 1:
                logger.error(f"API call failed after {max_retries} retries: {e}")
                return "", 0.0
            sleep_sec = min(30, 2 ** attempt)
            logger.warning(f"API call failed (attempt {attempt+1}), retrying in {sleep_sec}s...")
            time.sleep(sleep_sec)
    
    return "", 0.0


def extract_answer_letter(response: str, n_options: int = 4) -> str:
    """Extract single answer letter from response."""
    valid = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_options])
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


def load_videomme_data() -> pd.DataFrame:
    """Load Video-MME dataset from local parquet."""
    pq_path = os.path.join(DATA_ROOT, 'videomme/test-00000-of-00001.parquet')
    logger.info(f"Loading Video-MME from: {pq_path}")
    df = pd.read_parquet(pq_path)
    logger.info(f"Loaded {len(df)} samples")
    return df


def find_video_path(video_id: str) -> str:
    """Find video path given video_id (e.g., '001' or 'fFjv93ACGo8')."""
    # Try direct match
    path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if os.path.exists(path):
        return path
    
    # Try looking up by videoID from dataset
    return ""


def build_video_path_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build mapping from video_id or videoID to video path."""
    video_map = {}
    
    # First try matching by videoID (YouTube ID)
    for _, row in df.iterrows():
        video_id = row['video_id']
        youtube_id = row.get('videoID', video_id)
        
        # Try different naming patterns
        for vid in [youtube_id, video_id]:
            # Direct file
            path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
            if os.path.exists(path):
                video_map[video_id] = path
                break
            # YouTube ID often has - prefix
            path = os.path.join(VIDEO_DIR, f"-{vid}.mp4")
            if os.path.exists(path):
                video_map[video_id] = path
                break
    
    logger.info(f"Found {len(video_map)} videos in {VIDEO_DIR}")
    return video_map


def evaluate_sample(sample: Dict, video_path: str, video_map: Dict) -> Dict:
    """Evaluate a single Video-MME sample using Gemini 2.5 Pro."""
    try:
        question = sample['question']
        options = sample['options']
        ground_truth = sample['answer']
        question_id = sample['question_id']
        video_id = sample['video_id']
        
        # Get video path
        if video_path:
            actual_video_path = video_path
        else:
            actual_video_path = video_map.get(video_id, "")
        
        if not actual_video_path or not os.path.exists(actual_video_path):
            return {
                'question_id': question_id,
                'video_id': video_id,
                'predicted': '',
                'ground_truth': ground_truth,
                'correct': False,
                'error': f'video_not_found: {video_id} -> {actual_video_path}'
            }
    except Exception as e:
        logger.error(f"Error preparing sample: {e}")
        return {
            'question_id': sample.get('question_id', 'unknown'),
            'video_id': sample.get('video_id', 'unknown'),
            'predicted': '',
            'ground_truth': sample.get('answer', ''),
            'correct': False,
            'error': f'prepare_error: {e}'
        }
    
    # Build prompt with options
    options_text = ""
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        options_text += f"{letter}. {opt}\n"
    
    prompt = f"""Answer the following multiple choice question about the video.
Question: {question}
Options:
{options_text}
Respond with only the letter (A, B, C, or D) that best answers the question."""
    
    try:
        # Use fewer frames (8) and smaller images (384) to reduce payload size
        response, confidence = call_gemini25pro_vision(prompt, actual_video_path, max_tokens=128, max_frames=8)
        predicted = extract_answer_letter(response, n_options=4)
        correct = (predicted.upper() == ground_truth.upper())
        
        return {
            'question_id': question_id,
            'video_id': video_id,
            'question': question,
            'options': options.tolist() if hasattr(options, 'tolist') else list(options),
            'predicted': predicted,
            'ground_truth': ground_truth,
            'correct': correct,
            'confidence': confidence,
            'raw_response': response
        }
    except Exception as e:
        logger.error(f"Error evaluating {question_id}: {e}")
        logger.error(traceback.format_exc())
        return {
            'question_id': question_id,
            'video_id': video_id,
            'predicted': '',
            'ground_truth': ground_truth,
            'correct': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--output_dir', default='outputs/benchmark_videomme_gemini25pro')
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Video-MME evaluation with Gemini 2.5 Pro")
    logger.info(f"GPU {args.gpu_id}/{args.num_gpus}, output: {output_dir}")
    
    # Load data
    df = load_videomme_data()
    
    # Build video path map
    logger.info("Building video path map...")
    video_map = build_video_path_map(df)
    logger.info(f"Found {len(video_map)} videos")
    
    # Shard for this GPU
    samples = df.to_dict('records')
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    shard_size = len(samples) // args.num_gpus
    start_idx = args.gpu_id * shard_size
    end_idx = start_idx + shard_size if args.gpu_id < args.num_gpus - 1 else len(samples)
    my_samples = samples[start_idx:end_idx]
    
    logger.info(f"Processing {len(my_samples)} samples (shard {args.gpu_id})")
    
    # Evaluate
    results = []
    correct = 0
    start_time = time.time()
    
    for i, sample in enumerate(my_samples):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(my_samples) - i) / rate if rate > 0 else 0
            logger.info(f"Progress: {i}/{len(my_samples)} ({100*i/len(my_samples):.1f}%), "
                       f"Rate: {rate:.2f}/s, ETA: {eta/60:.1f}min")
        
        result = evaluate_sample(sample, "", video_map)
        results.append(result)
        
        if result['correct']:
            correct += 1
        
        # Save intermediate results
        if (i + 1) % 100 == 0:
            interim_acc = correct / (i + 1)
            logger.info(f"Intermediate accuracy: {interim_acc:.4f} ({correct}/{i+1})")
    
    # Save results
    elapsed = time.time() - start_time
    accuracy = correct / len(results) if results else 0
    
    result_file = output_dir / f"results_gpu{args.gpu_id}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'gpu_id': args.gpu_id,
            'total': len(results),
            'correct': correct,
            'accuracy': accuracy,
            'elapsed_seconds': elapsed,
            'samples_per_second': len(results) / elapsed if elapsed > 0 else 0,
            'results': results
        }, f, indent=2)
    
    logger.info(f"GPU {args.gpu_id} done: {correct}/{len(results)} = {accuracy:.4f}")
    logger.info(f"Elapsed: {elapsed/60:.1f} min")
    
    return accuracy


if __name__ == '__main__':
    main()
