#!/usr/bin/env python3
"""
VSIBench V21 Pipeline with Gemini 3 Pro API (via NEWAPI)

Replace local VL model (Qwen3.5-9B) with Gemini 3 Pro API calls.
High concurrency supported (no GPU needed for VL).
"""

import os
import sys
import json
import re
import gc
import time
import logging
import traceback
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# NEWAPI configuration
os.environ.setdefault('NEWAPI_API_KEY', 'sk-gVBzVgUnsXCGL0mOTY4dim5lywqHA5Y2ec84EHaZGU4NLLh8')
os.environ.setdefault('NEWAPI_BASE_URL', 'https://api.chataiapi.com/v1')
os.environ.setdefault('NEWAPI_GEMINI25_MODEL', 'gemini-3-pro-preview')
os.environ.setdefault('https_proxy', 'http://172.31.255.9:8000')
os.environ.setdefault('http_proxy', 'http://172.31.255.9:8000')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_gemini3pro_vision(prompt: str, video_path: str, max_tokens: int = 512) -> Tuple[str, float]:
    """Call Gemini 3 Pro API via NEWAPI for video understanding.
    
    Returns: (response_text, confidence)
    Confidence is estimated based on response clarity.
    """
    import requests
    import base64
    
    base_url = os.environ.get("NEWAPI_BASE_URL", "https://api.chataiapi.com/v1").rstrip("/")
    api_key = os.environ.get("NEWAPI_API_KEY")
    model = os.environ.get("NEWAPI_GEMINI25_MODEL", "gemini-3-pro-preview")
    
    if not api_key:
        raise RuntimeError("NEWAPI_API_KEY not set")
    
    # Read video and encode as base64
    try:
        with open(video_path, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode()
    except Exception as e:
        logger.error(f"Failed to read video {video_path}: {e}")
        return "", 0.0
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # Construct message with video
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "video_url",
            "video_url": {
                "url": f"data:video/mp4;base64,{video_data}"
            }
        }
    ]
    
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    
    max_retries = int(os.environ.get("NEWAPI_MAX_RETRIES", "5"))
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            
            msg = data["choices"][0]["message"]
            text = msg.get("content", "").strip()
            
            # Estimate confidence based on response characteristics
            confidence = 0.7  # Base confidence for API model
            if text and len(text) < 10:  # Short direct answer
                confidence = 0.85
            if re.match(r'^[A-D]\b', text):  # Direct letter answer
                confidence = 0.9
                
            return text, confidence
            
        except Exception as e:
            if attempt >= max_retries - 1:
                logger.error(f"API call failed after {max_retries} retries: {e}")
                return "", 0.0
            sleep_sec = min(30, 2 ** attempt)
            logger.warning(f"API call failed (attempt {attempt+1}), retrying in {sleep_sec}s...")
            time.sleep(sleep_sec)
    
    return "", 0.0


class Gemini3ProVLModel:
    """VL Model wrapper using Gemini 3 Pro API"""
    
    def __init__(self, device='cuda'):
        self.device = device  # Not used for API model
        self.model = True  # Dummy to indicate loaded
        self.processor = True
        
    def load(self, model_path: str = None):
        """Verify API key is available"""
        if not os.environ.get("NEWAPI_API_KEY"):
            raise RuntimeError("NEWAPI_API_KEY not set")
        logger.info("Gemini 3 Pro API model ready")
        
    def unload(self):
        pass  # Nothing to unload for API model
        
    def call(self, prompt: str, video_path: str, max_tokens: int = 512, **kwargs) -> str:
        """Single call to Gemini 3 Pro"""
        text, _ = call_gemini3pro_vision(prompt, video_path, max_tokens)
        return text
    
    def call_with_confidence(self, prompt: str, video_path: str, n_options: int = 4,
                             max_tokens: int = 128, **kwargs) -> Tuple[str, str, float, Dict]:
        """Call with confidence estimation (for V21 compatibility)"""
        text, conf = call_gemini3pro_vision(prompt, video_path, max_tokens)
        
        # Extract answer letter for confidence
        conf_letter = ""
        match = re.match(r'^([A-D])\b', text.strip().upper())
        if match:
            conf_letter = match.group(1)
            
        return text, conf_letter, conf, {}


# Import V21 pipeline components
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.grid64_agentic_pipeline_v21 import (
    Grid256, Grid256Builder, UNIFIED_FPS, VL_DEFAULT_MAX_PIXELS,
    generate_grid_slice, extract_focused_frames
)


class VSIBenchGemini3ProV21:
    """V21 Pipeline with Gemini 3 Pro as VL backend"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.builder = Grid256Builder(device=device, num_frames=16)
        self.vl = Gemini3ProVLModel()
        self.max_rounds = 3
        self.grid_max_frames = 128
        
    def load_models(self):
        self.builder.load_models()
        self.vl.load()
        
    def unload(self):
        self.builder.unload()
        self.vl.unload()


def main():
    """Run VSIBench evaluation with Gemini 3 Pro"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--output_dir', default='outputs/vsibench_v21_gemini3pro')
    args = parser.parse_args()
    
    logger.info(f"Starting VSIBench V21 + Gemini 3 Pro (GPU {args.gpu_id})")
    
    # TODO: Load VSIBench dataset and run evaluation
    # This is a placeholder - full implementation would integrate with VSIBench loader
    
    pipeline = VSIBenchGemini3ProV21()
    pipeline.load_models()
    
    logger.info("Models loaded, ready for evaluation")
    

if __name__ == '__main__':
    main()
