#!/usr/bin/env python3
"""
256³ Grid Mind Map — Agentic Pipeline V17 (Co-Evolving Spatial Reasoning)

核心创新: VL ↔ Grid 双向互进化
  Round 1: VL独立推理(全帧) + CODER独立推理(初始Grid) + Grid质量诊断
  Round 2: 双向进化
    - Grid侧: ADD缺失实体 / FILTER低置信检测 → re-CODER
    - VL侧:  Frame Focusing — 用Grid检测信息筛选关键实体共现帧 → VL重新审视
  Round 3: Manager收敛决策 — 根据质量信号综合判断，不hardcode

  设计原则:
    1. 无hardcode type-adaptive路径 — Manager看统一信号做决策
    2. VL进化 = 改变VL看到的内容（聚焦帧），不是给VL更多metadata
    3. Grid进化 = 提升3D表征质量（ADD/FILTER），不只是error handling
    4. 收敛判据 = Grid质量(conf) + VL-CODER一致性 + 轮次
    5. 数值题保持FastPath（已验证有效）
"""

import os, sys, json, re, gc, copy, time, math, logging, traceback
import base64, io
import numpy as np, cv2, torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

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

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

from scripts.grid64_real_test import (
    Grid64, GridEntity, Grid64Builder,
    EXTENDED_VOCABULARY, SYNONYMS, CALIBRATION_OBJECTS,
    _match_name, find_video_path, evaluate_sample, mean_relative_accuracy,
    grid_answer_counting, grid_answer_size, grid_answer_room_size,
    grid_answer_abs_distance, grid_answer_direction, grid_answer_rel_distance,
    grid_answer_appearance_order, grid_answer_route,
)

class Grid256(Grid64):
    GRID_SIZE = 256

class Grid256Builder(Grid64Builder):
    GRID_CLASS = Grid256

    def __init__(self, device='cuda', num_frames=16):
        super().__init__(device=device, num_frames=num_frames)
        # num_frames will be overridden per-video by build_grid_fps

    def build_grid_fps(self, video_path: str, fps: float = 2.0,
                       max_frames: int = 128, target_objects=None):
        """Build Grid using fps-based frame sampling (unified with VL).
        Dynamically sets num_frames based on video duration × fps,
        then delegates to the parent build_grid()."""
        try:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vfps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if vfps > 0 and total > 0:
                n = int((total / vfps) * fps)
                n = max(8, min(n, max_frames, total))
            else:
                n = min(32, max_frames)
        except:
            n = min(32, max_frames)
        self.num_frames = n
        logger.info(f"  Grid fps={fps}: {n} frames (max={max_frames})")
        return self.build_grid(video_path, target_objects)

UNIFIED_FPS = 2.0          # VL and Grid share the same fps
GRID_MAX_FRAMES = 128      # DA3 safe limit on 192GB VRAM
VL_DEFAULT_MAX_PIXELS = 640 * 480

# ============================================================================
# GRID_SLICE — 2D Top-Down Projection
# ============================================================================

OBJECT_COLORS = {
    'chair': (65,105,225), 'table': (139,69,19), 'sofa': (255,140,0), 'couch': (255,140,0),
    'bed': (148,103,189), 'desk': (139,69,19), 'door': (128,128,128), 'window': (135,206,235),
    'tv': (220,20,60), 'television': (220,20,60), 'toilet': (255,182,193), 'sink': (0,191,255),
    'refrigerator': (0,128,128), 'fridge': (0,128,128), 'lamp': (255,215,0), 'shelf': (107,142,35),
    'cabinet': (85,107,47), 'pillow': (255,192,203), 'rug': (188,143,143), 'trash': (169,169,169),
    'microwave': (100,149,237), 'oven': (178,34,34), 'plant': (34,139,34), 'curtain': (216,191,216),
    'monitor': (220,20,60), 'backpack': (128,0,0), 'default': (150,150,150),
}

def _get_entity_color(category):
    cat = category.lower()
    for k, c in OBJECT_COLORS.items():
        if k in cat: return c
    return OBJECT_COLORS['default']


def generate_grid_slice(grid, question="", options=None, image_size=640, margin=60):
    """从Grid256生成2D俯视投影图 (XZ平面)"""
    if options is None: options = []
    img = Image.new('RGB', (image_size, image_size), (255,255,255))
    draw = ImageDraw.Draw(img)
    eff = image_size - 2 * margin

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        tfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        sfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except: font = tfont = sfont = ImageFont.load_default()

    positions = {}
    for eid, e in grid.entities.items():
        if e.position_3d is not None:
            positions[eid] = (float(e.position_3d[0]), float(e.position_3d[2]), e)

    if not positions:
        draw.text((image_size//2-50, image_size//2), "No data", fill=(128,128,128), font=font)
        return img

    cam_xz = []
    if grid.camera_positions:
        for c in grid.camera_positions:
            p = c.get('position')
            if p is not None and len(p) >= 3: cam_xz.append((float(p[0]), float(p[2])))

    all_xs = [p[0] for p in positions.values()] + [c[0] for c in cam_xz]
    all_zs = [p[1] for p in positions.values()] + [c[1] for c in cam_xz]
    x_min, x_max = min(all_xs), max(all_xs)
    z_min, z_max = min(all_zs), max(all_zs)
    xr = max(x_max - x_min, 0.5); zr = max(z_max - z_min, 0.5)
    x_min -= xr*0.05; x_max += xr*0.05; z_min -= zr*0.05; z_max += zr*0.05
    xr = x_max - x_min; zr = z_max - z_min
    scale = eff / max(xr, zr)

    def to_px(x, z):
        px = int(margin + (x - x_min) * scale)
        pz = int(image_size - margin - (z - z_min) * scale)
        return max(margin, min(image_size-margin, px)), max(margin, min(image_size-margin, pz))

    # Grid lines + border
    for i in range(margin, image_size-margin+1, max(50, eff//10)):
        draw.line([(i,margin),(i,image_size-margin)], fill=(240,240,240))
        draw.line([(margin,i),(image_size-margin,i)], fill=(240,240,240))
    draw.rectangle([margin,margin,image_size-margin,image_size-margin], outline=(200,200,200), width=2)

    # Camera trajectory
    if len(cam_xz) >= 2:
        cpx = [to_px(cx,cz) for cx,cz in cam_xz]
        for i in range(len(cpx)-1):
            draw.line([cpx[i], cpx[i+1]], fill=(200,200,230), width=1)
        sx,sy = cpx[0]
        draw.polygon([(sx,sy-8),(sx-5,sy+4),(sx+5,sy+4)], fill=(100,100,180))
        draw.text((sx+6,sy-4), "Start", fill=(100,100,180), font=sfont)

    # Highlight entities
    hl_eids = set()
    if question:
        rel_names = _extract_question_entities(question, options)
        for eid, e in grid.entities.items():
            if any(_match_name(rn, e.category) for rn in rel_names):
                hl_eids.add(eid)

    # Distance lines between highlighted
    hl_items = [(eid, positions[eid]) for eid in hl_eids if eid in positions]
    if len(hl_items) >= 2:
        for i in range(len(hl_items)):
            for j in range(i+1, len(hl_items)):
                e1, (x1,z1,_) = hl_items[i]; e2, (x2,z2,_) = hl_items[j]
                p1, p2 = to_px(x1,z1), to_px(x2,z2)
                draw.line([p1,p2], fill=(255,100,100), width=2)
                d = grid.physical_distance(e1, e2)
                if d is not None:
                    mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
                    draw.text((mx-12,my-10), f"{d:.1f}m", fill=(200,0,0), font=font)

    # Draw entities
    for hl_pass in [False, True]:
        for eid, (x,z,entity) in positions.items():
            is_hl = eid in hl_eids
            if is_hl != hl_pass: continue
            px, pz = to_px(x, z)
            color = _get_entity_color(entity.category)
            r = 10 if is_hl else 6
            oc = (255,0,0) if is_hl else (80,80,80)
            ow = 3 if is_hl else 1
            draw.ellipse([px-r,pz-r,px+r,pz+r], fill=color, outline=oc, width=ow)
            tc = (180,0,0) if is_hl else (60,60,60)
            draw.text((px+r+2,pz-5), entity.category[:14], fill=tc, font=font if is_hl else sfont)

    draw.text((8,6), "Room Layout — Top-Down (XZ)", fill=(0,0,0), font=tfont)

    # Scale bar
    if scale > 0:
        sm = 1.0; sp = sm * scale
        if sp > eff*0.3: sm, sp = 0.5, 0.5*scale
        elif sp < 30: sm, sp = 2.0, 2.0*scale
        draw.line([(margin, image_size-18),(margin+int(sp), image_size-18)], fill=(0,0,0), width=2)
        draw.text((margin, image_size-32), f"{sm:.1f}m", fill=(0,0,0), font=sfont)

    return img


# ============================================================================
# VL Model Wrapper — fps=2
# ============================================================================

def _get_video_fps_nframes(video_path, target_fps=2.0):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 32
        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vfps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        if vfps <= 0: return 32
        n = max(8, int((tf / vfps) * target_fps))
        return min(n, 300)
    except: return 32


def extract_focused_frames(video_path, frame_indices, max_frames=8):
    """Extract specific frames from video by index. Returns list of PIL Images."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sorted(set(fi for fi in frame_indices if 0 <= fi < total))
        if len(indices) > max_frames:
            step = len(indices) / max_frames
            indices = [indices[int(i * step)] for i in range(max_frames)]
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
    except Exception as e:
        logger.warning(f"extract_focused_frames error: {e}")
    return frames


class VLModel:
    def __init__(self, device='cuda'):
        self.device = device; self.model = None; self.processor = None

    def load(self, model_path):
        if self.model is not None: return
        logger.info(f"Loading VL model: {model_path}")
        try:
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
        except Exception as e:
            logger.error(f"Failed to load VL: {e}"); traceback.print_exc()

    def unload(self):
        if self.model: del self.model; self.model = None
        if self.processor: del self.processor; self.processor = None
        gc.collect(); torch.cuda.empty_cache()

    def call(self, prompt, video_path, max_tokens=512, nframes=None,
             max_pixels=VL_DEFAULT_MAX_PIXELS, fps=None, temperature=0.0,
             images=None):
        if self.model is None: return ""
        try:
            from qwen_vl_utils import process_vision_info
            if nframes is None:
                nframes = _get_video_fps_nframes(video_path, fps if fps else UNIFIED_FPS)
            content = [{"type":"video","video":video_path,"max_pixels":max_pixels,"nframes":nframes}]
            if images:
                for im in images:
                    buf = io.BytesIO(); im.save(buf, format='PNG'); buf.seek(0)
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    content.append({"type":"image","image":f"data:image/png;base64,{b64}"})
            content.append({"type":"text","text":prompt})
            messages = [{"role":"user","content":content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                    padding=True, return_tensors="pt").to(self.device)
            gk = dict(max_new_tokens=max_tokens)
            if temperature > 0: gk.update(do_sample=True, temperature=temperature, top_p=0.8, top_k=20)
            else: gk['do_sample'] = False
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gk)
            resp = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            return resp.strip()
        except Exception as e:
            logger.warning(f"VL call failed: {e}"); return ""

    def call_with_frames(self, prompt, frames, max_tokens=256, temperature=0.0,
                         max_pixels=VL_DEFAULT_MAX_PIXELS, images=None):
        """Call VL with a list of PIL Image frames instead of video file.
        Frames are passed as a list-of-images video to qwen_vl_utils."""
        if self.model is None or not frames: return ""
        try:
            from qwen_vl_utils import process_vision_info
            frame_data = []
            for f in frames:
                buf = io.BytesIO(); f.save(buf, format='JPEG', quality=85); buf.seek(0)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                frame_data.append(f"data:image/jpeg;base64,{b64}")
            content = [{"type":"video","video":frame_data,"max_pixels":max_pixels}]
            if images:
                for im in images:
                    buf = io.BytesIO(); im.save(buf, format='PNG'); buf.seek(0)
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    content.append({"type":"image","image":f"data:image/png;base64,{b64}"})
            content.append({"type":"text","text":prompt})
            messages = [{"role":"user","content":content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                    padding=True, return_tensors="pt").to(self.device)
            gk = dict(max_new_tokens=max_tokens)
            if temperature > 0: gk.update(do_sample=True, temperature=temperature, top_p=0.8, top_k=20)
            else: gk['do_sample'] = False
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gk)
            resp = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            return resp.strip()
        except Exception as e:
            logger.warning(f"VL call_with_frames failed: {e}"); return ""

    def call_sampled(self, prompt, video_path, max_tokens=128, n_samples=3,
                     temperature=0.7, top_p=0.9, nframes=None,
                     max_pixels=VL_DEFAULT_MAX_PIXELS):
        """SC投票: 采样多次用于Self-Consistency. 优先num_return_sequences, fallback逐次."""
        if self.model is None: return [""]
        try:
            from qwen_vl_utils import process_vision_info
            if nframes is None:
                nframes = _get_video_fps_nframes(video_path, UNIFIED_FPS)
            messages = [{"role":"user","content":[
                {"type":"video","video":video_path,"max_pixels":max_pixels,"nframes":nframes},
                {"type":"text","text":prompt}
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
                return [r.strip() for r in self.processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)]
            except Exception:
                responses = []
                with torch.no_grad():
                    for _ in range(n_samples):
                        outputs = self.model.generate(
                            **inputs, max_new_tokens=max_tokens,
                            do_sample=True, temperature=temperature, top_p=top_p)
                        resp = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                        responses.append(resp.strip())
                return responses
        except Exception as e:
            logger.warning(f"VL call_sampled failed: {e}"); return [""]


# ============================================================================
# Context + Entity Extraction + Grid Text
# ============================================================================

class ToolExecutionContext:
    def __init__(self, grid, vl, video_path, builder, question, options, question_type=""):
        self.grid = grid; self.vl = vl; self.video_path = video_path
        self.builder = builder; self.question = question; self.options = options
        self.question_type = question_type
        self.tool_trace = []; self.vl_calls = 0; self._final_answer = None


def _extract_question_entities(question, options):
    entities = []; q = question.lower()
    patterns = [
        r'standing by (?:the )?(.+?)\s+and\s+facing (?:the )?(.+?),.*?(?:is|are)\s+(?:the )?(.+?)\s+(?:to|on)\s',
        r'distance between (?:the )?(.+?)\s+and\s+(?:the )?(.+?)[\s\?\.]',
        r'from (?:the )?(\w+(?:\s+\w+)*?) to (?:the )?(\w+(?:\s+\w+)*)',
        r'how (?:big|large|tall|long|wide) (?:is|are) (?:the )?(.+?)[\?\.]',
        r'how many (.+?)(?:\s+are|\s+in|\s+do|\?)',
        r'size of (?:the )?(.+?)[\?\s]', r'closest to (?:the )?(.+?)[\?\s]',
        r'farthest from (?:the )?(.+?)[\?\s]',
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            for g in m.groups():
                if g: entities.append(g.strip())
    for opt in (options or []):
        oc = opt.strip()
        if len(oc) >= 3 and oc[1] in '.、':
            c = oc[3:].strip().lower()
            skip = {'left','right','behind','front','back','front-left','front-right','back-left','back-right','yes','no'}
            if c and c not in skip and not c.replace('.','').isdigit(): entities.append(c)
    return list(set(entities))


def _grid_to_text_focused(grid, question, options):
    rel = _extract_question_entities(question, options)
    lines = [f"Scene: {len(grid.entities)} objects, mpg={grid.meters_per_grid:.4f}m, span≈{grid.meters_per_grid*grid.GRID_SIZE:.1f}m"]
    related, other = [], []
    for eid, e in sorted(grid.entities.items()):
        if any(_match_name(r, e.category) for r in rel): related.append((eid, e))
        else: other.append((eid, e))
    if related:
        lines.append(f"\n[Relevant: {len(related)} entities]")
        for eid, e in related:
            p = grid.grid_to_physical(e.grid_position); ps = grid.physical_size(eid)
            sz = f", size≈{ps:.2f}m" if ps else ""
            nf = len(set(d['frame_order'] for d in e.detections)) if e.detections else 0
            tf = len(grid.camera_positions) if grid.camera_positions else 16
            lines.append(f"  {eid}: pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})m{sz}, conf={e.confidence:.2f}, "
                        f"seen={nf}/{tf}, count={e.count_in_frame}, first={e.first_seen_frame}")
        # Pairwise distances between relevant entities
        if len(related) >= 2:
            lines.append(f"\n[Pairwise Distances]")
            for i in range(len(related)):
                for j in range(i+1, len(related)):
                    d = grid.physical_distance(related[i][0], related[j][0])
                    if d is not None:
                        lines.append(f"  {related[i][1].category} ↔ {related[j][1].category}: {d:.2f}m")
    if other:
        lines.append(f"\n[Other: {len(other)}]")
        s = [f"{eid}@({grid.grid_to_physical(e.grid_position)[0]:.1f},{grid.grid_to_physical(e.grid_position)[1]:.1f},{grid.grid_to_physical(e.grid_position)[2]:.1f})" for eid, e in other[:10]]
        if len(other) > 10: s.append(f"...+{len(other)-10}")
        lines.append("  " + ", ".join(s))
    return "\n".join(lines)


def _grid_to_text_full(grid):
    lines = [f"Scene: {len(grid.entities)} objects, mpg={grid.meters_per_grid:.4f}m"]
    for eid, e in sorted(grid.entities.items()):
        p = grid.grid_to_physical(e.grid_position); ps = grid.physical_size(eid)
        sz = f", size≈{ps:.2f}m" if ps else ""
        nf = len(set(d['frame_order'] for d in e.detections)) if e.detections else 0
        lines.append(f"  {eid}: pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})m{sz}, conf={e.confidence:.2f}, "
                    f"seen={nf}, count={e.count_in_frame}, first={e.first_seen_frame}")
    return "\n".join(lines)


# ============================================================================
# Tools: CODER, CRITIC, EVOLUTOR (same as V14)
# ============================================================================

def coder_tool(ctx, computation, **kwargs):
    grid = ctx.grid; c = computation.strip().lower()
    try:
        if c == 'direction':
            p, r = grid_answer_direction(grid, ctx.question, ctx.options)
            ctx.tool_trace.append({'tool':'coder','comp':c,'result':p}); return f"Direction: answer={p}, detail={r}"
        elif c == 'distance':
            o1, o2 = kwargs.get('obj1','').strip(), kwargs.get('obj2','').strip()
            if o1 and o2:
                e1, e2 = grid.get_by_category(o1), grid.get_by_category(o2)
                if e1 and e2:
                    d = grid.physical_distance(e1[0].entity_id, e2[0].entity_id)
                    if d is not None:
                        ctx.tool_trace.append({'tool':'coder','comp':c,'result':f"{d:.2f}m"})
                        return f"Distance({o1},{o2})={d:.2f}m"
                return f"Not found: {[n for n,e in [(o1,grid.get_by_category(o1)),(o2,grid.get_by_category(o2))] if not e]}"
            p, r = grid_answer_abs_distance(grid, ctx.question)
            ctx.tool_trace.append({'tool':'coder','comp':c,'result':p}); return f"Distance: answer={p}m, detail={r}"
        elif c == 'rel_distance':
            p, r = grid_answer_rel_distance(grid, ctx.question, ctx.options)
            ctx.tool_trace.append({'tool':'coder','comp':c,'result':p}); return f"RelDist: answer={p}, detail={r}"
        elif c == 'count':
            p, r = grid_answer_counting(grid, ctx.question)
            ctx.tool_trace.append({'tool':'coder','comp':c,'result':p}); return f"Count: answer={p}, detail={r}"
        elif c == 'size':
            p, r = grid_answer_size(grid, ctx.question)
            ctx.tool_trace.append({'tool':'coder','comp':c,'result':p}); return f"Size: answer={p}cm, detail={r}"
        elif c == 'room_size':
            p, r = grid_answer_room_size(grid, ctx.question)
            ctx.tool_trace.append({'tool':'coder','comp':c,'result':p}); return f"RoomSize: answer={p}sqm, detail={r}"
        elif c == 'appearance_order':
            p, r = grid_answer_appearance_order(grid, ctx.question, ctx.options)
            ctx.tool_trace.append({'tool':'coder','comp':c,'result':p}); return f"Appear: answer={p}, detail={r}"
        elif c == 'route':
            p, r = grid_answer_route(grid, ctx.question, ctx.options)
            ctx.tool_trace.append({'tool':'coder','comp':c,'result':p}); return f"Route: answer={p}, detail={r}"
        else: return f"Unknown '{c}'"
    except Exception as e: return f"Coder error ({c}): {e}"


def critic_tool(ctx, focus_entities="", checkpoints=""):
    gt = _grid_to_text_full(ctx.grid)
    fp = f"\nFocus: {focus_entities}" if focus_entities.strip() else ""
    prompt = f"""You are a quality reviewer. Find errors ONLY.
=== 3D DATA ===
{gt}
=== QUESTION ===
{ctx.question}{fp}
Report: ISSUE: entity=<name>|problem=<desc>|confidence=<h/m/l>
If none: ISSUE: none
SUMMARY: <sentence>"""
    resp = ctx.vl.call(prompt, ctx.video_path, max_tokens=400); ctx.vl_calls += 1
    issues = []; summary = ""
    for line in resp.split('\n'):
        ls = line.strip()
        if ls.upper().startswith('ISSUE:'):
            body = ls.split(':',1)[1].strip()
            if body.lower() in ('none','no issues','n/a',''): continue
            iss = {"raw": body}
            for part in body.split('|'):
                p = part.strip()
                if p.startswith('entity='): iss['entity'] = p.split('=',1)[1].strip()
                elif p.startswith('problem='): iss['problem'] = p.split('=',1)[1].strip()
                elif p.startswith('confidence='): iss['confidence'] = p.split('=',1)[1].strip().lower()
            issues.append(iss)
        elif ls.upper().startswith('SUMMARY:'): summary = ls.split(':',1)[1].strip()
    ctx.tool_trace.append({'tool':'critic','n_issues':len(issues),'issues':issues,'summary':summary})
    if not issues: return f"No issues. {summary}"
    parts = [f"Found {len(issues)} issue(s):"]
    for iss in issues: parts.append(f"  - {iss.get('entity','?')}: {iss.get('problem','?')}")
    return "\n".join(parts)


def evolutor_tool(ctx, action, target, reason=""):
    grid = ctx.grid; action = action.strip().upper()
    if action == 'DELETE':
        eid = target.strip().replace(' ','_')
        if eid not in grid.entities:
            cands = grid.get_by_category(target.strip())
            if cands: eid = cands[0].entity_id
            else:
                ctx.tool_trace.append({'tool':'evolutor','action':'DELETE','target':target,'ok':False})
                return f"DELETE failed: '{target}' not found."
        del grid.entities[eid]
        ctx.tool_trace.append({'tool':'evolutor','action':'DELETE','target':eid,'ok':True})
        return f"DELETE '{eid}' done. {len(grid.entities)} entities left."
    elif action == 'ADD':
        name = target.strip().lower()
        if grid.get_by_category(name): return f"ADD skipped: '{name}' exists."
        if ctx.builder is None: return f"ADD failed: no builder."
        added = ctx.builder.search_and_add_entity(grid, name)
        if added:
            p = grid.grid_to_physical(added.grid_position)
            ctx.tool_trace.append({'tool':'evolutor','action':'ADD','target':name,'ok':True,'eid':added.entity_id})
            return f"ADD '{name}' as '{added.entity_id}' at ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})m"
        ctx.tool_trace.append({'tool':'evolutor','action':'ADD','target':name,'ok':False})
        return f"ADD failed: '{name}' not found."
    elif action == 'FILTER_FRAMES':
        try:
            parts = target.split(':',1)
            if len(parts) != 2: return f"FILTER_FRAMES: bad format"
            ename, fspec = parts[0].strip().lower(), parts[1].strip()
            bad = set()
            for seg in fspec.split(','):
                seg = seg.strip()
                if '-' in seg:
                    try: s,e = map(int, seg.split('-',1)); bad.update(range(s,e+1))
                    except: pass
                else:
                    try: bad.add(int(seg))
                    except: pass
            ents = grid.get_by_category(ename)
            if not ents: return f"FILTER_FRAMES: '{ename}' not found."
            ent = ents[0]; orig = len(ent.detections)
            filt = [d for d in ent.detections if d.get('frame_order') not in bad]
            if len(filt) < orig:
                if filt:
                    pos = np.array([d['position_3d'] for d in filt])
                    ent.position_3d = np.median(pos, axis=0)
                    ent.grid_position = grid.world_to_grid(ent.position_3d)
                    ent.confidence = np.mean([d['confidence'] for d in filt])
                    ent.detections = filt
                    ctx.tool_trace.append({'tool':'evolutor','action':'FILTER_FRAMES','target':target,'ok':True,'frames_removed':orig-len(filt)})
                    return f"FILTER_FRAMES '{ename}': removed {orig-len(filt)} dets"
                else:
                    del grid.entities[ent.entity_id]
                    ctx.tool_trace.append({'tool':'evolutor','action':'FILTER_FRAMES','target':target,'ok':True,'result':'deleted'})
                    return f"FILTER_FRAMES '{ename}': all filtered, removed."
            return f"FILTER_FRAMES '{ename}': no change."
        except Exception as e: return f"FILTER_FRAMES error: {e}"
    elif action == 'SCALE_ADJUST':
        tv = target.strip().lower(); old = grid.meters_per_grid
        if tv == 'auto':
            grid._meters_per_grid = None; grid.calibrate_scale()
            ctx.tool_trace.append({'tool':'evolutor','action':'SCALE_ADJUST','target':'auto','ok':True})
            return f"SCALE_ADJUST auto: {old:.4f}→{grid.meters_per_grid:.4f}"
        try:
            f = float(tv)
            if f <= 0 or f > 100: return f"SCALE_ADJUST: bad factor {f}"
            grid.meters_per_grid = old * f
            ctx.tool_trace.append({'tool':'evolutor','action':'SCALE_ADJUST','target':target,'ok':True})
            return f"SCALE_ADJUST: {old:.4f}×{f}→{grid.meters_per_grid:.4f}"
        except: return f"SCALE_ADJUST: invalid '{target}'"
    return f"Unknown action '{action}'"


# ============================================================================
# Tools: DISTANCE_QUERY, TEMPORAL_INFO (lightweight, no VL calls)
# ============================================================================

def distance_query_tool(ctx, obj1, obj2):
    """Query distance between any two entities. Zero VL cost."""
    grid = ctx.grid
    o1, o2 = obj1.strip(), obj2.strip()
    e1_list, e2_list = grid.get_by_category(o1), grid.get_by_category(o2)
    if not e1_list: return f"'{o1}' not found in scene."
    if not e2_list: return f"'{o2}' not found in scene."
    results = []
    for e1 in e1_list[:3]:
        for e2 in e2_list[:3]:
            d = grid.physical_distance(e1.entity_id, e2.entity_id)
            if d is not None:
                results.append(f"{e1.entity_id} ↔ {e2.entity_id}: {d:.2f}m")
    ctx.tool_trace.append({'tool':'distance_query','obj1':o1,'obj2':o2,'n_results':len(results)})
    if not results: return f"Cannot compute distance between '{o1}' and '{o2}'."
    return "Distance results:\n" + "\n".join(results)


def temporal_info_tool(ctx, entities_str=""):
    """Show temporal appearance timeline for entities. Zero VL cost."""
    grid = ctx.grid
    if entities_str.strip():
        names = [n.strip().lower() for n in entities_str.split(',')]
    else:
        names = [n.lower() for n in _extract_question_entities(ctx.question, ctx.options)]
    if not names:
        names = [e.category.lower() for _, e in sorted(grid.entities.items())][:8]
    total_frames = len(grid.camera_positions) if grid.camera_positions else 16
    lines = [f"Temporal info ({total_frames} total frames):"]
    found_any = False
    for name in names[:10]:
        ents = grid.get_by_category(name)
        if not ents: lines.append(f"  {name}: NOT FOUND"); continue
        for ent in ents[:2]:
            found_any = True
            frames = sorted(set(d.get('frame_order', -1) for d in ent.detections if d.get('frame_order', -1) >= 0))
            first = frames[0] if frames else -1
            last = frames[-1] if frames else -1
            n_seen = len(frames)
            # Compute visibility span
            span_pct = f"{n_seen/total_frames*100:.0f}%" if total_frames > 0 else "?"
            lines.append(f"  {ent.entity_id} ({ent.category}): first_frame={first}, last_frame={last}, "
                        f"seen_in={n_seen}/{total_frames} frames ({span_pct}), count_per_frame={ent.count_in_frame}")
    if not found_any:
        lines.append("  No matching entities found.")
    ctx.tool_trace.append({'tool':'temporal_info','entities':names[:5]})
    return "\n".join(lines)


# ============================================================================
# Auto-ADD, Targeted Filter, Confidence, CODER Type Selection
# ============================================================================

def _extract_not_found(r):
    missing = []
    for m in re.finditer(r"'([^']+)' not found", r): missing.append(m.group(1))
    return list(dict.fromkeys(missing))

def _auto_add(ctx, coder_result):
    missing = _extract_not_found(coder_result)
    if not missing: return False, ""
    logs = []; mod = False
    for n in missing[:3]:
        nc = n.strip().lower()
        if len(nc)<2 or any(w in nc for w in ['the room','into','toward']): continue
        if ctx.grid.get_by_category(nc): continue
        if not ctx.builder: continue
        r = evolutor_tool(ctx, 'ADD', nc)
        logs.append(f"[AUTO_ADD] {r}")
        if 'failed' not in r and 'skipped' not in r: mod = True
    return mod, "\n".join(logs)

def _targeted_filter(ctx, names):
    mod = False; logs = []
    for n in names[:3]:
        found = ctx.grid.get_by_category(n)
        if not found: continue
        e = found[0]
        if not e.detections or len(e.detections) < 4: continue
        ds = sorted(e.detections, key=lambda d: d.get('confidence',0))
        nr = max(1, len(ds)//3)
        bf = [d.get('frame_order',-1) for d in ds[:nr]]
        bf = [f for f in bf if f >= 0]
        if not bf: continue
        r = evolutor_tool(ctx, 'FILTER_FRAMES', f"{n}:{','.join(str(f) for f in bf)}")
        logs.append(f"[FILTER] {n}: {r[:60]}")
        if 'failed' not in r.lower() and 'no change' not in r.lower(): mod = True
    return mod, "\n".join(logs)


# ============================================================================
# V17 NEW: Grid Quality Diagnosis + Frame Co-occurrence
# ============================================================================

def _grid_quality_diagnosis(grid, question, options):
    """Diagnose Grid quality for entities relevant to the question."""
    rel = _extract_question_entities(question, options)
    diag = {'entities': {}, 'missing': [], 'low_conf': [], 'low_coverage': [],
            'high_variance': [], 'overall_quality': 'normal'}
    for name in rel:
        ents = grid.get_by_category(name)
        if not ents:
            diag['missing'].append(name); continue
        e = ents[0]
        nf = len(set(d.get('frame_order', -1) for d in e.detections if d.get('frame_order', -1) >= 0))
        total_frames = len(grid.camera_positions) if grid.camera_positions else 16
        coverage = nf / max(total_frames, 1)
        pos_var = 0.0
        if len(e.detections) >= 2:
            positions = np.array([d['position_3d'] for d in e.detections if 'position_3d' in d])
            if len(positions) >= 2:
                pos_var = float(np.mean(np.std(positions, axis=0)))
        eq = 'good'
        if e.confidence < 0.3: eq = 'poor'; diag['low_conf'].append(name)
        elif coverage < 0.15: eq = 'weak'; diag['low_coverage'].append(name)
        elif pos_var > 0.5: eq = 'unstable'; diag['high_variance'].append(name)
        elif e.confidence < 0.5 or coverage < 0.3: eq = 'fair'
        diag['entities'][name] = {'confidence': round(e.confidence,3), 'coverage': round(coverage,3),
            'n_detections': nf, 'pos_variance': round(pos_var,3), 'quality': eq}
    n_issues = len(diag['missing']) + len(diag['low_conf']) + len(diag['low_coverage']) + len(diag['high_variance'])
    n_total = max(len(rel), 1)
    diag['overall_quality'] = 'good' if n_issues == 0 else ('fair' if n_issues <= n_total*0.3 else 'poor')
    return diag


def _get_cooccurrence_frames(grid, question, options):
    """Get video frame indices where key entities co-appear."""
    rel = _extract_question_entities(question, options)
    if not rel: return [], []
    entity_frames = {}
    for name in rel:
        ents = grid.get_by_category(name)
        if not ents: continue
        frames = set()
        for e in ents[:2]:
            for d in e.detections:
                fi = d.get('frame_idx', -1)
                if fi >= 0: frames.add(fi)
        if frames: entity_frames[name] = frames
    if len(entity_frames) < 2:
        all_frames = set()
        for fs in entity_frames.values(): all_frames.update(fs)
        return sorted(all_frames), list(entity_frames.keys())
    from itertools import combinations
    cooccur = set()
    for n1, n2 in combinations(entity_frames.keys(), 2):
        cooccur.update(entity_frames[n1] & entity_frames[n2])
    if not cooccur:
        all_frames = set()
        for fs in entity_frames.values(): all_frames.update(fs)
        return sorted(all_frames), list(entity_frames.keys())
    return sorted(cooccur), list(entity_frames.keys())


def _extract_involved_entities(coder_result, question, options):
    ents = []
    for m in re.finditer(r'(?:obs|fac|tgt|ref|start|waypoint\d*)=(\S+)', coder_result.lower()):
        n = m.group(1).strip("',\"")
        if n and n not in ('not','found','none'): ents.append(n)
    for m in re.finditer(r'ref=([^,]+)', coder_result.lower()): ents.append(m.group(1).strip())
    for m in re.finditer(r'(\w[\w\s]*?)=[\d.]+m', coder_result.lower()): ents.append(m.group(1).strip())
    ents.extend(_extract_question_entities(question, options))
    seen = set(); result = []
    for e in ents:
        ec = e.strip().lower()
        if ec and len(ec)>=2 and ec not in seen: seen.add(ec); result.append(ec)
    return result[:5]

def _coder_confidence(comp, result, grid=None):
    r = result.lower(); c = comp.strip().lower()
    # Failure signals → always low
    for sig in ['not found','cannot parse','fallback','same 3d position','error','failed','n/a','insufficient data']:
        if sig in r: return 'low'
    # Route: need clear margin
    if 'route_sim' in r:
        scores = [float(m) for m in re.findall(r'score=([-\d.]+)', r)]
        if len(scores) >= 2:
            ss = sorted(scores, reverse=True)
            if ss[0]-ss[1] > 0.3 and ss[0] > 0: return 'normal'
        return 'low'
    # Boundary clamp values → low
    CLAMP_SIGS = ['answer=20.00m','answer=0.10m','answer=200.0cm','answer=5.0cm']
    for sig in CLAMP_SIGS:
        if sig in r: return 'low'
    # room_size: always low (calibration-dependent, high error rate)
    if c == 'room_size': return 'low'
    # distance: always low (calibration-dependent, data shows FastPath 0.509 vs VL 0.783)
    if c == 'distance': return 'low'
    # mpg quality check
    if grid and grid.meters_per_grid > 0.15 and re.search(r'answer=[\d.]+\s*(?:m|cm|sq)', r): return 'low'
    # Reliable types
    if c in {'count','direction','appearance_order','size'}: return 'normal'
    return 'low'  # rel_distance, route, unknown

def _auto_coder_type(question, options):
    q = question.lower()
    if any(k in q for k in ['distance between','how far','meters apart','measuring from the closest point']): return 'rel_distance' if options else 'distance'
    # ROUTE must be checked BEFORE direction — route questions contain "facing" (e.g. "robot facing the TV") 
    # which would match direction. "navigate" is the unique route keyword.
    if any(k in q for k in ['navigate','route','walk from','go from']) and 'robot' in q: return 'route'
    if any(k in q for k in ['standing by','facing','to the left','to the right','to my']): return 'direction'
    if any(k in q for k in ['how many','count','number of']): return 'count'
    # room_size MUST be checked before size (both match "how big")
    if any(k in q for k in ['room size','floor area','square meter','area of the room','size of the room','big is the room']): return 'room_size'
    if any(k in q for k in ['how big','how large','how tall','how long','how wide','size of']): return 'size'
    if any(k in q for k in ['appear first','appears first','seen first']): return 'appearance_order'
    if any(k in q for k in ['route','path','navigate','walk from','go from','turn']): return 'route'
    return None



def _build_vl_independent_prompt(ctx):
    """VL独立判断prompt — 不展示任何Grid/CODER信息"""
    scale_ref = "Reference sizes: Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. Table ~75cm high. Bed ~200cm long. Sofa ~85cm high."
    if ctx.options:
        opts = f"\n=== OPTIONS ===\n{chr(10).join(ctx.options)}"
        inst = "Answer with ONLY the option letter (A, B, C, or D)."
    else:
        opts = ""; inst = "Respond with ONLY a single number (no units, no explanation)."
    return f"""You are analyzing a video of an indoor scene.

=== SCALE REFERENCES ===
{scale_ref}

=== QUESTION ===
{ctx.question}
{opts}

Watch the video carefully. Locate the relevant objects and reason about spatial relationships.
{inst}

Answer:"""


def _build_vl_focused_prompt(ctx, entity_names, n_focused_frames):
    """V17 Round 2 VL Evolving: 聚焦帧prompt — VL知道自己在看关键帧"""
    scale_ref = "Reference sizes: Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. Table ~75cm high. Bed ~200cm long."
    ent_str = ", ".join(entity_names[:5])
    if ctx.options:
        opts = f"\n=== OPTIONS ===\n{chr(10).join(ctx.options)}"
        inst = "Answer with ONLY the option letter (A, B, C, or D)."
    else:
        opts = ""; inst = "Respond with ONLY a single number (no units, no explanation)."
    return f"""You are analyzing selected key frames from an indoor scene video.

=== FRAME SELECTION ===
These {n_focused_frames} frames were specifically selected because the key objects ({ent_str}) are visible together. Pay close attention to the spatial relationships between these objects.

=== SCALE REFERENCES ===
{scale_ref}

=== QUESTION ===
{ctx.question}
{opts}

Focus on the relative positions of [{ent_str}] as shown in these frames.
{inst}

Answer:"""


def _build_convergence_prompt(ctx, round_info):
    """V17 Round 3 Manager convergence: 综合所有轮次信息做最终决策"""
    if ctx.options:
        opts = f"\n=== OPTIONS ===\n{chr(10).join(ctx.options)}"
        inst = "Answer with ONLY the option letter (A, B, C, or D)."
    else:
        opts = ""; inst = "Respond with ONLY a single number (no units, no explanation)."
    return f"""You are a spatial intelligence agent making a final decision by synthesizing multiple rounds of analysis.

=== QUESTION ===
{ctx.question}
{opts}

=== MULTI-ROUND EVIDENCE ===
{round_info}

=== INSTRUCTIONS ===
Consider all evidence above. When sources agree, you can be confident. When they disagree, weigh:
- Visual evidence from focused key frames (most reliable for relative positions)
- 3D computation (reliable when quality is 'good', unreliable when 'poor')
- Full-video visual judgment (good overview but may miss details)

{inst}

ANSWER: <your answer>
CONFIDENCE: <high/medium/low>"""


def _build_room_size_vl_prompt(ctx):
    """专门针对room_size的VL prompt — 完全不提及CODER数值，纯视觉估计"""
    return f"""You are analyzing a video of an indoor scene. Estimate the room's floor area.

=== SCALE REFERENCES ===
A standard door is ~200cm tall, ~90cm wide. A single bed is ~200cm × 100cm (2 sq meters).
A dining table seats 4-6 people and is ~150cm × 90cm (~1.35 sq meters).
A typical bathroom is 4-8 sq meters. A bedroom is 10-20 sq meters. A living room is 15-40 sq meters.

=== QUESTION ===
{ctx.question}

=== INSTRUCTIONS ===
Look at the video and estimate the room size by:
1. Identify the room type (bathroom, bedroom, living room, etc.)
2. Count how many large objects fit in the room
3. Use reference objects to estimate dimensions
4. Calculate approximate floor area

Respond with ONLY a single number in square meters (no units, no explanation).

Answer:"""


def _build_abs_distance_vl_prompt(ctx):
    """专门针对abs_distance的VL prompt — 完全不展示CODER数值，纯视觉估计 (仿V14 _build_decide_prompt low conf)"""
    n_entities = len(ctx.grid.entities) if ctx.grid else 0
    return f"""You are analyzing a video of an indoor scene.

=== SCALE REFERENCES ===
Reference sizes: Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. Table ~75cm high. Bed ~200cm long. Sofa ~85cm high.
Scene has {n_entities} detected objects.

=== 3D PERCEPTION NOTE ===
The 3D system computation may be unreliable for this question. Rely primarily on your visual judgment.

=== QUESTION ===
{ctx.question}

Watch the video carefully. Locate the relevant objects and reason about spatial relationships.
Respond with ONLY a single number (no units, no explanation).

Answer:"""


# ============================================================================
# ============================================================================
# Answer Cleaning
# ============================================================================

def _clean(raw, ctx):
    raw = str(raw).strip()
    if not ctx.options:
        m = re.search(r'[\d.]+', raw); return m.group() if m else '0'
    rc = raw.split('\n')[0].strip()
    for px in ['Answer submitted:','answer:','Answer:','ANSWER:','Final Answer:']:
        if rc.lower().startswith(px.lower()): rc = rc[len(px):].strip()
    m = re.search(r'^([A-Da-d])', rc)
    if m: return m.group(1).upper()
    for line in raw.split('\n'):
        line = line.strip()
        if line and line[0].upper() in 'ABCD' and (len(line)==1 or line[1] in '.、) ,'): return line[0].upper()
    rl = raw.lower()
    for i, opt in enumerate(ctx.options):
        oc = opt[3:].strip().lower() if len(opt)>=3 and opt[1] in '.、' else opt.lower()
        if oc and oc in rl: return chr(65+i)
    m = re.search(r'[\d.]+', raw)
    if m: return m.group()
    return "A"


def _parse_answer(resp, ctx):
    answer = ""; conf = "medium"
    for line in resp.split('\n'):
        ls = line.strip()
        if ls.upper().startswith('ANSWER:'): answer = ls.split(':',1)[1].strip()
        elif ls.upper().startswith('CONFIDENCE:'):
            c = ls.split(':',1)[1].strip().lower()
            if c in ('high','medium','low'): conf = c
    if not answer: answer = resp.strip()
    return _clean(answer, ctx), conf


# ============================================================================
# V17 Co-Evolving Manager Loop
# ============================================================================

def _exec_coder(ctx, coder_type):
    """Execute CODER with auto-ADD evolving. Returns (result, answer, confidence, add_log)."""
    if not coder_type: return '', '', 'low', ''
    r = coder_tool(ctx, coder_type)
    add_log = ''
    if 'not found' in r.lower():
        mod, log = _auto_add(ctx, r)
        if mod:
            r = coder_tool(ctx, coder_type); add_log = log
    cc = _coder_confidence(coder_type, r, ctx.grid)
    ca = ""
    m = re.search(r'answer=([A-D])', r)
    if m: ca = m.group(1)
    elif not ctx.options:
        m = re.search(r'answer=([\d.]+)', r)
        if m: ca = m.group(1)
    return r, ca, cc, add_log


def _quality_based_pick(vl_r1, vl_r2, coder_ans, coder_conf, grid_quality, vl_evolved):
    """When no convergence, pick answer based on quality signals."""
    if coder_ans and coder_conf == 'normal' and grid_quality == 'good':
        return coder_ans
    if vl_evolved and vl_r2 and vl_r2 in 'ABCD':
        return vl_r2
    if vl_r1 and vl_r1 in 'ABCD':
        return vl_r1
    return coder_ans or vl_r2 or 'A'


def _numerical_path(ctx, ct, rp):
    """Numerical tasks: streamlined path (same as V16, proven effective)."""
    rp.append("[numerical]")
    cr, ca, cc, add_log = _exec_coder(ctx, ct)
    if add_log: rp.append(f"[grid_evo] {add_log[:60]}")
    rp.append(f"[coder] ans={ca} conf={cc}")

    if ca and cc in ('normal', 'verified'):
        rp.append(f"[num_fastpath] coder={ca}")
        logger.info(f"  NumFastPath: coder={ca} conf={cc}")
        ctx._final_answer = ca
        return ca, " | ".join(rp)

    if ct == 'room_size':
        room_prompt = _build_room_size_vl_prompt(ctx)
        responses = ctx.vl.call_sampled(room_prompt, ctx.video_path, max_tokens=64, n_samples=3, temperature=0.7)
        ctx.vl_calls += len(responses)
        values = []
        for resp in responses:
            m = re.search(r'[\d.]+', resp)
            if m:
                try:
                    v = float(m.group())
                    if 1.0 <= v <= 200.0: values.append(v)
                except: pass
        if values:
            ans = f"{np.median(values):.1f}"
            rp.append(f"[room_vl_3vote] {[f'{v:.1f}' for v in values]} → {ans}")
        else:
            resp = ctx.vl.call(room_prompt, ctx.video_path, max_tokens=64); ctx.vl_calls += 1
            m = re.search(r'[\d.]+', resp)
            ans = m.group() if m else (ca if ca else '15')
            rp.append(f"[room_fallback] {ans}")
        ctx._final_answer = ans
        return ans, " | ".join(rp)

    if ct == 'distance':
        dist_prompt = _build_abs_distance_vl_prompt(ctx)
        responses = ctx.vl.call_sampled(dist_prompt, ctx.video_path, max_tokens=64, n_samples=3, temperature=0.7)
        ctx.vl_calls += len(responses)
        values = []
        for resp in responses:
            m = re.search(r'[\d.]+', resp)
            if m:
                try:
                    v = float(m.group())
                    if 0.1 <= v <= 30.0: values.append(v)
                except: pass
        if values:
            ans = f"{np.median(values):.1f}"
            rp.append(f"[dist_vl_3vote] {[f'{v:.1f}' for v in values]} → {ans}")
        else:
            resp = ctx.vl.call(dist_prompt, ctx.video_path, max_tokens=64); ctx.vl_calls += 1
            m = re.search(r'[\d.]+', resp)
            ans = m.group() if m else (ca if ca else '2.0')
            rp.append(f"[dist_fallback] {ans}")
        ctx._final_answer = ans
        return ans, " | ".join(rp)

    vl_prompt = _build_vl_independent_prompt(ctx)
    resp = ctx.vl.call(vl_prompt, ctx.video_path, max_tokens=128); ctx.vl_calls += 1
    ans = _clean(resp, ctx)
    rp.append(f"[num_vl] {ans}")
    if ca:
        try:
            if float(ans) <= 0 and float(ca) > 0:
                ans = ca; rp.append(f"[fallback_coder] {ca}")
        except: pass
    ctx._final_answer = ans
    return ans, " | ".join(rp)


def coevolving_loop(ctx, max_rounds=3):
    """V17 Co-Evolving Spatial Reasoning Loop.
    R1: VL(full video) + CODER(initial grid) independently
    R2: Grid evolving(ADD/FILTER) + VL evolving(focused frames)
    R3: Manager convergence — synthesize all evidence
    """
    rp = []
    ct = _auto_coder_type(ctx.question, ctx.options) or ''

    # Numerical tasks: streamlined
    if not ctx.options:
        return _numerical_path(ctx, ct, rp)

    # ──── Round 1: Independent Reasoning ────
    rp.append("[R1:independent]")

    # 1a. VL independent (full video, no CODER info)
    vl_ind_prompt = _build_vl_independent_prompt(ctx)
    vl_r1 = _clean(ctx.vl.call(vl_ind_prompt, ctx.video_path, max_tokens=128), ctx)
    ctx.vl_calls += 1
    rp.append(f"[vl_r1] {vl_r1}")

    # 1b. CODER independent (with auto-ADD grid evolving)
    cr, ca, cc, add_log = _exec_coder(ctx, ct)
    if add_log: rp.append(f"[grid_evo] {add_log[:80]}")
    rp.append(f"[coder_r1] ans={ca} conf={cc}")

    # 1c. Grid quality diagnosis
    diag = _grid_quality_diagnosis(ctx.grid, ctx.question, ctx.options)
    gq = diag['overall_quality']
    rp.append(f"[grid_q] {gq}")

    # 1d. Agreement check
    agree = (vl_r1 == ca) if (vl_r1 and ca and vl_r1 in 'ABCD' and ca in 'ABCD') else False
    rp.append(f"[agree_r1] {'Y' if agree else 'N'}")
    logger.info(f"  R1: VL={vl_r1} CODER={ca}({cc}) agree={agree} grid={gq}")

    # Early exit: agree + good grid + normal CODER conf
    if agree and gq == 'good' and cc == 'normal':
        rp.append(f"[converge_r1] {vl_r1}")
        ctx._final_answer = vl_r1
        return vl_r1, " | ".join(rp)

    if max_rounds < 2:
        ctx._final_answer = vl_r1
        return vl_r1, " | ".join(rp)

    # ──── Round 2: Co-Evolving ────
    rp.append("[R2:co-evolving]")

    # 2a. Grid Evolving: fix quality issues
    grid_evolved = False; ca_r2 = ca; cc_r2 = cc; cr_r2 = cr

    if diag['missing'] and ctx.builder:
        for name in diag['missing']:
            nc = name.strip().lower()
            if ctx.grid.get_by_category(nc): continue
            r = evolutor_tool(ctx, 'ADD', nc)
            if 'failed' not in r and 'skipped' not in r:
                grid_evolved = True; rp.append(f"[grid_evo:add] {r[:60]}")

    if diag['low_conf'] or diag['high_variance']:
        fmod, flog = _targeted_filter(ctx, diag['low_conf'] + diag['high_variance'])
        if fmod:
            grid_evolved = True; rp.append(f"[grid_evo:filter] {flog[:80]}")

    if grid_evolved and ct:
        cr_r2 = coder_tool(ctx, ct)
        cc_r2 = _coder_confidence(ct, cr_r2, ctx.grid)
        m = re.search(r'answer=([A-D])', cr_r2)
        ca_r2 = m.group(1) if m else ca
        rp.append(f"[coder_r2] ans={ca_r2} conf={cc_r2}")
        logger.info(f"  R2 GridEvo: CODER {ca}→{ca_r2} conf={cc}→{cc_r2}")

    # 2b. VL Evolving: Frame Focusing or Temporal-aware
    # Temporal questions (appearance order) need full timeline, not spatial co-occurrence
    q_lower = ctx.question.lower()
    is_temporal = any(k in q_lower for k in ['appear first', 'appears first', 'appearance order',
                                               'seen first', 'first-time appearance', 'show up first'])
    cooc_frames, cooc_ents = _get_cooccurrence_frames(ctx.grid, ctx.question, ctx.options)
    vl_r2 = ''; vl_evolved = False

    if is_temporal:
        # Temporal questions: use entity-relevant frames spread across timeline (not co-occurrence)
        if cooc_frames and len(cooc_frames) >= 3:
            # Use ALL entity-visible frames (union, not intersection) — preserves temporal spread
            all_ent_frames = set()
            rel = _extract_question_entities(ctx.question, ctx.options)
            for name in rel:
                ents = ctx.grid.get_by_category(name)
                if ents:
                    for e in ents[:2]:
                        for d in e.detections:
                            fi = d.get('frame_idx', -1)
                            if fi >= 0: all_ent_frames.add(fi)
            temporal_frames = sorted(all_ent_frames) if all_ent_frames else cooc_frames
            focused_pil = extract_focused_frames(ctx.video_path, temporal_frames, max_frames=12)
            if focused_pil and len(focused_pil) >= 3:
                ent_str = ", ".join(cooc_ents[:5])
                temporal_prompt = f"""You are analyzing selected frames from an indoor scene video, shown in chronological order.

=== FRAME SELECTION ===
These {len(focused_pil)} frames span the full video timeline and show when key objects ({ent_str}) appear. Earlier frames are shown first.

=== QUESTION ===
{ctx.question}

=== OPTIONS ===
{chr(10).join(ctx.options)}

Pay attention to WHEN each object first becomes visible across these frames. The frame order matches the video timeline.
Answer with ONLY the option letter (A, B, C, or D).

Answer:"""
                vl_r2 = _clean(ctx.vl.call_with_frames(temporal_prompt, focused_pil, max_tokens=128), ctx)
                ctx.vl_calls += 1; vl_evolved = True
                rp.append(f"[vl_r2_temporal] {vl_r2} ({len(focused_pil)}f/{len(temporal_frames)}ent)")
                logger.info(f"  R2 VLEvo: temporal {len(focused_pil)}f → {vl_r2}")
    else:
        # Spatial questions: co-occurrence Frame Focusing
        if cooc_frames and len(cooc_frames) >= 2:
            focused_pil = extract_focused_frames(ctx.video_path, cooc_frames, max_frames=8)
            if focused_pil and len(focused_pil) >= 2:
                vl_focused_prompt = _build_vl_focused_prompt(ctx, cooc_ents, len(focused_pil))
                vl_r2 = _clean(ctx.vl.call_with_frames(vl_focused_prompt, focused_pil, max_tokens=128), ctx)
                ctx.vl_calls += 1; vl_evolved = True
                rp.append(f"[vl_r2_focused] {vl_r2} ({len(focused_pil)}f/{len(cooc_frames)}cooc)")
                logger.info(f"  R2 VLEvo: focused {len(focused_pil)}f → {vl_r2}")

    if not vl_evolved:
        si = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
        vl_r2 = _clean(ctx.vl.call(vl_ind_prompt, ctx.video_path, max_tokens=128, images=[si]), ctx)
        ctx.vl_calls += 1
        rp.append(f"[vl_r2_slice] {vl_r2}")

    # 2c. Convergence check (majority of 3 sources)
    votes = Counter()
    for v in [vl_r1, vl_r2, ca_r2]:
        if v and v in 'ABCD': votes[v] += 1

    if votes and votes.most_common(1)[0][1] >= 2:
        ans = votes.most_common(1)[0][0]
        rp.append(f"[converge_r2] {dict(votes)}→{ans}")
        logger.info(f"  R2 Converge: {dict(votes)} → {ans}")
        ctx._final_answer = ans
        return ans, " | ".join(rp)

    if max_rounds < 3:
        ans = _quality_based_pick(vl_r1, vl_r2, ca_r2, cc_r2, gq, vl_evolved)
        rp.append(f"[pick_r2] {ans}")
        ctx._final_answer = ans
        return ans, " | ".join(rp)

    # ──── Round 3: Manager Convergence ────
    rp.append("[R3:convergence]")

    ri = []
    ri.append(f"Round 1 — Full Video VL: Answer={vl_r1}")
    ri.append(f"Round 1 — 3D Computation: Answer={ca} Confidence={cc}")
    ri.append(f"Grid Quality: {gq}")
    if grid_evolved:
        ri.append(f"Round 2 — After Grid repair: CODER Answer={ca_r2} Confidence={cc_r2}")
    if vl_evolved:
        ri.append(f"Round 2 — Focused Frames VL ({len(cooc_frames)} key frames): Answer={vl_r2}")
    else:
        ri.append(f"Round 2 — VL with spatial layout: Answer={vl_r2}")

    valid = [(n, a) for n, a in [('Full-video VL', vl_r1), ('Focused VL', vl_r2), ('3D Computation', ca_r2)] if a and a in 'ABCD']
    unique = set(a for _, a in valid)
    if len(unique) == 1:
        ri.append(f"\nAll sources agree: {list(unique)[0]}")
    else:
        ri.append(f"\nDisagreement: " + ", ".join(f"{n}={a}" for n, a in valid))

    si = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
    conv_resp = ctx.vl.call(_build_convergence_prompt(ctx, "\n".join(ri)),
                            ctx.video_path, max_tokens=256, images=[si])
    ctx.vl_calls += 1
    ans, conf = _parse_answer(conv_resp, ctx)
    rp.append(f"[manager_r3] {ans} conf={conf}")
    logger.info(f"  R3 Manager: {ans} conf={conf}")

    ctx._final_answer = ans
    return ans, " | ".join(rp)


# ============================================================================
# Pipeline + Main
# ============================================================================

class AgenticPipelineV17:
    def __init__(self, device='cuda:0', vl_model_path=None, max_rounds=3, grid_max_frames=128):
        self.device = device
        self.vl_model_path = vl_model_path or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        self.builder = Grid256Builder(device=device)
        self.vl = VLModel(device=device); self.max_rounds = max_rounds
        self.grid_max_frames = grid_max_frames

    def load_models(self): self.builder.load_models(); self.vl.load(self.vl_model_path)
    def unload(self): self.builder.unload(); self.vl.unload()

    def process_scene(self, video_path, samples):
        t0 = time.time()
        grid = self.builder.build_grid_fps(video_path, fps=UNIFIED_FPS, max_frames=self.grid_max_frames)
        bt = time.time()-t0
        nf = self.builder.num_frames
        logger.info(f"  Grid256: {len(grid.entities)} ents, mpg={grid.meters_per_grid:.4f}m, {nf}f ({bt:.1f}s)")
        results = []
        for s in samples:
            gc_ = copy.deepcopy(grid)
            results.append(self._process(gc_, s, video_path))
        return results

    def _process(self, grid, sample, video_path):
        qt = sample['question_type']; q = sample['question']
        opts = sample.get('options') or []; gt = sample['ground_truth']
        ctx = ToolExecutionContext(grid, self.vl, video_path, self.builder, q, opts, qt)
        t0 = time.time()
        pred, reasoning = coevolving_loop(ctx, self.max_rounds)
        elapsed = time.time()-t0
        score = evaluate_sample(qt, pred, gt)
        evo = []; ffc = 0; ci = 0; gm = False; cu = False
        for e in ctx.tool_trace:
            t = e.get('tool','')
            if t == 'critic': ci += e.get('n_issues',0)
            elif t == 'evolutor' and e.get('ok'):
                evo.append(f"{e['action']} {e['target']}"); gm = True
                if e.get('action') == 'FILTER_FRAMES': ffc += e.get('frames_removed',0)
            elif t == 'coder': cu = True
        return {
            'scene_name': sample.get('scene_name',''), 'question_type': qt,
            'question': q, 'ground_truth': gt, 'options': opts,
            'prediction': pred, 'answer': pred, 'reasoning': reasoning[:600], 'score': score,
            'critic_issues_count': ci, 'critic_has_issues': ci>0,
            'grid_modified': gm, 'evolution_actions': evo,
            'filter_frames_count': ffc, 'coder_used': cu,
            'grid_evolved': gm,
            'vl_focused_used': '[vl_r2_focused]' in reasoning or '[vl_r2_temporal]' in reasoning,
            'converged_round': 1 if '[converge_r1]' in reasoning else (2 if '[converge_r2]' in reasoning else 3),
            'vl_calls': ctx.vl_calls, 'elapsed_s': round(elapsed,1),
            'tool_trace': [{'tool':e.get('tool'),'action':e.get('action',''),'ok':e.get('ok',''),'n_issues':e.get('n_issues','')} for e in ctx.tool_trace],
            'v7_vl_score': sample.get('vl_score',0), 'v7_rule_score': sample.get('rule_score',0),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V17 Co-Evolving Spatial Reasoning Pipeline")
    parser.add_argument('--n_per_type', type=int, default=10)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--max_rounds', type=int, default=3)
    parser.add_argument('--grid_max_frames', type=int, default=128)
    parser.add_argument('--vl-model', type=str, default='/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()

    if args.gpu_id is not None:
        vis = os.environ.get('CUDA_VISIBLE_DEVICES','')
        args.device = 'cuda:0' if vis else f'cuda:{args.gpu_id}'

    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    logger.info(f"Loading V7: {v7_path}")
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
                for idx in np.linspace(0, len(avail)-1, n, dtype=int): test_samples.append(avail[idx])
    logger.info(f"Test: {len(test_samples)} samples")

    by_scene = defaultdict(list)
    for s in test_samples: by_scene[s['scene_name']].append(s)
    scene_list = sorted(by_scene.keys())

    if args.gpu_id is not None:
        total = len(scene_list); chunk = (total+args.num_gpus-1)//args.num_gpus
        start = args.gpu_id * chunk; end = min(start+chunk, total)
        my_scenes = scene_list[start:end]
        logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: {len(my_scenes)} scenes")
    else:
        my_scenes = scene_list

    vl_model = getattr(args,'vl_model',None) or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
    pipe = AgenticPipelineV17(device=args.device, vl_model_path=vl_model,
                              max_rounds=args.max_rounds, grid_max_frames=args.grid_max_frames)
    pipe.load_models()

    all_results = []; total_scenes = len(my_scenes)
    for si, sn in enumerate(my_scenes):
        samples = by_scene[sn]; vp = find_video_path(sn)
        if not vp:
            for s in samples:
                all_results.append({'scene_name':sn,'question_type':s['question_type'],'question':s['question'],
                    'ground_truth':s['ground_truth'],'options':s.get('options',[]),'prediction':'0','reasoning':'no video',
                    'score':0.0,'critic_has_issues':False,'critic_issues_count':0,'vl_calls':0,
                    'v7_vl_score':s.get('vl_score',0),'v7_rule_score':s.get('rule_score',0)})
            continue
        logger.info(f"[{si+1}/{total_scenes}] {sn} ({len(samples)} q)")
        try:
            results = pipe.process_scene(vp, samples)
            for r in results:
                all_results.append(r)
                d = r['score']-r['v7_vl_score']; mk = "+" if d>0 else ("-" if d<0 else "=")
                evo = "E!" if r.get('grid_modified') else "  "
                cod = "C" if r.get('coder_used') else " "
                logger.info(f"  {r['question_type'][:25]:25s} [VL:{r['vl_calls']} {evo}{cod}] "
                    f"Score={r['score']:.3f} V7={r['v7_vl_score']:.3f} {mk} | pred={str(r['prediction'])[:15]} gt={str(r['ground_truth'])[:12]} ({r['elapsed_s']:.0f}s)")
        except Exception as e:
            logger.error(f"  Error: {e}"); traceback.print_exc()
            for s in samples:
                all_results.append({'scene_name':sn,'question_type':s['question_type'],'question':s['question'],
                    'ground_truth':s['ground_truth'],'options':s.get('options',[]),'prediction':'0',
                    'reasoning':f'error: {str(e)[:100]}','score':0.0,'critic_has_issues':False,'critic_issues_count':0,
                    'vl_calls':0,'v7_vl_score':s.get('vl_score',0),'v7_rule_score':s.get('rule_score',0)})

    pipe.unload()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.gpu_id is not None:
        od = PROJECT_ROOT / "outputs" / "agentic_pipeline_v17_coevo" / f"gpu{args.gpu_id}"
    else:
        od = PROJECT_ROOT / "outputs" / f"agentic_pipeline_v17_{timestamp}"
    od.mkdir(parents=True, exist_ok=True)

    clean = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)): cr[k] = float(v)
            elif isinstance(v, np.ndarray): cr[k] = v.tolist()
            else: cr[k] = v
        clean.append(cr)
    with open(od / "detailed_results.json", 'w') as f: json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info(f"Results: {od} ({len(all_results)} samples)")

    if args.gpu_id is None:
        _print_summary(all_results, od, timestamp)


def _print_summary(all_results, od, ts):
    print("\n" + "="*120)
    print("Agentic Pipeline V17 — Co-Evolving Spatial Reasoning")
    print(f"Architecture: R1(VL+CODER independent) → R2(Grid Evo + VL Frame Focusing) → R3(Manager Convergence)")
    print(f"Unified fps={UNIFIED_FPS}, Grid max_frames={GRID_MAX_FRAMES}, max_rounds=3 | Samples: {len(all_results)}")
    print("="*120)
    tts = sorted(set(r['question_type'] for r in all_results))
    print(f"\n{'Task':<35} {'N':>4} {'V7':>6} {'V17':>6} {'Δ':>6}  {'VL#':>4} {'GE%':>5} {'VF%':>5} {'R̄':>4} {'t/s':>5}")
    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}  {'-'*4} {'-'*5} {'-'*5} {'-'*4} {'-'*5}")
    a7, a17 = [], []
    for qt in tts:
        qr = [r for r in all_results if r['question_type']==qt]
        v7 = np.mean([r['v7_vl_score'] for r in qr])
        v17 = np.mean([r['score'] for r in qr])
        d = v17-v7; vl = np.mean([r.get('vl_calls',0) for r in qr])
        ge = np.mean([1 if r.get('grid_evolved') else 0 for r in qr])*100
        vf = np.mean([1 if r.get('vl_focused_used') else 0 for r in qr])*100
        avg_r = np.mean([r.get('converged_round',3) for r in qr])
        tavg = np.mean([r.get('elapsed_s',0) for r in qr])
        mk = "+" if d>0.01 else ("-" if d<-0.01 else "=")
        print(f"  {qt:<35} {len(qr):>4} {v7:>5.3f} {v17:>5.3f} {d:>+5.3f}{mk} {vl:>4.1f} {ge:>4.0f}% {vf:>4.0f}% {avg_r:>3.1f} {tavg:>4.0f}s")
        a7.extend([r['v7_vl_score'] for r in qr]); a17.extend([r['score'] for r in qr])
    ov7, ov17 = np.mean(a7), np.mean(a17)
    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'Overall':<35} {len(all_results):>4} {ov7:>5.3f} {ov17:>5.3f} {ov17-ov7:>+5.3f}")
    tvl = sum(r.get('vl_calls',0) for r in all_results)
    avl = tvl/len(all_results) if all_results else 0
    at = np.mean([r.get('elapsed_s',0) for r in all_results])
    r1 = sum(1 for r in all_results if r.get('converged_round')==1)
    r2 = sum(1 for r in all_results if r.get('converged_round')==2)
    r3 = sum(1 for r in all_results if r.get('converged_round')==3)
    ge_n = sum(1 for r in all_results if r.get('grid_evolved'))
    vf_n = sum(1 for r in all_results if r.get('vl_focused_used'))
    print(f"\n  VL: total={tvl}, avg={avl:.1f}/sample | Time: {at:.0f}s/sample")
    print(f"  Convergence: R1={r1} ({100*r1/len(all_results):.1f}%), R2={r2} ({100*r2/len(all_results):.1f}%), R3={r3} ({100*r3/len(all_results):.1f}%)")
    print(f"  Grid Evolved: {ge_n} ({100*ge_n/len(all_results):.1f}%), VL Frame Focused: {vf_n} ({100*vf_n/len(all_results):.1f}%)")
    tc = Counter()
    for r in all_results:
        for e in r.get('tool_trace',[]): tc[e.get('tool','?')] += 1
    print(f"  Tools: {dict(tc)}")
    print(f"\n{'='*60}\n  V17 Overall = {ov17:.4f}  vs  V7 = {ov7:.4f}  (Δ = {ov17-ov7:+.4f})\n{'='*60}")
    summary = {'timestamp':ts,'version':'v17_coevolving','n_samples':len(all_results),
        'overall':{'v7':float(ov7),'v17':float(ov17),'delta':float(ov17-ov7)},
        'avg_vl_calls':float(avl),'avg_time_s':float(at),
        'convergence':{'r1':r1,'r2':r2,'r3':r3},
        'evolving':{'grid_evolved':ge_n,'vl_focused':vf_n},'tool_usage':dict(tc),
        'by_task':{qt:{'n':len([r for r in all_results if r['question_type']==qt]),
            'v7':float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type']==qt])),
            'v17':float(np.mean([r['score'] for r in all_results if r['question_type']==qt])),
        } for qt in tts}}
    with open(od / "summary.json", 'w') as f: json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
