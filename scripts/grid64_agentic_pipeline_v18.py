#!/usr/bin/env python3
"""
256³ Grid Mind Map — Agentic Pipeline V18 (Mind-of-Thought: True Closed-Loop)

核心架构 — 认知闭环 (Cognitive Loop):
  0. Belief Construction: Build 3D Grid (Mental Map) from video
  1. Hypothesis Generation: Grid/CODER produces spatial hypothesis + Active Focusing selects key frames
  2. Grounded Verification: VL verifies hypothesis on focused frames WITH spatial context
  3. Belief Correction: VL feedback drives Grid correction → iterate until convergence

设计原则:
  1. VL是唯一决策者 — Grid/CODER提供结构化上下文，不参与投票
  2. 真正的信息交换 — VL看到CODER的推理过程（坐标、空间关系），不只是答案
  3. VL反馈驱动Belief修正 — 当VL否定假设时，修正Grid并重新生成假设
  4. Active Focusing — Grid的3D信念索引关键共现帧，解决注意力稀释
  5. 数值题: CODER-then-VL (CODER先试，不可靠时VL兜底)
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


# ============================================================================
# Grid256 + fps-based builder (same as V17)
# ============================================================================

class Grid256(Grid64):
    GRID_SIZE = 256

class Grid256Builder(Grid64Builder):
    GRID_CLASS = Grid256

    def __init__(self, device='cuda', num_frames=16):
        super().__init__(device=device, num_frames=num_frames)

    def build_grid_fps(self, video_path: str, fps: float = 2.0,
                       max_frames: int = 128, target_objects=None):
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

UNIFIED_FPS = 2.0
GRID_MAX_FRAMES = 128

# Max pixels for VL model (default: 640*480, can override)
VL_DEFAULT_MAX_PIXELS = 640 * 480


# ============================================================================
# GRID_SLICE — 2D Top-Down Projection (same as V17)
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

    for i in range(margin, image_size-margin+1, max(50, eff//10)):
        draw.line([(i,margin),(i,image_size-margin)], fill=(240,240,240))
        draw.line([(margin,i),(image_size-margin,i)], fill=(240,240,240))
    draw.rectangle([margin,margin,image_size-margin,image_size-margin], outline=(200,200,200), width=2)

    if len(cam_xz) >= 2:
        cpx = [to_px(cx,cz) for cx,cz in cam_xz]
        for i in range(len(cpx)-1):
            draw.line([cpx[i], cpx[i+1]], fill=(200,200,230), width=1)
        sx,sy = cpx[0]
        draw.polygon([(sx,sy-8),(sx-5,sy+4),(sx+5,sy+4)], fill=(100,100,180))
        draw.text((sx+6,sy-4), "Start", fill=(100,100,180), font=sfont)

    hl_eids = set()
    if question:
        rel_names = _extract_question_entities(question, options)
        for eid, e in grid.entities.items():
            if any(_match_name(rn, e.category) for rn in rel_names):
                hl_eids.add(eid)

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

    if scale > 0:
        sm = 1.0; sp = sm * scale
        if sp > eff*0.3: sm, sp = 0.5, 0.5*scale
        elif sp < 30: sm, sp = 2.0, 2.0*scale
        draw.line([(margin, image_size-18),(margin+int(sp), image_size-18)], fill=(0,0,0), width=2)
        draw.text((margin, image_size-32), f"{sm:.1f}m", fill=(0,0,0), font=sfont)

    return img


# ============================================================================
# VL Model Wrapper (same as V17)
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
                nframes = _get_video_fps_nframes(video_path, UNIFIED_FPS)
            content = [{"type": "video", "video": f"file://{video_path}",
                       "nframes": nframes, "max_pixels": max_pixels}]
            if images:
                for img in images:
                    buf = io.BytesIO(); img.save(buf, format='PNG')
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    content.append({"type": "image", "image": f"data:image/png;base64,{b64}"})
            content.append({"type": "text", "text": prompt})
            msgs = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(msgs)
            inputs = self.processor(text=[text], images=img_inputs, videos=vid_inputs,
                                   return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                              do_sample=(temperature > 0), temperature=max(temperature, 0.01))
            resp = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            return resp.strip()
        except Exception as e:
            logger.warning(f"VL call failed: {e}"); return ""

    def call_with_frames(self, prompt, frames_pil, max_tokens=512, max_pixels=VL_DEFAULT_MAX_PIXELS, images=None):
        if self.model is None or not frames_pil: return ""
        try:
            from qwen_vl_utils import process_vision_info
            content = []
            for f in frames_pil:
                buf = io.BytesIO(); f.save(buf, format='JPEG', quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({"type": "image", "image": f"data:image/jpeg;base64,{b64}"})
            if images:
                for img in images:
                    buf = io.BytesIO(); img.save(buf, format='PNG')
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    content.append({"type": "image", "image": f"data:image/png;base64,{b64}"})
            content.append({"type": "text", "text": prompt})
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
            logger.warning(f"VL call_with_frames failed: {e}"); return ""

    def call_sampled(self, prompt, video_path, max_tokens=64, n_samples=3,
                     temperature=0.7, top_p=0.9, max_pixels=VL_DEFAULT_MAX_PIXELS):
        if self.model is None: return [""]
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
            responses = []
            with torch.no_grad():
                for _ in range(n_samples):
                    outputs = self.model.generate(**inputs, max_new_tokens=max_tokens,
                        do_sample=True, temperature=temperature, top_p=top_p)
                    resp = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                    responses.append(resp.strip())
            return responses
        except Exception as e:
            logger.warning(f"VL call_sampled failed: {e}"); return [""]


# ============================================================================
# Context + Entity Extraction (same as V17)
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


# ============================================================================
# Grid Text Representations
# ============================================================================

def _grid_to_text_focused(grid, question, options):
    rel = _extract_question_entities(question, options)
    lines = [f"Scene: {len(grid.entities)} objects, scale={grid.meters_per_grid:.4f}m/grid, span≈{grid.meters_per_grid*grid.GRID_SIZE:.1f}m"]
    related, other = [], []
    for eid, e in sorted(grid.entities.items()):
        if any(_match_name(r, e.category) for r in rel): related.append((eid, e))
        else: other.append((eid, e))
    if related:
        lines.append(f"\n[Key Objects: {len(related)}]")
        for eid, e in related:
            p = grid.grid_to_physical(e.grid_position); ps = grid.physical_size(eid)
            sz = f", size≈{ps:.2f}m" if ps else ""
            nf = len(set(d['frame_order'] for d in e.detections)) if e.detections else 0
            tf = len(grid.camera_positions) if grid.camera_positions else 16
            lines.append(f"  {eid}: pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})m{sz}, conf={e.confidence:.2f}, "
                        f"seen={nf}/{tf} frames, count={e.count_in_frame}, first_frame={e.first_seen_frame}")
        if len(related) >= 2:
            lines.append(f"\n[Distances Between Key Objects]")
            for i in range(len(related)):
                for j in range(i+1, len(related)):
                    d = grid.physical_distance(related[i][0], related[j][0])
                    if d is not None:
                        lines.append(f"  {related[i][1].category} ↔ {related[j][1].category}: {d:.2f}m")
    if other:
        lines.append(f"\n[Other Objects: {len(other)}]")
        s = [f"{eid}@({grid.grid_to_physical(e.grid_position)[0]:.1f},{grid.grid_to_physical(e.grid_position)[1]:.1f},{grid.grid_to_physical(e.grid_position)[2]:.1f})" for eid, e in other[:10]]
        if len(other) > 10: s.append(f"...+{len(other)-10}")
        lines.append("  " + ", ".join(s))
    return "\n".join(lines)


# ============================================================================
# Tools: CODER, EVOLUTOR (reused from V17)
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


def evolutor_tool(ctx, action, target):
    grid = ctx.grid; action = action.strip().upper()
    if action == 'ADD':
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
    return f"Unknown action '{action}'"


# ============================================================================
# Helper: Auto-ADD, Targeted Filter, CODER Type, Co-occurrence Frames
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
        logs.append(f"[ADD] {r}")
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


def _auto_coder_type(question, options):
    q = question.lower()
    if any(k in q for k in ['distance between','how far','meters apart','measuring from the closest point']): return 'rel_distance' if options else 'distance'
    if any(k in q for k in ['navigate','route','walk from','go from']) and 'robot' in q: return 'route'
    if any(k in q for k in ['standing by','facing','to the left','to the right','to my']): return 'direction'
    if any(k in q for k in ['how many','count','number of']): return 'count'
    if any(k in q for k in ['room size','floor area','square meter','area of the room','size of the room','big is the room']): return 'room_size'
    if any(k in q for k in ['how big','how large','how tall','how long','how wide','size of']): return 'size'
    if any(k in q for k in ['appear first','appears first','seen first']): return 'appearance_order'
    if any(k in q for k in ['route','path','navigate','walk from','go from','turn']): return 'route'
    return None


def _get_cooccurrence_frames(grid, question, options):
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


def _get_entity_union_frames(grid, question, options):
    """Get union of all frames where any key entity is visible (for temporal questions)."""
    rel = _extract_question_entities(question, options)
    all_frames = set(); names = []
    for name in rel:
        ents = grid.get_by_category(name)
        if not ents: continue
        names.append(name)
        for e in ents[:2]:
            for d in e.detections:
                fi = d.get('frame_idx', -1)
                if fi >= 0: all_frames.add(fi)
    return sorted(all_frames), names


def _is_temporal_question(question):
    q = question.lower()
    return any(k in q for k in ['appear first', 'appears first', 'appearance order',
                                 'seen first', 'first-time appearance', 'show up first'])


# ============================================================================
# Answer Cleaning (same as V17)
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


# ============================================================================
# V18 CORE: Spatial Hypothesis Generation
# ============================================================================

def _generate_spatial_hypothesis(ctx, coder_type, coder_result):
    """Extract a human-readable spatial hypothesis from CODER's computation.
    This is the 'top-down expectation' that VL will verify."""
    c = (coder_type or '').strip().lower()
    r = coder_result

    if c == 'direction':
        # Parse: "obs=X, fac=Y, tgt=Z ... answer=A"
        m_ans = re.search(r'answer=([A-D])', r)
        ans = m_ans.group(1) if m_ans else '?'
        # Extract the spatial reasoning
        m_detail = re.search(r'detail=(.*)', r)
        detail = m_detail.group(1).strip() if m_detail else r
        # Build hypothesis
        m_obs = re.search(r'obs=(\S+)', r.lower())
        m_fac = re.search(r'fac=(\S+)', r.lower())
        m_tgt = re.search(r'tgt=(\S+)', r.lower())
        obs = m_obs.group(1).strip("',") if m_obs else '?'
        fac = m_fac.group(1).strip("',") if m_fac else '?'
        tgt = m_tgt.group(1).strip("',") if m_tgt else '?'
        # Map answer to direction word
        dir_map = {}
        for opt in (ctx.options or []):
            if len(opt) >= 3 and opt[1] in '.、':
                dir_map[opt[0].upper()] = opt[3:].strip().lower()
        dir_word = dir_map.get(ans, '?')
        return (f"3D computation suggests: If you stand at {obs} and face {fac}, "
                f"the {tgt} should be to the {dir_word} (answer={ans}).\n"
                f"Reasoning: {detail[:200]}")

    elif c == 'rel_distance':
        m_ans = re.search(r'answer=([A-D])', r)
        ans = m_ans.group(1) if m_ans else '?'
        m_detail = re.search(r'detail=(.*)', r)
        detail = m_detail.group(1).strip() if m_detail else r
        return (f"3D distance computation suggests answer={ans}.\n"
                f"Reasoning: {detail[:250]}")

    elif c == 'appearance_order':
        m_ans = re.search(r'answer=([A-D])', r)
        ans = m_ans.group(1) if m_ans else '?'
        m_detail = re.search(r'detail=(.*)', r)
        detail = m_detail.group(1).strip() if m_detail else r
        return (f"Frame analysis suggests appearance order answer={ans}.\n"
                f"Reasoning: {detail[:250]}")

    elif c == 'route':
        m_ans = re.search(r'answer=([A-D])', r)
        ans = m_ans.group(1) if m_ans else '?'
        m_detail = re.search(r'detail=(.*)', r)
        detail = m_detail.group(1).strip() if m_detail else r
        return (f"3D route simulation suggests answer={ans}.\n"
                f"Reasoning: {detail[:250]}")

    elif c in ('distance', 'count', 'size', 'room_size'):
        m_ans = re.search(r'answer=([\d.]+)', r)
        ans = m_ans.group(1) if m_ans else '?'
        return f"3D computation result: {ans} ({r[:200]})"

    return f"Computation result: {r[:300]}"


# ============================================================================
# V18 CORE: Prompts — VL as sole decision maker WITH spatial context
# ============================================================================

def _build_grounded_vl_prompt(ctx, hypothesis, focused_info="", round_num=1):
    """VL prompt with spatial hypothesis as context — VL decides, not CODER.
    The hypothesis provides 'top-down expectation'; VL verifies with visual evidence."""
    scale_ref = "Reference sizes: Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. Table ~75cm high. Bed ~200cm long."

    if ctx.options:
        opts = f"\n=== OPTIONS ===\n{chr(10).join(ctx.options)}"
        inst = "Answer with ONLY the option letter (A, B, C, or D)."
    else:
        opts = ""
        inst = "Respond with ONLY a single number (no units, no explanation)."

    hyp_section = ""
    if hypothesis:
        hyp_section = f"""
=== 3D SPATIAL HYPOTHESIS (from scene reconstruction — may contain errors) ===
{hypothesis}

IMPORTANT: This hypothesis comes from noisy 3D estimation and may be WRONG.
Use it as a reference, but TRUST YOUR VISUAL JUDGMENT when it conflicts with what you see."""

    focus_section = ""
    if focused_info:
        focus_section = f"\n=== FRAME SELECTION ===\n{focused_info}"

    round_note = ""
    if round_num > 1:
        round_note = "\nThis is a follow-up analysis. Look more carefully at the spatial relationships."

    return f"""You are analyzing {'selected key frames from ' if focused_info else ''}a video of an indoor scene.

=== SCALE REFERENCES ===
{scale_ref}
{focus_section}
{hyp_section}

=== QUESTION ===
{ctx.question}
{opts}
{round_note}
Examine the visual evidence carefully. {inst}

Answer:"""


def _build_verification_prompt(ctx, hypothesis, entity_names):
    """Ask VL to verify a specific spatial claim from the hypothesis.
    Returns a simple agree/disagree signal for belief correction."""
    ent_str = ", ".join(entity_names[:5])
    return f"""You are verifying a spatial claim about objects in an indoor scene video.

=== CLAIM TO VERIFY ===
{hypothesis}

=== KEY OBJECTS ===
{ent_str}

=== QUESTION ===
Based on what you see in the video, is the spatial claim above CORRECT or INCORRECT?
Consider the relative positions of the objects carefully.

Reply with exactly one word: CORRECT or INCORRECT

Answer:"""


def _build_temporal_vl_prompt(ctx, entity_names, n_frames):
    """Temporal question prompt: VL looks at chronologically ordered frames."""
    ent_str = ", ".join(entity_names[:5])
    return f"""You are analyzing {n_frames} frames from an indoor scene video, shown in chronological order.

=== FRAME SELECTION ===
These frames span the full video timeline. Earlier frames are shown first.
Key objects to track: {ent_str}

=== QUESTION ===
{ctx.question}

=== OPTIONS ===
{chr(10).join(ctx.options)}

Pay attention to WHEN each object first becomes visible across these frames.
The frame order matches the video timeline.
Answer with ONLY the option letter (A, B, C, or D).

Answer:"""


def _build_numerical_vl_prompt(ctx, coder_type):
    """Numerical tasks: VL with scale references."""
    if coder_type == 'room_size':
        return f"""You are analyzing a video of an indoor scene. Estimate the room's floor area.

=== SCALE REFERENCES ===
A standard door is ~200cm tall, ~90cm wide. A single bed is ~200cm × 100cm (2 sq meters).
A dining table seats 4-6 people and is ~150cm × 90cm (~1.35 sq meters).
A typical bathroom is 4-8 sq meters. A bedroom is 10-20 sq meters. A living room is 15-40 sq meters.

=== QUESTION ===
{ctx.question}

Look at the video, identify the room type, count reference objects, and estimate the floor area.
Respond with ONLY a single number in square meters (no units, no explanation).

Answer:"""
    else:
        n_ent = len(ctx.grid.entities) if ctx.grid else 0
        return f"""You are analyzing a video of an indoor scene.

=== SCALE REFERENCES ===
Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. Table ~75cm high. Bed ~200cm long. Sofa ~85cm high.
Scene has {n_ent} detected objects.

=== QUESTION ===
{ctx.question}

Watch the video carefully and use reference objects for scale estimation.
Respond with ONLY a single number (no units, no explanation).

Answer:"""


# ============================================================================
# V18 CORE: Numerical Path (CODER-then-VL, proven in V16/V17)
# ============================================================================

def _numerical_path(ctx, ct, rp):
    """Numerical tasks: CODER tries first, VL verifies/overrides."""
    rp.append("[numerical]")

    # CODER attempt
    cr = coder_tool(ctx, ct)
    add_log = ''
    if 'not found' in cr.lower():
        mod, log = _auto_add(ctx, cr)
        if mod:
            cr = coder_tool(ctx, ct); add_log = log
    if add_log: rp.append(f"[belief_fix] {add_log[:60]}")

    m = re.search(r'answer=([\d.]+)', cr)
    ca = m.group(1) if m else ''
    rp.append(f"[coder] ans={ca}")

    # For room_size/distance: CODER is unreliable, always use VL
    # For count/size: CODER is more reliable
    if ct in ('count', 'size') and ca:
        # Trust CODER for count/size (proven reliable)
        rp.append(f"[num_coder_path] {ca}")
        ctx._final_answer = ca
        return ca, " | ".join(rp)

    # VL 3-vote for room_size, distance; single for others
    if ct in ('room_size', 'distance'):
        prompt = _build_numerical_vl_prompt(ctx, ct)
        responses = ctx.vl.call_sampled(prompt, ctx.video_path, max_tokens=64, n_samples=3, temperature=0.7)
        ctx.vl_calls += len(responses)
        values = []
        for resp in responses:
            m = re.search(r'[\d.]+', resp)
            if m:
                try: values.append(float(m.group()))
                except: pass
        if values:
            ans = str(round(float(np.median(values)), 1))
            rp.append(f"[vl_3vote] {[str(round(v,1)) for v in values]} → {ans}")
            ctx._final_answer = ans
            return ans, " | ".join(rp)

    # Single VL call for other numerical tasks
    prompt = _build_numerical_vl_prompt(ctx, ct)
    resp = ctx.vl.call(prompt, ctx.video_path, max_tokens=64)
    ctx.vl_calls += 1
    ans = _clean(resp, ctx)
    rp.append(f"[vl_num] {ans}")
    ctx._final_answer = ans
    return ans, " | ".join(rp)


# ============================================================================
# V18 CORE: Mind-of-Thought Closed Loop (Choice Tasks)
# ============================================================================

def mot_loop(ctx, max_rounds=3):
    """V18 Mind-of-Thought: True Closed-Loop Spatial Reasoning.

    Loop:
      1. Belief (Grid) generates spatial hypothesis + active focusing
      2. VL verifies hypothesis on focused frames with spatial context
      3. If VL disagrees → correct Belief → re-generate hypothesis
      4. Converge when VL is confident, or max_rounds reached

    VL is the SOLE decision maker. CODER/Grid provide structured context only.
    """
    rp = []
    ct = _auto_coder_type(ctx.question, ctx.options) or ''

    # Numerical tasks: streamlined path
    if not ctx.options:
        return _numerical_path(ctx, ct, rp)

    is_temporal = _is_temporal_question(ctx.question)

    # ──── Phase 0: Initial Belief → Hypothesis ────
    rp.append("[P0:belief]")

    # Ensure key entities exist in Grid
    cr = coder_tool(ctx, ct) if ct else ''
    if cr and 'not found' in cr.lower():
        mod, log = _auto_add(ctx, cr)
        if mod:
            cr = coder_tool(ctx, ct); rp.append(f"[belief_fix] {log[:60]}")

    hypothesis = _generate_spatial_hypothesis(ctx, ct, cr) if cr else ''
    m_coder_ans = re.search(r'answer=([A-D])', cr)
    coder_ans = m_coder_ans.group(1) if m_coder_ans else ''
    rp.append(f"[hypothesis] coder={coder_ans}")
    logger.info(f"  P0: coder_type={ct}, coder_ans={coder_ans}")

    # ──── Phase 1: VL Full-Video Grounded Judgment (with hypothesis as context) ────
    rp.append("[P1:grounded_vl]")

    vl_prompt = _build_grounded_vl_prompt(ctx, hypothesis)
    vl_ans_1 = _clean(ctx.vl.call(vl_prompt, ctx.video_path, max_tokens=128), ctx)
    ctx.vl_calls += 1
    rp.append(f"[vl_full] {vl_ans_1}")
    logger.info(f"  P1: VL(full+hyp)={vl_ans_1}")

    # ──── Phase 2: Active Focusing — Belief guides attention ────
    rp.append("[P2:active_focus]")

    if is_temporal:
        # Temporal: use union of entity-visible frames (preserve timeline)
        focus_frames, focus_ents = _get_entity_union_frames(ctx.grid, ctx.question, ctx.options)
        focus_type = 'temporal'
    else:
        # Spatial: use co-occurrence frames (where entities appear together)
        focus_frames, focus_ents = _get_cooccurrence_frames(ctx.grid, ctx.question, ctx.options)
        focus_type = 'cooccur'

    focused_pil = None
    if focus_frames and len(focus_frames) >= 2:
        max_ff = 12 if is_temporal else 8
        focused_pil = extract_focused_frames(ctx.video_path, focus_frames, max_frames=max_ff)

    vl_ans_2 = ''
    if focused_pil and len(focused_pil) >= 2:
        if is_temporal:
            vl_focus_prompt = _build_temporal_vl_prompt(ctx, focus_ents, len(focused_pil))
        else:
            focus_info = (f"These {len(focused_pil)} frames were selected because the key objects "
                         f"({', '.join(focus_ents[:5])}) are visible together. "
                         f"Pay close attention to their spatial relationships.")
            vl_focus_prompt = _build_grounded_vl_prompt(
                ctx, hypothesis,
                focused_info=focus_info,
                round_num=2)

        vl_ans_2 = _clean(ctx.vl.call_with_frames(vl_focus_prompt, focused_pil, max_tokens=128), ctx)
        ctx.vl_calls += 1
        rp.append(f"[vl_focused_{focus_type}] {vl_ans_2} ({len(focused_pil)}f)")
        logger.info(f"  P2: VL(focused)={vl_ans_2} ({len(focused_pil)} frames, {focus_type})")
    else:
        # Fallback: VL with grid slice image
        si = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
        vl_focus_prompt = _build_grounded_vl_prompt(ctx, hypothesis, round_num=2)
        vl_ans_2 = _clean(ctx.vl.call(vl_focus_prompt, ctx.video_path, max_tokens=128, images=[si]), ctx)
        ctx.vl_calls += 1
        rp.append(f"[vl_slice] {vl_ans_2}")
        logger.info(f"  P2: VL(slice)={vl_ans_2}")

    # ──── Phase 3: Convergence Check + Belief Correction ────
    # If VL full and VL focused agree → high confidence, done
    if vl_ans_1 == vl_ans_2 and vl_ans_1 in 'ABCD':
        rp.append(f"[converge_p2] VL_agree={vl_ans_1}")
        logger.info(f"  P2 converge: both VL agree on {vl_ans_1}")
        ctx._final_answer = vl_ans_1
        return vl_ans_1, " | ".join(rp)

    if max_rounds < 3:
        # Prefer focused VL (it saw more targeted evidence)
        ans = vl_ans_2 if vl_ans_2 in 'ABCD' else vl_ans_1
        rp.append(f"[early_stop] {ans}")
        ctx._final_answer = ans
        return ans, " | ".join(rp)

    # ──── Phase 3: Belief Correction + Re-verification ────
    rp.append("[P3:belief_correction]")

    # VL disagreed with itself → the evidence is ambiguous
    # Try to correct belief: filter low-confidence detections for key entities
    rel_names = _extract_question_entities(ctx.question, ctx.options)
    belief_modified = False

    # Targeted filter: remove bottom-1/3 confidence detections
    for name in rel_names:
        found = ctx.grid.get_by_category(name)
        if not found: continue
        e = found[0]
        if not e.detections or len(e.detections) < 4: continue
        # Check position variance — high variance means unstable detection
        positions = np.array([d['position_3d'] for d in e.detections if 'position_3d' in d])
        if len(positions) < 3: continue
        pos_std = float(np.mean(np.std(positions, axis=0)))
        if pos_std > 0.3:  # Unstable — filter outliers
            ds = sorted(e.detections, key=lambda d: d.get('confidence', 0))
            nr = max(1, len(ds) // 3)
            bf = [d.get('frame_order', -1) for d in ds[:nr]]
            bf = [f for f in bf if f >= 0]
            if bf:
                r = evolutor_tool(ctx, 'FILTER_FRAMES', f"{name}:{','.join(str(f) for f in bf)}")
                if 'removed' in r:
                    belief_modified = True
                    rp.append(f"[filter] {name}: {r[:50]}")
                    logger.info(f"  P3 belief fix: {r}")

    # Also try ADD if any entity is missing
    for name in rel_names:
        if not ctx.grid.get_by_category(name) and ctx.builder:
            r = evolutor_tool(ctx, 'ADD', name)
            if 'failed' not in r and 'skipped' not in r:
                belief_modified = True
                rp.append(f"[add] {r[:50]}")

    # Re-compute hypothesis with corrected belief
    if belief_modified and ct:
        cr_new = coder_tool(ctx, ct)
        hypothesis_new = _generate_spatial_hypothesis(ctx, ct, cr_new)
        m_new = re.search(r'answer=([A-D])', cr_new)
        coder_ans_new = m_new.group(1) if m_new else coder_ans
        rp.append(f"[re_hypothesis] coder={coder_ans_new}")
        hypothesis = hypothesis_new
    else:
        hypothesis_new = hypothesis

    # Final VL call: comprehensive judgment with all evidence
    # Give VL the full picture: video + grid slice + updated hypothesis + summary of disagreement
    si = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
    evidence_summary = (
        f"Previous analysis found conflicting signals:\n"
        f"- Full video analysis suggested: {vl_ans_1}\n"
        f"- Focused frame analysis suggested: {vl_ans_2}\n"
        f"{'- Belief was corrected (noisy detections filtered).' if belief_modified else ''}\n"
        f"Make your final judgment based on the visual evidence."
    )

    final_prompt = f"""You are making a final spatial reasoning judgment about an indoor scene.

=== SCALE REFERENCES ===
Door ~200cm tall, ~90cm wide. Chair seat ~45cm high. Table ~75cm high. Bed ~200cm long.

=== 3D SPATIAL HYPOTHESIS (may contain errors) ===
{hypothesis}

=== PREVIOUS ANALYSIS ===
{evidence_summary}

=== QUESTION ===
{ctx.question}

=== OPTIONS ===
{chr(10).join(ctx.options)}

Look at the video one more time. Focus on the key objects and their spatial relationships.
Trust what you SEE over any computation.
Answer with ONLY the option letter (A, B, C, or D).

Answer:"""

    vl_ans_3 = _clean(ctx.vl.call(final_prompt, ctx.video_path, max_tokens=128, images=[si]), ctx)
    ctx.vl_calls += 1
    rp.append(f"[vl_final] {vl_ans_3}")
    logger.info(f"  P3: VL(final)={vl_ans_3}")

    # Final answer: VL decides. If 2/3 VL calls agree, use that; otherwise use the final call.
    votes = Counter([v for v in [vl_ans_1, vl_ans_2, vl_ans_3] if v in 'ABCD'])
    if votes and votes.most_common(1)[0][1] >= 2:
        ans = votes.most_common(1)[0][0]
        rp.append(f"[vl_consensus] {dict(votes)}→{ans}")
    else:
        ans = vl_ans_3  # Trust the final, most-informed VL call
        rp.append(f"[vl_final_pick] {ans}")

    ctx._final_answer = ans
    return ans, " | ".join(rp)


# ============================================================================
# Pipeline + Main
# ============================================================================

class AgenticPipelineV18:
    def __init__(self, device='cuda:0', vl_model_path=None, max_rounds=3, grid_max_frames=128):
        self.device = device
        self.vl_model_path = vl_model_path or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        self.builder = Grid256Builder(device=device)
        self.vl = VLModel(device=device); self.max_rounds = max_rounds
        self.grid_max_frames = grid_max_frames

    def load_models(self): self.builder.load_models(); self.vl.load(self.vl_model_path)
    def unload(self): self.builder.unload(); self.vl.unload()

    def process_scene(self, video_path, questions, grid=None):
        if grid is None:
            grid = self.builder.build_grid_fps(video_path, fps=UNIFIED_FPS,
                                               max_frames=self.grid_max_frames)
        results = []
        for sample in questions:
            q = sample['question']; opts = sample.get('options') or []
            gt = sample['ground_truth']; qt = sample['question_type']

            ctx = ToolExecutionContext(grid, self.vl, video_path, self.builder, q, opts, qt)
            t0 = time.time()

            try:
                ans, reasoning = mot_loop(ctx, max_rounds=self.max_rounds)
            except Exception as e:
                logger.error(f"  Error: {e}"); traceback.print_exc()
                ans = 'A'; reasoning = f"[error] {e}"

            elapsed = time.time() - t0
            score = evaluate_sample(qt, ans, gt)

            # Track metadata
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
                'converged_phase': (1 if '[converge_p1]' in reasoning else
                                   2 if '[converge_p2]' in reasoning else 3),
                'vl_calls': ctx.vl_calls, 'elapsed_s': round(elapsed, 1),
                'tool_trace': [{'tool': e.get('tool'), 'action': e.get('action', ''),
                               'ok': e.get('ok', ''), 'n_issues': e.get('n_issues', '')}
                              for e in ctx.tool_trace],
                'v7_vl_score': sample.get('vl_score', 0),
                'v7_rule_score': sample.get('rule_score', 0),
            })

            logger.info(f"  [{qt}] ans={ans} gt={gt} score={score:.3f} "
                        f"vl={ctx.vl_calls} t={elapsed:.0f}s | {reasoning[:100]}")
        return results


# ============================================================================
# Summary Printing
# ============================================================================

def _print_summary(all_results, od, ts):
    print("\n" + "=" * 120)
    print("Agentic Pipeline V18 — Mind-of-Thought (True Closed-Loop)")
    print(f"Architecture: P0(Belief) → P1(Grounded VL) → P2(Active Focus) → P3(Belief Correction + Re-verify)")
    print(f"Unified fps={UNIFIED_FPS}, Grid max_frames={GRID_MAX_FRAMES}, max_rounds=3 | Samples: {len(all_results)}")
    print("=" * 120)
    tts = sorted(set(r['question_type'] for r in all_results))
    print(f"  {'Task':<35} {'N':>4} {'V7':>6} {'V18':>6} {'Δ':>6}  {'VL':>4} {'Bel%':>4} {'Foc%':>4} {'AvgP':>4} {'T':>5}")
    print("-" * 110)
    for qt in tts:
        qr = [r for r in all_results if r['question_type'] == qt]
        v7 = np.mean([r.get('v7_vl_score', 0) for r in qr])
        v18 = np.mean([r['score'] for r in qr])
        d = v18 - v7; vl = np.mean([r.get('vl_calls', 0) for r in qr])
        bm = np.mean([1 if r.get('belief_modified') else 0 for r in qr]) * 100
        vf = np.mean([1 if r.get('vl_focused_used') else 0 for r in qr]) * 100
        avg_p = np.mean([r.get('converged_phase', 3) for r in qr])
        tavg = np.mean([r.get('elapsed_s', 0) for r in qr])
        mk = "+" if d > 0.01 else ("-" if d < -0.01 else "=")
        print(f"  {qt:<35} {len(qr):>4} {v7:>5.3f} {v18:>5.3f} {d:>+5.3f}{mk} {vl:>4.1f} {bm:>3.0f}% {vf:>3.0f}% {avg_p:>3.1f} {tavg:>4.0f}s")

    ov7 = np.mean([r.get('v7_vl_score', 0) for r in all_results])
    ov18 = np.mean([r['score'] for r in all_results])
    print("-" * 110)
    print(f"  {'Overall':<35} {len(all_results):>4} {ov7:>5.3f} {ov18:>5.3f} {ov18-ov7:>+5.3f}")

    tvl = sum(r.get('vl_calls', 0) for r in all_results)
    avl = tvl / len(all_results) if all_results else 0
    at = np.mean([r.get('elapsed_s', 0) for r in all_results])
    p1 = sum(1 for r in all_results if r.get('converged_phase') == 1)
    p2 = sum(1 for r in all_results if r.get('converged_phase') == 2)
    p3 = sum(1 for r in all_results if r.get('converged_phase') == 3)
    bm_n = sum(1 for r in all_results if r.get('belief_modified'))
    vf_n = sum(1 for r in all_results if r.get('vl_focused_used'))
    print(f"\n  VL: total={tvl}, avg={avl:.1f}/sample | Time: {at:.0f}s/sample")
    print(f"  Convergence: P1={p1}, P2={p2} ({100*p2/max(1,len(all_results)):.1f}%), P3={p3} ({100*p3/max(1,len(all_results)):.1f}%)")
    print(f"  Belief Modified: {bm_n} ({100*bm_n/max(1,len(all_results)):.1f}%), VL Focused: {vf_n} ({100*vf_n/max(1,len(all_results)):.1f}%)")

    summary = {'timestamp': ts, 'version': 'v18_mot_closedloop', 'n_samples': len(all_results),
               'overall': {'v7': float(ov7), 'v18': float(ov18), 'delta': float(ov18 - ov7)},
               'avg_vl_calls': float(avl), 'avg_time_s': float(at),
               'convergence': {'p1': p1, 'p2': p2, 'p3': p3},
               'belief_modified': bm_n, 'vl_focused': vf_n,
               'by_task': {qt: {'n': len([r for r in all_results if r['question_type'] == qt]),
                                'v7': float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type'] == qt])),
                                'v18': float(np.mean([r['score'] for r in all_results if r['question_type'] == qt]))}
                           for qt in tts}}
    json.dump(summary, open(f"{od}/summary.json", 'w'), indent=2)
    return summary


# ============================================================================
# Main Entry
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V18 Mind-of-Thought Pipeline")
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
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
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
        total = len(scene_list); chunk = (total + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * chunk; end = min(start + chunk, total)
        my_scenes = scene_list[start:end]
        logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: {len(my_scenes)} scenes")
    else:
        my_scenes = scene_list

    vl_model = getattr(args, 'vl_model', None) or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
    pipe = AgenticPipelineV18(device=args.device, vl_model_path=vl_model,
                               max_rounds=args.max_rounds, grid_max_frames=args.grid_max_frames)
    pipe.load_models()

    all_results = []; total_scenes = len(my_scenes)
    for si, sn in enumerate(my_scenes):
        samples = by_scene[sn]; vp = find_video_path(sn)
        if not vp:
            for s in samples:
                all_results.append({'scene_name': sn, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []), 'prediction': '0', 'reasoning': 'no video',
                    'score': 0.0, 'vl_calls': 0,
                    'v7_vl_score': s.get('vl_score', 0), 'v7_rule_score': s.get('rule_score', 0)})
            continue
        logger.info(f"[{si+1}/{total_scenes}] {sn} ({len(samples)} q)")
        try:
            results = pipe.process_scene(vp, samples)
            for r in results:
                all_results.append(r)
                d = r['score'] - r['v7_vl_score']; mk = "+" if d > 0 else ("-" if d < 0 else "=")
                bm = "B!" if r.get('belief_modified') else "  "
                logger.info(f"  {r['question_type'][:25]:25s} [VL:{r['vl_calls']} {bm}] "
                    f"Score={r['score']:.3f} V7={r['v7_vl_score']:.3f} {mk} | "
                    f"pred={str(r['prediction'])[:15]} gt={str(r['ground_truth'])[:12]} ({r['elapsed_s']:.0f}s)")
        except Exception as e:
            logger.error(f"  Error: {e}"); traceback.print_exc()
            for s in samples:
                all_results.append({'scene_name': sn, 'question_type': s['question_type'],
                    'question': s['question'], 'ground_truth': s['ground_truth'],
                    'options': s.get('options', []), 'prediction': '0',
                    'reasoning': f'error: {str(e)[:100]}', 'score': 0.0, 'vl_calls': 0,
                    'v7_vl_score': s.get('vl_score', 0), 'v7_rule_score': s.get('rule_score', 0)})

    pipe.unload()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.gpu_id is not None:
        od = PROJECT_ROOT / "outputs" / "agentic_pipeline_v18_mot" / f"gpu{args.gpu_id}"
    else:
        od = PROJECT_ROOT / "outputs" / f"agentic_pipeline_v18_{timestamp}"
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
    logger.info(f"Results: {od} ({len(all_results)} samples)")

    if args.gpu_id is None:
        _print_summary(all_results, str(od), timestamp)


if __name__ == '__main__':
    main()
