#!/usr/bin/env python3
"""
256³ Grid Mind Map — Agentic Pipeline V16 (Unified Manager + Type-Adaptive Decision)

核心改进 (V14→V16): 统一Manager驱动工具选择 + 类型自适应Phase C决策

  设计原则:
    1. 统一Manager流程 — 所有题目走Phase A(工具选择)→Execute→Phase C(类型自适应决策)
    2. fps=2 — 对齐Qwen3-VL官方VLMEvalKit信息预算
    3. Manager自选工具 — GRID_INFO / GRID_SLICE / CODER / CRITIC / DISTANCE_QUERY / TEMPORAL_INFO / NONE
    4. 闭环自进化 — Auto-ADD / Self-Verify / Targeted FILTER / Confidence-Driven重试
    5. 类型自适应投票:
       - room_size: 纯VL 3-vote (CODER完全隐藏，避免数字污染)
       - rel_distance: Pairwise Condorcet(2票) + VL-only(1票), CODER excluded
       - direction: SC 3-vote with Phase C prompt
       - route: VL-only SC 3-vote
       - appearance_order: SC 3-vote with CODER+TEMPORAL_INFO
       - count/size: CODER FastPath (confident→直接返回)
       - distance: VL synthesis (CODER suppressed)

  工具集:
    GRID_INFO:      Grid数据文本摘要 (focused on question entities) + pairwise distances
    GRID_SLICE:     2D俯视投影图 (XZ平面, PIL Image)
    CODER:          确定性3D空间计算 (8种类型)
    CRITIC:         VL审查Grid数据一致性
    DISTANCE_QUERY: 零VL成本距离查询
    TEMPORAL_INFO:  零VL成本时间线查询
    NONE:           纯VL视觉推理
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

VL_FPS = 2
VL_GRID_NFRAMES = 16
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
                nframes = _get_video_fps_nframes(video_path, fps if fps else VL_FPS)
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

    def call_sampled(self, prompt, video_path, max_tokens=128, n_samples=3,
                     temperature=0.7, top_p=0.9, nframes=None,
                     max_pixels=VL_DEFAULT_MAX_PIXELS):
        """SC投票: 采样多次用于Self-Consistency. 优先num_return_sequences, fallback逐次."""
        if self.model is None: return [""]
        try:
            from qwen_vl_utils import process_vision_info
            if nframes is None:
                nframes = _get_video_fps_nframes(video_path, VL_FPS)
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


# ============================================================================
# Pairwise Condorcet + SC Vote helpers (from V14)
# ============================================================================

def _pairwise_condorcet_rel_distance(ctx, coder_result):
    """Pairwise Condorcet for rel_distance: 两两比较 + Condorcet投票。CODER large margin直接信任."""
    if not ctx.options or len(ctx.options) < 2:
        return '', ''
    q = ctx.question.lower()
    is_farthest = 'farthest' in q or 'furthest' in q
    comparison_word = "farther from" if is_farthest else "closer to"
    # Extract reference object
    ref_name = ''
    m_ref = re.search(r'(?:closest|nearest|farthest|furthest) to (?:the )?(.+?)[\?\.]', q)
    if m_ref: ref_name = m_ref.group(1).strip()
    if not ref_name:
        m_ref = re.search(r'distance .+? (?:the )?(.+?)[\?\.]', q)
        if m_ref: ref_name = m_ref.group(1).strip()
    if not ref_name:
        m_coder_ref = re.search(r'ref=([^,]+)', coder_result) if coder_result else None
        if m_coder_ref: ref_name = m_coder_ref.group(1).strip()
    if not ref_name: return '', 'no ref found'
    # Extract option objects
    option_objects = []
    for opt in ctx.options:
        m = re.match(r'^([A-D])\.?\s*(.+)', opt.strip())
        if m: option_objects.append((m.group(1), m.group(2).strip()))
    if len(option_objects) < 2: return '', ''
    # Parse CODER distances
    coder_dists = {}
    if coder_result:
        for m_d in re.finditer(r'(\w[\w\s]*?)=([\d.]+)m', coder_result):
            coder_dists[m_d.group(1).strip().lower()] = float(m_d.group(2))
    # Pairwise comparisons
    from itertools import combinations
    wins = Counter(); comparisons = []
    for (l1, name1), (l2, name2) in combinations(option_objects, 2):
        d1 = coder_dists.get(name1.lower()); d2 = coder_dists.get(name2.lower())
        if d1 is not None and d2 is not None and abs(d1-d2) > 1.5:
            winner = (l1 if d1 > d2 else l2) if is_farthest else (l1 if d1 < d2 else l2)
            wins[winner] += 1; comparisons.append(f"{winner}(coder)"); continue
        prompt = f"""Look at the video carefully. Compare the distances of two objects to the {ref_name}.

Which object is {comparison_word} the {ref_name}: the {name1} or the {name2}?

Think about where each object is relative to the {ref_name} in the scene.
Answer with ONLY the object name (either "{name1}" or "{name2}"):"""
        response = ctx.vl.call(prompt, ctx.video_path, max_tokens=50); ctx.vl_calls += 1
        resp_lower = response.lower().strip()
        n1_low, n2_low = name1.lower(), name2.lower()
        winner = ''
        if n1_low in resp_lower and n2_low not in resp_lower: winner = l1
        elif n2_low in resp_lower and n1_low not in resp_lower: winner = l2
        elif n1_low in resp_lower and n2_low in resp_lower:
            winner = l1 if resp_lower.index(n1_low) < resp_lower.index(n2_low) else l2
        if winner: wins[winner] += 1; comparisons.append(f"{winner}")
    if not wins: return '', 'pairwise inconclusive'
    best = wins.most_common(1)[0][0]
    return best, f"pw_wins={dict(wins)} comps={comparisons}"


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


def _build_direction_prompt(ctx, grid_info_text=''):
    """Direction专用prompt: VL为主 + GRID_INFO位置参考(不含CODER答案)"""
    opts = "\n".join(ctx.options) if ctx.options else ""
    grid_section = ""
    if grid_info_text:
        grid_section = f"""
=== OBJECT POSITIONS (from 3D reconstruction) ===
{grid_info_text}
Note: A top-down room layout image is also provided. Use it to understand relative positions.
Positions may have errors. Trust your visual observation from the video when it contradicts the data."""
    
    return f"""You are analyzing a video of an indoor scene. You need to determine spatial direction/orientation.

=== QUESTION ===
{ctx.question}

=== OPTIONS ===
{opts}
{grid_section}

=== INSTRUCTIONS ===
Watch the video carefully. Pay attention to:
1. The camera/person's position and facing direction
2. Where each referenced object is located relative to the observer
3. Left/right are from the observer's perspective (the camera's view)

Answer with ONLY the option letter (A, B, C, or D).

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


def _build_rel_distance_prompt(ctx, grid_info_text, coder_result):
    """专门针对rel_distance: 只展示距离数据(不展示CODER答案字母), 让VL自行判断"""
    opts = "\n".join(ctx.options) if ctx.options else ""
    # Extract distance data from CODER result, but strip the answer letter
    dist_info = ""
    if coder_result:
        # Keep distance data lines but remove answer= line
        lines = []
        for line in coder_result.split(','):
            line = line.strip()
            if 'answer=' in line.lower(): continue  # Skip answer
            if '=' in line and 'm' in line: lines.append(line)
        if lines:
            dist_info = f"\n=== MEASURED DISTANCES (from 3D reconstruction) ===\n" + "\n".join(f"  {l}" for l in lines)
            dist_info += "\nNote: These distances may have 2-5x error due to calibration. Use as rough reference only."
    
    grid_section = ""
    if grid_info_text:
        grid_section = f"\n=== 3D SPATIAL DATA ===\n{grid_info_text}\nNote: Positions from 3D reconstruction; distances may have 2-5x error."
    
    return f"""You are analyzing a video of an indoor scene. You need to determine which object is closest/farthest to a reference object.

=== QUESTION ===
{ctx.question}

=== OPTIONS ===
{opts}
{grid_section}
{dist_info}

=== INSTRUCTIONS ===
Watch the video carefully. Look at where each candidate object is relative to the reference object.
Consider both visual observation and any distance data provided.
The distance measurements have errors, so if visual observation clearly contradicts the data, trust your eyes.

Answer with ONLY the option letter (A, B, C, or D).

ANSWER: <letter>
CONFIDENCE: <high/medium/low>"""


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


# ============================================================================
# V16 Manager Prompts
# ============================================================================

def _phase_a_prompt(ctx):
    grid = ctx.grid; n = len(grid.entities); span = grid.meters_per_grid * grid.GRID_SIZE
    opts = "\n".join(ctx.options) if ctx.options else "(numerical answer expected)"
    enames = [f"{eid}({e.category})" for eid,e in sorted(grid.entities.items())]
    elist = ", ".join(enames[:15]) + (f", ...+{len(enames)-15}" if len(enames)>15 else "")
    return f"""You are a spatial intelligence agent analyzing an indoor scene video with a 3D perception system.

=== QUESTION ===
{ctx.question}

=== OPTIONS ===
{opts}

=== SCENE OVERVIEW ===
{n} detected objects, scene_span≈{span:.1f}m
Objects: {elist}

=== AVAILABLE TOOLS (select 0-4) ===
1. GRID_INFO — Detailed 3D position/size/confidence data + pairwise distances for relevant objects
2. GRID_SLICE — 2D top-down view image showing object layout with distances
3. CODER — Deterministic 3D computation. Types: count/distance/rel_distance/direction/size/room_size/appearance_order/route
4. DISTANCE_QUERY — Quick distance lookup between two objects. Params: obj1, obj2
5. TEMPORAL_INFO — Appearance timeline: when objects first/last appear, visibility span. Good for appearance_order/route questions
6. CRITIC — Video reviewer verifies Grid data consistency
7. NONE — No 3D tools needed; answer from video only

=== INSTRUCTIONS ===
REASONING: <analyze what spatial info this question needs>
TOOLS: [<JSON list of tool selections>]

Examples:
  TOOLS: [{{"tool":"CODER","type":"direction"}}, {{"tool":"GRID_SLICE"}}]
  TOOLS: [{{"tool":"GRID_INFO"}}, {{"tool":"CODER","type":"count"}}]
  TOOLS: [{{"tool":"DISTANCE_QUERY","obj1":"chair","obj2":"table"}}]
  TOOLS: [{{"tool":"TEMPORAL_INFO","entities":"chair,table,door"}}]
  TOOLS: [{{"tool":"NONE"}}]"""


def _phase_c_prompt(ctx, tool_results, has_slice=False, coder_confidence='low'):
    opts = "\n".join(ctx.options) if ctx.options else "(numerical answer expected)"
    info = []
    if 'GRID_INFO' in tool_results:
        info.append(f"=== 3D SPATIAL DATA ===\n{tool_results['GRID_INFO']}\nNote: Positions from 3D reconstruction; distances/sizes may have 2-5x error.")
    if 'CODER' in tool_results:
        cr = tool_results['CODER']
        ct = tool_results.get('_ct', '')
        unit_map = {'distance':'meters','size':'centimeters','room_size':'square meters','count':'count',
                    'direction':'option letter (A/B/C/D)','rel_distance':'option letter (A/B/C/D)',
                    'appearance_order':'option letter (A/B/C/D)','route':'option letter (A/B/C/D)'}
        unit_hint = f" | Computation type: {ct}, unit: {unit_map.get(ct,'')}" if ct else ""
        note = ""
        if 'not found' in cr.lower(): note = "\n⚠ Some entities not found. Result may be incomplete."
        # If CODER is low confidence for numerical task, suppress the answer to avoid polluting VL
        if coder_confidence == 'low' and not ctx.options:
            # Strip the specific answer value; keep only detail/context
            cr_display = re.sub(r'answer=[\d.]+\S*', 'answer=<unreliable, use your own estimate>', cr)
            info.append(f"=== 3D COMPUTATION (CODER) ===\n{cr_display}{note}{unit_hint}\n⚠ LOW reliability — calibration error likely. Use your own visual estimate.")
        else:
            trust_map = {'count':'high','direction':'medium-high','appearance_order':'medium',
                         'distance':'medium (calibration-dependent)','size':'medium (calibration-dependent)',
                         'room_size':'low-medium (calibration-dependent)','rel_distance':'medium','route':'medium'}
            trust = trust_map.get(ct, 'medium')
            info.append(f"=== 3D COMPUTATION (CODER) ===\n{cr}{note}{unit_hint}\nReliability: {trust}. Deterministic geometry on 3D Grid data.")
    if 'CRITIC' in tool_results:
        info.append(f"=== QUALITY REVIEW ===\n{tool_results['CRITIC']}")
    if 'DISTANCE_QUERY' in tool_results:
        info.append(f"=== DISTANCE QUERY ===\n{tool_results['DISTANCE_QUERY']}")
    if 'TEMPORAL_INFO' in tool_results:
        info.append(f"=== TEMPORAL INFO (appearance timeline) ===\n{tool_results['TEMPORAL_INFO']}")
    if has_slice:
        info.append("=== ROOM LAYOUT ===\nA top-down view image is provided. Highlighted objects (red outline) are relevant. Red lines show distances.")
    if not info:
        info.append("=== NO 3D DATA REQUESTED ===\nAnswer from video observation only.")

    if ctx.options:
        ans_inst = "Answer with the option letter (A/B/C/D). Trust CODER if consistent with video; otherwise use your judgment."
    else:
        ans_inst = "Answer with a single number. Use CODER result as primary reference if available."

    return f"""You are a spatial intelligence agent making a final decision.

{chr(10).join(info)}

=== REFERENCES ===
Door ~200cm tall. Chair seat ~45cm. Table ~75cm. Bed ~200cm. Sofa ~85cm.

=== QUESTION ===
{ctx.question}

=== OPTIONS ===
{opts}

=== INSTRUCTIONS ===
Watch the video carefully and consider all information above.
{ans_inst}

ANSWER: <your answer>
CONFIDENCE: <high/medium/low>"""


# ============================================================================
# V16 Unified Manager Loop
# ============================================================================

def _parse_tools(resp):
    m = re.search(r'TOOLS:\s*\[(.+?)\]', resp, re.DOTALL)
    if not m: m = re.search(r'\[(\s*\{.+?\}\s*(?:,\s*\{.+?\}\s*)*)\]', resp, re.DOTALL)
    if not m: return []
    try:
        t = '[' + m.group(1) + ']'; t = t.replace("'", '"')
        a = json.loads(t); return a if isinstance(a, list) else []
    except:
        acts = []
        for om in re.finditer(r'\{[^}]+\}', m.group(1)):
            try: acts.append(json.loads(om.group().replace("'",'"')))
            except: pass
        return acts

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

def _exec_tools(ctx, sels):
    results = {}; slice_img = None
    for sel in sels[:4]:
        tn = sel.get('tool','').upper()
        if tn == 'GRID_INFO':
            results['GRID_INFO'] = _grid_to_text_focused(ctx.grid, ctx.question, ctx.options)
        elif tn == 'GRID_SLICE':
            slice_img = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
            results['GRID_SLICE'] = "(image provided)"
        elif tn == 'CODER':
            ct = sel.get('type','').lower() or _auto_coder_type(ctx.question, ctx.options) or ''
            if ct:
                if ct == 'distance' and ctx.options: ct = 'rel_distance'
                r = coder_tool(ctx, ct); results['CODER'] = r; results['_ct'] = ct
                if 'not found' in r.lower():
                    mod, log = _auto_add(ctx, r)
                    if mod:
                        r2 = coder_tool(ctx, ct); results['CODER'] = r2; results['_add_log'] = log
        elif tn == 'CRITIC':
            results['CRITIC'] = critic_tool(ctx, sel.get('focus',''))
        elif tn == 'DISTANCE_QUERY':
            o1 = sel.get('obj1','').strip(); o2 = sel.get('obj2','').strip()
            if o1 and o2:
                results['DISTANCE_QUERY'] = distance_query_tool(ctx, o1, o2)
            else:
                # Auto-extract entity pair from question
                ents = _extract_question_entities(ctx.question, ctx.options)
                if len(ents) >= 2:
                    results['DISTANCE_QUERY'] = distance_query_tool(ctx, ents[0], ents[1])
                else:
                    results['DISTANCE_QUERY'] = "DISTANCE_QUERY: need obj1 and obj2 params"
        elif tn == 'TEMPORAL_INFO':
            results['TEMPORAL_INFO'] = temporal_info_tool(ctx, sel.get('entities',''))
    return results, slice_img


def manager_unified_loop(ctx, max_rounds=3):
    rp = []

    # Phase A: Tool selection
    pa = ctx.vl.call(_phase_a_prompt(ctx), ctx.video_path, max_tokens=512); ctx.vl_calls += 1
    rm = re.search(r'REASONING:\s*(.+?)(?=TOOLS:|$)', pa, re.DOTALL)
    if rm: rp.append(f"[reason] {rm.group(1).strip()[:100]}")
    tsels = _parse_tools(pa)
    tnames = [s.get('tool','?').upper() for s in tsels]
    rp.append(f"[tools] {tnames}"); logger.info(f"  PhaseA: tools={tnames}")

    # Auto-add CODER: for numerical tasks always inject; for choice inject if not NONE
    at = _auto_coder_type(ctx.question, ctx.options)
    has_coder = any(t=='CODER' for t in tnames)
    has_none = any(t=='NONE' for t in tnames)
    if not has_coder:
        if not ctx.options and at:
            tsels.append({'tool':'CODER','type':at}); rp.append(f"[force_coder] {at}")
        elif not has_none and at:
            tsels.append({'tool':'CODER','type':at}); rp.append(f"[auto_coder] {at}")
        elif at in ('rel_distance','direction','appearance_order','route'):
            # Always inject CODER for key choice types to ensure ct is set
            tsels.append({'tool':'CODER','type':at}); rp.append(f"[force_coder_choice] {at}")

    # Execute tools
    tr, si = _exec_tools(ctx, tsels)
    if '_add_log' in tr: rp.append(f"[auto_add] {tr['_add_log'][:60]}")
    used = {k for k in tr if not k.startswith('_')}

    # Extract CODER answer + confidence
    ct = tr.get('_ct',''); cr = tr.get('CODER','')
    cc = _coder_confidence(ct, cr, ctx.grid) if ct else 'low'
    ca = ""
    if cr:
        m = re.search(r'answer=([A-D])', cr)
        if m: ca = m.group(1)
        elif not ctx.options:
            m = re.search(r'answer=([\d.]+)', cr)
            if m: ca = m.group(1)
    rp.append(f"[coder] ans={ca} conf={cc}" if ct else "[coder] none")

    # ══════════════════════════════════════════════════════════
    # NUMERICAL FAST PATH: confident CODER → skip VL synthesis
    # ══════════════════════════════════════════════════════════
    if not ctx.options and ca and cc in ('normal', 'verified'):
        rp.append(f"[num_fastpath] coder={ca}")
        logger.info(f"  NumFastPath: coder={ca} conf={cc} (skip VL synthesis)")
        ctx._final_answer = ca
        return ca, " | ".join(rp)

    # ══════════════════════════════════════════════════════════
    # ROOM_SIZE SPECIAL PATH: hide CODER completely, pure VL
    # ══════════════════════════════════════════════════════════
    if not ctx.options and ct == 'room_size':
        # Use SC 3-vote with dedicated room_size prompt (no CODER info at all)
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
            rp.append(f"[room_size_vl_3vote] vals={[f'{v:.1f}' for v in values]} median={ans}")
            logger.info(f"  RoomSize VL 3-vote: {values} → median={ans}")
        else:
            # Fallback: single VL call
            resp = ctx.vl.call(room_prompt, ctx.video_path, max_tokens=64); ctx.vl_calls += 1
            m = re.search(r'[\d.]+', resp)
            ans = m.group() if m else (ca if ca else '15')
            rp.append(f"[room_size_vl_fallback] ans={ans}")
        ctx._final_answer = ans
        return ans, " | ".join(rp)

    # ══════════════════════════════════════════════════════════
    # ABS_DISTANCE SPECIAL PATH: pure VL 3-vote (like V14 decide_num:low)
    # V14 achieved 0.796 with single VL call, no CODER info.
    # V16 TA4 got 0.616 with Phase C (CODER info polluted VL).
    # Strategy: 3-vote + median, completely hide CODER.
    # ══════════════════════════════════════════════════════════
    if not ctx.options and ct == 'distance':
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
            rp.append(f"[abs_dist_vl_3vote] vals={[f'{v:.1f}' for v in values]} median={ans}")
            logger.info(f"  AbsDist VL 3-vote: {values} → median={ans}")
        else:
            resp = ctx.vl.call(dist_prompt, ctx.video_path, max_tokens=64); ctx.vl_calls += 1
            m = re.search(r'[\d.]+', resp)
            ans = m.group() if m else (ca if ca else '2.0')
            rp.append(f"[abs_dist_vl_fallback] ans={ans}")
        ctx._final_answer = ans
        return ans, " | ".join(rp)

    # ══════════════════════════════════════════════════════════
    # NUMERICAL LOW-CONF (size low-conf, etc): VL synthesis with suppressed CODER
    # ══════════════════════════════════════════════════════════
    if not ctx.options and cc == 'low':
        imgs = [si] if si else None
        pc = ctx.vl.call(_phase_c_prompt(ctx, tr, si is not None, coder_confidence='low'),
                         ctx.video_path, max_tokens=256, images=imgs)
        ctx.vl_calls += 1
        ans, conf = _parse_answer(pc, ctx)
        rp.append(f"[num_low_vl] ans={ans} conf={conf}")
        # Fallback if VL returns 0 but CODER has value
        if ca:
            try:
                if float(ans) <= 0 and float(ca) > 0:
                    ans = ca; rp.append(f"[num_fallback] VL=0→coder={ca}")
            except: pass
        ctx._final_answer = ans
        return ans, " | ".join(rp)

    # ══════════════════════════════════════════════════════════
    # CHOICE TASKS: Type-adaptive decision
    # ══════════════════════════════════════════════════════════

    # Determine effective task type: prefer at (auto-detected from question), fallback to ct (from CODER)
    eff_type = at if at else (ct or '')

    # Step 1: VL independent judgment (no CODER info)
    vl_ind_prompt = _build_vl_independent_prompt(ctx)
    vl_ind_resp = ctx.vl.call(vl_ind_prompt, ctx.video_path, max_tokens=128); ctx.vl_calls += 1
    vl_ind_answer = _clean(vl_ind_resp, ctx)
    rp.append(f"[vl_ind] {vl_ind_answer}")

    # Step 2: Type-adaptive decision
    if eff_type == 'rel_distance':
        # ── REL_DISTANCE: VL_ind + GRID_SLICE image ──
        # Analysis: SC vote=30%, single VL+distance data=38%, VL_ind pure=40%
        # Distance data from CODER (20% accurate) misleads VL.
        # SC voting amplifies wrong answers. Best: VL_ind with GRID_SLICE for spatial context.
        if not si:
            si = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
        rd_prompt = _build_vl_independent_prompt(ctx)
        pc = ctx.vl.call(rd_prompt, ctx.video_path, max_tokens=128, images=[si]); ctx.vl_calls += 1
        ans = _clean(pc, ctx)
        rp.append(f"[reldist_vl_slice] ans={ans}")
        logger.info(f"  RelDist VL+Slice: ans={ans}")

    elif eff_type == 'direction':
        # ── DIRECTION: VL_ind + GRID_SLICE image only (no GRID_INFO text) ──
        # TA4: VL+Slice=52.9%. TA5 with GRID_INFO text: 48.7% (worse).
        # GRID_INFO text with inaccurate positions misleads VL. Only use GRID_SLICE image.
        if not si:
            si = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
        dir_prompt = _build_vl_independent_prompt(ctx)
        pc = ctx.vl.call(dir_prompt, ctx.video_path, max_tokens=128, images=[si]); ctx.vl_calls += 1
        ans = _clean(pc, ctx)
        rp.append(f"[dir_vl_slice] ans={ans}")
        logger.info(f"  Direction VL+Slice: ans={ans}")

    elif eff_type == 'route':
        # ── ROUTE: VL SC 3-vote with GRID_SLICE for spatial context ──
        # Route planning benefits from seeing the room layout (top-down view).
        if not si:
            si = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
        route_prompt = _build_vl_independent_prompt(ctx)
        # First deterministic call with GRID_SLICE
        pc = ctx.vl.call(route_prompt, ctx.video_path, max_tokens=128, images=[si]); ctx.vl_calls += 1
        first_ans = _clean(pc, ctx)
        # SC 3-vote: sample 2 more with GRID_SLICE
        responses = ctx.vl.call_sampled(route_prompt, ctx.video_path, max_tokens=128,
                                         n_samples=2, temperature=0.7)
        ctx.vl_calls += len(responses)
        votes = []
        if first_ans and first_ans in 'ABCD': votes.append(first_ans)
        for r in responses:
            c = _clean(r, ctx)
            if c and c in 'ABCD': votes.append(c)
        if not votes: votes = [vl_ind_answer]
        vote_counts = Counter(votes)
        ans = vote_counts.most_common(1)[0][0]
        rp.append(f"[route_slice_vote:{len(votes)}] {dict(vote_counts)} → {ans}")
        logger.info(f"  Route VL+Slice vote: {dict(vote_counts)} → {ans}")

    elif eff_type == 'appearance_order':
        # ── APPEARANCE_ORDER: SC 3-vote with Phase C (includes CODER + TEMPORAL_INFO) ──
        imgs = [si] if si else None
        phase_c = _phase_c_prompt(ctx, tr, si is not None, coder_confidence=cc)
        responses = ctx.vl.call_sampled(phase_c, ctx.video_path, max_tokens=128, n_samples=3, temperature=0.7)
        ctx.vl_calls += len(responses)
        votes = []
        for r in responses:
            c = _clean(r, ctx)
            if c and c in 'ABCD': votes.append(c)
        if not votes: votes = [vl_ind_answer]
        vote_counts = Counter(votes)
        ans = vote_counts.most_common(1)[0][0]
        rp.append(f"[appear_sc3vote:{len(votes)}] {dict(vote_counts)} → {ans}")
        logger.info(f"  Appear SC 3-vote: {dict(vote_counts)} → {ans}")

    else:
        # ── DEFAULT CHOICE: Phase C synthesis + retry loop (original V16 logic) ──
        imgs = [si] if si else None
        pc = ctx.vl.call(_phase_c_prompt(ctx, tr, si is not None, coder_confidence=cc),
                         ctx.video_path, max_tokens=256, images=imgs)
        ctx.vl_calls += 1
        ans, conf = _parse_answer(pc, ctx)
        rp.append(f"[r1] {ans} conf={conf}")

        # Detect CODER conflict
        conflict = False
        if ca and ans and cc in ('normal','verified'):
            conflict = ca.upper() != ans.upper()

        for ri in range(1, max_rounds):
            if conf != 'low' and not (conflict and ri == 1): break
            rp.append(f"[retry_r{ri+1}] conf={conf} conflict={conflict}")
            if conflict and 'CRITIC' not in used:
                ie = _extract_involved_entities(cr, ctx.question, ctx.options)
                critic_r = critic_tool(ctx, ", ".join(ie[:3]))
                tr['CRITIC'] = critic_r; used.add('CRITIC')
                rp.append(f"[conflict_critic] {critic_r[:60]}")
                if 'issue' in critic_r.lower() and 'no issues' not in critic_r.lower():
                    fm, fl = _targeted_filter(ctx, ie)
                    if fm and ct:
                        nr = coder_tool(ctx, ct); tr['CODER'] = nr; cr = nr
                        rp.append(f"[repair] {nr[:60]}")
            elif conf == 'low' and 'GRID_SLICE' not in used:
                si = generate_grid_slice(ctx.grid, ctx.question, ctx.options)
                tr['GRID_SLICE'] = "(image)"; used.add('GRID_SLICE'); imgs = [si]
            elif conf == 'low' and 'GRID_INFO' not in used:
                tr['GRID_INFO'] = _grid_to_text_focused(ctx.grid, ctx.question, ctx.options)
                used.add('GRID_INFO')
            imgs = [si] if si else None
            pc = ctx.vl.call(_phase_c_prompt(ctx, tr, si is not None, coder_confidence=cc),
                             ctx.video_path, max_tokens=256, images=imgs)
            ctx.vl_calls += 1
            ans, conf = _parse_answer(pc, ctx)
            rp.append(f"[r{ri+1}] {ans} conf={conf}")
            conflict = False

    ctx._final_answer = ans
    return ans, " | ".join(rp)


# ============================================================================
# Pipeline + Main
# ============================================================================

class AgenticPipelineV16:
    def __init__(self, device='cuda:0', vl_model_path=None, max_rounds=3):
        self.device = device
        self.vl_model_path = vl_model_path or '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        self.builder = Grid256Builder(device=device, num_frames=VL_GRID_NFRAMES)
        self.vl = VLModel(device=device); self.max_rounds = max_rounds

    def load_models(self): self.builder.load_models(); self.vl.load(self.vl_model_path)
    def unload(self): self.builder.unload(); self.vl.unload()

    def process_scene(self, video_path, samples):
        t0 = time.time(); grid = self.builder.build_grid(video_path); bt = time.time()-t0
        logger.info(f"  Grid256: {len(grid.entities)} ents, mpg={grid.meters_per_grid:.4f}m ({bt:.1f}s)")
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
        pred, reasoning = manager_unified_loop(ctx, self.max_rounds)
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
            'prediction': pred, 'reasoning': reasoning[:500], 'score': score,
            'critic_issues_count': ci, 'critic_has_issues': ci>0,
            'grid_modified': gm, 'evolution_actions': evo,
            'filter_frames_count': ffc, 'coder_used': cu,
            'auto_add_triggered': '[auto_add]' in reasoning.lower(),
            'verify_triggered': '[conflict_critic]' in reasoning.lower(),
            'vl_calls': ctx.vl_calls, 'elapsed_s': round(elapsed,1),
            'tool_trace': [{'tool':e.get('tool'),'action':e.get('action',''),'ok':e.get('ok',''),'n_issues':e.get('n_issues','')} for e in ctx.tool_trace],
            'v7_vl_score': sample.get('vl_score',0), 'v7_rule_score': sample.get('rule_score',0),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V16 Unified Manager-Driven Pipeline")
    parser.add_argument('--n_per_type', type=int, default=10)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--max_rounds', type=int, default=3)
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
    pipe = AgenticPipelineV16(device=args.device, vl_model_path=vl_model, max_rounds=args.max_rounds)
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
        od = PROJECT_ROOT / "outputs" / "agentic_pipeline_v16_full" / f"gpu{args.gpu_id}"
    else:
        od = PROJECT_ROOT / "outputs" / f"agentic_pipeline_v16_{timestamp}"
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
    print("Agentic Pipeline V16 — Unified Manager-Driven")
    print(f"Architecture: All→Manager(tool selection)→GRID_INFO/GRID_SLICE/CODER/CRITIC/NONE→Synthesis")
    print(f"fps=2, no SC voting, max_rounds=3 | Samples: {len(all_results)}")
    print("="*120)
    tts = sorted(set(r['question_type'] for r in all_results))
    print(f"\n{'Task':<35} {'N':>4} {'V7':>6} {'V16':>6} {'Δ':>6}  {'VL#':>4} {'Cod%':>5} {'t/s':>5}")
    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}  {'-'*4} {'-'*5} {'-'*5}")
    a7, a16 = [], []
    for qt in tts:
        qr = [r for r in all_results if r['question_type']==qt]
        v7 = np.mean([r['v7_vl_score'] for r in qr])
        v16 = np.mean([r['score'] for r in qr])
        d = v16-v7; vl = np.mean([r.get('vl_calls',0) for r in qr])
        cod = np.mean([1 if r.get('coder_used') else 0 for r in qr])*100
        tavg = np.mean([r.get('elapsed_s',0) for r in qr])
        mk = "+" if d>0.01 else ("-" if d<-0.01 else "=")
        print(f"  {qt:<35} {len(qr):>4} {v7:>5.3f} {v16:>5.3f} {d:>+5.3f}{mk} {vl:>4.1f} {cod:>4.0f}% {tavg:>4.0f}s")
        a7.extend([r['v7_vl_score'] for r in qr]); a16.extend([r['score'] for r in qr])
    ov7, ov16 = np.mean(a7), np.mean(a16)
    print(f"{'-'*35} {'-'*4} {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'Overall':<35} {len(all_results):>4} {ov7:>5.3f} {ov16:>5.3f} {ov16-ov7:>+5.3f}")
    tvl = sum(r.get('vl_calls',0) for r in all_results)
    avl = tvl/len(all_results) if all_results else 0
    at = np.mean([r.get('elapsed_s',0) for r in all_results])
    print(f"\n  VL: total={tvl}, avg={avl:.1f}/sample | Time: {at:.0f}s/sample")
    tc = Counter()
    for r in all_results:
        for e in r.get('tool_trace',[]): tc[e.get('tool','?')] += 1
    print(f"  Tools: {dict(tc)}")
    print(f"\n{'='*60}\n  V16 Overall = {ov16:.4f}  vs  V7 = {ov7:.4f}  (Δ = {ov16-ov7:+.4f})\n{'='*60}")
    summary = {'timestamp':ts,'version':'v16_unified_manager','n_samples':len(all_results),
        'overall':{'v7':float(ov7),'v16':float(ov16),'delta':float(ov16-ov7)},
        'avg_vl_calls':float(avl),'avg_time_s':float(at),'tool_usage':dict(tc),
        'by_task':{qt:{'n':len([r for r in all_results if r['question_type']==qt]),
            'v7':float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type']==qt])),
            'v16':float(np.mean([r['score'] for r in all_results if r['question_type']==qt])),
        } for qt in tts}}
    with open(od / "summary.json", 'w') as f: json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
