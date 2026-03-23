#!/usr/bin/env python3
"""Route Planning修复测试 — 全量route_planning评估

修复点:
1. 物体主体名词提取: "the wardrobe with two mirrors on it" -> "wardrobe"
2. get_by_category严格匹配: 过滤停用词, 优先精确匹配
3. facing=start时fallback到VL
"""
import os, sys, json, re, logging, argparse, time
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from pathlib import Path

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from scripts.grid64_real_test import (
    Grid64, Grid64Builder, GridEntity, find_video_path,
    evaluate_sample, mean_relative_accuracy, _match_name, SYNONYMS
)

# ============================================================================
# Fix 1: 主体名词提取
# ============================================================================
STOP_WORDS = {'the', 'a', 'an', 'of', 'with', 'on', 'at', 'near', 'in', 'to',
              'it', 'its', 'is', 'that', 'this', 'from', 'by', 'for', 'and',
              'two', 'three', 'four', 'one', 'next', 'close', 'closest',
              'farthest', 'front', 'back', 'left', 'right', 'behind',
              'above', 'below', 'between', 'opposite', 'across', 'end',
              'corner', 'bottom', 'top', 'side', 'middle', 'center', 'edge'}

def extract_noun_core(phrase: str) -> str:
    """从描述性短语提取核心名词
    
    'the wardrobe with two mirrors on it' -> 'wardrobe'
    'end of the bed' -> 'bed'
    'bottom-right corner of the bed' -> 'bed'
    'the white cabinet near the blue box' -> 'white cabinet'
    'the refrigerator' -> 'refrigerator'
    'wall 3' -> 'wall 3'
    """
    phrase = phrase.lower().strip()
    
    # 先去掉介词短语: "X with/near/on/of Y" -> 取X或Y中更像主体的
    # 策略: 如果有 "of the X", 主体是X (如 "end of the bed" -> "bed")
    m_of = re.search(r'\bof\s+(?:the\s+)?(.+?)$', phrase)
    if m_of:
        # "end of the bed" -> bed; "bottom-right corner of the bed" -> bed  
        core_after_of = m_of.group(1).strip()
        # 但 "side of the room" -> room, 这是对的
        # 去掉停用词检查是否有实质名词
        words = [w for w in core_after_of.split() if w not in STOP_WORDS and len(w) > 1]
        if words:
            return ' '.join(words)
    
    # 去掉 "with/near/on/next to ..." 后缀
    phrase = re.sub(r'\s+(?:with|near|next to|close to|on|at|in|facing)\s+.*$', '', phrase)
    
    # 去掉开头的 the/a/an
    phrase = re.sub(r'^(?:the|a|an)\s+', '', phrase)
    
    # 去掉纯修饰词但保留颜色+名词 (如 "white cabinet")
    # 不动了，直接返回
    return phrase.strip()


# ============================================================================
# Fix 2: 严格的物体匹配
# ============================================================================
def _match_name_strict(target: str, label: str) -> float:
    """返回匹配得分 (0=不匹配, 越高越好), 而非bool
    
    过滤停用词，避免 "the" 匹配一切
    """
    target = target.lower().strip()
    label = label.lower().strip()
    
    # 完全匹配
    if target == label:
        return 10.0
    
    # 包含关系 (但要求长度合理)
    if target in label and len(target) > 2:
        return 5.0 + len(target) / max(len(label), 1)
    if label in target and len(label) > 2:
        return 4.0 + len(label) / max(len(target), 1)
    
    # 分词匹配 (过滤停用词)
    tw = set(target.replace('_', ' ').replace('-', ' ').split()) - STOP_WORDS
    lw = set(label.replace('_', ' ').replace('-', ' ').split()) - STOP_WORDS
    
    if not tw or not lw:
        return 0.0
    
    overlap = tw & lw
    if overlap:
        # 有实质词重叠
        score = 2.0 * len(overlap) / max(len(tw | lw), 1)
        return score
    
    # 同义词匹配
    for t in tw:
        if t in SYNONYMS:
            for s in SYNONYMS[t]:
                if s in lw:
                    return 1.0
    
    return 0.0


def get_entity_strict(grid: Grid64, name: str) -> List[GridEntity]:
    """用严格匹配从grid找物体, 按匹配得分排序"""
    # 先提取主体名词
    core = extract_noun_core(name)
    
    scored = []
    for eid, e in grid.entities.items():
        # 用core匹配
        s1 = _match_name_strict(core, e.category)
        # 也用原始名匹配
        s2 = _match_name_strict(name, e.category)
        score = max(s1, s2)
        if score > 0:
            scored.append((score, e))
    
    # 按得分排序, 取最高的
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored] if scored else []


# ============================================================================
# Fix 3: 修复后的route算法
# ============================================================================
FACING_START_MARKER = "__FACING_EQUALS_START__"

def grid_answer_route_fixed(grid: Grid64, question: str, options: List[str]) -> Tuple[str, str]:
    """修复后的路线规划算法
    
    修复:
    1. 用extract_noun_core提取主体名词
    2. 用get_entity_strict做严格匹配
    3. facing=start时返回特殊标记
    """
    if not options:
        return "A", "no options"
    
    q = question.lower()
    
    # 解析起点和朝向
    m_start = re.search(r'beginning at (?:the )?(.+?)\s+(?:and\s+)?facing (?:the )?(.+?)\.', q)
    if not m_start:
        m_start = re.search(r'beginning at (?:the )?(.+?)\s+facing (?:the )?(.+?)\.', q)
    if not m_start:
        return "A", "cannot parse start/facing"
    
    start_name = m_start.group(1).strip()
    facing_name = m_start.group(2).strip()
    
    # 解析步骤序列
    steps = re.findall(r'\d+\.\s+(.+?)(?=\d+\.|You have reached|$)', q, re.DOTALL)
    
    waypoints = []
    for step in steps:
        step = step.strip().rstrip('.')
        if 'please fill in' in step:
            waypoints.append(('fill_in', None))
        elif 'go forward' in step:
            m_fwd = re.search(r'go forward\s+until\s+(?:you\s+(?:reach|get to)\s+)?(?:the\s+)?(.+?)(?:\s+is\s+on|\s*$)', step)
            if not m_fwd:
                m_fwd = re.search(r'go forward\s+until\s+(?:passing\s+)?(?:the\s+)?(.+?)(?:\s+on\s+|\s*$)', step)
            if m_fwd:
                wp_name = m_fwd.group(1).strip().rstrip('.')
                waypoints.append(('go_forward', wp_name))
            else:
                waypoints.append(('go_forward', None))
    
    if not waypoints:
        return "A", "no waypoints parsed"
    
    # Fix 2: 用严格匹配获取物体
    e_start = get_entity_strict(grid, start_name)
    e_facing = get_entity_strict(grid, facing_name)
    
    if not e_start or not e_facing:
        missing = []
        if not e_start: missing.append(f"start='{start_name}'")
        if not e_facing: missing.append(f"facing='{facing_name}'")
        return "A", f"not found: {', '.join(missing)}"
    
    start_pos = np.array(e_start[0].position_3d, dtype=float)
    facing_pos = np.array(e_facing[0].position_3d, dtype=float)
    
    # 初始朝向
    init_facing = np.array([facing_pos[0] - start_pos[0], facing_pos[2] - start_pos[2]])
    
    # Fix 3: facing=start检测 — 判定标准: XZ距离 < 0.5m
    if np.linalg.norm(init_facing) < 0.5:
        return FACING_START_MARKER, f"facing=start (dist={np.linalg.norm(init_facing):.3f}m), fallback to VL"
    
    init_facing = init_facing / np.linalg.norm(init_facing)
    
    # 收集waypoint位置 (用严格匹配)
    wp_positions = {}
    for wtype, wname in waypoints:
        if wname and wname not in wp_positions:
            e_wp = get_entity_strict(grid, wname)
            if e_wp:
                wp_positions[wname] = np.array(e_wp[0].position_3d, dtype=float)
    
    n_fill_ins = sum(1 for wtype, _ in waypoints if wtype == 'fill_in')
    
    # 对每个选项模拟
    best_score = -999
    best_opt = "A"
    score_details = []
    
    for opt in options:
        letter = opt[0]
        opt_content = re.sub(r'^[A-D]\.\s*', '', opt).strip()
        turns = [t.strip().lower() for t in opt_content.split(',')]
        
        if len(turns) != n_fill_ins:
            score_details.append(f"{letter}: turns={len(turns)} != fill_ins={n_fill_ins}")
            continue
        
        current_pos = start_pos.copy()
        current_facing = init_facing.copy()
        turn_idx = 0
        score = 0.0
        
        for wi, (wtype, wname) in enumerate(waypoints):
            if wtype == 'fill_in':
                turn_cmd = turns[turn_idx]
                turn_idx += 1
                
                if 'back' in turn_cmd:
                    current_facing = -current_facing
                elif 'left' in turn_cmd:
                    current_facing = np.array([-current_facing[1], current_facing[0]])
                elif 'right' in turn_cmd:
                    current_facing = np.array([current_facing[1], -current_facing[0]])
                
                # 检查转向后朝向是否大致指向下一个waypoint
                next_wp_name = None
                for future_type, future_name in waypoints[wi+1:]:
                    if future_type == 'go_forward' and future_name:
                        next_wp_name = future_name
                        break
                
                if next_wp_name and next_wp_name in wp_positions:
                    next_wp_pos = wp_positions[next_wp_name]
                    to_next = np.array([next_wp_pos[0] - current_pos[0], next_wp_pos[2] - current_pos[2]])
                    if np.linalg.norm(to_next) > 1e-8:
                        to_next_norm = to_next / np.linalg.norm(to_next)
                        dot = float(np.dot(current_facing, to_next_norm))
                        if dot > 0:
                            score += dot
                        else:
                            score -= 0.5
                
            elif wtype == 'go_forward' and wname:
                if wname in wp_positions:
                    new_pos = wp_positions[wname]
                    move_dir = np.array([new_pos[0] - current_pos[0], new_pos[2] - current_pos[2]])
                    if np.linalg.norm(move_dir) > 1e-8:
                        current_facing = move_dir / np.linalg.norm(move_dir)
                    current_pos = new_pos.copy()
        
        score_details.append(f"{letter}: score={score:.3f}")
        if score > best_score:
            best_score = score
            best_opt = letter
    
    reasoning = f"route_fixed: start={start_name}→{e_start[0].category} facing={facing_name}→{e_facing[0].category}, " + "; ".join(score_details)
    return best_opt, reasoning


# ============================================================================
# VL Model (轻量, 仅用于route fallback)
# ============================================================================
class VLModel:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.processor = None
    
    def load(self, model_path: str):
        if self.model is not None:
            return
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        logger.info("VL model loaded for route fallback")
    
    def call(self, prompt: str, video_path: str, max_tokens: int = 80,
             nframes: int = 32, max_pixels: int = 360*420) -> str:
        if self.model is None:
            return ""
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": max_pixels, "nframes": nframes},
            {"type": "text", "text": prompt}
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        out_ids = ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


# ============================================================================
# VL Route prompt
# ============================================================================
def vl_answer_route(vl: VLModel, question: str, options: List[str], video_path: str) -> str:
    """纯VL回答route问题"""
    opts_text = "\n".join(options)
    prompt = f"""Read the question carefully and watch the video.
{question}
{opts_text}
Think about the spatial layout of the objects in the scene. 
Imagine yourself as the robot and reason about which turns would lead you to each waypoint.
Answer with ONLY the option letter (A, B, C, or D):"""
    
    response = vl.call(prompt, video_path, max_tokens=50, nframes=32)
    # 提取选项字母
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response:
            return letter
    return 'A'


# ============================================================================
# Main test
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
        args.device = 'cuda:0'
        logger.info(f"GPU {args.gpu_id}: CUDA_VISIBLE_DEVICES={visible_devices}, using cuda:0")
    else:
        args.device = f'cuda:{args.gpu_id}'
    
    # Load V7 baseline
    v7_path = PROJECT_ROOT / "outputs" / "evolving_agent_v7_20260203_134612" / "detailed_results.json"
    with open(v7_path) as f:
        v7_results = json.load(f)
    
    # 只取route_planning
    route_samples = [s for s in v7_results if s['question_type'] == 'route_planning' and find_video_path(s['scene_name'])]
    logger.info(f"Total route_planning samples: {len(route_samples)}")
    
    # 分GPU
    by_scene = defaultdict(list)
    for s in route_samples:
        by_scene[s['scene_name']].append(s)
    scene_list = sorted(by_scene.keys())
    my_scenes = [scene_list[i] for i in range(len(scene_list)) if i % args.num_gpus == args.gpu_id]
    my_samples = sum(len(by_scene[s]) for s in my_scenes)
    logger.info(f"GPU {args.gpu_id}/{args.num_gpus}: {len(my_scenes)} scenes, {my_samples} samples")
    
    # Load models
    builder = Grid64Builder(device=args.device, num_frames=32)
    builder.load_models()
    
    vl = VLModel(device=args.device)
    vl_model_path = '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
    vl.load(vl_model_path)
    
    results = []
    stats = {'grid_used': 0, 'vl_fallback': 0, 'grid_correct': 0, 'vl_correct': 0,
             'old_algo_correct': 0, 'total': 0}
    
    for si, scene_name in enumerate(my_scenes):
        samples = by_scene[scene_name]
        video_path = find_video_path(scene_name)
        if not video_path:
            continue
        
        logger.info(f"[{si+1}/{len(my_scenes)}] {scene_name} ({len(samples)} route q)")
        
        # Build grid once per scene
        grid = builder.build_grid(video_path, target_objects=None)
        
        for s in samples:
            t0 = time.time()
            question = s['question']
            options = s.get('options', [])
            gt = s['ground_truth']
            
            # Method 1: 修复后的Grid route
            pred_grid, reason_grid = grid_answer_route_fixed(grid, question, options)
            
            # Method 2: 纯VL回答
            pred_vl = vl_answer_route(vl, question, options, video_path)
            
            # 决策: facing=start -> 用VL, 否则用Grid
            if pred_grid == FACING_START_MARKER:
                final_pred = pred_vl
                method = 'vl_fallback'
                stats['vl_fallback'] += 1
                if final_pred == gt:
                    stats['vl_correct'] += 1
            else:
                final_pred = pred_grid
                method = 'grid'
                stats['grid_used'] += 1
                if final_pred == gt:
                    stats['grid_correct'] += 1
            
            # 也跑一下旧算法对比
            from scripts.grid64_real_test import grid_answer_route as old_route
            pred_old, _ = old_route(grid, question, options)
            if pred_old == gt:
                stats['old_algo_correct'] += 1
            
            score = 1.0 if final_pred == gt else 0.0
            elapsed = time.time() - t0
            stats['total'] += 1
            
            results.append({
                'scene_name': scene_name,
                'question_type': 'route_planning',
                'question': question[:200],
                'ground_truth': gt,
                'prediction': final_pred,
                'pred_grid': pred_grid,
                'pred_vl': pred_vl,
                'pred_old': pred_old,
                'method': method,
                'reason': reason_grid[:200],
                'score': score,
                'elapsed_s': round(elapsed, 1),
            })
            
            tag = '✓' if score == 1 else '✗'
            old_tag = '✓' if pred_old == gt else '✗'
            logger.info(f"  {tag} [{method}] pred={final_pred} gt={gt} | old={pred_old}{old_tag} | {reason_grid[:80]}")
    
    # Save results
    out_dir = PROJECT_ROOT / "outputs" / "route_fix_test" / f"gpu{args.gpu_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    total = stats['total']
    if total > 0:
        new_correct = sum(1 for r in results if r['score'] == 1)
        vl_only = sum(1 for r in results if r['pred_vl'] == r['ground_truth'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"GPU {args.gpu_id} SUMMARY ({total} route samples)")
        logger.info(f"{'='*60}")
        logger.info(f"  NEW (grid+vl_fallback): {new_correct}/{total} = {new_correct/total:.4f}")
        logger.info(f"    Grid used: {stats['grid_used']}, correct: {stats['grid_correct']}")
        logger.info(f"    VL fallback: {stats['vl_fallback']}, correct: {stats['vl_correct']}")
        logger.info(f"  OLD algorithm:          {stats['old_algo_correct']}/{total} = {stats['old_algo_correct']/total:.4f}")
        logger.info(f"  Pure VL:                {vl_only}/{total} = {vl_only/total:.4f}")
        logger.info(f"Results saved: {out_dir}")

if __name__ == '__main__':
    main()
