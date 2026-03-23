#!/usr/bin/env python3
"""
V7.3 - 方向问题修复版

修复内容：
1. 正则表达式支持 hard 格式 (standing by X, facing Y, is Z...)
2. 正确的方向计算算法（三物体局部坐标系）
3. 保存完整的 position_3d 坐标
4. 仅测试方向问题，快速验证
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
import cv2
from tqdm import tqdm

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从 V7 导入基础组件
from tests.test_evolving_agent_v7_dual_reasoning import (
    MindMapBuilder, ScaleCalibrator, MindMapEvolver, MindMapEntity,
    find_video_path, match_object_name, NUMERICAL_TASKS, CHOICE_TASKS,
    CALIBRATION_OBJECTS, EXTENDED_VOCABULARY
)


# ============================================================================
# 修复的问题解析 - 支持所有方向问题格式
# ============================================================================

def parse_direction_question(question: str) -> Dict:
    """
    解析方向问题，提取物体名称
    
    支持格式：
    1. hard: "standing by X, facing Y, is Z to my front-left/..."
    2. medium: "direction of X from Y"
    3. easy: "is X to the left/right of Y"
    """
    q = question.lower()
    
    # hard格式: 三物体 (standing, facing, target)
    m = re.search(r"standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+) to my", q)
    if m:
        return {
            'format': 'hard',
            'standing': m.group(1),
            'facing': m.group(2),
            'target': m.group(3),
        }
    
    # medium格式: direction of X from Y
    m = re.search(r"direction of (?:the )?(\w+) from (?:the )?(\w+)", q)
    if m:
        return {
            'format': 'medium',
            'target': m.group(1),
            'ref': m.group(2),
        }
    
    # easy格式: is X to the left/right of Y  
    m = re.search(r"is (?:the )?(\w+) to the.*?of (?:the )?(\w+)", q)
    if m:
        return {
            'format': 'easy',
            'target': m.group(1),
            'ref': m.group(2),
        }
    
    # 另一种hard格式: standing by X, facing Y, is Z to the left/right
    m = re.search(r"standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+) to the", q)
    if m:
        return {
            'format': 'hard_lr',
            'standing': m.group(1),
            'facing': m.group(2),
            'target': m.group(3),
        }
    
    return None


# ============================================================================
# 修复的方向计算算法
# ============================================================================

def compute_direction_hard(standing_pos: np.ndarray, facing_pos: np.ndarray, 
                           target_pos: np.ndarray) -> Tuple[str, Dict]:
    """
    计算 hard 格式的方向（三物体问题）
    
    站在 standing_pos，面朝 facing_pos，目标在 target_pos
    返回: front-left, front-right, back-left, back-right
    
    使用俯视图 (x-z 平面):
    - x: 左右（正为右）
    - z: 深度（正为远/前）
    """
    debug_info = {}
    
    # 面朝方向向量 (俯视图: x, z)
    face_dir = np.array([facing_pos[0] - standing_pos[0], 
                         facing_pos[2] - standing_pos[2]])
    face_len = np.linalg.norm(face_dir)
    
    if face_len < 1e-6:
        return "unknown", {'error': 'face_dir too short'}
    
    face_dir = face_dir / face_len
    debug_info['face_dir'] = face_dir.tolist()
    
    # 右方向 (face_dir 顺时针旋转90度)
    # 如果 face = (fx, fz)，右 = (fz, -fx)
    right_dir = np.array([face_dir[1], -face_dir[0]])
    debug_info['right_dir'] = right_dir.tolist()
    
    # 目标方向
    target_dir = np.array([target_pos[0] - standing_pos[0],
                           target_pos[2] - standing_pos[2]])
    debug_info['target_dir'] = target_dir.tolist()
    
    # 投影到局部坐标系
    front_component = float(np.dot(target_dir, face_dir))
    right_component = float(np.dot(target_dir, right_dir))
    debug_info['front_component'] = front_component
    debug_info['right_component'] = right_component
    
    # 判断方位
    fb = "front" if front_component > 0 else "back"
    lr = "right" if right_component > 0 else "left"
    
    return f"{fb}-{lr}", debug_info


def compute_direction_simple(ref_pos: np.ndarray, target_pos: np.ndarray) -> Tuple[List[str], Dict]:
    """
    计算简单相对方向（两物体问题）
    
    返回多个可能的方向描述
    """
    diff = target_pos - ref_pos
    debug_info = {'diff': diff.tolist()}
    
    directions = []
    
    # 左右 (x轴)
    if abs(diff[0]) > 0.3:
        directions.append("right" if diff[0] > 0 else "left")
    
    # 上下 (y轴)
    if abs(diff[1]) > 0.3:
        directions.append("below" if diff[1] > 0 else "above")
    
    # 前后 (z轴 - 深度)
    if abs(diff[2]) > 0.3:
        directions.append("behind" if diff[2] > 0 else "in front")
    
    return directions if directions else ["same"], debug_info


# ============================================================================
# 物体匹配
# ============================================================================

def find_object_in_mindmap(obj_name: str, mind_map: Dict[str, MindMapEntity]) -> Optional[MindMapEntity]:
    """在心智地图中查找匹配的物体"""
    if not obj_name:
        return None
    
    obj_lower = obj_name.lower().strip()
    
    for label, entity in mind_map.items():
        if match_object_name(obj_lower, label):
            return entity
    
    return None


# ============================================================================
# 修复的方向规则推理
# ============================================================================

def rule_answer_direction_fixed(
    mind_map: Dict[str, MindMapEntity],
    question: str,
    options: List[str],
) -> Tuple[str, str, Dict]:
    """
    修复的方向规则推理
    
    返回: (prediction, reasoning, extra_info)
    extra_info 包含完整的坐标和计算过程
    """
    extra_info = {
        'parsed': None,
        'positions': {},
        'computed_direction': None,
        'debug': {},
    }
    
    # 解析问题
    parsed = parse_direction_question(question)
    extra_info['parsed'] = parsed
    
    if not parsed:
        return options[0][0] if options else "A", "Could not parse question format", extra_info
    
    if parsed['format'] in ['hard', 'hard_lr']:
        # 三物体格式
        standing_e = find_object_in_mindmap(parsed['standing'], mind_map)
        facing_e = find_object_in_mindmap(parsed['facing'], mind_map)
        target_e = find_object_in_mindmap(parsed['target'], mind_map)
        
        # 记录匹配结果
        extra_info['positions'] = {
            'standing': {
                'query': parsed['standing'],
                'found': standing_e.label if standing_e else None,
                'pos': standing_e.position_3d.tolist() if standing_e and standing_e.position_3d is not None else None,
            },
            'facing': {
                'query': parsed['facing'],
                'found': facing_e.label if facing_e else None,
                'pos': facing_e.position_3d.tolist() if facing_e and facing_e.position_3d is not None else None,
            },
            'target': {
                'query': parsed['target'],
                'found': target_e.label if target_e else None,
                'pos': target_e.position_3d.tolist() if target_e and target_e.position_3d is not None else None,
            },
        }
        
        # 检查是否找到所有物体
        if not all([standing_e, facing_e, target_e]):
            missing = []
            if not standing_e: missing.append(parsed['standing'])
            if not facing_e: missing.append(parsed['facing'])
            if not target_e: missing.append(parsed['target'])
            return options[0][0] if options else "A", f"Objects not found: {missing}", extra_info
        
        # 检查是否有坐标
        if any(e.position_3d is None for e in [standing_e, facing_e, target_e]):
            return options[0][0] if options else "A", "Missing position_3d data", extra_info
        
        # 计算方向
        computed, debug = compute_direction_hard(
            standing_e.position_3d, facing_e.position_3d, target_e.position_3d
        )
        extra_info['computed_direction'] = computed
        extra_info['debug'] = debug
        
        # 匹配选项
        reasoning = f"standing={standing_e.label}@{standing_e.position_3d.tolist()}, "
        reasoning += f"facing={facing_e.label}@{facing_e.position_3d.tolist()}, "
        reasoning += f"target={target_e.label}@{target_e.position_3d.tolist()} -> {computed}"
        
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            if computed in opt_lower:
                return chr(65 + i), reasoning + f" -> matched option {chr(65 + i)}", extra_info
        
        return options[0][0] if options else "A", reasoning + " (no option match)", extra_info
    
    elif parsed['format'] in ['medium', 'easy']:
        # 两物体格式
        ref_e = find_object_in_mindmap(parsed['ref'], mind_map)
        target_e = find_object_in_mindmap(parsed['target'], mind_map)
        
        extra_info['positions'] = {
            'ref': {
                'query': parsed['ref'],
                'found': ref_e.label if ref_e else None,
                'pos': ref_e.position_3d.tolist() if ref_e and ref_e.position_3d is not None else None,
            },
            'target': {
                'query': parsed['target'],
                'found': target_e.label if target_e else None,
                'pos': target_e.position_3d.tolist() if target_e and target_e.position_3d is not None else None,
            },
        }
        
        if not ref_e or not target_e:
            return options[0][0] if options else "A", "Objects not found", extra_info
        
        if ref_e.position_3d is None or target_e.position_3d is None:
            return options[0][0] if options else "A", "Missing position_3d", extra_info
        
        # 计算方向
        directions, debug = compute_direction_simple(ref_e.position_3d, target_e.position_3d)
        extra_info['computed_direction'] = directions
        extra_info['debug'] = debug
        
        reasoning = f"ref={ref_e.label}, target={target_e.label} -> {directions}"
        
        # 匹配选项
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            for d in directions:
                if d in opt_lower:
                    return chr(65 + i), reasoning + f" -> matched '{d}' in option {chr(65 + i)}", extra_info
        
        return options[0][0] if options else "A", reasoning + " (no match)", extra_info
    
    return options[0][0] if options else "A", f"Unknown format: {parsed['format']}", extra_info


# ============================================================================
# Worker 进程
# ============================================================================

def worker_process(gpu_id: int, samples: List[Dict], result_queue: mp.Queue):
    """单GPU工作进程"""
    # 关键：设置 CUDA_VISIBLE_DEVICES 而不是直接使用 cuda:X
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda'
    
    # 初始化
    builder = MindMapBuilder(device=device, num_frames=32)
    calibrator = ScaleCalibrator()
    
    results = []
    scene_cache = {}  # 缓存同场景的心智地图
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}", position=gpu_id):
        scene_name = sample['scene_name']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 获取或构建心智地图
            if scene_name not in scene_cache:
                video_path = sample['video_path']
                mind_map, total_frames, _ = builder.build_from_video(video_path)
                
                # 校准
                calibration = calibrator.calibrate(mind_map)
                if calibration.scale_factor != 1.0:
                    for entity in mind_map.values():
                        if entity.position_3d is not None:
                            entity.position_3d = entity.position_3d * calibration.scale_factor
                        if entity.size_3d is not None:
                            entity.size_3d = entity.size_3d * calibration.scale_factor
                
                scene_cache[scene_name] = {
                    'mind_map': mind_map,
                    'calibration': calibration,
                    'total_frames': total_frames,
                }
            
            cached = scene_cache[scene_name]
            mind_map = cached['mind_map']
            calibration = cached['calibration']
            
            # 规则推理
            rule_pred, rule_reasoning, extra_info = rule_answer_direction_fixed(
                mind_map, question, options
            )
            rule_correct = rule_pred == gt
            
            # 保存完整的心智地图坐标
            mind_map_full = {}
            for label, entity in mind_map.items():
                mind_map_full[label] = {
                    'count': entity.count,
                    'confidence': round(entity.avg_confidence, 3),
                    'first_seen_frame': entity.first_seen_frame,
                    'position_3d': entity.position_3d.tolist() if entity.position_3d is not None else None,
                    'size_3d': entity.size_3d.tolist() if entity.size_3d is not None else None,
                }
            
            results.append({
                'scene_name': scene_name,
                'question': question,
                'question_type': question_type,
                'ground_truth': gt,
                'options': options,
                'rule_prediction': rule_pred,
                'rule_reasoning': rule_reasoning,
                'rule_correct': rule_correct,
                'parsed': extra_info.get('parsed'),
                'positions': extra_info.get('positions'),
                'computed_direction': extra_info.get('computed_direction'),
                'debug': extra_info.get('debug'),
                'calibration': {
                    'object': calibration.calibration_object,
                    'scale_factor': round(calibration.scale_factor, 4),
                },
                'mind_map_full': mind_map_full,
            })
            
        except Exception as e:
            import traceback
            results.append({
                'scene_name': scene_name,
                'question': question,
                'question_type': question_type,
                'ground_truth': gt,
                'options': options,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })
    
    # 清理
    builder.unload()
    gc.collect()
    torch.cuda.empty_cache()
    
    result_queue.put(results)


# ============================================================================
# 主函数
# ============================================================================

def load_direction_data() -> List[Dict]:
    """加载方向问题数据"""
    from datasets import load_dataset
    
    logger.info("加载 VSI-Bench 数据集...")
    ds = load_dataset(
        'nyu-visionx/VSI-Bench',
        split='test',
        cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench'
    )
    
    samples = []
    for item in ds:
        # 只筛选方向问题
        if 'direction' not in item['question_type']:
            continue
        
        scene_name = item['scene_name']
        video_path = find_video_path(scene_name)
        
        if not video_path:
            continue
        
        samples.append({
            'scene_name': scene_name,
            'video_path': video_path,
            'question': item['question'],
            'question_type': item['question_type'],
            'options': item.get('options', []),
            'ground_truth': item['ground_truth'],
        })
    
    logger.info(f"筛选方向问题: {len(samples)} 条")
    return samples


def main():
    parser = argparse.ArgumentParser(description='V7.3 方向问题修复测试')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    # 加载数据
    samples = load_direction_data()
    
    if args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"限制为: {len(samples)} 条")
    
    # 分配到 GPU
    num_gpus = min(args.num_gpus, torch.cuda.device_count(), len(samples))
    chunks = [samples[i::num_gpus] for i in range(num_gpus)]
    
    logger.info(f"使用 {num_gpus} 个 GPU 并行处理")
    
    # 并行处理
    result_queue = mp.Queue()
    processes = []
    
    for gpu_id in range(num_gpus):
        if len(chunks[gpu_id]) > 0:
            p = mp.Process(target=worker_process, args=(gpu_id, chunks[gpu_id], result_queue))
            p.start()
            processes.append(p)
    
    # 收集结果
    all_results = []
    for _ in range(len(processes)):
        results = result_queue.get()
        all_results.extend(results)
    
    for p in processes:
        p.join()
    
    # 统计
    print("\n" + "="*80)
    print("V7.3 方向问题修复测试结果")
    print("="*80)
    
    # 按问题类型统计
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r.get('question_type', 'unknown')].append(r)
    
    for qtype, items in sorted(by_type.items()):
        total = len(items)
        errors = sum(1 for r in items if 'error' in r)
        parsed = sum(1 for r in items if r.get('parsed'))
        computed = sum(1 for r in items if r.get('computed_direction'))
        correct = sum(1 for r in items if r.get('rule_correct'))
        
        print(f"\n{qtype}: {total} 样本")
        print(f"  错误: {errors} ({100*errors/total:.1f}%)")
        print(f"  成功解析: {parsed} ({100*parsed/total:.1f}%)")
        print(f"  成功计算: {computed} ({100*computed/total:.1f}%)")
        print(f"  规则正确: {correct} ({100*correct/total:.1f}%)")
    
    # 总体统计
    total = len(all_results)
    errors = sum(1 for r in all_results if 'error' in r)
    correct = sum(1 for r in all_results if r.get('rule_correct'))
    
    print(f"\n{'='*80}")
    print(f"总计: {total} 样本")
    print(f"错误: {errors} ({100*errors/total:.1f}%)")
    print(f"正确: {correct} ({100*correct/total:.1f}%)")
    print(f"{'='*80}")
    
    # 保存结果
    output_dir = Path(args.output_dir) / f"v73_direction_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果
    with open(output_dir / "detailed_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存汇总
    summary = {
        'total_samples': total,
        'errors': errors,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'by_type': {
            qtype: {
                'total': len(items),
                'correct': sum(1 for r in items if r.get('rule_correct')),
                'accuracy': sum(1 for r in items if r.get('rule_correct')) / len(items) if items else 0,
            }
            for qtype, items in by_type.items()
        }
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"结果已保存到: {output_dir}")
    
    # 打印典型案例
    print("\n" + "="*80)
    print("典型案例分析")
    print("="*80)
    
    # 正确案例
    correct_cases = [r for r in all_results if r.get('rule_correct') and r.get('positions')][:3]
    if correct_cases:
        print("\n【正确案例】")
        for r in correct_cases:
            print(f"\n场景: {r['scene_name']}, 类型: {r['question_type']}")
            print(f"问题: {r['question'][:80]}...")
            print(f"解析: {r.get('parsed')}")
            print(f"坐标: standing={r['positions'].get('standing', {}).get('pos')}")
            print(f"       facing={r['positions'].get('facing', {}).get('pos')}")
            print(f"       target={r['positions'].get('target', {}).get('pos')}")
            print(f"计算: {r.get('computed_direction')}")
            print(f"结果: GT={r['ground_truth']}, Pred={r['rule_prediction']} ✓")
    
    # 错误案例（有计算结果但不正确）
    wrong_cases = [r for r in all_results 
                   if not r.get('rule_correct') and r.get('computed_direction') 
                   and 'error' not in r][:3]
    if wrong_cases:
        print("\n【错误案例（有计算但不正确）】")
        for r in wrong_cases:
            print(f"\n场景: {r['scene_name']}, 类型: {r['question_type']}")
            print(f"问题: {r['question'][:80]}...")
            print(f"解析: {r.get('parsed')}")
            print(f"坐标: standing={r['positions'].get('standing', {}).get('pos')}")
            print(f"       facing={r['positions'].get('facing', {}).get('pos')}")
            print(f"       target={r['positions'].get('target', {}).get('pos')}")
            print(f"计算: {r.get('computed_direction')}")
            print(f"选项: {r.get('options')}")
            print(f"结果: GT={r['ground_truth']}, Pred={r['rule_prediction']} ✗")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
