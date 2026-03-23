#!/usr/bin/env python3
"""
VSIBench V5 测试 - 统一框架（体素+特征+概率）

作者: tianjungu
日期: 2026-01-29
"""

import os
import sys
import json
import gc
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import re
import logging

import numpy as np

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from tests.test_vsibench_directqa import (
    normalize_number, mean_relative_accuracy,
    NUMERICAL_TASKS, CHOICE_TASKS,
    find_video_path, get_scene_source,
    EXTENDED_VOCABULARY,
)

from core.mind_map_v5 import (
    MindMapBuilderV5,
    MindMapReasonerV5,
    SparseVoxelMap,
    MindMapEntityV5,
)


def worker_process(gpu_id: int, samples: List[Dict], result_queue: mp.Queue,
                   extract_features: bool = False):
    """GPU Worker 进程"""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 创建 V5 构建器
    builder = MindMapBuilderV5(
        device='cuda',
        num_frames=32,
        box_threshold=0.25,
        voxel_size=0.1,
        extract_features=extract_features,
    )
    
    results = []
    total = len(samples)
    
    for i, sample in enumerate(samples):
        video_path = sample['video_path']
        question = sample['question']
        question_type = sample['question_type']
        gt = sample['ground_truth']
        options = sample.get('options', [])
        
        try:
            # 提取目标物体
            target_objects = []
            if question_type == 'object_counting':
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            # 任务感知的尺度校准策略
            # room_size_estimation 不使用动态尺度校准（避免回归）
            use_dynamic_scale = (question_type != 'room_size_estimation')
            
            # 构建心智地图
            entities, voxel_map = builder.build_from_video(
                video_path, target_objects, 
                use_dynamic_scale=use_dynamic_scale
            )
            
            # 使用统一推理器回答问题
            confidence = 0.5
            
            if question_type == 'object_counting':
                pred, confidence = MindMapReasonerV5.answer_counting(entities, question)
            
            elif question_type == 'object_size_estimation':
                pred, confidence = MindMapReasonerV5.answer_object_size(entities, question)
            
            elif question_type == 'room_size_estimation':
                # 关键改进：使用体素地图
                pred, confidence = MindMapReasonerV5.answer_room_size(entities, voxel_map, question)
            
            elif question_type == 'object_abs_distance':
                pred, confidence = MindMapReasonerV5.answer_abs_distance(entities, question)
            
            elif question_type.startswith('object_rel_direction'):
                difficulty = question_type.split('_')[-1]
                pred, confidence = MindMapReasonerV5.answer_rel_direction(
                    entities, question, options, difficulty
                )
            
            elif question_type == 'object_rel_distance':
                pred, confidence = MindMapReasonerV5.answer_rel_distance(entities, question, options)
            
            elif question_type == 'obj_appearance_order':
                pred, confidence = MindMapReasonerV5.answer_appearance_order(entities, question, options)
            
            elif question_type == 'route_planning':
                pred, confidence = MindMapReasonerV5.answer_route_planning(
                    entities, voxel_map, question, options
                )
            
            else:
                pred = str(options[0]) if options else "0"
            
            # 计算指标
            if question_type in NUMERICAL_TASKS:
                pred_num = normalize_number(pred)
                gt_num = normalize_number(str(gt))
                score = mean_relative_accuracy(pred_num, gt_num) if pred_num and gt_num else 0.0
                correct = score > 0.5
            else:
                pred_letter = None
                gt_letter = str(gt).strip().upper()
                
                if len(gt_letter) > 1:
                    for idx, opt in enumerate(options):
                        opt_content = re.sub(r'^[A-D]\.\s*', '', opt).strip()
                        if gt_letter.lower() == opt_content.lower():
                            gt_letter = chr(65 + idx)
                            break
                
                if pred:
                    letter_match = re.match(r'^([A-D])[\.\s]', pred.strip().upper())
                    if letter_match:
                        pred_letter = letter_match.group(1)
                    else:
                        pred_clean = re.sub(r'^[A-D]\.\s*', '', pred).lower().strip()
                        for idx, opt in enumerate(options):
                            opt_content = re.sub(r'^[A-D]\.\s*', '', opt).lower().strip()
                            if pred_clean == opt_content or pred_clean in opt_content or opt_content in pred_clean:
                                pred_letter = chr(65 + idx)
                                break
                
                correct = pred_letter == gt_letter if pred_letter else False
                score = 1.0 if correct else 0.0
            
            result = {
                'sample_id': sample['id'],
                'question_type': question_type,
                'question': question,
                'prediction': pred,
                'ground_truth': gt,
                'score': score,
                'correct': bool(correct),
                'confidence': float(confidence),
                'num_entities': len(entities),
                'num_voxels': len(voxel_map.voxels),
            }
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} 样本 {sample['id']} 错误: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'sample_id': sample['id'],
                'question_type': question_type,
                'question': question,
                'prediction': '',
                'ground_truth': gt,
                'score': 0.0,
                'correct': False,
                'confidence': 0.0,
                'error': str(e),
            }
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"GPU {gpu_id}: {i+1}/{total} 完成")
    
    builder.unload()
    result_queue.put((gpu_id, results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--task-type', type=str, default='all')
    parser.add_argument('--extract-features', action='store_true', help='提取 DINOv2 特征')
    args = parser.parse_args()
    
    print("=" * 70)
    print("VSIBench V5 测试 - 统一框架（体素+特征+概率）")
    print("=" * 70)
    print(f"GPU数量: {args.num_gpus}")
    print(f"提取特征: {args.extract_features}")
    
    from datasets import load_dataset
    dataset = load_dataset(
        "nyu-visionx/VSI-Bench",
        split="test",
        cache_dir="/home/tione/notebook/tianjungu/hf_cache/vsibench"
    )
    
    # 准备样本
    samples = []
    for idx, item in enumerate(dataset):
        scene_name = item['scene_name']
        question_type = item['question_type']
        
        if args.task_type != 'all' and question_type != args.task_type:
            continue
        
        video_path = find_video_path(scene_name)
        if not video_path:
            continue
        
        samples.append({
            'id': idx,
            'scene_name': scene_name,
            'source': get_scene_source(scene_name),
            'video_path': video_path,
            'question': item['question'],
            'question_type': question_type,
            'ground_truth': item['ground_truth'],
            'options': item.get('options', []),
        })
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"总样本数: {len(samples)}")
    
    type_counts = defaultdict(int)
    for s in samples:
        type_counts[s['question_type']] += 1
    print("\n任务类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    
    # 分配到 GPU
    num_gpus = min(args.num_gpus, len(samples))
    samples_per_gpu = len(samples) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])
    
    # 启动多进程
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    
    start_time = datetime.now()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_samples[gpu_id], result_queue, args.extract_features)
        )
        p.start()
        processes.append(p)
        logger.info(f"启动 GPU {gpu_id}: {len(gpu_samples[gpu_id])} 样本")
    
    # 收集结果
    all_results = []
    for _ in range(num_gpus):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        logger.info(f"GPU {gpu_id} 完成: {len(results)} 结果")
    
    for p in processes:
        p.join()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n总耗时: {duration:.1f}秒")
    
    # 统计
    type_scores = defaultdict(list)
    type_confidences = defaultdict(list)
    
    for r in all_results:
        type_scores[r['question_type']].append(r['score'])
        type_confidences[r['question_type']].append(r.get('confidence', 0.5))
    
    print("\n" + "="*70)
    print("V5 统一框架测试结果")
    print("="*70)
    
    total_score = 0
    total_count = 0
    
    for q_type in sorted(type_scores.keys()):
        scores = type_scores[q_type]
        confidences = type_confidences[q_type]
        
        if q_type in NUMERICAL_TASKS:
            avg = np.mean(scores) * 100
            metric = "MRA"
        else:
            avg = np.mean(scores) * 100
            metric = "Acc"
        
        avg_conf = np.mean(confidences) * 100
        
        print(f"{q_type:30s}: {avg:6.2f}% {metric} (置信度: {avg_conf:.1f}%, {len(scores)} 样本)")
        total_score += sum(scores)
        total_count += len(scores)
    
    overall = total_score / total_count * 100 if total_count > 0 else 0
    print("-"*70)
    print(f"{'Overall':30s}: {overall:6.2f}% ({total_count} 样本)")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/v5_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    cleaned_results = []
    for r in all_results:
        cleaned = {k: convert_numpy(v) for k, v in r.items()}
        cleaned_results.append(cleaned)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'config': 'V5: voxel_map + probabilistic_position + semantic_features',
            'summary': {
                q: {
                    'mean': float(np.mean(s)),
                    'count': len(s),
                    'avg_confidence': float(np.mean(type_confidences[q]))
                }
                for q, s in type_scores.items()
            },
            'overall': float(overall),
            'details': cleaned_results,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果保存到: {output_dir}")
    
    # 打印置信度分析
    print("\n" + "="*70)
    print("置信度分析")
    print("="*70)
    
    # 高置信度样本的准确率
    high_conf_correct = []
    low_conf_correct = []
    
    for r in all_results:
        conf = r.get('confidence', 0.5)
        if conf > 0.6:
            high_conf_correct.append(r['score'])
        else:
            low_conf_correct.append(r['score'])
    
    if high_conf_correct:
        print(f"高置信度 (>0.6) 样本准确率: {np.mean(high_conf_correct)*100:.2f}% ({len(high_conf_correct)} 样本)")
    if low_conf_correct:
        print(f"低置信度 (<=0.6) 样本准确率: {np.mean(low_conf_correct)*100:.2f}% ({len(low_conf_correct)} 样本)")


if __name__ == '__main__':
    main()
