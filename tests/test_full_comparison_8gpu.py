#!/usr/bin/env python3
"""
完整对比测试 - 8卡并行

测试 VSIBench 全部 565 个 object_counting 样本
对比方法:
1. MindMap 方法 (帧间最大计数)
2. DINO+DA3+3D聚类 (之前效果好的方法)

作者: tianjungu
日期: 2026-01-25
"""

import os
import sys
import json
import time
import gc
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import re

import numpy as np
import cv2

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# VSIBench 评测指标
# ============================================================================

def normalize_number(text: str) -> Optional[float]:
    """提取数字"""
    if text is None:
        return None
    text = re.sub(r'(meters?|m|cm|feet|ft|inches?|in)\b', '', text.lower())
    match = re.search(r'[-+]?\d*\.?\d+', text)
    if match:
        try:
            return float(match.group())
        except:
            return None
    return None


def mean_relative_accuracy(pred: float, target: float, start=0.05, end=0.5, interval=0.05) -> float:
    """VSIBench 官方 MRA 指标"""
    if pred is None or target is None:
        return 0.0
    epsilon = 1e-8
    rel_error = abs(pred - target) / (abs(target) + epsilon)
    thresholds = np.arange(start, end + interval / 2, interval)
    conditions = rel_error < (1 - thresholds)
    return conditions.astype(float).mean()


# ============================================================================
# 物体类型自适应阈值
# ============================================================================

OBJECT_TYPE_THRESHOLDS = {
    'chair': 0.35,
    'table': 0.45,
    'sofa': 0.45,
    'bed': 0.45,
    'stool': 0.40,
    'door': 0.45,
    'window': 0.40,
    'washer': 0.35,
    'lamp': 0.40,
    'tv': 0.45,
    'monitor': 0.45,
    'pillow': 0.45,
    'toilet': 0.45,
    'bathtub': 0.45,
    'refrigerator': 0.45,
    'sink': 0.40,
    'mirror': 0.40,
    'towel': 0.35,
    'backpack': 0.35,
    'trash': 0.35,
    'default': 0.45
}


# ============================================================================
# Worker 进程
# ============================================================================

def worker_mindmap(gpu_id: int, samples: List[Dict], result_queue: mp.Queue):
    """MindMap 方法 Worker"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    import torch
    
    # 添加项目路径
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from core.semantic_labeler import GroundingDINOLabeler
    
    print(f"[GPU {gpu_id}] Starting MindMap worker with {len(samples)} samples")
    
    # 加载模型
    labeler = GroundingDINOLabeler(
        model_id="IDEA-Research/grounding-dino-base",
        device="cuda",
        box_threshold=0.25,
        text_threshold=0.25,
    )
    labeler.load_model()
    
    results = []
    num_frames = 15
    
    for i, sample in enumerate(samples):
        video_path = sample['video_path']
        target_object = sample['target_object']
        gt = sample['ground_truth']
        
        try:
            # 提取帧
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                results.append({'sample_id': sample['id'], 'error': 'Cannot open video', 'mra': 0.0})
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            if not frames:
                results.append({'sample_id': sample['id'], 'error': 'No frames', 'mra': 0.0})
                continue
            
            # 检测
            text_prompt = f"{target_object} ."
            detections_per_frame = []
            
            for frame in frames:
                dets = labeler.detect(frame, text_prompt)
                detections_per_frame.append(len(dets))
            
            # 取最大值
            pred = max(detections_per_frame) if detections_per_frame else 0
            mra = mean_relative_accuracy(pred, gt)
            
            results.append({
                'sample_id': sample['id'],
                'scene_name': sample['scene_name'],
                'target_object': target_object,
                'ground_truth': gt,
                'prediction': pred,
                'mra': mra,
            })
            
            if (i + 1) % 10 == 0:
                avg_mra = np.mean([r['mra'] for r in results if 'mra' in r])
                print(f"[GPU {gpu_id}] Progress: {i+1}/{len(samples)}, Avg MRA: {avg_mra:.4f}")
                
        except Exception as e:
            print(f"[GPU {gpu_id}] Error on sample {sample['id']}: {e}")
            results.append({'sample_id': sample['id'], 'error': str(e), 'mra': 0.0})
    
    result_queue.put((gpu_id, results))
    print(f"[GPU {gpu_id}] Finished")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full Comparison Test (8 GPUs)')
    parser.add_argument('--num-gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples (None = all)')
    args = parser.parse_args()
    
    output_dir = str(PROJECT_ROOT / "outputs" / f"full_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    print("Loading VSIBench dataset...")
    from datasets import load_dataset
    
    dataset = load_dataset(
        "nyu-visionx/VSI-Bench",
        split="test",
        cache_dir="/home/tione/notebook/tianjungu/hf_cache/vsibench"
    )
    print(f"Dataset size: {len(dataset)}")
    
    # 获取 object_counting 样本
    video_dirs = {
        'ARKitScenes': '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
        'ScanNet': '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
        'HM3D': '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/hm3d',
    }
    
    samples = []
    for item in dataset:
        if item['question_type'] != 'object_counting':
            continue
        
        scene_name = item['scene_name']
        
        if scene_name.startswith('scene'):
            source = 'ScanNet'
        elif scene_name.isdigit():
            source = 'ARKitScenes'
        else:
            source = 'HM3D'
        
        video_path = os.path.join(video_dirs[source], f"{scene_name}.mp4")
        
        if not os.path.exists(video_path):
            continue
        
        question = item['question']
        match = re.search(r'How many (\w+)\(s\)', question)
        if not match:
            continue
        
        target_object = match.group(1)
        gt = normalize_number(item['ground_truth'])
        
        if gt is None:
            continue
        
        samples.append({
            'id': item['id'],
            'scene_name': scene_name,
            'source': source,
            'video_path': video_path,
            'question': question,
            'target_object': target_object,
            'ground_truth': gt,
        })
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"Found {len(samples)} object_counting samples")
    
    # 分配样本到各 GPU
    samples_per_gpu = len(samples) // args.num_gpus
    gpu_samples = []
    for i in range(args.num_gpus):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < args.num_gpus - 1 else len(samples)
        gpu_samples.append(samples[start:end])
    
    # 启动多进程
    print(f"\nStarting {args.num_gpus} workers...")
    result_queue = mp.Queue()
    processes = []
    
    for gpu_id, gpu_sample_list in enumerate(gpu_samples):
        if not gpu_sample_list:
            continue
        p = mp.Process(target=worker_mindmap, args=(gpu_id, gpu_sample_list, result_queue))
        p.start()
        processes.append(p)
    
    # 收集结果
    print("Waiting for results...")
    gpu_results = {}
    for _ in processes:
        gpu_id, results = result_queue.get()
        gpu_results[gpu_id] = results
        print(f"Received results from GPU {gpu_id}: {len(results)} samples")
    
    # 等待所有进程结束
    for p in processes:
        p.join()
    
    # 合并结果
    all_results = []
    for gpu_id in sorted(gpu_results.keys()):
        all_results.extend(gpu_results[gpu_id])
    
    # 计算统计
    mras = [r['mra'] for r in all_results if 'mra' in r]
    preds = [r.get('prediction', 0) for r in all_results if 'prediction' in r]
    gts = [r.get('ground_truth', 0) for r in all_results if 'ground_truth' in r]
    
    over_count = sum(1 for p, g in zip(preds, gts) if p > g)
    under_count = sum(1 for p, g in zip(preds, gts) if p < g)
    exact = sum(1 for p, g in zip(preds, gts) if p == g)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'output_dir': output_dir,
        'num_samples': len(all_results),
        'num_gpus': args.num_gpus,
        'method': 'mindmap',
        'results': {
            'avg_mra': np.mean(mras) if mras else 0,
            'std_mra': np.std(mras) if mras else 0,
            'over_counting': over_count,
            'under_counting': under_count,
            'exact_match': exact,
            'over_counting_pct': over_count / len(all_results) * 100 if all_results else 0,
            'under_counting_pct': under_count / len(all_results) * 100 if all_results else 0,
            'exact_match_pct': exact / len(all_results) * 100 if all_results else 0,
        }
    }
    
    # 保存结果
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 打印报告
    print("\n" + "="*80)
    print("Full Comparison Results - Object Counting Task")
    print("="*80)
    print(f"Total samples: {len(all_results)}")
    print(f"MRA: {summary['results']['avg_mra']:.4f} ± {summary['results']['std_mra']:.4f}")
    print(f"Over-counting: {over_count} ({summary['results']['over_counting_pct']:.1f}%)")
    print(f"Under-counting: {under_count} ({summary['results']['under_counting_pct']:.1f}%)")
    print(f"Exact match: {exact} ({summary['results']['exact_match_pct']:.1f}%)")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
