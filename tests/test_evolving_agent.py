#!/usr/bin/env python3
"""
Self-Evolving Agent VSI-Bench 测试脚本

测试自进化心智地图智能体在 VSI-Bench 上的表现
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import torch

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_HUB_CACHE'] = '/home/tione/notebook/tianjungu/hf_cache/hub'
os.environ['MODELSCOPE_CACHE'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.evolving_agent import (
    SelfEvolvingAgent, TaskType, TaskManager,
    DirectionOptimizer, RouteOptimizer
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 视频路径配置
# ============================================================================

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]


def find_video_path(scene_name: str) -> Optional[str]:
    """查找视频路径"""
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def get_scene_source(scene_name: str) -> str:
    """判断场景来源"""
    if scene_name.startswith('scene'):
        return 'ScanNet'
    elif scene_name.startswith('4'):
        return 'ARKitScenes'
    else:
        return 'ScanNetPP'


# ============================================================================
# 数据集加载
# ============================================================================

def load_vsibench_dataset(data_path: str = None, max_samples: int = None) -> List[Dict]:
    """加载 VSI-Bench 数据集并构建样本"""
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "nyu-visionx/VSI-Bench", 
            split="test",
            cache_dir="/home/tione/notebook/tianjungu/hf_cache/vsibench"
        )
    except Exception as e:
        logger.error(f"无法加载数据集: {e}")
        return []
    
    samples = []
    for idx, item in enumerate(dataset):
        scene_name = item['scene_name']
        question_type = item['question_type']
        
        # 查找视频路径
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
            'ground_truth': str(item['ground_truth']),
            'options': item.get('options', []),
        })
        
        if max_samples and len(samples) >= max_samples:
            break
    
    logger.info(f"加载了 {len(samples)} 个有效样本")
    return samples


# ============================================================================
# 评估指标
# ============================================================================

def mean_relative_accuracy(pred: float, target: float, start=0.05, end=0.5, interval=0.05) -> float:
    """VSIBench 官方 MRA 指标 (多阈值版本)"""
    if pred is None or target is None:
        return 0.0
    epsilon = 1e-8
    rel_error = abs(pred - target) / (abs(target) + epsilon)
    thresholds = np.arange(start, end + interval / 2, interval)
    conditions = rel_error < (1 - thresholds)
    return conditions.astype(float).mean()


def compute_mra(pred: str, gt: str) -> float:
    """计算 Mean Relative Accuracy (MRA) 用于数值答案"""
    try:
        pred_val = float(pred)
        gt_val = float(gt)
        return mean_relative_accuracy(pred_val, gt_val)
    except ValueError:
        return 0.0


def compute_accuracy(pred: str, gt: str, options: List[str] = None) -> float:
    """计算准确率（选择题）"""
    pred_lower = pred.lower().strip()
    gt_lower = gt.lower().strip()
    
    # 直接匹配
    if pred_lower == gt_lower:
        return 1.0
    
    # 选项字母匹配
    if options:
        for i, opt in enumerate(options):
            if opt.lower() == gt_lower:
                if pred_lower == chr(65 + i).lower() or pred_lower == opt.lower():
                    return 1.0
    
    # 部分匹配
    if gt_lower in pred_lower or pred_lower in gt_lower:
        return 0.5
    
    return 0.0


# ============================================================================
# 单 GPU 工作进程
# ============================================================================

def process_sample(args_tuple):
    """处理单个样本"""
    sample, gpu_id, agent_config = args_tuple
    
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # 创建 Agent（每个进程独立创建）
        agent = SelfEvolvingAgent(**agent_config)
        
        # 提取信息
        video_path = sample.get("video_path", "")
        question = sample.get("question", "")
        question_type = sample.get("question_type", "")
        gt_answer = sample.get("ground_truth", "")
        options = sample.get("options", None)
        
        # 处理
        start_time = time.time()
        result = agent.process(
            video_path=video_path,
            question=question,
            question_type=question_type,
            options=options
        )
        elapsed = time.time() - start_time
        
        return {
            "sample_id": sample.get("id", ""),
            "question_type": question_type,
            "pred": result["answer"],
            "gt": gt_answer,
            "options": options,
            "confidence": result["confidence"],
            "evolved": result["evolved"],
            "evolution_rounds": result.get("evolution_rounds", 0),
            "elapsed": elapsed,
            "success": True,
        }
        
    except Exception as e:
        logger.error(f"处理样本失败: {e}")
        return {
            "sample_id": sample.get("id", ""),
            "question_type": sample.get("question_type", ""),
            "pred": "",
            "gt": sample.get("answer", ""),
            "options": sample.get("options", None),
            "success": False,
            "error": str(e),
        }


# ============================================================================
# 主测试函数
# ============================================================================

def run_benchmark(args):
    """运行基准测试"""
    # 加载数据集
    logger.info("加载数据集...")
    dataset = load_vsibench_dataset(args.data_path, args.max_samples)
    
    if not dataset:
        logger.error("数据集为空")
        return
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # Agent 配置
    agent_config = {
        "device": "cuda",
        "num_frames": args.num_frames,
        "confidence_threshold": args.confidence_threshold,
        "max_evolution_rounds": args.max_evolution,
        "use_vl_model": args.use_vl,
        "vl_model_path": args.vl_model,
    }
    
    # 分配到多 GPU
    num_gpus = args.num_gpus
    gpu_assignments = [(sample, i % num_gpus, agent_config) for i, sample in enumerate(dataset)]
    
    # 多进程处理
    results = []
    if num_gpus > 1:
        with mp.Pool(num_gpus) as pool:
            for i, result in enumerate(pool.imap(process_sample, gpu_assignments)):
                results.append(result)
                if (i + 1) % 10 == 0:
                    logger.info(f"进度: {i + 1}/{len(dataset)}")
    else:
        for i, assignment in enumerate(gpu_assignments):
            result = process_sample(assignment)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"进度: {i + 1}/{len(dataset)}")
    
    # 计算指标
    compute_metrics(results, args.output)


def compute_metrics(results: List[Dict], output_path: str = None):
    """计算评估指标"""
    # 按任务类型分组
    by_type = defaultdict(list)
    for r in results:
        if r.get("success", False):
            by_type[r["question_type"]].append(r)
    
    # MRA 任务类型
    mra_types = [
        "object_counting", "object_abs_distance", 
        "object_size_estimation", "room_size_estimation"
    ]
    
    # 计算指标
    metrics = {}
    total_score = 0
    total_count = 0
    
    for qtype, samples in by_type.items():
        if not samples:
            continue
        
        if qtype in mra_types:
            # MRA 指标
            scores = [compute_mra(s["pred"], s["gt"]) for s in samples]
        else:
            # Accuracy 指标
            scores = [compute_accuracy(s["pred"], s["gt"], s.get("options")) for s in samples]
        
        avg_score = np.mean(scores) * 100
        metrics[qtype] = {
            "score": avg_score,
            "count": len(samples),
            "evolved_rate": np.mean([s.get("evolved", False) for s in samples]) * 100,
            "avg_confidence": np.mean([s.get("confidence", 0) for s in samples]),
        }
        
        total_score += avg_score * len(samples)
        total_count += len(samples)
    
    # Overall
    overall = total_score / total_count if total_count > 0 else 0
    
    # 打印结果
    print("\n" + "=" * 80)
    print("【Self-Evolving Agent VSI-Bench 评估结果】")
    print("=" * 80)
    
    print(f"\n{'任务类型':<35} {'得分':>10} {'样本数':>8} {'演化率':>10} {'平均置信度':>12}")
    print("-" * 80)
    
    for qtype in sorted(metrics.keys()):
        m = metrics[qtype]
        print(f"{qtype:<35} {m['score']:>9.2f}% {m['count']:>8} {m['evolved_rate']:>9.2f}% {m['avg_confidence']:>11.3f}")
    
    print("-" * 80)
    print(f"{'Overall':<35} {overall:>9.2f}% {total_count:>8}")
    print("=" * 80)
    
    # 保存结果
    if output_path:
        output_data = {
            "overall": overall,
            "by_type": metrics,
            "results": results,
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到: {output_path}")


# ============================================================================
# 单样本测试（调试用）
# ============================================================================

def test_single_sample(args):
    """测试单个样本"""
    logger.info("单样本测试模式")
    
    # 创建 Agent
    agent = SelfEvolvingAgent(
        device=args.device,
        num_frames=args.num_frames,
        confidence_threshold=args.confidence_threshold,
        max_evolution_rounds=args.max_evolution,
        use_vl_model=args.use_vl,
        vl_model_path=args.vl_model,
    )
    
    # 如果提供了视频和问题
    if args.video and args.question:
        result = agent.process(
            video_path=args.video,
            question=args.question,
            question_type=args.question_type,
            options=args.options.split(",") if args.options else None
        )
        
        print("\n" + "=" * 60)
        print("【推理结果】")
        print(f"答案: {result['answer']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"是否演化: {result['evolved']}")
        if result['evolved']:
            print(f"演化轮数: {result['evolution_rounds']}")
        print(f"\n推理步骤:")
        for step in result['reasoning_steps']:
            print(f"  - {step['step_name']}: {step.get('description', '')[:100]}")
        print(f"\n批判反馈: {result['critic_feedback']}")
        print("=" * 60)
        
        return
    
    # 否则从数据集加载第一个样本
    dataset = load_vsibench_dataset(args.data_path, max_samples=10)
    if not dataset:
        logger.error("无法加载数据集")
        return
    
    sample = dataset[0]
    logger.info(f"测试样本: {sample.get('id', 'N/A')}")
    logger.info(f"问题类型: {sample.get('question_type', 'N/A')}")
    logger.info(f"问题: {sample.get('question', 'N/A')}")
    
    result = agent.process(
        video_path=sample.get("video_path", ""),
        question=sample.get("question", ""),
        question_type=sample.get("question_type", ""),
        options=sample.get("options", None)
    )
    
    gt = sample.get("ground_truth", "")
    
    print("\n" + "=" * 60)
    print(f"预测: {result['answer']}")
    print(f"真值: {gt}")
    print(f"置信度: {result['confidence']:.3f}")
    print(f"演化: {result['evolved']}")
    print("=" * 60)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Self-Evolving Agent VSI-Bench Test")
    
    # 数据集参数
    parser.add_argument("--data-path", type=str, default=None, help="数据集路径")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    
    # Agent 参数
    parser.add_argument("--num-frames", type=int, default=32, help="视频采样帧数")
    parser.add_argument("--confidence-threshold", type=float, default=0.4, help="置信度阈值")
    parser.add_argument("--max-evolution", type=int, default=2, help="最大演化轮数")
    parser.add_argument("--use-vl", action="store_true", help="使用 VL 模型")
    parser.add_argument("--vl-model", type=str, default=None, help="VL 模型路径")
    
    # 运行参数
    parser.add_argument("--num-gpus", type=int, default=1, help="GPU 数量")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    
    # 调试参数
    parser.add_argument("--single", action="store_true", help="单样本测试模式")
    parser.add_argument("--video", type=str, default=None, help="视频路径（单样本）")
    parser.add_argument("--question", type=str, default=None, help="问题（单样本）")
    parser.add_argument("--question-type", type=str, default=None, help="问题类型（单样本）")
    parser.add_argument("--options", type=str, default=None, help="选项，逗号分隔（单样本）")
    
    args = parser.parse_args()
    
    # 设置输出路径
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(PROJECT_ROOT, "outputs", f"evolving_agent_{timestamp}.json")
    
    # 运行
    if args.single:
        test_single_sample(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
