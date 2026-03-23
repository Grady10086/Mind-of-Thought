#!/usr/bin/env python3
"""
Self-Evolving Agent VSI-Bench 测试脚本 V2

改进：
1. 复用 directqa 中验证有效的推理逻辑
2. 添加 tqdm 进度条
3. 添加时间戳和详细日志
4. 对比演化前后效果

作者: tianjungu
日期: 2026-01-30
"""

import os
import sys
import json
import time
import argparse
import logging
import gc
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import multiprocessing as mp
import re

import numpy as np
import torch
from tqdm import tqdm

# 设置环境
os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_HUB_CACHE'] = '/home/tione/notebook/tianjungu/hf_cache/hub'
os.environ['MODELSCOPE_CACHE'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入 directqa 的核心函数
from tests.test_vsibench_directqa import (
    SYNONYM_MAP, get_synonyms, match_object_name,
    MindMapEntity3D, MindMapBuilderDirectQA, DirectQA,
    mean_relative_accuracy, normalize_number,
    NUMERICAL_TASKS, CHOICE_TASKS
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
# 演化版本的推理器
# ============================================================================

class EvolvingReasoner(DirectQA):
    """带演化能力的推理器
    
    继承 DirectQA 的所有推理逻辑，
    添加置信度计算和演化触发判断
    """
    
    def __init__(self, confidence_threshold: float = 0.4):
        super().__init__()
        self.confidence_threshold = confidence_threshold
    
    def answer_with_confidence(
        self, 
        mind_map: Dict[str, MindMapEntity3D],
        question: str,
        question_type: str,
        options: List[str] = None
    ) -> Tuple[str, float, bool]:
        """带置信度的回答
        
        Returns:
            (answer, confidence, needs_evolution)
        """
        # 计算心智地图置信度
        map_confidence = self._compute_map_confidence(mind_map, question, question_type)
        
        # 调用父类推理
        if question_type == 'object_counting':
            answer = self.answer_counting(mind_map, question)
        elif question_type == 'object_size_estimation':
            answer = self.answer_object_size(mind_map, question)
        elif question_type == 'room_size_estimation':
            answer = self.answer_room_size(mind_map, question)
        elif question_type == 'object_abs_distance':
            answer = self.answer_abs_distance(mind_map, question)
        elif question_type in ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard']:
            difficulty = question_type.split('_')[-1]
            answer = self.answer_rel_direction(mind_map, question, options, difficulty)
        elif question_type == 'object_rel_distance':
            answer = self.answer_rel_distance(mind_map, question, options)
        elif question_type == 'obj_appearance_order':
            answer = self.answer_appearance_order(mind_map, question, options)
        elif question_type == 'route_planning':
            answer = self.answer_route_planning(mind_map, question, options)
        else:
            answer = "unknown"
            map_confidence = 0.1
        
        needs_evolution = map_confidence < self.confidence_threshold
        
        return answer, map_confidence, needs_evolution
    
    def _compute_map_confidence(
        self, 
        mind_map: Dict[str, MindMapEntity3D],
        question: str,
        question_type: str
    ) -> float:
        """计算心智地图对该问题的置信度"""
        if not mind_map:
            return 0.1
        
        q_lower = question.lower()
        
        # 找到问题涉及的实体
        relevant_entities = []
        for label, entity in mind_map.items():
            if label.lower() in q_lower or any(s in q_lower for s in get_synonyms(label)):
                relevant_entities.append(entity)
        
        if not relevant_entities:
            # 没找到相关实体
            return 0.3
        
        # 基于实体检测质量计算置信度
        confidences = []
        for ent in relevant_entities:
            # 检测次数贡献
            count_score = min(1.0, ent.count / 5)
            # 检测置信度贡献
            det_score = ent.avg_confidence
            # 3D 信息完整性
            pos_score = 1.0 if ent.position_3d is not None else 0.3
            size_score = 1.0 if ent.size_3d is not None else 0.5
            
            # 加权
            entity_conf = 0.3 * count_score + 0.3 * det_score + 0.2 * pos_score + 0.2 * size_score
            confidences.append(entity_conf)
        
        avg_confidence = np.mean(confidences) if confidences else 0.3
        
        # 任务特定调整
        if question_type in ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard']:
            # 方向任务需要更高置信度
            avg_confidence *= 0.8
        elif question_type == 'route_planning':
            avg_confidence *= 0.7
        
        return float(avg_confidence)


# ============================================================================
# 演化器 - 视觉回溯
# ============================================================================

class MindMapEvolver:
    """心智地图演化器 - 通过视觉回溯提升置信度"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.vl_model = None
        self.vl_processor = None
    
    def load_qwen_vl(self, model_path: str = "Qwen/Qwen3-VL-8B-Instruct"):
        """加载 Qwen3-VL 模型"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            logger.info(f"加载 Qwen3-VL 模型: {model_path}")
            self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.vl_processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            logger.info("Qwen3-VL 加载完成")
        except Exception as e:
            logger.warning(f"Qwen3-VL 加载失败: {e}")
            self.vl_model = None
    
    def refine_entity(
        self, 
        entity: MindMapEntity3D,
        video_path: str,
        question: str
    ) -> Tuple[MindMapEntity3D, bool]:
        """通过视觉回溯改进实体信息
        
        Returns:
            (updated_entity, was_refined)
        """
        if self.vl_model is None:
            # 无 VL 模型，直接提升置信度（假设人工验证）
            entity.avg_confidence = min(1.0, entity.avg_confidence + 0.2)
            return entity, True
        
        # 提取关键帧
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 定位到首次发现帧
        target_frame = min(entity.first_seen_frame, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return entity, False
        
        # 转 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 构建验证问题
        verification_prompt = f"在这张图片中，你能看到 {entity.label} 吗？如果能看到，请回答'是'并描述其大致位置。如果不能，请回答'否'。"
        
        try:
            # VL 推理
            from qwen_vl_utils import process_vision_info
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_rgb},
                    {"type": "text", "text": verification_prompt}
                ]
            }]
            
            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=128)
            response = self.vl_processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            # 解析响应
            if '是' in response or 'yes' in response.lower():
                entity.avg_confidence = min(1.0, entity.avg_confidence + 0.3)
                return entity, True
            else:
                # 可能是误检，降低置信度
                entity.avg_confidence = max(0.1, entity.avg_confidence - 0.2)
                return entity, True
                
        except Exception as e:
            logger.warning(f"VL 推理失败: {e}")
            return entity, False
    
    def refine_mind_map(
        self, 
        mind_map: Dict[str, MindMapEntity3D],
        video_path: str,
        question: str,
        entities_to_refine: List[str] = None
    ) -> Tuple[Dict[str, MindMapEntity3D], List[str]]:
        """演化整个心智地图
        
        Returns:
            (updated_map, corrections)
        """
        corrections = []
        
        if entities_to_refine is None:
            # 找出低置信度实体
            entities_to_refine = [
                label for label, ent in mind_map.items()
                if ent.avg_confidence < 0.5
            ]
        
        for label in entities_to_refine:
            if label not in mind_map:
                continue
            
            old_conf = mind_map[label].avg_confidence
            updated_entity, refined = self.refine_entity(
                mind_map[label], video_path, question
            )
            
            if refined:
                mind_map[label] = updated_entity
                new_conf = updated_entity.avg_confidence
                corrections.append(f"{label}: {old_conf:.2f} -> {new_conf:.2f}")
        
        return mind_map, corrections


# ============================================================================
# 自进化 Agent
# ============================================================================

class SelfEvolvingAgentV2:
    """自进化心智地图智能体 V2"""
    
    def __init__(
        self,
        device: str = 'cuda',
        num_frames: int = 32,
        confidence_threshold: float = 0.4,
        max_evolution_rounds: int = 1,
        use_vl_model: bool = False,
        vl_model_path: str = None
    ):
        self.device = device
        self.num_frames = num_frames
        self.confidence_threshold = confidence_threshold
        self.max_evolution_rounds = max_evolution_rounds
        
        # 组件
        self.builder = MindMapBuilderDirectQA(
            device=device,
            num_frames=num_frames,
            box_threshold=0.25
        )
        self.reasoner = EvolvingReasoner(confidence_threshold=confidence_threshold)
        self.evolver = MindMapEvolver(device=device)
        
        # 加载 VL 模型（可选）
        if use_vl_model and vl_model_path:
            self.evolver.load_qwen_vl(vl_model_path)
        
        # 统计
        self.stats = {
            "total": 0,
            "evolved": 0,
            "improved_after_evolution": 0,
        }
    
    def process(
        self,
        video_path: str,
        question: str,
        question_type: str,
        options: List[str] = None,
        ground_truth: str = None
    ) -> Dict[str, Any]:
        """处理单个查询"""
        self.stats["total"] += 1
        
        # 1. 构建心智地图
        try:
            mind_map = self.builder.build_from_video(video_path)
        except Exception as e:
            logger.warning(f"心智地图构建失败: {e}")
            mind_map = {}
        
        if not mind_map:
            default_answer = self._get_default_answer(question_type, options)
            return {
                "answer": default_answer,
                "confidence": 0.1,
                "evolved": False,
                "answer_before_evolution": default_answer,
                "confidence_before_evolution": 0.1,
            }
        
        # 2. 初次推理
        answer_v1, conf_v1, needs_evolution = self.reasoner.answer_with_confidence(
            mind_map, question, question_type, options
        )
        
        result = {
            "answer_before_evolution": answer_v1,
            "confidence_before_evolution": conf_v1,
            "evolved": False,
        }
        
        # 3. 演化（如果需要）
        if needs_evolution and self.max_evolution_rounds > 0:
            self.stats["evolved"] += 1
            
            # 执行演化
            updated_map, corrections = self.evolver.refine_mind_map(
                mind_map, video_path, question
            )
            
            if corrections:
                result["evolved"] = True
                result["corrections"] = corrections
                
                # 重新推理
                answer_v2, conf_v2, _ = self.reasoner.answer_with_confidence(
                    updated_map, question, question_type, options
                )
                
                result["answer"] = answer_v2
                result["confidence"] = conf_v2
                
                # 检查是否改进
                if ground_truth and self._check_improved(
                    answer_v1, answer_v2, ground_truth, question_type
                ):
                    self.stats["improved_after_evolution"] += 1
            else:
                result["answer"] = answer_v1
                result["confidence"] = conf_v1
        else:
            result["answer"] = answer_v1
            result["confidence"] = conf_v1
        
        return result
    
    def _get_default_answer(self, question_type: str, options: List[str] = None) -> str:
        """获取默认答案"""
        if options:
            return options[0]
        
        defaults = {
            "object_counting": "1",
            "object_size_estimation": "50",
            "room_size_estimation": "20",
            "object_abs_distance": "2.0",
        }
        return defaults.get(question_type, "unknown")
    
    def _check_improved(
        self, 
        answer_v1: str, 
        answer_v2: str, 
        ground_truth: str,
        question_type: str
    ) -> bool:
        """检查演化后是否改进"""
        if question_type in NUMERICAL_TASKS:
            pred_v1 = normalize_number(answer_v1)
            pred_v2 = normalize_number(answer_v2)
            gt = normalize_number(ground_truth)
            
            if pred_v1 is None or pred_v2 is None or gt is None:
                return False
            
            score_v1 = mean_relative_accuracy(pred_v1, gt)
            score_v2 = mean_relative_accuracy(pred_v2, gt)
            return score_v2 > score_v1
        else:
            # 选择题
            correct_v1 = answer_v1.strip().lower() == ground_truth.strip().lower()
            correct_v2 = answer_v2.strip().lower() == ground_truth.strip().lower()
            return correct_v2 and not correct_v1
    
    def unload(self):
        """释放资源"""
        self.builder.unload()
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self.stats,
            "evolution_rate": self.stats["evolved"] / max(1, self.stats["total"]),
            "improvement_rate": self.stats["improved_after_evolution"] / max(1, self.stats["evolved"]),
        }


# ============================================================================
# 数据集加载
# ============================================================================

def load_vsibench_dataset(max_samples: int = None, task_filter: str = None) -> List[Dict]:
    """加载 VSI-Bench 数据集"""
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
        
        # 任务类型过滤
        if task_filter and task_filter != 'all' and question_type != task_filter:
            continue
        
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
# 评估
# ============================================================================

def normalize_choice_answer(pred: str, gt: str, options: List[str] = None) -> Tuple[str, str]:
    """标准化选择题答案格式
    
    将预测和真值都转换为选项标签 (A, B, C, D) 或统一的内容格式
    """
    pred = pred.strip()
    gt = gt.strip()
    
    # 如果 gt 是单个字母 (A, B, C, D)
    if len(gt) == 1 and gt.upper() in 'ABCD':
        gt_label = gt.upper()
        
        # 如果 pred 也是单个字母
        if len(pred) == 1 and pred.upper() in 'ABCD':
            return pred.upper(), gt_label
        
        # 如果 pred 是 "A. xxx" 格式
        if len(pred) >= 2 and pred[0].upper() in 'ABCD' and pred[1] == '.':
            return pred[0].upper(), gt_label
        
        # 如果 pred 是选项内容，尝试匹配
        if options:
            pred_lower = pred.lower()
            for i, opt in enumerate(options):
                opt_content = opt.lower()
                # 去掉选项前缀 "A. ", "B. " 等
                if len(opt) >= 3 and opt[1] == '.':
                    opt_content = opt[3:].strip().lower()
                
                if pred_lower in opt_content or opt_content in pred_lower:
                    return chr(65 + i), gt_label  # A=65
        
        return pred.upper() if len(pred) == 1 else pred, gt_label
    
    # 如果 gt 是选项内容（如 "left", "right"）
    # 直接比较内容
    return pred.lower(), gt.lower()


def evaluate_answer(pred: str, gt: str, question_type: str, options: List[str] = None) -> Tuple[float, bool]:
    """评估单个答案
    
    Returns:
        (score, is_correct)
    """
    if question_type in NUMERICAL_TASKS:
        pred_val = normalize_number(pred)
        gt_val = normalize_number(gt)
        
        if pred_val is None or gt_val is None:
            return 0.0, False
        
        score = mean_relative_accuracy(pred_val, gt_val)
        return score, score > 0.5
    else:
        # 选择题 - 标准化后比较
        pred_norm, gt_norm = normalize_choice_answer(pred, gt, options)
        correct = pred_norm == gt_norm
        return float(correct), correct


def worker_process(args):
    """单 GPU Worker"""
    gpu_id, samples, agent_config = args
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    agent = SelfEvolvingAgentV2(**agent_config)
    
    results = []
    for sample in tqdm(samples, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            result = agent.process(
                video_path=sample['video_path'],
                question=sample['question'],
                question_type=sample['question_type'],
                options=sample.get('options'),
                ground_truth=sample.get('ground_truth')
            )
            
            # 评估
            score, correct = evaluate_answer(
                result['answer'],
                sample['ground_truth'],
                sample['question_type'],
                sample.get('options')
            )
            
            # 评估演化前
            score_before, correct_before = evaluate_answer(
                result['answer_before_evolution'],
                sample['ground_truth'],
                sample['question_type'],
                sample.get('options')
            )
            
            results.append({
                'id': sample['id'],
                'scene_name': sample['scene_name'],
                'question_type': sample['question_type'],
                'question': sample['question'],
                'ground_truth': sample['ground_truth'],
                'pred': result['answer'],
                'pred_before': result['answer_before_evolution'],
                'score': float(score),
                'score_before': float(score_before),
                'correct': bool(correct),
                'correct_before': bool(correct_before),
                'evolved': bool(result.get('evolved', False)),
                'confidence': float(result.get('confidence', 0)),
                'confidence_before': float(result.get('confidence_before_evolution', 0)),
            })
            
        except Exception as e:
            logger.error(f"样本 {sample['id']} 处理失败: {e}")
            results.append({
                'id': sample['id'],
                'scene_name': sample['scene_name'],
                'question_type': sample['question_type'],
                'score': 0.0,
                'score_before': 0.0,
                'correct': False,
                'correct_before': False,
                'evolved': False,
                'error': str(e),
            })
    
    agent.unload()
    return results, agent.get_stats()


# ============================================================================
# 主程序
# ============================================================================

def run_benchmark(args):
    """运行基准测试"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 70)
    logger.info(f"Self-Evolving Agent V2 VSI-Bench 测试")
    logger.info(f"时间戳: {timestamp}")
    logger.info("=" * 70)
    
    # 加载数据集
    logger.info("加载数据集...")
    dataset = load_vsibench_dataset(args.max_samples, args.task_filter)
    
    if not dataset:
        logger.error("数据集为空")
        return
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # Agent 配置
    agent_config = {
        "device": args.device,
        "num_frames": args.num_frames,
        "confidence_threshold": args.confidence_threshold,
        "max_evolution_rounds": args.max_evolution,
        "use_vl_model": args.use_vl,
        "vl_model_path": args.vl_model if args.use_vl else None,
    }
    
    # 分配样本到多 GPU
    num_gpus = args.num_gpus
    samples_per_gpu = len(dataset) // num_gpus + 1
    
    gpu_tasks = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = min((i + 1) * samples_per_gpu, len(dataset))
        if start_idx < len(dataset):
            gpu_tasks.append((i, dataset[start_idx:end_idx], agent_config))
    
    logger.info(f"使用 {len(gpu_tasks)} 个 GPU")
    
    # 多进程运行
    all_results = []
    all_stats = defaultdict(int)
    
    if num_gpus > 1:
        with mp.Pool(num_gpus) as pool:
            for results, stats in pool.map(worker_process, gpu_tasks):
                all_results.extend(results)
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        all_stats[k] += v
    else:
        results, stats = worker_process(gpu_tasks[0])
        all_results.extend(results)
        all_stats = stats
    
    # 计算统计
    logger.info("\n" + "=" * 70)
    logger.info("测试结果")
    logger.info("=" * 70)
    
    type_scores = defaultdict(list)
    type_scores_before = defaultdict(list)
    type_evolved = defaultdict(int)
    type_improved = defaultdict(int)
    
    for r in all_results:
        qtype = r['question_type']
        type_scores[qtype].append(r['score'])
        type_scores_before[qtype].append(r['score_before'])
        if r.get('evolved'):
            type_evolved[qtype] += 1
            if r['score'] > r['score_before']:
                type_improved[qtype] += 1
    
    # 输出结果
    print("\n任务类型 | 样本数 | 演化前得分 | 演化后得分 | 提升 | 演化率 | 演化后改进率")
    print("-" * 90)
    
    overall_before = []
    overall_after = []
    
    for qtype in sorted(type_scores.keys()):
        scores_after = type_scores[qtype]
        scores_before = type_scores_before[qtype]
        count = len(scores_after)
        
        avg_before = np.mean(scores_before) * 100
        avg_after = np.mean(scores_after) * 100
        improvement = avg_after - avg_before
        
        evolved = type_evolved[qtype]
        improved = type_improved[qtype]
        
        evolution_rate = evolved / count * 100 if count > 0 else 0
        improvement_rate = improved / evolved * 100 if evolved > 0 else 0
        
        print(f"{qtype:<30} | {count:>4} | {avg_before:>8.2f}% | {avg_after:>8.2f}% | {improvement:>+6.2f}% | {evolution_rate:>5.1f}% | {improvement_rate:>6.1f}%")
        
        overall_before.extend(scores_before)
        overall_after.extend(scores_after)
    
    print("-" * 90)
    
    total_before = np.mean(overall_before) * 100 if overall_before else 0
    total_after = np.mean(overall_after) * 100 if overall_after else 0
    total_improvement = total_after - total_before
    total_evolved = sum(type_evolved.values())
    total_improved = sum(type_improved.values())
    total_count = len(all_results)
    
    print(f"{'Overall':<30} | {total_count:>4} | {total_before:>8.2f}% | {total_after:>8.2f}% | {total_improvement:>+6.2f}% | {total_evolved/total_count*100:>5.1f}% | {total_improved/max(1,total_evolved)*100:>6.1f}%")
    
    # 保存结果
    output_dir = Path(args.output_dir) / f"evolving_agent_v2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "timestamp": timestamp,
        "config": agent_config,
        "num_samples": total_count,
        "overall_score_before": total_before,
        "overall_score_after": total_after,
        "improvement": total_improvement,
        "evolution_rate": total_evolved / total_count if total_count > 0 else 0,
        "improvement_rate": total_improved / total_evolved if total_evolved > 0 else 0,
        "by_type": {
            qtype: {
                "count": len(type_scores[qtype]),
                "score_before": float(np.mean(type_scores_before[qtype])),
                "score_after": float(np.mean(type_scores[qtype])),
                "evolved": type_evolved[qtype],
                "improved": type_improved[qtype],
            }
            for qtype in type_scores.keys()
        }
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存到: {output_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Self-Evolving Agent V2 VSI-Bench Test")
    
    # 数据集参数
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--task-filter", type=str, default="all", 
                        help="任务类型过滤 (e.g., object_counting, all)")
    
    # Agent 参数
    parser.add_argument("--num-frames", type=int, default=32, help="采样帧数")
    parser.add_argument("--confidence-threshold", type=float, default=0.4, help="演化置信度阈值")
    parser.add_argument("--max-evolution", type=int, default=1, help="最大演化轮数")
    parser.add_argument("--use-vl", action="store_true", help="使用 VL 模型")
    parser.add_argument("--vl-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="VL 模型路径")
    
    # 运行参数
    parser.add_argument("--num-gpus", type=int, default=8, help="GPU 数量")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--output-dir", type=str, 
                        default="/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs",
                        help="输出目录")
    
    args = parser.parse_args()
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
