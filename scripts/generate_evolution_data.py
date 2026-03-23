#!/usr/bin/env python3
"""
从已生成的Mind Map数据中添加Evolution步骤
用于实验2: 带Evolution的训练数据
"""
import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')

import json
import os
import gc
import torch
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from tests.test_v7_with_finetuned_vl import MindMapEvolver, ScaleCalibrator, MindMapBuilder
import traceback

def load_original_data():
    """加载原始VSIBench格式数据"""
    data_file = "data/vsi590k_subset/train_balanced_1k_scannet_vsibench.json"
    with open(data_file, 'r') as f:
        return json.load(f)

def worker_add_evolution(gpu_id, samples, output_file):
    """为每个样本添加Evolution步骤"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        print(f"[GPU {gpu_id}] 初始化Evolution组件...")
        builder = MindMapBuilder(device='cuda', num_frames=32, box_threshold=0.25)
        calibrator = ScaleCalibrator()
        evolver = MindMapEvolver(device='cuda')
        print(f"[GPU {gpu_id}] ✓ 组件初始化完成")
    except Exception as e:
        print(f"[GPU {gpu_id}] ❌ 初始化失败: {e}")
        traceback.print_exc()
        return
    
    results = []
    errors = 0
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        try:
            video_path = sample['video_path']
            question = sample['question']
            question_type = sample['question_type']
            answer = sample['ground_truth']
            
            # 1. 构建Mind Map (与实验1相同)
            target_objects = []
            if 'counting' in question_type:
                import re
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map, total_frames, sampled_frames = builder.build_from_video(video_path, target_objects)
            
            # 2. 校准
            calibration = calibrator.calculate_scale_factor(mind_map)
            mind_map = calibrator.apply_calibration(mind_map, calibration)
            
            # 3. ⭐ Evolution步骤 (实验2特有)
            if question_type == 'object_counting' and target_objects:
                frame_indices = list(range(len(sampled_frames)))
                mind_map, actions = evolver.evolve_for_counting(
                    mind_map, target_objects[0], sampled_frames, frame_indices
                )
            elif question_type == 'appearance_order':
                # 对appearance_order任务也应用evolution
                mind_map, _ = evolver.refine_temporal_info(mind_map, sampled_frames)
            
            # 4. 格式化Mind Map
            mind_map_text = format_mind_map(mind_map)
            
            # 5. 生成训练样本
            training_sample = {
                'conversations': [
                    {
                        'from': 'human',
                        'value': f"""You are a spatial intelligence assistant analyzing a video of an indoor scene.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM (WITH EVOLUTION) ===
{mind_map_text}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Please analyze the video frames and the detected objects to answer the question.
Answer with the option's letter from the given choices directly.
"""
                    },
                    {'from': 'gpt', 'value': answer}
                ],
                'question_type': question_type
            }
            
            results.append(training_sample)
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {e}")
            if errors < 3:
                traceback.print_exc()
            errors += 1
            continue
        
        if len(results) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 保存结果
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"[GPU {gpu_id}] ✅ 完成 {len(results)} 个样本 (失败 {errors} 个)")

def format_mind_map(mind_map):
    """格式化Mind Map为文本"""
    import numpy as np
    lines = []
    for label, entity in mind_map.items():
        if entity.position_3d is None:
            continue
        
        pos = entity.position_3d
        size = entity.size_3d if entity.size_3d is not None else 0.0
        count = len(entity.detections)
        
        line = f"{label}: position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        
        if isinstance(size, (list, tuple, np.ndarray)):
            line += f", size ({size[0]:.2f}m × {size[1]:.2f}m × {size[2]:.2f}m)"
        elif size > 0:
            line += f", size ({size:.2f}m)"
        
        if count > 1:
            line += f", count: {count}"
        
        lines.append(line)
    
    return "\n".join(lines) if lines else "No objects detected."

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print("=" * 80)
    print("生成带Evolution的Mind Map训练数据")
    print("=" * 80)
    print()
    
    # 加载原始数据
    data = load_original_data()
    print(f"加载 {len(data)} 条原始数据")
    print()
    
    # 8卡并行处理
    num_gpus = 8
    chunk_size = (len(data) + num_gpus - 1) // num_gpus
    
    processes = []
    output_files = []
    
    print(f"🚀 启动 {num_gpus} 个GPU并行处理")
    print()
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(data))
        subset = data[start_idx:end_idx]
        
        output_file = f"outputs/mindmap_evolution_gpu{gpu_id}.jsonl"
        output_files.append(output_file)
        
        import time
        time.sleep(gpu_id * 1.0)
        
        p = mp.Process(target=worker_add_evolution, args=(gpu_id, subset, output_file))
        p.start()
        processes.append(p)
        
        print(f"  GPU {gpu_id}: {len(subset)} 样本")
    
    print()
    print("⏳ 等待所有GPU完成...")
    print()
    
    for p in processes:
        p.join()
    
    # 合并输出
    print("=" * 80)
    print("合并输出文件...")
    print("=" * 80)
    
    output_path = Path("outputs/mindmap_9908_with_evolution.jsonl")
    with open(output_path, 'w') as out_f:
        for gpu_file in output_files:
            if Path(gpu_file).exists():
                with open(gpu_file, 'r') as in_f:
                    out_f.write(in_f.read())
    
    total_samples = sum(1 for _ in open(output_path))
    print(f"✅ 完成! 总计 {total_samples} 条训练数据")
    print(f"   保存在: {output_path}")
    print()
    print("=" * 80)
