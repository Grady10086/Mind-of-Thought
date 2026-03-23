#!/usr/bin/env python3
"""
扩大Evolution覆盖范围 - 阶段2实验
目标：将Evolution应用到所有主要任务类型，覆盖率从3%提升到80%+
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
import traceback
import re

def worker_process_with_expanded_evolution(gpu_id: int, samples, output_file: str):
    """
    GPU worker - 应用扩大的Evolution策略
    """
    # 在子进程中添加路径
    import sys
    sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
    sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')
    
    import torch
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    print(f"GPU {gpu_id}: 初始化组件...")
    
    from core.mind_map_builder import MindMapBuilder
    from core.scale_calibrator import ScaleCalibrator
    from core.mind_map_evolver import MindMapEvolver
    
    builder = MindMapBuilder(device=device, num_frames=32, box_threshold=0.25)
    calibrator = ScaleCalibrator()
    evolver = MindMapEvolver(device=device)
    
    builder.load_models()
    print(f"GPU {gpu_id}: 组件初始化完成")
    
    results = []
    evolution_count = 0
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        try:
            video_path = sample['video_path']
            question = sample['question']
            question_type = sample['question_type']
            answer = sample['ground_truth']
            
            # 1. 构建Mind Map
            target_objects = []
            if 'counting' in question_type:
                match = re.search(r'How many (\w+)', question)
                if match:
                    target_objects = [match.group(1)]
            
            mind_map, total_frames, sampled_frames = builder.build_from_video(
                video_path, target_objects
            )
            
            # 2. 校准
            calibration = calibrator.calculate_scale_factor(mind_map)
            mind_map = calibrator.apply_calibration(mind_map, calibration)
            
            # 3. 🔥 扩大Evolution策略覆盖
            evolution_applied = False
            
            # 3.1 Counting任务 - 去重演化
            if question_type == 'object_counting' and target_objects:
                frame_indices = list(range(len(sampled_frames)))
                mind_map, actions = evolver.evolve_for_counting(
                    mind_map, target_objects[0], sampled_frames, frame_indices
                )
                if actions:
                    evolution_applied = True
            
            # 3.2 Appearance Order - 时序演化
            elif question_type == 'obj_appearance_order':
                mind_map, actions = evolver.refine_temporal_info(mind_map, sampled_frames)
                if actions:
                    evolution_applied = True
            
            # 3.3 Size Estimation - 尺度演化
            elif question_type in ['object_size_estimation', 'room_size_estimation']:
                # 应用更激进的尺度校准
                if calibration.confidence > 0.5:
                    mind_map, actions = evolver.refine_scale_measurements(
                        mind_map, calibration
                    )
                    if actions:
                        evolution_applied = True
            
            # 3.4 Distance任务 - 距离演化
            elif question_type in ['object_abs_distance', 'object_rel_distance']:
                # 修正距离测量
                mind_map, actions = evolver.refine_distance_measurements(
                    mind_map, calibration
                )
                if actions:
                    evolution_applied = True
            
            # 3.5 Direction任务 - 空间关系演化
            elif 'direction' in question_type:
                # 修正空间方向关系
                mind_map, actions = evolver.refine_spatial_relations(mind_map)
                if actions:
                    evolution_applied = True
            
            # 3.6 Route Planning - 拓扑演化
            elif question_type == 'route_planning':
                # 构建拓扑图并优化路径
                mind_map, actions = evolver.refine_topology(mind_map)
                if actions:
                    evolution_applied = True
            
            if evolution_applied:
                evolution_count += 1
            
            # 4. 格式化Mind Map
            mind_map_text = format_mind_map(mind_map, calibration)
            
            # 5. 生成训练样本
            evolution_tag = "WITH EVOLUTION" if evolution_applied else "INITIAL PERCEPTION"
            
            training_sample = {
                'conversations': [
                    {
                        'from': 'human',
                        'value': f"""You are a spatial intelligence assistant analyzing a video of an indoor scene.

=== DETECTED OBJECTS ({evolution_tag}) ===
{mind_map_text}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Please analyze the video frames and the detected objects to answer the question accurately.
"""
                    },
                    {'from': 'gpt', 'value': answer}
                ],
                'videos': [video_path],
                'question_type': question_type,
                'evolution_applied': evolution_applied
            }
            
            results.append(training_sample)
            
        except Exception as e:
            print(f"GPU {gpu_id}: Error processing {sample.get('scene_name', 'unknown')}: {e}")
            traceback.print_exc()
            continue
    
    # 保存结果
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    builder.unload()
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"GPU {gpu_id}: 完成 {len(results)} 个样本，Evolution应用于 {evolution_count} 个样本 ({evolution_count/len(results)*100:.1f}%)")


def format_mind_map(mind_map, calibration):
    """格式化Mind Map为文本"""
    lines = []
    
    # 校准信息
    if calibration.calibration_object:
        lines.append(f"Calibration: Using '{calibration.calibration_object}' (confidence: {calibration.confidence:.2f})")
        lines.append(f"Scale factor: {calibration.scale_factor:.3f}")
    
    lines.append("\nDetected Objects:")
    
    for label, entity in sorted(mind_map.items(), 
                                  key=lambda x: x[1].first_seen_frame):
        avg_pos = entity.avg_position
        avg_size = entity.avg_size
        
        lines.append(f"- {label} (count: {entity.count}, confidence: {entity.avg_confidence:.2f})")
        lines.append(f"  Position: ({avg_pos[0]:.2f}, {avg_pos[1]:.2f}, {avg_pos[2]:.2f}) meters")
        lines.append(f"  Size: {avg_size[0]:.2f}m × {avg_size[1]:.2f}m × {avg_size[2]:.2f}m")
        lines.append(f"  First seen: frame {entity.first_seen_frame}")
    
    return '\n'.join(lines)


def main():
    print("="*80)
    print("扩大Evolution覆盖范围 - 生成阶段2训练数据")
    print("="*80)
    
    # 加载原始数据
    input_file = "data/vsi590k_subset/train_balanced_1k_scannet_vsibench.json"
    print(f"\n加载数据: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 统计任务类型分布
    from collections import Counter
    task_dist = Counter(d['question_type'] for d in data)
    print("\n任务类型分布:")
    for task, count in sorted(task_dist.items()):
        print(f"  {task}: {count}")
    
    # 多GPU并行处理
    num_gpus = 8
    samples_per_gpu = len(data) // num_gpus
    
    processes = []
    output_files = []
    
    for gpu_id in range(num_gpus):
        start = gpu_id * samples_per_gpu
        end = start + samples_per_gpu if gpu_id < num_gpus - 1 else len(data)
        gpu_samples = data[start:end]
        
        output_file = f"outputs/mindmap_expanded_evo_gpu{gpu_id}.jsonl"
        output_files.append(output_file)
        
        p = mp.Process(
            target=worker_process_with_expanded_evolution,
            args=(gpu_id, gpu_samples, output_file)
        )
        p.start()
        processes.append(p)
    
    # 等待完成
    for p in processes:
        p.join()
    
    # 合并结果
    print("\n合并结果...")
    final_output = "outputs/mindmap_9908_expanded_evolution.jsonl"
    
    all_samples = []
    evolution_count = 0
    
    for output_file in output_files:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    all_samples.append(sample)
                    if sample.get('evolution_applied', False):
                        evolution_count += 1
    
    with open(final_output, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\n✅ 完成！")
    print(f"总样本数: {len(all_samples)}")
    print(f"Evolution应用数: {evolution_count} ({evolution_count/len(all_samples)*100:.1f}%)")
    print(f"输出文件: {final_output}")
    print("="*80)


if __name__ == '__main__':
    main()
