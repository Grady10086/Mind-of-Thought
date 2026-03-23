#!/usr/bin/env python3
"""
运行V7感知流程,导出Mind Map训练数据 (4391条平衡采样)
"""
import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')

from tests.test_v7_with_finetuned_vl import *
import multiprocessing as mp

# 加载平衡采样的数据 (scannet only, 9908条)
def load_training_data():
    import json
    data_file = "data/vsi590k_subset/train_balanced_1k_scannet_vsibench.json"
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data  # 全部9908条

# 创建一个简化的worker,不运行VL推理
def worker_export_mindmap(gpu_id, samples, output_file):
    import os
    import gc
    import torch
    from tqdm import tqdm
    import traceback
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # 只创建感知组件
        print(f"[GPU {gpu_id}] 开始初始化组件...")
        builder = MindMapBuilder(device='cuda', num_frames=32, box_threshold=0.25)
        print(f"[GPU {gpu_id}] ✓ MindMapBuilder 初始化完成")
        
        calibrator = ScaleCalibrator()
        print(f"[GPU {gpu_id}] ✓ ScaleCalibrator 初始化完成")
        
        evolver = MindMapEvolver(device='cuda')
        print(f"[GPU {gpu_id}] ✓ MindMapEvolver 初始化完成")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] ❌ 组件初始化失败: {e}")
        traceback.print_exc()
        # 创建空文件表示失败
        with open(output_file, 'w') as f:
            f.write('')
        return
    
    results = []
    errors = 0
    
    for sample in tqdm(samples, desc=f"GPU {gpu_id}"):
        try:
            video_path = sample['video_path']
            question = sample['question']
            question_type = sample['question_type']
            
            # 1. 构建Mind Map
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
            
            # 3. 演化
            if question_type == 'object_counting' and target_objects:
                frame_indices = list(range(len(sampled_frames)))
                mind_map, actions = evolver.evolve_for_counting(
                    mind_map, target_objects[0], sampled_frames, frame_indices
                )
            
            # 4. 格式化Mind Map为文本
            mind_map_text = format_mind_map_for_prompt(mind_map)
            
            # 5. 生成训练样本
            training_sample = generate_training_prompt(
                question=question,
                question_type=question_type,
                mind_map_text=mind_map_text,
                answer=sample['ground_truth']
            )
            
            results.append(training_sample)
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error on sample {sample.get('scene_name', '?')}: {e}")
            if errors < 3:  # 只打印前3个错误的详细信息
                import traceback
                traceback.print_exc()
            errors += 1
            continue
        
        # 清理显存
        if len(results) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 保存结果
    import json
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"[GPU {gpu_id}] ✅ 完成 {len(results)} 个样本 (失败 {errors} 个, 成功率 {len(results)/(len(results)+errors)*100:.1f}%)")

def format_mind_map_for_prompt(mind_map):
    """格式化Mind Map为V7 prompt格式"""
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

def generate_training_prompt(question, question_type, mind_map_text, answer):
    """生成V7格式的训练prompt"""
    prompt = f"""You are a spatial intelligence assistant analyzing a video of an indoor scene.

=== DETECTED OBJECTS FROM PERCEPTION SYSTEM ===
{mind_map_text}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Please analyze the video frames and the detected objects to answer the question.
"""
    
    if 'Options:' in question or 'A.' in question:
        prompt += "Answer with the option's letter from the given choices directly.\n"
    
    return {
        'conversations': [
            {'from': 'human', 'value': prompt},
            {'from': 'gpt', 'value': answer}
        ],
        'question_type': question_type
    }

# 主流程
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    # 加载数据
    data = load_training_data()
    print(f"=" * 80)
    print(f"加载 {len(data)} 条训练数据")
    print(f"=" * 80)
    print()
    
    # 8卡并行
    num_gpus = 8
    chunk_size = (len(data) + num_gpus - 1) // num_gpus
    
    # 清理之前的输出
    import subprocess
    subprocess.run("rm -f outputs/mindmap_balanced_gpu*.jsonl", shell=True)
    
    processes = []
    output_files = []
    
    print(f"🚀 启动 {num_gpus} 个GPU并行处理")
    print(f"   每个GPU约 {chunk_size} 个样本")
    print()
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(data))
        subset = data[start_idx:end_idx]
        
        output_file = f"outputs/mindmap_balanced_gpu{gpu_id}.jsonl"
        output_files.append(output_file)
        
        # 添加延迟,让GPU顺序初始化,避免竞争
        import time
        time.sleep(gpu_id * 1.0)  # 每个GPU延迟1秒
        
        p = mp.Process(target=worker_export_mindmap, args=(gpu_id, subset, output_file))
        p.start()
        processes.append(p)
        
        print(f"  GPU {gpu_id}: {len(subset):4d} 样本 ({start_idx:4d}-{end_idx:4d})")
    
    print()
    print("⏳ 等待所有GPU完成...")
    print()
    
    # 等待完成
    for p in processes:
        p.join()
    
    # 合并结果并验证
    print()
    print("=" * 80)
    print("📦 合并结果...")
    print("=" * 80)
    
    total_samples = 0
    for gpu_id, output_file in enumerate(output_files):
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                count = sum(1 for _ in f)
            print(f"  GPU {gpu_id}: {count:4d} 样本")
            total_samples += count
        else:
            print(f"  GPU {gpu_id}: ❌ 文件不存在")
    
    print()
    print(f"  总计: {total_samples}/{len(data)} 样本 ({total_samples/len(data)*100:.1f}%)")
    
    # 合并
    subprocess.run("cat outputs/mindmap_balanced_gpu*.jsonl > data/vsi590k_subset/train_balanced_4391_real_mindmap.jsonl", shell=True)
    
    print()
    print("=" * 80)
    print("✅ 完成!")
    print("=" * 80)
    print()
    print(f"输出文件: data/vsi590k_subset/train_balanced_4391_real_mindmap.jsonl")
    print(f"样本数: {total_samples}")
    print(f"成功率: {total_samples/len(data)*100:.1f}%")
