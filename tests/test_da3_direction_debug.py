#!/usr/bin/env python3
"""
深入分析 DA3 方向计算问题

测试一个简单的案例，验证:
1. 物体检测是否正确
2. 3D 坐标是否合理
3. 方向计算是否正确
"""
import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DA3_PATH = PROJECT_ROOT.parent / "Depth-Anything-3"
if str(DA3_PATH / "src") not in sys.path:
    sys.path.insert(0, str(DA3_PATH / "src"))


def compute_direction_in_viewer_frame(
    target_pos: np.ndarray,
    standing_pos: np.ndarray,
    facing_pos: np.ndarray,
) -> tuple:
    """
    在观察者坐标系中计算目标方向（修复版）
    
    使用叉积和点积来判断方向:
    - 点积 (forward · target_rel) > 0: 前方; < 0: 后方
    - 叉积 (forward × target_rel) > 0: 左边; < 0: 右边
    """
    forward = facing_pos[:2] - standing_pos[:2]
    forward_norm = np.linalg.norm(forward)
    
    if forward_norm < 0.01:
        return "same-position", 0, 0
    
    forward = forward / forward_norm
    target_rel = target_pos[:2] - standing_pos[:2]
    target_norm = np.linalg.norm(target_rel)
    
    if target_norm < 0.01:
        return "same-position", 0, 0
    
    # 点积判断前后
    dot_product = np.dot(forward, target_rel)
    
    # 叉积判断左右 (> 0 是左边, < 0 是右边)
    cross_product = forward[0] * target_rel[1] - forward[1] * target_rel[0]
    
    threshold = target_norm * 0.1
    directions = []
    
    if dot_product > threshold:
        directions.append("front")
    elif dot_product < -threshold:
        directions.append("back")
    
    if cross_product > threshold:
        directions.append("left")
    elif cross_product < -threshold:
        directions.append("right")
    
    if len(directions) == 0:
        result = "same-position"
    elif len(directions) == 1:
        result = directions[0]
    else:
        result = "-".join(directions)
    
    return result, dot_product, cross_product


def main():
    from datasets import load_dataset
    from core.perception_da3_full import MindMapBuilder3D
    
    # 加载一个方向任务样本
    print("加载 VSI-Bench 数据集...")
    ds = load_dataset(
        'nyu-visionx/VSI-Bench',
        split='test',
        cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench'
    )
    
    # 找一个简单的方向任务
    VIDEO_DIRS = [
        '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
        '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
        '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
        '/home/tione/notebook/tianjungu/hf_cache/vsibench/arkitscenes',
        '/home/tione/notebook/tianjungu/hf_cache/vsibench/scannet',
        '/home/tione/notebook/tianjungu/hf_cache/vsibench/scannetpp',
    ]
    
    def find_video_path(scene_name):
        for dir_path in VIDEO_DIRS:
            video_path = os.path.join(dir_path, f"{scene_name}.mp4")
            if os.path.exists(video_path):
                return video_path
        return None
    
    # 收集方向任务
    direction_samples = []
    for item in ds:
        if 'object_rel_direction' in item['question_type']:
            scene_name = item['scene_name']
            video_path = find_video_path(scene_name)
            if video_path:
                direction_samples.append({
                    'scene_name': scene_name,
                    'video_path': video_path,
                    'question': item['question'],
                    'question_type': item['question_type'],
                    'options': item.get('options', []),
                    'ground_truth': item['ground_truth'],
                })
    
    print(f"找到 {len(direction_samples)} 个方向任务样本")
    
    # 取前几个样本深入分析
    samples_to_analyze = direction_samples[:5]
    
    # 初始化 builder
    print("\n初始化 MindMapBuilder3D...")
    builder = MindMapBuilder3D(
        device='cuda',
        num_frames=16,
        box_threshold=0.25,
        model_name="da3nested-giant-large",
        use_ray_pose=True,
    )
    
    EXTENDED_VOCABULARY = [
        "chair", "table", "sofa", "couch", "stove", "tv", "television",
        "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator", "fridge",
        "sink", "toilet", "bathtub", "door", "window", "picture", "painting",
        "pillow", "cushion", "monitor", "backpack", "bag", "heater",
        "trash can", "trash bin", "mirror", "towel", "plant", "cup",
    ]
    
    import re
    
    for idx, sample in enumerate(samples_to_analyze):
        print(f"\n{'='*80}")
        print(f"样本 {idx+1}: {sample['scene_name']}")
        print(f"问题: {sample['question']}")
        print(f"答案: {sample['ground_truth']}")
        print(f"选项: {sample['options']}")
        print(f"{'='*80}")
        
        # 解析问题
        q_lower = sample['question'].lower()
        pattern1 = r'standing by (?:the )?(\w+).*?facing (?:the )?(\w+).*?is (?:the )?(\w+)'
        match = re.search(pattern1, q_lower)
        
        if match:
            standing_by = match.group(1)
            facing = match.group(2)
            target = match.group(3)
            print(f"\n解析结果: standing={standing_by}, facing={facing}, target={target}")
        else:
            print(f"\n无法解析问题")
            continue
        
        # 构建心智地图
        mind_map_3d, da3_prediction, sampled_frames = builder.build_from_video(
            sample['video_path'],
            target_objects=None,
            extended_vocabulary=EXTENDED_VOCABULARY,
        )
        
        print(f"\n检测到的物体: {list(mind_map_3d.keys())}")
        
        # 查找物体
        def find_entity(obj_name):
            for label, entity in mind_map_3d.items():
                if obj_name in label.lower() or label.lower() in obj_name:
                    return label, entity
            return None, None
        
        standing_label, standing_entity = find_entity(standing_by)
        facing_label, facing_entity = find_entity(facing)
        target_label, target_entity = find_entity(target)
        
        print(f"\n物体查找结果:")
        print(f"  standing: {standing_label} -> {standing_entity is not None}")
        print(f"  facing: {facing_label} -> {facing_entity is not None}")
        print(f"  target: {target_label} -> {target_entity is not None}")
        
        if standing_entity and facing_entity and target_entity:
            standing_pos = standing_entity['position_3d']
            facing_pos = facing_entity['position_3d']
            target_pos = target_entity['position_3d']
            
            print(f"\n3D 坐标:")
            print(f"  {standing_label}: {standing_pos.round(3)}")
            print(f"  {facing_label}: {facing_pos.round(3)}")
            print(f"  {target_label}: {target_pos.round(3)}")
            
            # 计算方向
            direction, proj_fwd, proj_right = compute_direction_in_viewer_frame(
                target_pos, standing_pos, facing_pos
            )
            
            print(f"\n方向计算:")
            print(f"  前向投影: {proj_fwd:.3f}")
            print(f"  右向投影: {proj_right:.3f}")
            print(f"  预测方向: {direction}")
            
            # 分析正确答案
            gt = sample['ground_truth'].strip().upper()[0]
            gt_idx = ord(gt) - ord('A')
            gt_direction = sample['options'][gt_idx] if gt_idx < len(sample['options']) else "Unknown"
            print(f"  正确答案: {gt} ({gt_direction})")
            
            # 可视化
            print(f"\n坐标系可视化 (XY 平面):")
            print(f"  Standing: ({standing_pos[0]:.2f}, {standing_pos[1]:.2f})")
            print(f"  Facing: ({facing_pos[0]:.2f}, {facing_pos[1]:.2f})")
            print(f"  Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
            
            forward = facing_pos[:2] - standing_pos[:2]
            forward = forward / np.linalg.norm(forward)
            print(f"  Forward vector: ({forward[0]:.3f}, {forward[1]:.3f})")
        
        print("\n" + "-"*80)
    
    builder.unload()


if __name__ == '__main__':
    main()
