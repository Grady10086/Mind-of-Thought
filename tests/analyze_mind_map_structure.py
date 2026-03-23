#!/usr/bin/env python3
"""
心智地图结构分析 - 理解坐标系问题

分析内容：
1. 心智地图中物体的 3D 坐标分布
2. DA3 输出的相机位姿结构
3. 世界坐标系的特征
4. 为什么无法建立完整的 OCC 类地图
"""

import os
import sys
import json
import re
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from datetime import datetime

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 视频路径
VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
    '/home/tione/notebook/tianjungu/hf_cache/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/hf_cache/vsibench/scannet',
    '/home/tione/notebook/tianjungu/hf_cache/vsibench/scannetpp',
]

EXTENDED_VOCABULARY = [
    "chair", "table", "sofa", "couch", "stove", "tv", "television",
    "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator", "fridge",
    "sink", "toilet", "bathtub", "door", "window", "picture", "painting",
    "pillow", "cushion", "monitor", "backpack", "bag", "heater",
    "trash can", "trash bin", "mirror", "towel", "plant", "cup",
    "nightstand", "closet", "microwave", "printer", "washer", "dryer",
    "oven", "counter", "drawer", "curtain", "rug", "carpet", "clock",
    "fan", "air conditioner", "bookshelf", "armchair", "stool",
    "dishwasher", "telephone", "keyboard", "laptop", "whiteboard",
    "radiator", "fireplace", "vase", "bottle", "box", "basket",
]


def find_video_path(scene_name: str):
    for dir_path in VIDEO_DIRS:
        video_path = os.path.join(dir_path, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path
    return None


def analyze_single_video(video_path: str, num_frames: int = 16):
    """分析单个视频的心智地图结构"""
    from core.perception_da3_full import DA3FullEstimator
    from core.semantic_labeler import GroundingDINOLabeler
    
    # 加载模型
    da3 = DA3FullEstimator(
        model_name="/home/tione/notebook/tianjungu/hf_cache/DA3NESTED-GIANT-LARGE-1.1",
        device='cuda',
        use_ray_pose=True,
    )
    
    labeler = GroundingDINOLabeler(
        model_id="IDEA-Research/grounding-dino-base",
        device='cuda',
        box_threshold=0.25,
        text_threshold=0.25,
    )
    labeler.load_model()
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    sampled_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # DA3 推理
    prediction = da3.estimate_multiview(sampled_frames, ref_view_strategy="first")
    
    # 收集物体检测
    vocab = EXTENDED_VOCABULARY
    prompt = " . ".join(vocab) + " ."
    
    object_positions = defaultdict(list)
    proc_H, proc_W = prediction.depth_maps.shape[1:3]
    
    for i, frame_rgb in enumerate(sampled_frames):
        orig_H, orig_W = frame_rgb.shape[:2]
        detections = labeler.detect(frame_rgb, prompt)
        
        for det in detections:
            label = det.label.strip().lower()
            if label.startswith('##'):
                continue
            
            bbox = det.bbox_pixels
            scale_x = proc_W / orig_W
            scale_y = proc_H / orig_H
            
            bbox_scaled = (
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            )
            
            pos_3d = da3.compute_object_center_3d(prediction, i, bbox_scaled)
            if pos_3d is not None:
                object_positions[label].append({
                    'frame': i,
                    'position': pos_3d.tolist(),
                    'confidence': det.confidence,
                })
    
    # 构建心智地图
    mind_map = {}
    for label, dets in object_positions.items():
        if dets:
            positions = np.array([d['position'] for d in dets])
            median_pos = np.median(positions, axis=0)
            std_pos = np.std(positions, axis=0)
            
            mind_map[label] = {
                'position_3d': median_pos.tolist(),
                'position_std': std_pos.tolist(),
                'num_detections': len(dets),
                'detection_frames': [d['frame'] for d in dets],
                'all_positions': [d['position'] for d in dets],
            }
    
    # 分析 DA3 输出
    analysis = {
        'video_path': video_path,
        'num_frames': len(sampled_frames),
        'mind_map': mind_map,
        'da3_output': {
            'depth_map_shape': prediction.depth_maps.shape,
            'extrinsics_shape': prediction.extrinsics.shape,
            'intrinsics_shape': prediction.intrinsics.shape,
        },
        'camera_analysis': analyze_cameras(prediction),
        'coordinate_analysis': analyze_coordinates(mind_map, prediction),
    }
    
    return analysis


def analyze_cameras(prediction):
    """分析相机位姿"""
    extrinsics = prediction.extrinsics  # (N, 3, 4)
    intrinsics = prediction.intrinsics  # (N, 3, 3)
    
    N = extrinsics.shape[0]
    
    # 计算所有相机中心
    camera_centers = []
    camera_orientations = []
    
    for i in range(N):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        
        # 相机中心 = -R^T @ t
        center = -R.T @ t
        camera_centers.append(center)
        
        # 相机朝向 (Z 轴在世界坐标系中的方向)
        forward = R.T @ np.array([0, 0, 1])  # 相机 Z 轴
        down = R.T @ np.array([0, 1, 0])     # 相机 Y 轴 (向下)
        right = R.T @ np.array([1, 0, 0])    # 相机 X 轴 (向右)
        
        camera_orientations.append({
            'forward': forward.tolist(),
            'down': down.tolist(),
            'right': right.tolist(),
        })
    
    camera_centers = np.array(camera_centers)
    
    # 相机轨迹统计
    trajectory_length = 0
    for i in range(1, N):
        trajectory_length += np.linalg.norm(camera_centers[i] - camera_centers[i-1])
    
    # 检查 down 向量的一致性 (重力方向)
    down_vectors = np.array([o['down'] for o in camera_orientations])
    mean_down = np.mean(down_vectors, axis=0)
    mean_down = mean_down / np.linalg.norm(mean_down)
    
    down_angles = []
    for d in down_vectors:
        d_norm = d / np.linalg.norm(d)
        angle = np.arccos(np.clip(np.dot(d_norm, mean_down), -1, 1)) * 180 / np.pi
        down_angles.append(angle)
    
    # 焦距分析
    focal_lengths = []
    for i in range(N):
        K = intrinsics[i]
        fx, fy = K[0, 0], K[1, 1]
        focal_lengths.append((fx + fy) / 2)
    
    return {
        'num_cameras': N,
        'camera_centers': camera_centers.tolist(),
        'camera_center_range': {
            'x': [float(camera_centers[:, 0].min()), float(camera_centers[:, 0].max())],
            'y': [float(camera_centers[:, 1].min()), float(camera_centers[:, 1].max())],
            'z': [float(camera_centers[:, 2].min()), float(camera_centers[:, 2].max())],
        },
        'trajectory_length': float(trajectory_length),
        'mean_down_vector': mean_down.tolist(),
        'down_angle_consistency': {
            'max_deviation_deg': float(max(down_angles)),
            'mean_deviation_deg': float(np.mean(down_angles)),
        },
        'focal_length_range': [float(min(focal_lengths)), float(max(focal_lengths))],
        'sample_camera_orientations': camera_orientations[:3],  # 前3帧
    }


def analyze_coordinates(mind_map, prediction):
    """分析坐标系特征"""
    if not mind_map:
        return {'error': 'Empty mind map'}
    
    # 收集所有物体位置
    all_positions = []
    for label, data in mind_map.items():
        all_positions.append(data['position_3d'])
    
    positions = np.array(all_positions)
    
    # 坐标范围
    coord_range = {
        'x': [float(positions[:, 0].min()), float(positions[:, 0].max())],
        'y': [float(positions[:, 1].min()), float(positions[:, 1].max())],
        'z': [float(positions[:, 2].min()), float(positions[:, 2].max())],
    }
    
    # 检查 Y 轴是否为重力方向 (如果是，物体应该在相似的 Y 高度)
    y_values = positions[:, 1]
    y_std = float(np.std(y_values))
    
    # 检查 XZ 平面是否为地面平面
    xz_span = np.sqrt((positions[:, 0].max() - positions[:, 0].min())**2 + 
                       (positions[:, 2].max() - positions[:, 2].min())**2)
    
    # 物体分布分析
    center = np.mean(positions, axis=0)
    distances_from_center = np.linalg.norm(positions - center, axis=1)
    
    return {
        'num_objects': len(mind_map),
        'object_labels': list(mind_map.keys()),
        'coordinate_range': coord_range,
        'scene_size': {
            'x_span': float(positions[:, 0].max() - positions[:, 0].min()),
            'y_span': float(positions[:, 1].max() - positions[:, 1].min()),
            'z_span': float(positions[:, 2].max() - positions[:, 2].min()),
        },
        'y_axis_analysis': {
            'y_std': y_std,
            'is_likely_gravity': y_std < 1.0,  # 如果 Y 变化小于 1 米，可能是重力方向
        },
        'xz_plane_span': float(xz_span),
        'scene_center': center.tolist(),
        'mean_distance_from_center': float(np.mean(distances_from_center)),
    }


def main():
    from datasets import load_dataset
    
    # 加载数据集获取视频
    ds = load_dataset('nyu-visionx/VSI-Bench', split='test',
                      cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench')
    
    direction_types = ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard']
    
    # 选择几个代表性视频
    selected_scenes = []
    for item in ds:
        if item['question_type'] in direction_types:
            video_path = find_video_path(item['scene_name'])
            if video_path and item['scene_name'] not in [s[0] for s in selected_scenes]:
                selected_scenes.append((item['scene_name'], video_path, item['question']))
                if len(selected_scenes) >= 3:  # 分析 3 个视频
                    break
    
    print('='*80)
    print('心智地图结构分析')
    print('='*80)
    
    all_analyses = []
    
    for scene_name, video_path, question in selected_scenes:
        print(f'\n分析视频: {scene_name}')
        print(f'问题: {question[:100]}...')
        
        try:
            analysis = analyze_single_video(video_path)
            all_analyses.append(analysis)
            
            # 打印结果
            print(f'\n--- 心智地图 ---')
            for label, data in analysis['mind_map'].items():
                pos = data['position_3d']
                std = data['position_std']
                print(f'  {label}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), '
                      f'std=({std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}), '
                      f'detections={data["num_detections"]}')
            
            print(f'\n--- 相机分析 ---')
            cam = analysis['camera_analysis']
            print(f'  相机数: {cam["num_cameras"]}')
            print(f'  轨迹长度: {cam["trajectory_length"]:.2f}')
            print(f'  平均 down 向量: {cam["mean_down_vector"]}')
            print(f'  down 角度最大偏差: {cam["down_angle_consistency"]["max_deviation_deg"]:.1f}°')
            
            print(f'\n--- 坐标系分析 ---')
            coord = analysis['coordinate_analysis']
            print(f'  物体数: {coord["num_objects"]}')
            print(f'  场景尺寸: X={coord["scene_size"]["x_span"]:.2f}, '
                  f'Y={coord["scene_size"]["y_span"]:.2f}, Z={coord["scene_size"]["z_span"]:.2f}')
            print(f'  Y 轴标准差: {coord["y_axis_analysis"]["y_std"]:.2f}')
            print(f'  Y 轴可能是重力方向: {coord["y_axis_analysis"]["is_likely_gravity"]}')
            
        except Exception as e:
            print(f'  错误: {e}')
            import traceback
            traceback.print_exc()
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'outputs/mind_map_analysis_{timestamp}.json'
    os.makedirs('outputs', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_analyses, f, indent=2, default=str)
    print(f'\n结果已保存到: {output_file}')
    
    # 总结
    print('\n' + '='*80)
    print('总结')
    print('='*80)
    print("""
关键发现：

1. DA3 坐标系特性：
   - DA3 输出的是相对坐标系，以第一帧为参考
   - Y 轴大致对应重力方向（相机 down 向量）
   - 但不是绝对的重力对齐

2. 心智地图结构：
   - 每个物体有 3D 位置 (X, Y, Z)
   - 位置来自多帧检测的中位数
   - 标准差反映检测一致性

3. 方向判断问题：
   - 需要知道"观察者面向"的方向
   - 需要知道"上"的方向来计算"左右"
   - DA3 的 Y 轴近似重力，但有偏差

4. 改进方向：
   - 使用 DA3 的相机 down 向量估计重力方向
   - 在 XZ 平面（地面）上进行方向判断
   - 考虑多视图融合提高稳定性
""")


if __name__ == '__main__':
    main()
