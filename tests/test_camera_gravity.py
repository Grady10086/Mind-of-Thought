#!/usr/bin/env python3
"""
验证假设：相机是水平拍摄的

如果相机水平拍摄，那么：
1. 相机的 down 向量（Y轴）应该指向重力方向
2. DA3 extrinsics 中的 R[1, :] 就是相机 down 在世界坐标系中的表示
3. 如果所有帧的 camera_down 都指向同一个方向，那就是重力方向

检验方法：
- 对每个场景，提取所有帧的 camera_down 向量
- 检查这些向量是否一致（应该都指向重力方向）
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_camera_gravity(num_videos=5):
    """分析几个视频的相机重力方向"""
    import cv2
    from core.perception_da3_full import DA3FullEstimator
    
    VIDEO_DIRS = [
        '/home/tione/notebook/tianjungu/hf_cache/vsibench/arkitscenes',
        '/home/tione/notebook/tianjungu/hf_cache/vsibench/scannet',
    ]
    
    # 找几个视频
    videos = []
    for vdir in VIDEO_DIRS:
        if os.path.exists(vdir):
            for f in os.listdir(vdir)[:num_videos]:
                if f.endswith('.mp4'):
                    videos.append(os.path.join(vdir, f))
    videos = videos[:num_videos]
    
    print(f"分析 {len(videos)} 个视频的相机重力方向...")
    
    # 加载 DA3
    da3 = DA3FullEstimator(
        model_name="da3nested-giant-large",
        device='cuda',
        use_ray_pose=True,
    )
    
    results = []
    
    for video_path in videos:
        print(f"\n{'='*60}")
        print(f"Video: {os.path.basename(video_path)}")
        print('='*60)
        
        # 读取视频帧
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if len(frames) < 2:
            continue
        
        # DA3 推理
        prediction = da3.estimate_multiview(frames)
        
        # 提取每帧的 camera_down 向量
        camera_downs = []
        for i, ext in enumerate(prediction.extrinsics):
            R = ext[:3, :3]
            # R[1, :] 是相机 Y 轴（down）在世界坐标系中的方向
            camera_down = R[1, :]
            camera_down = camera_down / np.linalg.norm(camera_down)
            camera_downs.append(camera_down)
        
        camera_downs = np.array(camera_downs)
        
        # 计算 camera_down 的一致性
        mean_down = np.mean(camera_downs, axis=0)
        mean_down = mean_down / np.linalg.norm(mean_down)
        
        # 每帧与平均值的夹角
        angles = []
        for cd in camera_downs:
            cos_angle = np.clip(np.dot(cd, mean_down), -1, 1)
            angle_deg = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle_deg)
        
        print(f"\n相机 down 向量分析:")
        print(f"  平均 down 向量: [{mean_down[0]:.3f}, {mean_down[1]:.3f}, {mean_down[2]:.3f}]")
        print(f"  各帧与平均值的夹角: {np.mean(angles):.1f}° ± {np.std(angles):.1f}°")
        print(f"  最大偏差: {np.max(angles):.1f}°")
        
        # 检查是否近似某个标准轴
        axes = {
            '+Y': np.array([0, 1, 0]),
            '-Y': np.array([0, -1, 0]),
            '+Z': np.array([0, 0, 1]),
            '-Z': np.array([0, 0, -1]),
            '+X': np.array([1, 0, 0]),
            '-X': np.array([-1, 0, 0]),
        }
        
        print(f"\n  与标准轴的夹角:")
        for name, axis in axes.items():
            cos_angle = np.clip(np.dot(mean_down, axis), -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            print(f"    {name}: {angle:.1f}°")
        
        results.append({
            'video': os.path.basename(video_path),
            'mean_down': mean_down.tolist(),
            'consistency_std': float(np.std(angles)),
            'max_deviation': float(np.max(angles)),
        })
    
    # 总结
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    
    all_downs = np.array([r['mean_down'] for r in results])
    if len(all_downs) > 1:
        # 检查不同视频之间的 gravity 方向是否一致
        global_mean = np.mean(all_downs, axis=0)
        global_mean = global_mean / np.linalg.norm(global_mean)
        
        cross_video_angles = []
        for d in all_downs:
            cos_angle = np.clip(np.dot(d, global_mean), -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            cross_video_angles.append(angle)
        
        print(f"\n不同视频间的重力方向一致性:")
        print(f"  全局平均 down: [{global_mean[0]:.3f}, {global_mean[1]:.3f}, {global_mean[2]:.3f}]")
        print(f"  各视频与全局平均的夹角: {cross_video_angles}")
        print(f"  最大偏差: {np.max(cross_video_angles):.1f}°")
    
    da3.unload()
    return results


if __name__ == '__main__':
    analyze_camera_gravity(num_videos=5)
