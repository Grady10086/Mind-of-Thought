#!/usr/bin/env python3
"""
生成一个完整场景的心智地图示例

展示心智地图的全部内容，包括：
- 每个实体的详细信息
- 每个实体在各帧的检测结果
- 元信息（房间大小、物体3D位置等）
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_meta_info(scene_name: str) -> dict:
    """加载场景的 meta 信息"""
    meta_paths = [
        '/home/tione/notebook/tianjungu/projects/thinking-in-space/data/meta_info/arkitscenes_meta_info_val.json',
        '/home/tione/notebook/tianjungu/projects/thinking-in-space/data/meta_info/scannet_meta_info_val.json',
        '/home/tione/notebook/tianjungu/projects/thinking-in-space/data/meta_info/scannetpp_meta_info_val.json',
    ]
    
    for path in meta_paths:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                if scene_name in data:
                    return data[scene_name]
    return {}


def build_full_mindmap(video_path: str, scene_name: str) -> dict:
    """构建完整的心智地图"""
    from core.semantic_labeler import GroundingDINOLabeler
    
    # 加载模型
    print("Loading GroundingDINO...")
    labeler = GroundingDINOLabeler(
        model_id="IDEA-Research/grounding-dino-base",
        device="cuda",
        box_threshold=0.25,
        text_threshold=0.25,
    )
    labeler.load_model()
    
    # 提取帧
    print(f"Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Total frames: {len(frames)}")
    
    # 均匀采样 16 帧
    num_sample = min(16, len(frames))
    frame_indices = np.linspace(0, len(frames) - 1, num_sample).astype(int)
    
    # 检测词汇
    vocab = [
        "chair", "table", "sofa", "couch", "stove", "tv", "television",
        "bed", "cabinet", "shelf", "desk", "lamp", "refrigerator",
        "sink", "toilet", "bathtub", "door", "window", "picture",
        "pillow", "cushion", "monitor", "backpack", "bag",
        "trash can", "trash bin", "mirror", "towel", "plant",
        "nightstand", "closet", "microwave", "printer", "washer",
    ]
    text_prompt = " . ".join(vocab) + " ."
    
    # 收集所有检测
    print("Detecting objects in sampled frames...")
    all_detections = defaultdict(list)
    frame_detections = {}
    
    for i, idx in enumerate(frame_indices):
        frame = frames[idx]
        h, w = frame.shape[:2]
        
        result = labeler.detect_and_label_frame(frame, text_prompt)
        
        frame_dets = []
        for det in result.get('detections', []):
            label = det.label.lower().strip()
            det_info = {
                'label': label,
                'confidence': float(det.confidence),
                'bbox_pixels': det.bbox_pixels,
                'bbox_normalized': [
                    det.bbox_pixels[0] / w,
                    det.bbox_pixels[1] / h,
                    det.bbox_pixels[2] / w,
                    det.bbox_pixels[3] / h,
                ],
                'frame_idx': int(idx),
                'frame_time': idx / 30.0,  # 假设 30fps
            }
            all_detections[label].append(det_info)
            frame_dets.append(det_info)
        
        frame_detections[int(idx)] = frame_dets
        print(f"  Frame {i+1}/{num_sample} (idx={idx}): {len(frame_dets)} detections")
    
    # 标签归一化
    label_to_category = {
        'table': 'table', 'dining table': 'table', 'coffee table': 'table',
        'chair': 'chair', 'armchair': 'chair', 'office chair': 'chair',
        'sofa': 'sofa', 'couch': 'sofa',
        'stove': 'stove', 'oven': 'stove',
        'tv': 'tv', 'television': 'tv',
        'monitor': 'monitor', 'screen': 'monitor',
        'pillow': 'pillow', 'cushion': 'pillow',
        'trash can': 'trash bin', 'trash bin': 'trash bin',
    }
    
    # 构建实体
    entities = {}
    for label, dets in all_detections.items():
        if not dets:
            continue
        
        # 归一化标签
        category = label
        for key, cat in label_to_category.items():
            if key in label.lower():
                category = cat
                break
        
        # 按帧统计
        frame_counts = defaultdict(list)
        for det in dets:
            frame_counts[det['frame_idx']].append(det)
        
        max_count = max(len(fd) for fd in frame_counts.values())
        first_frame = min(det['frame_idx'] for det in dets)
        avg_confidence = np.mean([d['confidence'] for d in dets])
        
        entity = {
            'entity_id': f"entity_{category}",
            'label': category,
            'original_labels': list(set(d['label'] for d in dets)),
            'count': max_count,
            'first_seen_frame': first_frame,
            'first_seen_time': first_frame / 30.0,
            'total_detections': len(dets),
            'avg_confidence': float(avg_confidence),
            'frame_counts': {int(k): len(v) for k, v in frame_counts.items()},
            'all_detections': dets,
        }
        
        if category not in entities:
            entities[category] = entity
        else:
            # 合并相同类别
            entities[category]['total_detections'] += len(dets)
            entities[category]['all_detections'].extend(dets)
            entities[category]['count'] = max(entities[category]['count'], max_count)
    
    # 加载 meta 信息
    meta_info = load_meta_info(scene_name)
    
    # 构建完整心智地图
    mind_map = {
        'scene_info': {
            'scene_name': scene_name,
            'video_path': video_path,
            'total_frames': len(frames),
            'sampled_frames': num_sample,
            'frame_indices': [int(i) for i in frame_indices],
        },
        'meta_info': {
            'room_size': meta_info.get('room_size'),
            'object_bbox': {
                k: [{
                    'centroid': v[0].get('centroid') if v else None,
                    'axesLengths': v[0].get('axesLengths') if v else None,
                }] if v else []
                for k, v in meta_info.get('object_bbox', {}).items()
            } if meta_info.get('object_bbox') else {},
        },
        'detection_summary': {
            'total_entities': len(entities),
            'total_detections': sum(e['total_detections'] for e in entities.values()),
            'entity_counts': {k: v['count'] for k, v in entities.items()},
        },
        'entities': entities,
        'frame_detections': frame_detections,
    }
    
    return mind_map


def main():
    # 选择一个场景
    scene_name = "41069025"  # ARKitScenes
    video_path = f"/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes/{scene_name}.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    print(f"=" * 70)
    print(f"生成完整心智地图示例")
    print(f"场景: {scene_name}")
    print(f"=" * 70)
    
    mind_map = build_full_mindmap(video_path, scene_name)
    
    # 保存
    output_dir = PROJECT_ROOT / "outputs" / f"full_mindmap_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "full_mindmap.json", 'w') as f:
        json.dump(mind_map, f, indent=2, default=str)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("心智地图摘要")
    print("=" * 70)
    print(f"场景: {scene_name}")
    print(f"总帧数: {mind_map['scene_info']['total_frames']}")
    print(f"采样帧数: {mind_map['scene_info']['sampled_frames']}")
    print(f"房间面积: {mind_map['meta_info']['room_size']} 平方米")
    print(f"\n检测到的实体:")
    for label, entity in mind_map['entities'].items():
        print(f"  {label}: 数量={entity['count']}, 总检测数={entity['total_detections']}, 置信度={entity['avg_confidence']:.2f}")
    
    print(f"\n完整心智地图已保存: {output_dir / 'full_mindmap.json'}")


if __name__ == "__main__":
    main()
