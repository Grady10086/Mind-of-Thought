"""
GroundingDINO + VSIBench 完整测试

使用 GroundingDINO 为 VSIBench 视频场景添加语义标签，
并展示完整的心智地图输出。

使用方法:
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m tests.test_grounding_dino_vsibench
"""

import os
import sys
import time
import json
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

import numpy as np

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# VSIBench 视频路径
VSIBENCH_VIDEO_DIR = "/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes"


def extract_video_frames(video_path: str, sample_fps: float = 1.0) -> List[np.ndarray]:
    """从视频中提取帧"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    logger.info(f"Video: {Path(video_path).name}")
    logger.info(f"  FPS: {fps:.1f}, Duration: {duration:.1f}s, Total frames: {total_frames}")
    
    # 计算采样间隔
    sample_interval = int(fps / sample_fps)
    sample_interval = max(1, sample_interval)
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    
    logger.info(f"  Extracted {len(frames)} frames (sample_fps={sample_fps})")
    
    return frames


def create_mock_camera_poses(num_frames: int, scene_bounds: np.ndarray = None):
    """创建模拟相机位姿"""
    from core.scene import CameraPose
    
    poses = []
    
    # 模拟相机沿圆形路径移动
    radius = 3.0
    height = 1.5
    
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        
        x = radius * np.cos(angle)
        y = height
        z = radius * np.sin(angle)
        
        # 相机朝向场景中心
        forward = np.array([-x, 0, -z])
        forward = forward / np.linalg.norm(forward)
        
        pose = CameraPose(
            frame_id=i,
            position=np.array([x, y, z]),
            rotation=np.eye(3),  # 简化
            extrinsic=np.eye(4),
            intrinsic=np.array([
                [500, 0, 320],
                [0, 500, 240],
                [0, 0, 1]
            ]),
        )
        pose.forward = forward
        poses.append(pose)
    
    return poses


class GroundingDINOVSIBenchTest:
    """GroundingDINO + VSIBench 完整测试"""
    
    def __init__(
        self,
        video_id: str = "41069025",
        output_dir: str = None,
    ):
        self.video_id = video_id
        self.video_path = os.path.join(VSIBENCH_VIDEO_DIR, f"{video_id}.mp4")
        self.output_dir = output_dir or str(PROJECT_ROOT / "outputs" / "grounding_dino_test")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.frames = None
        self.scene = None
        self.mind_map = None
        self.labeler = None
    
    def step1_extract_frames(self, sample_fps: float = 1.0):
        """步骤 1: 提取视频帧"""
        logger.info("=" * 70)
        logger.info("Step 1: 提取视频帧")
        logger.info("=" * 70)
        
        self.frames = extract_video_frames(self.video_path, sample_fps)
        
        # 保存示例帧
        sample_frame_path = os.path.join(self.output_dir, f"{self.video_id}_sample_frame.jpg")
        cv2.imwrite(sample_frame_path, cv2.cvtColor(self.frames[0], cv2.COLOR_RGB2BGR))
        logger.info(f"Sample frame saved: {sample_frame_path}")
        
        return self.frames
    
    def step2_load_or_reconstruct_scene(self):
        """步骤 2: 加载或重建 3D 场景"""
        logger.info("\n" + "=" * 70)
        logger.info("Step 2: 加载 3D 场景")
        logger.info("=" * 70)
        
        from core.scene import SceneLoader
        
        # 尝试加载已有场景
        scene_path = str(PROJECT_ROOT / "outputs" / "da3_reconstruction" / "scene.glb")
        
        if os.path.exists(scene_path):
            self.scene = SceneLoader.load(scene_path)
            logger.info(f"Loaded existing scene: {scene_path}")
            logger.info(f"  Points: {self.scene.num_points:,}")
            logger.info(f"  Cameras: {self.scene.num_cameras}")
        else:
            logger.warning(f"Scene not found: {scene_path}")
            logger.info("Creating mock scene from video frames...")
            
            # 创建模拟场景
            from core.scene import Scene3D
            
            h, w = self.frames[0].shape[:2]
            
            # 从帧创建简单点云
            points = []
            colors = []
            
            for frame in self.frames[:10]:  # 只用前 10 帧
                # 随机采样像素
                sample_indices = np.random.choice(h * w, size=1000, replace=False)
                ys, xs = np.divmod(sample_indices, w)
                
                for x, y in zip(xs, ys):
                    # 简单深度估计 (mock)
                    z = 2.0 + np.random.random() * 3.0
                    points.append([
                        (x - w/2) / 500 * z,
                        (y - h/2) / 500 * z,
                        z
                    ])
                    colors.append(frame[y, x] / 255.0)
            
            points = np.array(points)
            colors = np.array(colors)
            
            # 创建相机位姿
            camera_poses = create_mock_camera_poses(len(self.frames))
            
            self.scene = Scene3D(
                points=points,
                colors=colors,
                camera_poses={i: p for i, p in enumerate(camera_poses)},
            )
            
            logger.info(f"Created mock scene with {len(points)} points")
        
        return self.scene
    
    def step3_build_mind_map(self, voxel_size: float = 0.1):
        """步骤 3: 构建心智地图"""
        logger.info("\n" + "=" * 70)
        logger.info("Step 3: 构建心智地图 (MindMapV2)")
        logger.info("=" * 70)
        
        from core.mind_map_v2 import MindMapBuilder
        
        builder = MindMapBuilder(
            voxel_size=voxel_size,
            min_entity_voxels=5,
        )
        
        start = time.time()
        self.mind_map = builder.build(self.scene)
        build_time = time.time() - start
        
        logger.info(f"心智地图构建完成: {self.mind_map}")
        logger.info(f"  实体数: {self.mind_map.entity_count}")
        logger.info(f"  轨迹点: {len(self.mind_map.trajectory)}")
        logger.info(f"  构建耗时: {build_time:.2f}s")
        
        return self.mind_map
    
    def step4_grounding_dino_detection(
        self,
        text_prompt: str = None,
        sample_frames: int = 10,
    ):
        """步骤 4: GroundingDINO 目标检测"""
        logger.info("\n" + "=" * 70)
        logger.info("Step 4: GroundingDINO 目标检测")
        logger.info("=" * 70)
        
        from core.semantic_labeler import GroundingDINOLabeler
        
        self.labeler = GroundingDINOLabeler(
            model_id="IDEA-Research/grounding-dino-tiny",
            device="cuda",
            box_threshold=0.3,
            text_threshold=0.3,
        )
        
        logger.info("Loading GroundingDINO model...")
        start = time.time()
        self.labeler.load_model()
        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        # 选择采样帧进行检测
        frame_indices = np.linspace(0, len(self.frames) - 1, sample_frames).astype(int)
        
        all_detections = {}
        detection_summary = {}
        
        for idx in frame_indices:
            frame = self.frames[idx]
            
            logger.info(f"\n检测帧 {idx}...")
            start = time.time()
            
            result = self.labeler.detect_and_label_frame(frame, text_prompt)
            
            detect_time = time.time() - start
            
            all_detections[int(idx)] = {
                'frame_idx': int(idx),
                'num_detections': result['num_detections'],
                'detections': [
                    {
                        'label': d.label,
                        'confidence': round(d.confidence, 3),
                        'bbox_pixels': [round(x, 1) for x in d.bbox_pixels],
                    }
                    for d in result['detections']
                ]
            }
            
            # 统计检测到的物体
            for det in result['detections']:
                label = det.label.lower().strip()
                if label not in detection_summary:
                    detection_summary[label] = {'count': 0, 'max_confidence': 0}
                detection_summary[label]['count'] += 1
                detection_summary[label]['max_confidence'] = max(
                    detection_summary[label]['max_confidence'],
                    det.confidence
                )
            
            logger.info(f"  检测到 {result['num_detections']} 个物体 ({detect_time:.2f}s)")
            for det in result['detections'][:5]:
                logger.info(f"    - {det.label}: {det.confidence:.2%}")
        
        # 打印检测汇总
        logger.info("\n" + "-" * 50)
        logger.info("检测汇总 (所有帧):")
        logger.info("-" * 50)
        
        sorted_labels = sorted(detection_summary.items(), key=lambda x: x[1]['count'], reverse=True)
        for label, stats in sorted_labels[:15]:
            logger.info(f"  {label}: {stats['count']} 次, 最高置信度 {stats['max_confidence']:.2%}")
        
        # 保存可视化
        self._save_detection_visualization(frame_indices[0], all_detections)
        
        return all_detections, detection_summary
    
    def _save_detection_visualization(self, frame_idx: int, all_detections: Dict):
        """保存检测可视化"""
        frame = self.frames[frame_idx].copy()
        detections = all_detections[frame_idx]['detections']
        
        # 绘制边界框
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det['bbox_pixels']]
            
            # 颜色根据置信度
            confidence = det['confidence']
            color = (
                int(255 * (1 - confidence)),
                int(255 * confidence),
                0
            )
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label_text = f"{det['label']}: {confidence:.0%}"
            cv2.putText(
                frame, label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
        
        # 保存
        vis_path = os.path.join(self.output_dir, f"{self.video_id}_detection_frame{frame_idx}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        logger.info(f"Detection visualization saved: {vis_path}")
    
    def step5_add_semantic_labels(self, all_detections: Dict):
        """步骤 5: 为实体添加语义标签 (基于检测结果聚合)"""
        logger.info("\n" + "=" * 70)
        logger.info("Step 5: 为实体添加语义标签")
        logger.info("=" * 70)
        
        if not self.mind_map or not self.mind_map.entities:
            logger.warning("No entities to label")
            return
        
        # 从检测结果中聚合最常见的标签
        label_counts = {}
        for frame_idx, data in all_detections.items():
            for det in data['detections']:
                label = det['label'].lower().strip()
                # 过滤掉场景级别的标签
                if label in ['floor', 'wall', 'ceiling', 'window', 'door', 'stairs']:
                    continue
                if label not in label_counts:
                    label_counts[label] = {'count': 0, 'max_confidence': 0, 'total_area': 0}
                label_counts[label]['count'] += 1
                label_counts[label]['max_confidence'] = max(
                    label_counts[label]['max_confidence'],
                    det['confidence']
                )
                # 计算框面积
                bbox = det['bbox_pixels']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                label_counts[label]['total_area'] += area
        
        # 按面积和频次排序
        sorted_labels = sorted(
            label_counts.items(),
            key=lambda x: (x[1]['total_area'], x[1]['count']),
            reverse=True
        )
        
        logger.info(f"物体级标签统计 (排除场景标签):")
        for label, stats in sorted_labels[:10]:
            logger.info(f"  {label}: 次数={stats['count']}, 置信度={stats['max_confidence']:.0%}")
        
        # 根据实体大小分配标签
        # entity_000 是大实体 (整个场景), entity_001, entity_002 是小物体
        for i, entity in enumerate(self.mind_map.entities):
            volume = entity.volume
            
            if volume > 10.0:
                # 大实体 - 可能是房间/场景
                entity.semantic_label = "room"
                entity.semantic_confidence = 0.8
                logger.info(f"  {entity.entity_id}: room (大型实体, 体积={volume:.1f}m³)")
            elif volume > 0.05 and sorted_labels:
                # 中等实体 - 分配最常见的家具标签
                # 根据实体索引分配不同标签
                label_idx = min(i - 1, len(sorted_labels) - 1) if i > 0 else 0
                if label_idx >= 0 and label_idx < len(sorted_labels):
                    label, stats = sorted_labels[label_idx]
                    entity.semantic_label = label
                    entity.semantic_confidence = stats['max_confidence']
                    logger.info(f"  {entity.entity_id}: {label} ({entity.semantic_confidence:.0%})")
                else:
                    entity.semantic_label = "furniture"
                    entity.semantic_confidence = 0.5
                    logger.info(f"  {entity.entity_id}: furniture (默认)")
            else:
                # 小实体
                if sorted_labels:
                    # 查找小物体类别
                    small_object_labels = ['lamp', 'cup', 'book', 'phone', 'vase', 'clock']
                    for label, stats in sorted_labels:
                        if any(s in label for s in small_object_labels):
                            entity.semantic_label = label
                            entity.semantic_confidence = stats['max_confidence']
                            break
                    else:
                        entity.semantic_label = "small object"
                        entity.semantic_confidence = 0.5
                else:
                    entity.semantic_label = "object"
                    entity.semantic_confidence = 0.3
                logger.info(f"  {entity.entity_id}: {entity.semantic_label} (小型实体)")
    
    def step6_generate_complete_output(self, detection_summary: Dict):
        """步骤 6: 生成完整输出"""
        logger.info("\n" + "=" * 70)
        logger.info("Step 6: 生成完整心智地图输出")
        logger.info("=" * 70)
        
        # 构建完整输出
        output = {
            "metadata": {
                "video_id": self.video_id,
                "video_path": self.video_path,
                "num_frames": len(self.frames),
                "frame_shape": list(self.frames[0].shape),
                "detection_model": "IDEA-Research/grounding-dino-tiny",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "scene_info": {
                "num_points": self.scene.num_points,
                "num_cameras": self.scene.num_cameras,
                "bounds": {
                    "min": self.scene.bounds[0].tolist() if hasattr(self.scene, 'bounds') else None,
                    "max": self.scene.bounds[1].tolist() if hasattr(self.scene, 'bounds') else None,
                }
            },
            "mind_map": {
                "voxel_size": self.mind_map.voxel_size,
                "entity_count": self.mind_map.entity_count,
                "trajectory_points": len(self.mind_map.trajectory),
                "voxel_count": self.mind_map.voxel_map.voxel_count,
            },
            "entities": [],
            "detection_summary": detection_summary,
            "trajectory_summary": [],
            "spatial_relations": [],
        }
        
        # 实体信息
        for entity in self.mind_map.entities:
            entity_data = {
                "id": entity.entity_id,
                "semantic_label": getattr(entity, 'semantic_label', 'unknown'),
                "semantic_confidence": round(getattr(entity, 'semantic_confidence', 0.0), 3),
                "center": [round(x, 3) for x in entity.centroid.tolist()],
                "size": [round(x, 3) for x in entity.size.tolist()],
                "volume": round(entity.volume, 4),
                "voxel_count": entity.voxel_count,
                "first_seen_frame": entity.first_seen_frame,
                "visible_frames_count": len(entity.visible_frames) if entity.visible_frames else 0,
                "average_color": [round(x, 3) for x in entity.mean_color.tolist()] if entity.mean_color is not None else None,
            }
            output["entities"].append(entity_data)
        
        # 轨迹摘要
        sample_indices = np.linspace(0, len(self.mind_map.trajectory) - 1, min(10, len(self.mind_map.trajectory))).astype(int)
        for idx in sample_indices:
            tp = self.mind_map.trajectory[idx]
            output["trajectory_summary"].append({
                "frame": tp.frame_id,
                "position": [round(x, 3) for x in tp.position.tolist()],
                "visible_entities": list(tp.visible_entities) if tp.visible_entities else [],
            })
        
        # 空间关系
        if self.mind_map.entity_count >= 2:
            for i, e1 in enumerate(self.mind_map.entities):
                for e2 in self.mind_map.entities[i+1:]:
                    dist = e1.distance_to(e2)
                    rel = self.mind_map.get_relations(e1.entity_id).get(e2.entity_id, {})
                    output["spatial_relations"].append({
                        "entity1": e1.entity_id,
                        "entity1_label": getattr(e1, 'semantic_label', 'unknown'),
                        "entity2": e2.entity_id,
                        "entity2_label": getattr(e2, 'semantic_label', 'unknown'),
                        "distance": round(dist, 3),
                        "relation": rel.get('relation', 'near'),
                    })
        
        # 保存 JSON
        output_path = os.path.join(self.output_dir, f"{self.video_id}_complete_mind_map.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Complete output saved: {output_path}")
        
        # 生成 LLM Prompt
        prompt = self._generate_llm_prompt(output)
        prompt_path = os.path.join(self.output_dir, f"{self.video_id}_llm_prompt.yaml")
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        logger.info(f"LLM prompt saved: {prompt_path}")
        
        return output
    
    def _generate_llm_prompt(self, output: Dict) -> str:
        """生成 LLM Prompt"""
        prompt = f"""# Spatial Mind Map - Video {output['metadata']['video_id']}

## Scene Overview
- Total Frames: {output['metadata']['num_frames']}
- Detection Model: {output['metadata']['detection_model']}
- Voxel Resolution: {output['mind_map']['voxel_size']}m

## Detected Objects Summary
"""
        # 检测汇总
        for label, stats in sorted(output['detection_summary'].items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            prompt += f"- {label}: {stats['count']} detections (max conf: {stats['max_confidence']:.0%})\n"
        
        prompt += f"""
## Entities ({output['mind_map']['entity_count']} total)
"""
        for entity in output['entities']:
            prompt += f"""
### {entity['id']}
- Label: {entity['semantic_label']} (confidence: {entity['semantic_confidence']:.0%})
- Center: ({entity['center'][0]:.2f}, {entity['center'][1]:.2f}, {entity['center'][2]:.2f})
- Size: {entity['size'][0]:.2f}m x {entity['size'][1]:.2f}m x {entity['size'][2]:.2f}m
- Volume: {entity['volume']:.3f}m³
- First seen: frame {entity['first_seen_frame']}
"""
        
        if output['spatial_relations']:
            prompt += "\n## Spatial Relations\n"
            for rel in output['spatial_relations'][:10]:
                prompt += f"- {rel['entity1_label']} ({rel['entity1']}) is {rel['relation']} {rel['entity2_label']} ({rel['entity2']}), distance: {rel['distance']:.2f}m\n"
        
        prompt += f"""
## Camera Trajectory ({len(output['trajectory_summary'])} samples)
"""
        for tp in output['trajectory_summary'][:5]:
            prompt += f"- Frame {tp['frame']}: pos=({tp['position'][0]:.2f}, {tp['position'][1]:.2f}, {tp['position'][2]:.2f}), visible: {len(tp['visible_entities'])} objects\n"
        
        return prompt
    
    def step7_print_summary(self, output: Dict):
        """步骤 7: 打印完整汇总"""
        logger.info("\n" + "=" * 70)
        logger.info("完整心智地图汇总")
        logger.info("=" * 70)
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    VSIBench + GroundingDINO 完整结果                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ 视频 ID: {output['metadata']['video_id']:<57} ║
║ 帧数: {output['metadata']['num_frames']:<60} ║
║ 分辨率: {output['metadata']['frame_shape'][1]}x{output['metadata']['frame_shape'][0]:<50} ║
╠══════════════════════════════════════════════════════════════════════╣
║ 检测模型: {output['metadata']['detection_model']:<56} ║
╠══════════════════════════════════════════════════════════════════════╣
║                           检测到的物体                               ║
╠══════════════════════════════════════════════════════════════════════╣""")
        
        for label, stats in sorted(output['detection_summary'].items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            conf_str = f"{stats['max_confidence']:.0%}"
            print(f"║  {label:<20} | 次数: {stats['count']:<5} | 最高置信度: {conf_str:<10} ║")
        
        print("""╠══════════════════════════════════════════════════════════════════════╣
║                           实体信息                                   ║
╠══════════════════════════════════════════════════════════════════════╣""")
        
        for entity in output['entities']:
            print(f"║  {entity['id']}: {entity['semantic_label']:<15} ({entity['semantic_confidence']:.0%})                     ║")
            print(f"║    中心: ({entity['center'][0]:.2f}, {entity['center'][1]:.2f}, {entity['center'][2]:.2f})                                    ║")
            print(f"║    尺寸: {entity['size'][0]:.2f}m x {entity['size'][1]:.2f}m x {entity['size'][2]:.2f}m                                 ║")
            print(f"║    体积: {entity['volume']:.3f}m³, 首次出现: 帧 {entity['first_seen_frame']:<20} ║")
        
        print("""╠══════════════════════════════════════════════════════════════════════╣
║                           空间关系                                   ║
╠══════════════════════════════════════════════════════════════════════╣""")
        
        for rel in output['spatial_relations'][:5]:
            print(f"║  {rel['entity1_label']} {rel['relation']} {rel['entity2_label']}, 距离: {rel['distance']:.2f}m           ║")
        
        print("""╚══════════════════════════════════════════════════════════════════════╝
""")
    
    def run(
        self,
        sample_fps: float = 1.0,
        voxel_size: float = 0.1,
        detection_frames: int = 10,
        text_prompt: str = None,
    ):
        """运行完整测试"""
        logger.info("=" * 70)
        logger.info(f"GroundingDINO + VSIBench 完整测试")
        logger.info(f"Video ID: {self.video_id}")
        logger.info("=" * 70)
        
        # 1. 提取帧
        self.step1_extract_frames(sample_fps)
        
        # 2. 加载场景
        self.step2_load_or_reconstruct_scene()
        
        # 3. 构建心智地图
        self.step3_build_mind_map(voxel_size)
        
        # 4. GroundingDINO 检测
        all_detections, detection_summary = self.step4_grounding_dino_detection(
            text_prompt=text_prompt,
            sample_frames=detection_frames,
        )
        
        # 5. 添加语义标签
        self.step5_add_semantic_labels(all_detections)
        
        # 6. 生成完整输出
        output = self.step6_generate_complete_output(detection_summary)
        
        # 7. 打印汇总
        self.step7_print_summary(output)
        
        logger.info("\n" + "=" * 70)
        logger.info("测试完成！")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 70)
        
        return output


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GroundingDINO + VSIBench Test")
    parser.add_argument("--video_id", type=str, default="41069025", help="VSIBench video ID")
    parser.add_argument("--sample_fps", type=float, default=1.0, help="Frame sampling FPS")
    parser.add_argument("--detection_frames", type=int, default=10, help="Number of frames for detection")
    parser.add_argument("--text_prompt", type=str, default=None, help="Custom detection prompt")
    
    args = parser.parse_args()
    
    # 检查依赖
    try:
        import torch
        import transformers
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"Transformers: {transformers.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return
    
    # 运行测试
    tester = GroundingDINOVSIBenchTest(video_id=args.video_id)
    output = tester.run(
        sample_fps=args.sample_fps,
        detection_frames=args.detection_frames,
        text_prompt=args.text_prompt,
    )
    
    return output


if __name__ == "__main__":
    main()
