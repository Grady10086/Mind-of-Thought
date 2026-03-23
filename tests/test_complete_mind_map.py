"""
完整心智地图测试 - 输出所有详细信息供核验

测试内容:
1. 基础 V2 功能
2. 房间边界检测 (新增)
3. 射线投射可见性 (新增)
4. 完整输出所有信息
"""

import os
import sys
import json
import logging
import time
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.scene import SceneLoader, Scene3D
from core.mind_map_v2 import MindMapBuilder, MindMapV2
from core.voxel_map import SparseVoxelMap
from core.room_boundary import RoomBoundaryDetector, detect_room_from_voxel_map
from core.visibility import RaycastVisibility, SimpleVisibility

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def separator(title: str):
    """打印分隔符"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_complete_test():
    """运行完整测试"""
    
    # GLB 文件路径
    glb_path = project_root / "outputs" / "da3_reconstruction" / "scene.glb"
    output_dir = project_root / "outputs" / "complete_mind_map_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "完整心智地图测试" + " " * 20 + "#")
    print("#" * 70)
    
    # ========================================
    # 1. 加载场景
    # ========================================
    separator("1. 场景加载")
    
    start_time = time.time()
    scene = SceneLoader.load(str(glb_path))
    load_time = time.time() - start_time
    
    print(f"文件路径: {glb_path}")
    print(f"加载耗时: {load_time:.2f}s")
    print(f"\n场景信息:")
    print(f"  点云数量: {len(scene.point_cloud):,} 点")
    print(f"  相机帧数: {scene.num_frames} 帧")
    print(f"  颜色信息: {'有' if scene.colors is not None else '无'}")
    
    # 计算场景边界
    bounds = scene.bounds
    if bounds:
        pmin, pmax = bounds
        print(f"\n场景边界:")
        print(f"  X: [{pmin[0]:.2f}, {pmax[0]:.2f}] m, 范围 = {pmax[0]-pmin[0]:.2f} m")
        print(f"  Y: [{pmin[1]:.2f}, {pmax[1]:.2f}] m, 范围 = {pmax[1]-pmin[1]:.2f} m")
        print(f"  Z: [{pmin[2]:.2f}, {pmax[2]:.2f}] m, 范围 = {pmax[2]-pmin[2]:.2f} m")
    
    # 打印部分相机位姿
    print(f"\n相机轨迹 (采样显示):")
    camera_names = sorted(scene.camera_poses.keys())
    sample_indices = [0, len(camera_names)//4, len(camera_names)//2, 3*len(camera_names)//4, len(camera_names)-1]
    for idx in sample_indices:
        if idx < len(camera_names):
            name = camera_names[idx]
            pose = scene.camera_poses[name]
            pos = pose.position
            print(f"  帧 {idx:3d}: position=[{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
    
    # ========================================
    # 2. 构建心智地图 (V2)
    # ========================================
    separator("2. 心智地图构建 (四步法)")
    
    start_time = time.time()
    builder = MindMapBuilder(
        voxel_size=0.1,          # 10cm 体素
        min_entity_voxels=10,    # 最小 10 个体素
        feature_threshold=0.5,
    )
    mind_map = builder.build(scene)
    build_time = time.time() - start_time
    
    print(f"\n构建耗时: {build_time:.2f}s")
    print(f"\n心智地图摘要:")
    print(f"  体素分辨率: {mind_map.voxel_size} m")
    print(f"  总体素数: {mind_map.voxel_map.voxel_count:,}")
    print(f"  占据体素: {mind_map.voxel_map.occupied_count:,}")
    print(f"  发现实体: {mind_map.entity_count}")
    print(f"  轨迹点数: {len(mind_map.trajectory)}")
    
    # ========================================
    # 3. 实体详细信息
    # ========================================
    separator("3. 实体详细信息")
    
    entities_data = []
    for i, entity in enumerate(mind_map.entities):
        entity_info = {
            'id': entity.entity_id,
            'centroid': entity.centroid.tolist(),
            'bbox_min': entity.bbox_min.tolist(),
            'bbox_max': entity.bbox_max.tolist(),
            'size': entity.size.tolist(),
            'volume': entity.volume,
            'voxel_count': entity.voxel_count,
            'first_seen_frame': entity.first_seen_frame,
            'last_seen_frame': entity.last_seen_frame,
            'visible_frames_count': len(entity.visible_frames),
            'visible_frames': entity.visible_frames[:20] if entity.visible_frames else [],
            'mean_color': entity.mean_color.tolist() if entity.mean_color is not None else None,
        }
        entities_data.append(entity_info)
        
        print(f"\n实体 {i+1}: {entity.entity_id}")
        print(f"  中心坐标: [{entity.centroid[0]:.3f}, {entity.centroid[1]:.3f}, {entity.centroid[2]:.3f}]")
        print(f"  包围盒:")
        print(f"    最小: [{entity.bbox_min[0]:.3f}, {entity.bbox_min[1]:.3f}, {entity.bbox_min[2]:.3f}]")
        print(f"    最大: [{entity.bbox_max[0]:.3f}, {entity.bbox_max[1]:.3f}, {entity.bbox_max[2]:.3f}]")
        print(f"  尺寸: [{entity.size[0]:.2f}, {entity.size[1]:.2f}, {entity.size[2]:.2f}] m")
        print(f"  体积: {entity.volume:.3f} m³")
        print(f"  体素数: {entity.voxel_count}")
        print(f"  首次出现: 帧 {entity.first_seen_frame}")
        print(f"  可见帧数: {len(entity.visible_frames)}")
        if entity.mean_color is not None:
            c = entity.mean_color
            print(f"  平均颜色: RGB({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
    
    # ========================================
    # 4. 房间边界检测 (新功能)
    # ========================================
    separator("4. 房间边界检测 (新功能)")
    
    try:
        start_time = time.time()
        room_detector = RoomBoundaryDetector(
            ransac_threshold=0.05,
            ransac_iterations=1000,
            min_plane_points=100,
        )
        
        # 获取点云
        points = mind_map.voxel_map.get_point_cloud()
        
        if points is not None and len(points) > 0:
            room_bounds = room_detector.detect(points, max_planes=8)
            detect_time = time.time() - start_time
            
            print(f"\n房间边界检测耗时: {detect_time:.2f}s")
            print(f"\n房间尺寸估计:")
            print(f"  地板高度: {room_bounds.floor_height:.3f} m")
            print(f"  天花板高度: {room_bounds.ceiling_height:.3f} m")
            print(f"  房间高度: {room_bounds.room_height:.3f} m")
            print(f"  房间包围盒: {[round(v, 3) for v in room_bounds.bbox]}")
            print(f"  房间尺寸 (宽x高x深): {room_bounds.dimensions[0]:.2f} x {room_bounds.dimensions[1]:.2f} x {room_bounds.dimensions[2]:.2f} m")
            print(f"  检测到墙壁数: {len(room_bounds.walls)}")
            print(f"  检测到地板: {'是' if room_bounds.floor_plane else '否'}")
            print(f"  检测到天花板: {'是' if room_bounds.ceiling_plane else '否'}")
            
            room_data = room_bounds.to_dict()
        else:
            print("无法获取点云进行房间检测")
            room_data = None
            
    except Exception as e:
        print(f"房间边界检测失败: {e}")
        import traceback
        traceback.print_exc()
        room_data = None
    
    # ========================================
    # 5. 射线投射可见性 (新功能)
    # ========================================
    separator("5. 射线投射可见性计算 (新功能)")
    
    raycast = None
    visibility_data = []
    
    try:
        start_time = time.time()
        raycast = RaycastVisibility(mind_map.voxel_map)
        raycast_build_time = time.time() - start_time
        print(f"\n射线投射加速结构构建: {raycast_build_time:.2f}s")
        print(f"占据体素数: {len(raycast.occupied_voxels):,}")
        
        # 对每个实体计算可见性
        print(f"\n各实体可见性分析 (从首帧相机):")
        
        if mind_map.trajectory:
            camera_positions = [mind_map.trajectory[0].position]
            # 添加中间帧和末帧
            if len(mind_map.trajectory) > 1:
                camera_positions.append(mind_map.trajectory[len(mind_map.trajectory)//2].position)
            if len(mind_map.trajectory) > 2:
                camera_positions.append(mind_map.trajectory[-1].position)
            
            for cam_idx, camera_pos in enumerate(camera_positions):
                frame_names = ['首帧', '中帧', '末帧']
                print(f"\n  从{frame_names[cam_idx]}相机 ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}):")
                
                for entity in mind_map.entities:
                    entity_dict = {
                        'center': entity.centroid.tolist(),
                        'size': entity.size.tolist(),
                    }
                    
                    start_time = time.time()
                    vis_result = raycast.compute_visibility(
                        camera_position=camera_pos,
                        entity=entity_dict,
                        max_distance=15.0,
                        num_rays=9,
                        occlusion_threshold=0.5,
                    )
                    compute_time = time.time() - start_time
                    
                    print(f"    {entity.entity_id}: 可见={vis_result.visible}, "
                          f"遮挡率={vis_result.occlusion_ratio:.1%}, "
                          f"距离={vis_result.distance:.2f}m")
                    
                    visibility_data.append({
                        'entity_id': entity.entity_id,
                        'camera_frame': frame_names[cam_idx],
                        'camera_position': camera_pos.tolist(),
                        'visible': vis_result.visible,
                        'occlusion_ratio': vis_result.occlusion_ratio,
                        'distance': vis_result.distance,
                        'reason': vis_result.reason,
                    })
        else:
            print("  无相机轨迹数据")
    
    except Exception as e:
        print(f"射线投射可见性计算失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # 6. 空间关系
    # ========================================
    separator("6. 空间关系图")
    
    spatial_relations = []
    if mind_map._spatial_graph:
        print(f"\n实体间空间关系:")
        for entity_id, relations in mind_map._spatial_graph.items():
            if relations:
                print(f"\n  {entity_id}:")
                for other_id, rel in relations.items():
                    print(f"    -> {other_id}: {rel['relation']}, 距离={rel['distance']}m")
                    spatial_relations.append({
                        'from': entity_id,
                        'to': other_id,
                        'relation': rel['relation'],
                        'distance': rel['distance'],
                    })
    else:
        print("无空间关系数据 (实体数量不足)")
    
    # ========================================
    # 7. 轨迹分析
    # ========================================
    separator("7. 相机轨迹分析")
    
    trajectory_data = []
    if mind_map.trajectory:
        # 计算轨迹统计
        positions = np.array([t.position for t in mind_map.trajectory])
        
        print(f"\n轨迹统计:")
        print(f"  总帧数: {len(mind_map.trajectory)}")
        print(f"  位置范围:")
        print(f"    X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
        print(f"    Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
        print(f"    Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
        
        # 计算总移动距离
        if len(positions) > 1:
            diffs = np.diff(positions, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_distance = np.sum(distances)
            print(f"  总移动距离: {total_distance:.2f} m")
            print(f"  平均帧间距离: {np.mean(distances):.3f} m")
        
        print(f"\n轨迹详情 (采样):")
        sample_frames = [0, len(mind_map.trajectory)//4, len(mind_map.trajectory)//2, 
                        3*len(mind_map.trajectory)//4, len(mind_map.trajectory)-1]
        
        for frame_id in sample_frames:
            if frame_id < len(mind_map.trajectory):
                tp = mind_map.trajectory[frame_id]
                trajectory_data.append({
                    'frame': tp.frame_id,
                    'position': tp.position.tolist(),
                    'visible_entities': tp.visible_entities,
                    'visible_count': len(tp.visible_entities),
                })
                
                print(f"  帧 {tp.frame_id:3d}: pos=[{tp.position[0]:7.3f}, {tp.position[1]:7.3f}, {tp.position[2]:7.3f}], "
                      f"可见实体数={len(tp.visible_entities)}")
    
    # ========================================
    # 8. LLM Prompt 输出
    # ========================================
    separator("8. LLM Prompt 输出")
    
    yaml_prompt = mind_map.to_prompt(format='yaml')
    print("\n--- YAML 格式 ---")
    print(yaml_prompt)
    
    md_prompt = mind_map.to_prompt(format='markdown')
    print("\n--- Markdown 格式 ---")
    print(md_prompt)
    
    # ========================================
    # 9. 保存完整输出
    # ========================================
    separator("9. 保存完整输出")
    
    # 汇总所有数据
    complete_output = {
        'metadata': {
            'source_file': str(glb_path),
            'voxel_size': mind_map.voxel_size,
            'load_time_sec': load_time,
            'build_time_sec': build_time,
        },
        'scene_info': {
            'point_count': len(scene.point_cloud),
            'camera_frames': scene.num_frames,
            'has_colors': scene.colors is not None,
            'bounds': {
                'min': bounds[0].tolist() if bounds else None,
                'max': bounds[1].tolist() if bounds else None,
            },
        },
        'voxel_map': {
            'total_voxels': mind_map.voxel_map.voxel_count,
            'occupied_voxels': mind_map.voxel_map.occupied_count,
            'resolution': mind_map.voxel_map.voxel_size,
        },
        'entities': entities_data,
        'room_bounds': room_data,
        'visibility_analysis': visibility_data,
        'spatial_relations': spatial_relations,
        'trajectory_summary': trajectory_data,
        'llm_prompts': {
            'yaml': yaml_prompt,
            'markdown': md_prompt,
        },
    }
    
    # 保存 JSON
    output_json = output_dir / "complete_mind_map_output.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(complete_output, f, indent=2, ensure_ascii=False)
    print(f"完整 JSON 输出: {output_json}")
    
    # 保存 YAML prompt
    output_yaml = output_dir / "mind_map_prompt.yaml"
    with open(output_yaml, 'w', encoding='utf-8') as f:
        f.write(yaml_prompt)
    print(f"YAML Prompt: {output_yaml}")
    
    # 保存 Markdown prompt  
    output_md = output_dir / "mind_map_prompt.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(md_prompt)
    print(f"Markdown Prompt: {output_md}")
    
    # 保存原始心智地图
    mind_map_json = output_dir / "mind_map_raw.json"
    mind_map.save(str(mind_map_json))
    print(f"原始心智地图: {mind_map_json}")
    
    # ========================================
    # 10. 测试总结
    # ========================================
    separator("测试总结")
    
    print(f"""
场景信息:
  - 点云: {len(scene.point_cloud):,} 点
  - 相机帧: {scene.num_frames} 帧
  - 加载耗时: {load_time:.2f}s

心智地图 V2:
  - 体素分辨率: {mind_map.voxel_size}m
  - 体素数量: {mind_map.voxel_map.voxel_count:,}
  - 发现实体: {mind_map.entity_count}
  - 构建耗时: {build_time:.2f}s

房间边界检测:
  - 状态: {'成功' if room_data else '失败'}
  - 房间尺寸: {room_data['dimensions'] if room_data else 'N/A'} m

射线投射可见性:
  - 加速结构体素: {len(raycast.occupied_voxels) if raycast else 'N/A'}
  - 可见性分析数: {len(visibility_data)}

输出文件:
  - {output_json}
  - {output_yaml}
  - {output_md}
  - {mind_map_json}
""")
    
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "测试完成" + " " * 26 + "#")
    print("#" * 70 + "\n")
    
    return complete_output


if __name__ == "__main__":
    run_complete_test()
