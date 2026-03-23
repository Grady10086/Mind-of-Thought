"""
MindMapV2 完整测试脚本

测试四步构建法的心智地图是否满足 VSIBench 视频 QA 空间理解任务的要求

测试内容：
1. 基础功能测试 - 场景加载、体素化、实体发现
2. VSIBench 任务模拟测试 - 物体计数、空间关系、路径规划等
3. 性能测试 - 大规模点云处理、内存占用
4. LLM Prompt 生成测试

使用方法：
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m tests.test_mind_map_v2
"""

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None


class MindMapV2Tester:
    """
    MindMapV2 测试器
    
    测试心智地图 V2 版本是否满足 VSIBench 任务需求
    """
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or str(PROJECT_ROOT / "outputs" / "test_mind_map_v2")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results: List[TestResult] = []
        self.scene = None
        self.mind_map = None
        
    def log_result(self, result: TestResult):
        """记录测试结果"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        logger.info(f"{status} | {result.name} | {result.duration:.2f}s | {result.message}")
    
    def run_test(self, test_func, name: str) -> TestResult:
        """运行单个测试"""
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            if isinstance(result, tuple):
                passed, message, details = result
            else:
                passed, message, details = result, "", None
            return TestResult(name=name, passed=passed, duration=duration, message=message, details=details)
        except Exception as e:
            duration = time.time() - start_time
            tb = traceback.format_exc()
            logger.error(f"Test {name} failed with exception:\n{tb}")
            return TestResult(name=name, passed=False, duration=duration, message=str(e))
    
    # ==================== 基础功能测试 ====================
    
    def test_scene_loading(self) -> Tuple[bool, str, Dict]:
        """测试场景加载"""
        from core.scene import SceneLoader
        
        # 查找可用的 GLB 文件
        da3_output = PROJECT_ROOT / "outputs" / "da3_reconstruction"
        glb_path = da3_output / "scene.glb"
        
        if not glb_path.exists():
            return False, f"GLB 文件不存在: {glb_path}", {}
        
        # 加载场景
        scene = SceneLoader.load_glb(str(glb_path))
        self.scene = scene
        
        details = {
            "num_points": scene.num_points,
            "num_cameras": scene.num_cameras,
            "has_mesh": scene.has_mesh,
            "bounds": [b.tolist() for b in scene.bounds] if scene.bounds else None,
        }
        
        # 验证基本属性
        if scene.point_cloud is None or len(scene.point_cloud) == 0:
            return False, "点云为空", details
        
        if scene.num_points < 1000:
            return False, f"点云过少: {scene.num_points}", details
        
        return True, f"加载成功: {scene.num_points} 点, {scene.num_cameras} 相机", details
    
    def test_voxelization(self) -> Tuple[bool, str, Dict]:
        """测试体素化 (第一步: 几何积木化)"""
        from core.voxel_map import SparseVoxelMap
        
        if self.scene is None:
            return False, "场景未加载", {}
        
        # 测试不同体素大小
        voxel_sizes = [0.05, 0.1, 0.2]
        results = {}
        
        for voxel_size in voxel_sizes:
            voxel_map = SparseVoxelMap(voxel_size=voxel_size)
            
            start = time.time()
            voxel_map.integrate_points(
                self.scene.point_cloud,
                colors=self.scene.colors,
                frame_id=0
            )
            integration_time = time.time() - start
            
            results[f"voxel_{voxel_size}m"] = {
                "num_voxels": voxel_map.num_voxels,
                "num_occupied": voxel_map.num_occupied,
                "integration_time": round(integration_time, 3),
                "compression_ratio": round(self.scene.num_points / max(voxel_map.num_voxels, 1), 2),
            }
        
        # 验证
        if results["voxel_0.1m"]["num_voxels"] == 0:
            return False, "体素化失败", results
        
        return True, f"体素化成功, 0.1m: {results['voxel_0.1m']['num_voxels']} 体素", results
    
    def test_connected_components(self) -> Tuple[bool, str, Dict]:
        """测试连通域分析 (第二步: 实体符号化)"""
        from core.voxel_map import SparseVoxelMap, connected_components_3d
        
        if self.scene is None:
            return False, "场景未加载", {}
        
        # 体素化
        voxel_map = SparseVoxelMap(voxel_size=0.1)
        voxel_map.integrate_points(self.scene.point_cloud, colors=self.scene.colors)
        
        # 连通域分析
        start = time.time()
        components = connected_components_3d(voxel_map, feature_threshold=0.5)
        cc_time = time.time() - start
        
        # 统计
        component_sizes = [len(c) for c in components.values()]
        
        details = {
            "num_components": len(components),
            "total_voxels": sum(component_sizes),
            "largest_component": max(component_sizes) if component_sizes else 0,
            "smallest_component": min(component_sizes) if component_sizes else 0,
            "mean_size": round(np.mean(component_sizes), 2) if component_sizes else 0,
            "analysis_time": round(cc_time, 3),
        }
        
        if len(components) == 0:
            return False, "未找到连通域", details
        
        return True, f"发现 {len(components)} 个连通域", details
    
    def test_mind_map_building(self) -> Tuple[bool, str, Dict]:
        """测试完整的四步构建流程"""
        from core.mind_map_v2 import MindMapBuilder
        
        if self.scene is None:
            return False, "场景未加载", {}
        
        # 构建心智地图
        builder = MindMapBuilder(
            voxel_size=0.1,
            min_entity_voxels=10,  # 过滤小噪声
            feature_threshold=0.5
        )
        
        start = time.time()
        mind_map = builder.build(self.scene)
        build_time = time.time() - start
        
        self.mind_map = mind_map
        
        details = {
            "entity_count": mind_map.entity_count,
            "trajectory_points": len(mind_map.trajectory),
            "voxel_stats": mind_map.voxel_map.to_dict() if mind_map.voxel_map else None,
            "build_time": round(build_time, 3),
        }
        
        if mind_map.entity_count == 0:
            return False, "未发现任何实体", details
        
        return True, f"构建完成: {mind_map.entity_count} 实体, {len(mind_map.trajectory)} 轨迹点", details
    
    # ==================== VSIBench 任务模拟测试 ====================
    
    def test_object_counting(self) -> Tuple[bool, str, Dict]:
        """测试物体计数能力 (VSIBench: object_counting)"""
        if self.mind_map is None:
            return False, "心智地图未构建", {}
        
        # 统计实体
        entities = self.mind_map.entities
        
        # 按尺寸分类
        small = [e for e in entities if e.volume < 0.01]  # < 10L
        medium = [e for e in entities if 0.01 <= e.volume < 0.1]  # 10L-100L
        large = [e for e in entities if e.volume >= 0.1]  # > 100L
        
        details = {
            "total_entities": len(entities),
            "small_objects": len(small),
            "medium_objects": len(medium),
            "large_objects": len(large),
            "entity_volumes": [round(e.volume, 4) for e in entities[:10]],
        }
        
        # 验证计数能力
        if len(entities) == 0:
            return False, "无法计数 - 没有实体", details
        
        return True, f"可计数: {len(entities)} 物体 (小:{len(small)}, 中:{len(medium)}, 大:{len(large)})", details
    
    def test_spatial_relations(self) -> Tuple[bool, str, Dict]:
        """测试空间关系推理 (VSIBench: object_rel_direction, object_rel_distance)"""
        if self.mind_map is None:
            return False, "心智地图未构建", {}
        
        # 计算空间关系
        self.mind_map.compute_spatial_relations(max_distance=5.0)
        
        # 统计关系类型
        relation_types = {}
        total_relations = 0
        
        for entity_id, relations in self.mind_map._spatial_graph.items():
            for other_id, rel_data in relations.items():
                rel_type = rel_data['relation']
                relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
                total_relations += 1
        
        # 测试方向查询
        if self.mind_map.entity_count >= 2:
            e1 = self.mind_map.entities[0]
            e2 = self.mind_map.entities[1]
            relations = self.mind_map.get_relations(e1.entity_id)
            
            details = {
                "total_relations": total_relations,
                "relation_types": relation_types,
                "sample_query": {
                    "from": e1.entity_id,
                    "to": e2.entity_id,
                    "relation": relations.get(e2.entity_id, {}).get('relation', 'N/A'),
                    "distance": relations.get(e2.entity_id, {}).get('distance', 'N/A'),
                },
            }
        else:
            details = {
                "total_relations": total_relations,
                "relation_types": relation_types,
            }
        
        if total_relations == 0 and self.mind_map.entity_count > 1:
            return False, "未建立空间关系", details
        
        return True, f"空间关系: {total_relations} 对, 类型: {list(relation_types.keys())}", details
    
    def test_object_appearance_order(self) -> Tuple[bool, str, Dict]:
        """测试物体出现顺序 (VSIBench: obj_appearance_order)"""
        if self.mind_map is None:
            return False, "心智地图未构建", {}
        
        # 按首次出现帧排序
        entities_by_appearance = sorted(
            self.mind_map.entities,
            key=lambda e: e.first_seen_frame if e.first_seen_frame >= 0 else float('inf')
        )
        
        appearance_order = [
            {
                "entity_id": e.entity_id,
                "first_seen": e.first_seen_frame,
                "position": [round(x, 2) for x in e.centroid.tolist()],
            }
            for e in entities_by_appearance[:10]
        ]
        
        details = {
            "appearance_order": appearance_order,
            "entities_with_timestamp": len([e for e in self.mind_map.entities if e.first_seen_frame >= 0]),
        }
        
        # 验证时序信息
        if not any(e.first_seen_frame >= 0 for e in self.mind_map.entities):
            return False, "无时序信息 - 需要多帧积分", details
        
        return True, f"时序信息可用: {details['entities_with_timestamp']} 实体有时间戳", details
    
    def test_trajectory_visibility(self) -> Tuple[bool, str, Dict]:
        """测试轨迹可见性 (VSIBench: route_planning)"""
        if self.mind_map is None:
            return False, "心智地图未构建", {}
        
        trajectory = self.mind_map.trajectory
        
        if not trajectory:
            return False, "无轨迹数据", {}
        
        # 分析可见性
        visibility_stats = []
        for tp in trajectory[:10]:  # 前 10 帧
            visibility_stats.append({
                "frame": tp.frame_id,
                "position": [round(x, 2) for x in tp.position.tolist()],
                "visible_count": len(tp.visible_entities),
            })
        
        # 测试可见性查询
        if trajectory:
            mid_frame = len(trajectory) // 2
            visible_at_mid = self.mind_map.query_visible_at_frame(mid_frame)
        else:
            visible_at_mid = []
        
        details = {
            "trajectory_length": len(trajectory),
            "visibility_stats": visibility_stats,
            "visible_at_mid_frame": [e.entity_id for e in visible_at_mid],
        }
        
        return True, f"轨迹: {len(trajectory)} 点, 中点可见: {len(visible_at_mid)} 物体", details
    
    def test_distance_estimation(self) -> Tuple[bool, str, Dict]:
        """测试距离估计 (VSIBench: object_abs_distance)"""
        if self.mind_map is None:
            return False, "心智地图未构建", {}
        
        entities = self.mind_map.entities
        if len(entities) < 2:
            return False, "实体不足，无法测试距离", {}
        
        # 计算实体间距离
        distances = []
        for i, e1 in enumerate(entities[:5]):
            for e2 in entities[i+1:6]:
                dist = e1.distance_to(e2)
                distances.append({
                    "from": e1.entity_id,
                    "to": e2.entity_id,
                    "distance_m": round(dist, 2),
                })
        
        # 统计距离分布
        all_distances = [d["distance_m"] for d in distances]
        
        details = {
            "sample_distances": distances[:10],
            "min_distance": min(all_distances) if all_distances else 0,
            "max_distance": max(all_distances) if all_distances else 0,
            "mean_distance": round(np.mean(all_distances), 2) if all_distances else 0,
        }
        
        return True, f"距离估计: {len(distances)} 对, 范围 {details['min_distance']}-{details['max_distance']}m", details
    
    def test_nearby_query(self) -> Tuple[bool, str, Dict]:
        """测试附近物体查询"""
        if self.mind_map is None:
            return False, "心智地图未构建", {}
        
        entities = self.mind_map.entities
        if not entities:
            return False, "无实体可查询", {}
        
        # 选择第一个实体查询附近
        target = entities[0]
        nearby = self.mind_map.query_nearby(target.entity_id, radius=2.0)
        
        details = {
            "target": target.entity_id,
            "target_position": [round(x, 2) for x in target.centroid.tolist()],
            "radius": 2.0,
            "nearby_count": len(nearby),
            "nearby_entities": [
                {
                    "id": e.entity_id,
                    "distance": round(e.distance_to(target), 2),
                }
                for e in nearby[:5]
            ],
        }
        
        return True, f"附近查询: {target.entity_id} 周围 2m 内有 {len(nearby)} 物体", details
    
    # ==================== LLM Prompt 生成测试 ====================
    
    def test_prompt_generation(self) -> Tuple[bool, str, Dict]:
        """测试 LLM Prompt 生成"""
        if self.mind_map is None:
            return False, "心智地图未构建", {}
        
        # YAML 格式
        yaml_prompt = self.mind_map.to_prompt(format='yaml')
        
        # Markdown 格式
        md_prompt = self.mind_map.to_prompt(format='markdown')
        
        # 保存到文件
        yaml_path = os.path.join(self.output_dir, "mind_map_prompt.yaml")
        md_path = os.path.join(self.output_dir, "mind_map_prompt.md")
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_prompt)
        
        with open(md_path, 'w') as f:
            f.write(md_prompt)
        
        details = {
            "yaml_length": len(yaml_prompt),
            "md_length": len(md_prompt),
            "yaml_preview": yaml_prompt[:500] + "..." if len(yaml_prompt) > 500 else yaml_prompt,
            "saved_to": [yaml_path, md_path],
        }
        
        # 验证 Prompt 质量
        if "Entities:" not in yaml_prompt:
            return False, "YAML Prompt 缺少 Entities", details
        
        return True, f"Prompt 生成: YAML {len(yaml_prompt)} 字符, MD {len(md_prompt)} 字符", details
    
    def test_json_export(self) -> Tuple[bool, str, Dict]:
        """测试 JSON 导出"""
        if self.mind_map is None:
            return False, "心智地图未构建", {}
        
        # 导出到 JSON
        json_path = os.path.join(self.output_dir, "mind_map.json")
        self.mind_map.save(json_path)
        
        # 重新加载验证
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        details = {
            "json_path": json_path,
            "keys": list(data.keys()),
            "entity_count": data.get('entity_count', 0),
            "file_size_kb": round(os.path.getsize(json_path) / 1024, 2),
        }
        
        return True, f"JSON 导出: {details['file_size_kb']} KB", details
    
    # ==================== 性能测试 ====================
    
    def test_large_scale_performance(self) -> Tuple[bool, str, Dict]:
        """测试大规模点云处理性能"""
        from core.voxel_map import SparseVoxelMap
        
        # 生成大规模模拟点云
        num_points = 1_000_000
        logger.info(f"生成 {num_points:,} 点的模拟点云...")
        
        np.random.seed(42)
        points = np.random.randn(num_points, 3) * 10  # 10m 范围
        colors = np.random.rand(num_points, 3)
        
        # 测试体素化性能
        voxel_sizes = [0.05, 0.1, 0.2]
        perf_results = {}
        
        for voxel_size in voxel_sizes:
            voxel_map = SparseVoxelMap(voxel_size=voxel_size)
            
            start = time.time()
            voxel_map.integrate_points(points, colors)
            duration = time.time() - start
            
            perf_results[f"{voxel_size}m"] = {
                "num_voxels": voxel_map.num_voxels,
                "time_seconds": round(duration, 3),
                "points_per_second": round(num_points / duration, 0),
            }
        
        details = {
            "num_points": num_points,
            "performance": perf_results,
        }
        
        # 验证性能
        if perf_results["0.1m"]["time_seconds"] > 60:
            return False, f"体素化过慢: {perf_results['0.1m']['time_seconds']}s", details
        
        return True, f"100万点体素化: {perf_results['0.1m']['time_seconds']}s", details
    
    def test_memory_efficiency(self) -> Tuple[bool, str, Dict]:
        """测试内存效率"""
        import gc
        
        if self.scene is None:
            return False, "场景未加载", {}
        
        # 测试不同表示方式的内存占用
        gc.collect()
        
        # 原始点云内存
        point_cloud_bytes = self.scene.point_cloud.nbytes if self.scene.point_cloud is not None else 0
        colors_bytes = self.scene.colors.nbytes if self.scene.colors is not None else 0
        raw_memory = point_cloud_bytes + colors_bytes
        
        # 体素地图内存估计
        from core.voxel_map import SparseVoxelMap
        voxel_map = SparseVoxelMap(voxel_size=0.1)
        voxel_map.integrate_points(self.scene.point_cloud, colors=self.scene.colors)
        
        # 估算体素内存 (每个体素约 100 字节)
        voxel_memory = voxel_map.num_voxels * 100
        
        compression = raw_memory / max(voxel_memory, 1)
        
        details = {
            "raw_memory_mb": round(raw_memory / 1024 / 1024, 2),
            "voxel_memory_mb": round(voxel_memory / 1024 / 1024, 2),
            "compression_ratio": round(compression, 2),
            "num_points": self.scene.num_points,
            "num_voxels": voxel_map.num_voxels,
        }
        
        return True, f"内存压缩比: {compression:.1f}x ({details['raw_memory_mb']}MB -> {details['voxel_memory_mb']}MB)", details
    
    # ==================== 主测试流程 ====================
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("=" * 70)
        logger.info("MindMapV2 完整测试开始")
        logger.info("=" * 70)
        
        # 基础功能测试
        logger.info("\n📌 基础功能测试")
        logger.info("-" * 50)
        
        result = self.run_test(self.test_scene_loading, "场景加载")
        self.log_result(result)
        
        result = self.run_test(self.test_voxelization, "体素化 (Step 1)")
        self.log_result(result)
        
        result = self.run_test(self.test_connected_components, "连通域分析 (Step 2)")
        self.log_result(result)
        
        result = self.run_test(self.test_mind_map_building, "心智地图构建 (完整流程)")
        self.log_result(result)
        
        # VSIBench 任务模拟测试
        logger.info("\n📌 VSIBench 任务模拟测试")
        logger.info("-" * 50)
        
        result = self.run_test(self.test_object_counting, "物体计数 (object_counting)")
        self.log_result(result)
        
        result = self.run_test(self.test_spatial_relations, "空间关系 (object_rel_direction/distance)")
        self.log_result(result)
        
        result = self.run_test(self.test_object_appearance_order, "出现顺序 (obj_appearance_order)")
        self.log_result(result)
        
        result = self.run_test(self.test_trajectory_visibility, "轨迹可见性 (route_planning)")
        self.log_result(result)
        
        result = self.run_test(self.test_distance_estimation, "距离估计 (object_abs_distance)")
        self.log_result(result)
        
        result = self.run_test(self.test_nearby_query, "附近物体查询")
        self.log_result(result)
        
        # LLM Prompt 生成测试
        logger.info("\n📌 LLM Prompt 生成测试")
        logger.info("-" * 50)
        
        result = self.run_test(self.test_prompt_generation, "Prompt 生成")
        self.log_result(result)
        
        result = self.run_test(self.test_json_export, "JSON 导出")
        self.log_result(result)
        
        # 性能测试
        logger.info("\n📌 性能测试")
        logger.info("-" * 50)
        
        result = self.run_test(self.test_large_scale_performance, "大规模点云处理")
        self.log_result(result)
        
        result = self.run_test(self.test_memory_efficiency, "内存效率")
        self.log_result(result)
        
        # 汇总报告
        self.print_summary()
        
        # 保存详细报告
        self.save_report()
    
    def print_summary(self):
        """打印测试汇总"""
        logger.info("\n" + "=" * 70)
        logger.info("测试汇总")
        logger.info("=" * 70)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_time = sum(r.duration for r in self.results)
        
        logger.info(f"通过: {passed}/{len(self.results)}")
        logger.info(f"失败: {failed}/{len(self.results)}")
        logger.info(f"总耗时: {total_time:.2f}s")
        
        if failed > 0:
            logger.info("\n失败的测试:")
            for r in self.results:
                if not r.passed:
                    logger.info(f"  ❌ {r.name}: {r.message}")
        
        # VSIBench 任务支持评估
        logger.info("\n" + "-" * 50)
        logger.info("VSIBench 任务支持评估:")
        logger.info("-" * 50)
        
        vsibench_tasks = {
            "object_counting": "物体计数 (object_counting)",
            "object_rel_direction": "空间关系 (object_rel_direction/distance)",
            "obj_appearance_order": "出现顺序 (obj_appearance_order)",
            "route_planning": "轨迹可见性 (route_planning)",
            "object_abs_distance": "距离估计 (object_abs_distance)",
        }
        
        for task_id, test_name in vsibench_tasks.items():
            result = next((r for r in self.results if test_name in r.name), None)
            if result:
                status = "✅ 支持" if result.passed else "❌ 不支持"
                logger.info(f"  {status} | {task_id}")
            else:
                logger.info(f"  ⚠️ 未测试 | {task_id}")
    
    def save_report(self):
        """保存详细报告"""
        report = {
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                "total_time": sum(r.duration for r in self.results),
            },
            "tests": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }
        
        report_path = os.path.join(self.output_dir, "test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n详细报告已保存: {report_path}")


def main():
    """主函数"""
    # 检查 GPU
    import torch
    if torch.cuda.is_available():
        logger.info(f"可用 GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("未检测到 GPU，将使用 CPU")
    
    # 运行测试
    tester = MindMapV2Tester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
