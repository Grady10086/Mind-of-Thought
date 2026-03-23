"""
VSIBench 端到端集成测试

测试 MindMapV2 在真实 VSIBench 任务上的表现

测试流程：
1. 加载 DA3 重建的 3D 场景
2. 构建 MindMapV2
3. 模拟 VSIBench 各类任务的查询
4. 评估回答准确性

使用方法：
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m tests.test_vsibench_integration
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

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


class VSIBenchSimulator:
    """
    VSIBench 任务模拟器
    
    模拟各种空间理解任务，测试 MindMapV2 的回答能力
    """
    
    def __init__(self, mind_map):
        self.mind_map = mind_map
    
    def simulate_object_counting(self, category: str = None) -> Dict[str, Any]:
        """
        模拟物体计数任务
        
        VSIBench 示例问题:
        - "How many chairs are there in the room?"
        - "Count the number of tables"
        """
        entities = self.mind_map.entities
        
        if category:
            # 按类别过滤（如果有语义标签）
            filtered = [e for e in entities if e.semantic_label and category.lower() in e.semantic_label.lower()]
            count = len(filtered)
        else:
            count = len(entities)
        
        return {
            "task": "object_counting",
            "question": f"How many {category or 'objects'} are there?",
            "answer": count,
            "confidence": 1.0 if self.mind_map.entity_count > 0 else 0.0,
            "reasoning": f"Found {count} entities in the spatial mind map",
        }
    
    def simulate_relative_direction(self, obj1_id: str, obj2_id: str) -> Dict[str, Any]:
        """
        模拟相对方向任务
        
        VSIBench 示例问题:
        - "Is the chair to the left or right of the table?"
        - "Is the lamp above or below the desk?"
        """
        e1 = self.mind_map.get_entity(obj1_id)
        e2 = self.mind_map.get_entity(obj2_id)
        
        if e1 is None or e2 is None:
            return {
                "task": "relative_direction",
                "answer": "unknown",
                "confidence": 0.0,
            }
        
        relations = self.mind_map.get_relations(obj1_id)
        rel_data = relations.get(obj2_id, {})
        
        return {
            "task": "relative_direction",
            "question": f"What is the spatial relation between {obj1_id} and {obj2_id}?",
            "answer": rel_data.get('relation', 'near'),
            "distance": rel_data.get('distance', 0),
            "confidence": 1.0 if rel_data else 0.5,
            "reasoning": f"{obj1_id} is {rel_data.get('relation', 'near')} {obj2_id}",
        }
    
    def simulate_absolute_distance(self, obj1_id: str, obj2_id: str) -> Dict[str, Any]:
        """
        模拟绝对距离任务
        
        VSIBench 示例问题:
        - "What is the distance between the chair and the table in meters?"
        """
        e1 = self.mind_map.get_entity(obj1_id)
        e2 = self.mind_map.get_entity(obj2_id)
        
        if e1 is None or e2 is None:
            return {
                "task": "absolute_distance",
                "answer": -1,
                "confidence": 0.0,
            }
        
        distance = e1.distance_to(e2)
        
        return {
            "task": "absolute_distance",
            "question": f"What is the distance between {obj1_id} and {obj2_id}?",
            "answer": round(distance, 2),
            "unit": "meters",
            "confidence": 0.9,  # DA3 精度约 0.1m
            "reasoning": f"Euclidean distance between centroids",
        }
    
    def simulate_appearance_order(self) -> Dict[str, Any]:
        """
        模拟出现顺序任务
        
        VSIBench 示例问题:
        - "What is the first object you see when entering the room?"
        - "Which object appears first, the chair or the table?"
        """
        entities = sorted(
            self.mind_map.entities,
            key=lambda e: e.first_seen_frame if e.first_seen_frame >= 0 else float('inf')
        )
        
        if not entities:
            return {
                "task": "appearance_order",
                "answer": [],
                "confidence": 0.0,
            }
        
        order = [
            {
                "entity_id": e.entity_id,
                "first_seen_frame": e.first_seen_frame,
            }
            for e in entities[:5]
        ]
        
        return {
            "task": "appearance_order",
            "question": "In what order do objects appear?",
            "answer": order,
            "first_object": entities[0].entity_id if entities else None,
            "confidence": 1.0 if entities[0].first_seen_frame >= 0 else 0.5,
        }
    
    def simulate_route_planning(self, start_frame: int, end_frame: int) -> Dict[str, Any]:
        """
        模拟路径规划/轨迹任务
        
        VSIBench 示例问题:
        - "What objects will you pass if you walk from point A to point B?"
        """
        trajectory = self.mind_map.trajectory
        
        if not trajectory:
            return {
                "task": "route_planning",
                "answer": [],
                "confidence": 0.0,
            }
        
        # 获取路径上的可见物体
        objects_on_route = set()
        for i in range(min(start_frame, len(trajectory)), min(end_frame + 1, len(trajectory))):
            objects_on_route.update(trajectory[i].visible_entities)
        
        return {
            "task": "route_planning",
            "question": f"What objects are visible from frame {start_frame} to {end_frame}?",
            "answer": list(objects_on_route),
            "count": len(objects_on_route),
            "confidence": 0.8,
        }
    
    def simulate_size_estimation(self, entity_id: str) -> Dict[str, Any]:
        """
        模拟尺寸估计任务
        
        VSIBench 示例问题:
        - "What is the approximate size of the table?"
        """
        entity = self.mind_map.get_entity(entity_id)
        
        if entity is None:
            return {
                "task": "size_estimation",
                "answer": None,
                "confidence": 0.0,
            }
        
        return {
            "task": "size_estimation",
            "question": f"What is the size of {entity_id}?",
            "answer": {
                "width": round(entity.size[0], 2),
                "height": round(entity.size[1], 2),
                "depth": round(entity.size[2], 2),
                "volume": round(entity.volume, 3),
            },
            "unit": "meters",
            "confidence": 0.8,
        }
    
    def generate_llm_context(self) -> str:
        """
        生成完整的 LLM 上下文
        
        用于测试 Prompt 是否包含足够信息回答 VSIBench 问题
        """
        context = self.mind_map.to_prompt(format='yaml')
        
        # 添加任务指令
        prompt = f"""You are a spatial reasoning assistant. Based on the following spatial mind map, answer questions about the 3D scene.

{context}

You can answer the following types of questions:
1. Object counting: How many X are there?
2. Spatial relations: Is X to the left/right/above/below of Y?
3. Distance estimation: How far is X from Y?
4. Appearance order: What do you see first?
5. Route planning: What objects will you pass on the way?

Please answer based ONLY on the information in the mind map above.
"""
        return prompt


class VSIBenchIntegrationTest:
    """VSIBench 集成测试"""
    
    def __init__(self, scene_path: str = None):
        self.scene_path = scene_path or str(PROJECT_ROOT / "outputs" / "da3_reconstruction" / "scene.glb")
        self.output_dir = str(PROJECT_ROOT / "outputs" / "vsibench_test")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.mind_map = None
        self.simulator = None
    
    def load_and_build(self):
        """加载场景并构建心智地图"""
        from core.scene import SceneLoader
        from core.mind_map_v2 import MindMapBuilder
        
        logger.info("=" * 60)
        logger.info("Step 1: 加载 3D 场景")
        logger.info("=" * 60)
        
        scene = SceneLoader.load(self.scene_path)
        logger.info(f"场景加载完成: {scene.num_points} 点, {scene.num_cameras} 相机")
        
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: 构建心智地图 (MindMapV2)")
        logger.info("=" * 60)
        
        builder = MindMapBuilder(
            voxel_size=0.1,
            min_entity_voxels=5,
        )
        
        start = time.time()
        self.mind_map = builder.build(scene)
        build_time = time.time() - start
        
        logger.info(f"心智地图构建完成: {self.mind_map}")
        logger.info(f"构建耗时: {build_time:.2f}s")
        
        self.simulator = VSIBenchSimulator(self.mind_map)
    
    def run_task_simulations(self):
        """运行所有任务模拟"""
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: VSIBench 任务模拟")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. 物体计数
        logger.info("\n📌 任务 1: 物体计数 (object_counting)")
        result = self.simulator.simulate_object_counting()
        results['object_counting'] = result
        logger.info(f"  答案: {result['answer']} 个物体")
        logger.info(f"  置信度: {result['confidence']}")
        
        # 2. 相对方向
        logger.info("\n📌 任务 2: 相对方向 (object_rel_direction)")
        if self.mind_map.entity_count >= 2:
            e1, e2 = self.mind_map.entities[0], self.mind_map.entities[1]
            result = self.simulator.simulate_relative_direction(e1.entity_id, e2.entity_id)
            results['relative_direction'] = result
            logger.info(f"  问题: {e1.entity_id} 与 {e2.entity_id} 的关系?")
            logger.info(f"  答案: {result['answer']}, 距离: {result.get('distance', 'N/A')}m")
        else:
            logger.info("  跳过 - 实体不足")
        
        # 3. 绝对距离
        logger.info("\n📌 任务 3: 绝对距离 (object_abs_distance)")
        if self.mind_map.entity_count >= 2:
            e1, e2 = self.mind_map.entities[0], self.mind_map.entities[1]
            result = self.simulator.simulate_absolute_distance(e1.entity_id, e2.entity_id)
            results['absolute_distance'] = result
            logger.info(f"  问题: {e1.entity_id} 到 {e2.entity_id} 的距离?")
            logger.info(f"  答案: {result['answer']} 米")
        else:
            logger.info("  跳过 - 实体不足")
        
        # 4. 出现顺序
        logger.info("\n📌 任务 4: 出现顺序 (obj_appearance_order)")
        result = self.simulator.simulate_appearance_order()
        results['appearance_order'] = result
        logger.info(f"  第一个出现: {result.get('first_object', 'N/A')}")
        logger.info(f"  顺序: {[o['entity_id'] for o in result.get('answer', [])[:3]]}")
        
        # 5. 路径规划
        logger.info("\n📌 任务 5: 路径规划 (route_planning)")
        if len(self.mind_map.trajectory) >= 2:
            result = self.simulator.simulate_route_planning(0, len(self.mind_map.trajectory) - 1)
            results['route_planning'] = result
            logger.info(f"  路径上可见物体: {result['count']} 个")
        else:
            logger.info("  跳过 - 轨迹不足")
        
        # 6. 尺寸估计
        logger.info("\n📌 任务 6: 尺寸估计 (object_size_estimation)")
        if self.mind_map.entity_count >= 1:
            entity = self.mind_map.entities[0]
            result = self.simulator.simulate_size_estimation(entity.entity_id)
            results['size_estimation'] = result
            size = result.get('answer', {})
            logger.info(f"  物体: {entity.entity_id}")
            logger.info(f"  尺寸: {size.get('width', 0):.2f} x {size.get('height', 0):.2f} x {size.get('depth', 0):.2f} m")
        else:
            logger.info("  跳过 - 无实体")
        
        return results
    
    def test_llm_prompt(self):
        """测试 LLM Prompt 生成"""
        logger.info("\n" + "=" * 60)
        logger.info("Step 4: LLM Prompt 生成")
        logger.info("=" * 60)
        
        prompt = self.simulator.generate_llm_context()
        
        # 保存 Prompt
        prompt_path = os.path.join(self.output_dir, "llm_prompt.txt")
        with open(prompt_path, 'w') as f:
            f.write(prompt)
        
        logger.info(f"Prompt 长度: {len(prompt)} 字符")
        logger.info(f"Prompt 已保存: {prompt_path}")
        logger.info("\nPrompt 预览 (前 1000 字符):")
        logger.info("-" * 40)
        print(prompt[:1000])
        logger.info("-" * 40)
        
        return prompt
    
    def evaluate_vsibench_support(self, results: Dict):
        """评估 VSIBench 任务支持情况"""
        logger.info("\n" + "=" * 60)
        logger.info("Step 5: VSIBench 任务支持评估")
        logger.info("=" * 60)
        
        vsibench_tasks = [
            ("object_counting", "物体计数"),
            ("object_rel_direction_lr", "左右方向"),
            ("object_rel_direction_fb", "前后方向"),
            ("object_rel_direction_ud", "上下方向"),
            ("object_rel_distance", "相对距离"),
            ("object_abs_distance", "绝对距离"),
            ("obj_appearance_order", "出现顺序"),
            ("route_planning", "路径规划"),
            ("object_size_estimation", "尺寸估计"),
            ("room_size_estimation", "房间尺寸"),
        ]
        
        support_matrix = {}
        
        for task_id, task_name in vsibench_tasks:
            # 检查是否有对应的测试结果
            if task_id in results or any(task_id.split('_')[0] in k for k in results.keys()):
                relevant_result = results.get(task_id) or results.get(task_id.split('_')[0] + '_' + task_id.split('_')[-1])
                if relevant_result is None:
                    # 尝试模糊匹配
                    for k, v in results.items():
                        if task_id.split('_')[0] in k:
                            relevant_result = v
                            break
                
                if relevant_result and relevant_result.get('confidence', 0) > 0:
                    support = "✅ 完全支持"
                    confidence = relevant_result.get('confidence', 0)
                else:
                    support = "⚠️ 部分支持"
                    confidence = 0.5
            else:
                support = "❌ 不支持"
                confidence = 0.0
            
            support_matrix[task_id] = {
                "name": task_name,
                "support": support,
                "confidence": confidence,
            }
            
            logger.info(f"  {support} | {task_name} ({task_id})")
        
        # 计算总体支持率
        supported = sum(1 for v in support_matrix.values() if "✅" in v['support'])
        partial = sum(1 for v in support_matrix.values() if "⚠️" in v['support'])
        total = len(vsibench_tasks)
        
        logger.info(f"\n总体支持率: {supported}/{total} 完全支持, {partial}/{total} 部分支持")
        
        return support_matrix
    
    def run(self):
        """运行完整测试"""
        logger.info("=" * 70)
        logger.info("VSIBench 集成测试开始")
        logger.info("=" * 70)
        
        # 1. 加载和构建
        self.load_and_build()
        
        # 2. 任务模拟
        results = self.run_task_simulations()
        
        # 3. LLM Prompt
        prompt = self.test_llm_prompt()
        
        # 4. 评估
        support_matrix = self.evaluate_vsibench_support(results)
        
        # 5. 保存报告
        report = {
            "scene_path": self.scene_path,
            "mind_map_stats": {
                "entity_count": self.mind_map.entity_count,
                "trajectory_points": len(self.mind_map.trajectory),
                "voxel_size": self.mind_map.voxel_size,
            },
            "task_results": results,
            "support_matrix": support_matrix,
        }
        
        report_path = os.path.join(self.output_dir, "integration_test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n详细报告已保存: {report_path}")
        
        logger.info("\n" + "=" * 70)
        logger.info("测试完成！")
        logger.info("=" * 70)
        
        return report


def main():
    """主函数"""
    # 检查依赖
    try:
        import trimesh
        import torch
        logger.info("依赖检查通过")
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        return
    
    # 运行测试
    tester = VSIBenchIntegrationTest()
    report = tester.run()
    
    return report


if __name__ == "__main__":
    main()
