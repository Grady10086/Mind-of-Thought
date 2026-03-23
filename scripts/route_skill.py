#!/usr/bin/env python3
"""
RouteSkill: 专业化的路径规划技能
无偏优化: 仅基于几何、物理约束、逻辑一致性
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import re


class RouteSkill:
    """
    Route规划专业化技能
    
    核心能力:
    1. 空间几何推理: 计算转向角度、朝向向量
    2. 物理约束检查: 路径可行性、障碍物避让
    3. 逻辑一致性: 验证转向序列的合理性
    """
    
    def __init__(self, grid):
        self.grid = grid
        self.obstacles = self._build_obstacle_map()
    
    def _build_obstacle_map(self) -> Dict:
        """构建障碍物地图 (排除起点、终点相关物体)"""
        obstacles = {}
        for eid, e in self.grid.entities.items():
            pos = np.array(e.position_3d, dtype=float)
            obstacles[eid] = {
                'pos': pos,
                'pos_2d': np.array([pos[0], pos[2]]),
                'bbox_size': getattr(e, 'bbox_size', [0.5, 0.5, 0.5])
            }
        return obstacles
    
    def parse_question(self, question: str) -> Optional[Dict]:
        """
        解析路径规划问题
        
        Returns:
            {
                'start_name': str,
                'facing_name': str,
                'target_name': str,
                'waypoints': List[(type, name)],
                'n_fill_ins': int
            }
        """
        q = question.lower()
        
        # 解析起点和朝向
        m_start = re.search(r'beginning at (?:the )?(.+?)(?:\s+and\s+)?facing (?:the )?(.+?)\.', q)
        if not m_start:
            return None
        
        start_name = m_start.group(1).strip()
        facing_name = m_start.group(2).strip()
        
        # 解析目标
        m_target = re.search(r'navigate to (?:the )?(.+?)(?:\.|,)', q)
        target_name = m_target.group(1).strip() if m_target else None
        
        # 解析步骤序列
        steps = re.findall(r'\d+\.\s+(.+?)(?=\d+\.|You have reached|$)', q, re.DOTALL)
        
        waypoints = []
        for step in steps:
            step = step.strip().rstrip('.')
            if 'please fill in' in step:
                waypoints.append(('fill_in', None))
            elif 'go forward' in step:
                m_fwd = re.search(r'go forward\s+until\s+(?:the\s+)?(.+?)(?:\s+is\s+on|\s*$)', step)
                if m_fwd:
                    wp_name = m_fwd.group(1).strip().rstrip('.')
                    waypoints.append(('go_forward', wp_name))
                else:
                    waypoints.append(('go_forward', None))
        
        n_fill_ins = sum(1 for wtype, _ in waypoints if wtype == 'fill_in')
        
        return {
            'start_name': start_name,
            'facing_name': facing_name,
            'target_name': target_name,
            'waypoints': waypoints,
            'n_fill_ins': n_fill_ins
        }
    
    def get_entity_position(self, name: str) -> Optional[np.ndarray]:
        """获取实体3D位置"""
        entities = self.grid.get_by_category(name)
        if entities:
            return np.array(entities[0].position_3d, dtype=float)
        return None
    
    def compute_facing_vector(self, start_pos: np.ndarray, facing_pos: np.ndarray) -> np.ndarray:
        """计算朝向向量 (XZ平面)"""
        vec = np.array([facing_pos[0] - start_pos[0], facing_pos[2] - start_pos[2]])
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return np.array([0.0, 1.0])  # 默认朝向
        return vec / norm
    
    def turn_left(self, facing: np.ndarray) -> np.ndarray:
        """向左转90度"""
        return np.array([-facing[1], facing[0]])
    
    def turn_right(self, facing: np.ndarray) -> np.ndarray:
        """向右转90度"""
        return np.array([facing[1], -facing[0]])
    
    def turn_back(self, facing: np.ndarray) -> np.ndarray:
        """向后转180度"""
        return -facing
    
    def apply_turn(self, facing: np.ndarray, turn_cmd: str) -> np.ndarray:
        """应用转向命令"""
        turn_cmd = turn_cmd.lower()
        if 'back' in turn_cmd:
            return self.turn_back(facing)
        elif 'left' in turn_cmd:
            return self.turn_left(facing)
        elif 'right' in turn_cmd:
            return self.turn_right(facing)
        return facing
    
    def angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个向量之间的角度 (弧度)"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def simulate_route(
        self,
        start_pos: np.ndarray,
        init_facing: np.ndarray,
        waypoints: List[Tuple],
        turns: List[str],
        wp_positions: Dict[str, np.ndarray]
    ) -> Dict:
        """
        模拟完整路径
        
        Returns:
            {
                'valid': bool,
                'final_pos': np.ndarray,
                'final_facing': np.ndarray,
                'turn_validity': List[bool],  # 每个fill_in转向是否合理
                'path_segments': List[Dict],  # 每段路径信息
                'score': float
            }
        """
        current_pos = start_pos.copy()
        current_facing = init_facing.copy()
        turn_idx = 0
        path_segments = []
        turn_validity = []
        total_score = 0.0
        
        for wi, (wtype, wname) in enumerate(waypoints):
            if wtype == 'fill_in':
                turn_cmd = turns[turn_idx]
                turn_idx += 1
                
                # 应用转向
                new_facing = self.apply_turn(current_facing, turn_cmd)
                
                # 检查转向后是否指向下一个waypoint
                next_wp_name = None
                for future_type, future_name in waypoints[wi+1:]:
                    if future_type == 'go_forward' and future_name:
                        next_wp_name = future_name
                        break
                
                turn_valid = True
                if next_wp_name and next_wp_name in wp_positions:
                    next_pos = wp_positions[next_wp_name]
                    target_dir = np.array([next_pos[0] - current_pos[0], 
                                          next_pos[2] - current_pos[2]])
                    if np.linalg.norm(target_dir) > 1e-8:
                        target_dir = target_dir / np.linalg.norm(target_dir)
                        angle = self.angle_between(new_facing, target_dir)
                        # 转向后应该大致朝向目标 (< 90度)
                        turn_valid = angle < np.pi / 2
                        
                        # 根据对齐程度给分
                        alignment_score = np.cos(angle)  # 1 = 完美对齐, 0 = 90度, -1 = 反向
                        total_score += max(0, alignment_score)
                
                turn_validity.append(turn_valid)
                current_facing = new_facing
            
            elif wtype == 'go_forward' and wname:
                if wname in wp_positions:
                    target_pos = wp_positions[wname]
                    
                    # 检查移动方向是否与当前朝向一致
                    move_dir = np.array([target_pos[0] - current_pos[0],
                                        target_pos[2] - current_pos[2]])
                    move_dist = np.linalg.norm(move_dir)
                    
                    if move_dist > 1e-8:
                        move_dir = move_dir / move_dist
                        angle = self.angle_between(current_facing, move_dir)
                        
                        # 检查路径是否朝向目标
                        path_segment = {
                            'from': current_pos.copy(),
                            'to': target_pos,
                            'distance': move_dist,
                            'angle_diff': angle,
                            'aligned': angle < np.pi / 3  # < 60度视为对齐
                        }
                        path_segments.append(path_segment)
                        
                        # 对齐给分
                        if path_segment['aligned']:
                            total_score += 1.0
                        else:
                            total_score -= 0.5  # 惩罚反向移动
                    
                    current_pos = target_pos.copy()
                    # 更新朝向为移动方向
                    if move_dist > 1e-8:
                        current_facing = move_dir
        
        # 总体有效性：所有转向都合理
        valid = all(turn_validity) if turn_validity else True
        
        return {
            'valid': valid,
            'final_pos': current_pos,
            'final_facing': current_facing,
            'turn_validity': turn_validity,
            'path_segments': path_segments,
            'score': total_score
        }
    
    def check_endpoint_reachability(
        self,
        final_pos: np.ndarray,
        target_name: str,
        threshold: float = 1.0
    ) -> Tuple[bool, float]:
        """
        检查终点是否可达
        
        Returns:
            (是否可达, 距离)
        """
        target_pos = self.get_entity_position(target_name)
        if target_pos is None:
            return False, float('inf')
        
        dist = np.linalg.norm(final_pos - target_pos)
        return dist < threshold, dist
    
    def evaluate_option(
        self,
        option_content: str,
        parsed: Dict,
        start_pos: np.ndarray,
        init_facing: np.ndarray,
        wp_positions: Dict[str, np.ndarray]
    ) -> Dict:
        """
        评估单个选项
        
        Returns:
            {
                'letter': str,
                'score': float,
                'valid': bool,
                'details': str
            }
        """
        turns = [t.strip().lower() for t in option_content.split(',')]
        
        # 检查fill_in数量是否匹配
        if len(turns) != parsed['n_fill_ins']:
            return {
                'letter': option_content[0] if option_content else '?',
                'score': -100,
                'valid': False,
                'details': f"fill_in mismatch: {len(turns)} vs {parsed['n_fill_ins']}"
            }
        
        # 模拟路径
        sim_result = self.simulate_route(
            start_pos, init_facing,
            parsed['waypoints'], turns, wp_positions
        )
        
        # 检查终点可达性
        target_reachable = True
        target_dist = 0.0
        if parsed['target_name']:
            reachable, dist = self.check_endpoint_reachability(
                sim_result['final_pos'], parsed['target_name']
            )
            target_reachable = reachable
            target_dist = dist
            
            if reachable:
                sim_result['score'] += 2.0
            else:
                sim_result['score'] -= 1.0
        
        valid = sim_result['valid'] and target_reachable
        
        details = (f"score={sim_result['score']:.2f}, "
                  f"valid_turns={all(sim_result['turn_validity'])}, "
                  f"target_dist={target_dist:.2f}m")
        
        return {
            'letter': option_content[0] if option_content else '?',
            'score': sim_result['score'],
            'valid': valid,
            'details': details
        }
    
    def solve(self, question: str, options: List[str]) -> Tuple[str, str]:
        """
        解决路径规划问题
        
        Returns:
            (答案字母, 推理过程)
        """
        if not options:
            return "A", "no options"
        
        # 解析问题
        parsed = self.parse_question(question)
        if not parsed:
            return "A", "failed to parse question"
        
        # 获取起点和朝向位置
        start_pos = self.get_entity_position(parsed['start_name'])
        facing_pos = self.get_entity_position(parsed['facing_name'])
        
        if start_pos is None or facing_pos is None:
            missing = []
            if start_pos is None:
                missing.append(f"start='{parsed['start_name']}'")
            if facing_pos is None:
                missing.append(f"facing='{parsed['facing_name']}'")
            return "A", f"not found: {', '.join(missing)}"
        
        # 计算初始朝向
        init_facing = self.compute_facing_vector(start_pos, facing_pos)
        
        # 收集所有waypoint位置
        wp_positions = {}
        for wtype, wname in parsed['waypoints']:
            if wname and wname not in wp_positions:
                pos = self.get_entity_position(wname)
                if pos is not None:
                    wp_positions[wname] = pos
        
        # 评估所有选项
        evaluations = []
        for opt in options:
            opt_content = re.sub(r'^[A-D]\.\s*', '', opt).strip()
            eval_result = self.evaluate_option(
                opt_content, parsed, start_pos, init_facing, wp_positions
            )
            eval_result['letter'] = opt[0]  # 使用选项的字母
            evaluations.append(eval_result)
        
        # 选择最佳选项
        # 优先选择有效的选项中得分最高的
        valid_evals = [e for e in evaluations if e['valid']]
        
        if valid_evals:
            best = max(valid_evals, key=lambda x: x['score'])
        else:
            # 如果没有有效的，选择得分最高的（即使无效）
            best = max(evaluations, key=lambda x: x['score'])
        
        # 构建推理信息
        details = "; ".join([
            f"{e['letter']}: {e['details']}"
            for e in evaluations
        ])
        
        reasoning = (f"RouteSkill: {parsed['n_fill_ins']} fill-ins, "
                    f"waypoints={len(parsed['waypoints'])}, "
                    f"chosen={best['letter']}(score={best['score']:.2f}), "
                    f"evaluations=[{details}]")
        
        return best['letter'], reasoning
