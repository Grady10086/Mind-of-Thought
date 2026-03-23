#!/usr/bin/env python3
"""
合并和比较V10三个优化方案的结果
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_results(glob_pattern):
    """加载所有GPU的结果并合并"""
    results = []
    for path in Path('.').glob(glob_pattern):
        try:
            with open(path) as f:
                data = json.load(f)
                results.extend(data)
                print(f"  Loaded {path}: {len(data)} samples")
        except Exception as e:
            print(f"  Error loading {path}: {e}")
    return results

def analyze_scheme(name, results):
    """分析一个scheme的结果"""
    if not results:
        return None
    
    stats = {
        'name': name,
        'n_total': len(results),
        'overall': np.mean([r['score'] for r in results]),
        'v7_baseline': np.mean([r['v7_vl_score'] for r in results]),
    }
    
    # 按任务类型分析
    by_type = defaultdict(list)
    for r in results:
        by_type[r['question_type']].append(r)
    
    # 关键任务
    key_tasks = ['route_planning', 'object_abs_distance', 'object_rel_direction', 
                 'room_size_estimation', 'object_counting']
    
    for task in key_tasks:
        if task in by_type:
            task_results = by_type[task]
            stats[f'{task}_n'] = len(task_results)
            stats[f'{task}_v10'] = np.mean([r['score'] for r in task_results])
            stats[f'{task}_v7'] = np.mean([r['v7_vl_score'] for r in task_results])
            stats[f'{task}_delta'] = stats[f'{task}_v10'] - stats[f'{task}_v7']
        else:
            stats[f'{task}_n'] = 0
            stats[f'{task}_v10'] = 0
            stats[f'{task}_v7'] = 0
            stats[f'{task}_delta'] = 0
    
    # V10特性统计
    stats['evo_rate'] = np.mean([1 if r.get('grid_modified') else 0 for r in results]) * 100
    stats['coder_rate'] = np.mean([1 if r.get('coder_used') else 0 for r in results]) * 100
    stats['verify_rate'] = np.mean([1 if r.get('verify_triggered') else 0 for r in results]) * 100
    stats['avg_vl_calls'] = np.mean([r.get('vl_calls', 0) for r in results])
    
    return stats

def print_comparison(all_stats):
    """打印对比表格"""
    print("\n" + "="*100)
    print("V10 Schemes Comparison Results")
    print("="*100)
    
    # 总体对比
    print("\n【Overall Performance】")
    print(f"{'Scheme':<25} {'N':>5} {'V10':>7} {'V7':>7} {'Δ':>7} {'VL#':>5} {'Evo%':>6} {'Vfy%':>6}")
    print("-" * 75)
    for s in all_stats:
        if s:
            marker = "★" if s['overall'] == max(st['overall'] for st in all_stats if st) else " "
            print(f"{marker} {s['name']:<23} {s['n_total']:>5} {s['overall']:>7.3f} {s['v7_baseline']:>7.3f} "
                  f"{s['overall']-s['v7_baseline']:>+7.3f} {s['avg_vl_calls']:>5.1f} {s['evo_rate']:>5.0f}% {s['verify_rate']:>5.0f}%")
    
    # Route Planning对比
    print("\n【Route Planning Performance】")
    print(f"{'Scheme':<25} {'N':>5} {'V10':>7} {'V7':>7} {'Δ':>7}")
    print("-" * 55)
    for s in all_stats:
        if s and s.get('route_planning_n', 0) > 0:
            marker = "★" if s['route_planning_v10'] == max(st['route_planning_v10'] for st in all_stats if st and st.get('route_planning_n',0)>0) else " "
            print(f"{marker} {s['name']:<23} {s['route_planning_n']:>5} {s['route_planning_v10']:>7.3f} "
                  f"{s['route_planning_v7']:>7.3f} {s['route_planning_delta']:>+7.3f}")
    
    # Abs Distance对比
    print("\n【Object Abs Distance Performance】")
    print(f"{'Scheme':<25} {'N':>5} {'V10':>7} {'V7':>7} {'Δ':>7}")
    print("-" * 55)
    for s in all_stats:
        if s and s.get('object_abs_distance_n', 0) > 0:
            marker = "★" if s['object_abs_distance_v10'] == max(st['object_abs_distance_v10'] for st in all_stats if st and st.get('object_abs_distance_n',0)>0) else " "
            print(f"{marker} {s['name']:<23} {s['object_abs_distance_n']:>5} {s['object_abs_distance_v10']:>7.3f} "
                  f"{s['object_abs_distance_v7']:>7.3f} {s['object_abs_distance_delta']:>+7.3f}")
    
    # 其他任务
    print("\n【Other Key Tasks】")
    other_tasks = ['object_rel_direction', 'room_size_estimation', 'object_counting']
    for task in other_tasks:
        task_short = task.replace('object_', '').replace('_estimation', '').replace('_', ' ')
        print(f"\n{task_short.title()}:")
        for s in all_stats:
            if s:
                n_key = f'{task}_n'
                v10_key = f'{task}_v10'
                v7_key = f'{task}_v7'
                delta_key = f'{task}_delta'
                if s.get(n_key, 0) > 0:
                    print(f"  {s['name']:<23}: V10={s[v10_key]:.3f} V7={s[v7_key]:.3f} Δ={s[delta_key]:+.3f} (n={s[n_key]})")
    
    print("\n" + "="*100)
    
    # 推荐
    print("\n【Recommendation】")
    if all_stats:
        best = max((s for s in all_stats if s), key=lambda x: x['overall'])
        print(f"Best Overall: {best['name']} ({best['overall']:.3f})")
        
        best_route = max((s for s in all_stats if s and s.get('route_planning_n',0)>0), 
                        key=lambda x: x['route_planning_v10'])
        print(f"Best Route: {best_route['name']} ({best_route['route_planning_v10']:.3f})")
        
        best_dist = max((s for s in all_stats if s and s.get('object_abs_distance_n',0)>0), 
                       key=lambda x: x['object_abs_distance_v10'])
        print(f"Best AbsDist: {best_dist['name']} ({best_dist['object_abs_distance_v10']:.3f})")
    
    print("="*100)

def main():
    schemes = [
        ('Baseline V10', 'outputs/agentic_pipeline_v10_full/gpu*/detailed_results.json'),
        ('Scheme 1 (Route Opt)', 'outputs/agentic_pipeline_v10_route_full/gpu*/detailed_results.json'),
        ('Scheme 2 (Scale Adj)', 'outputs/agentic_pipeline_v10_scale_full/gpu*/detailed_results.json'),
        ('Scheme 3 (Route Verify)', 'outputs/agentic_pipeline_v10_verify_full/gpu*/detailed_results.json'),
    ]
    
    print("Loading results...")
    all_stats = []
    for name, pattern in schemes:
        print(f"\n{name}:")
        results = load_results(pattern)
        if results:
            stats = analyze_scheme(name, results)
            all_stats.append(stats)
            
            # 保存合并结果
            output_dir = Path(pattern.replace('/gpu*/detailed_results.json', '/merged'))
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'detailed_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved merged: {output_dir}/detailed_results.json ({len(results)} samples)")
        else:
            print(f"  No results found")
            all_stats.append(None)
    
    # 打印对比
    print_comparison(all_stats)

if __name__ == '__main__':
    main()
