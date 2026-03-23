#!/usr/bin/env python3
"""
合并8GPU全量测试结果 + 生成V5总结报告
Usage: python scripts/merge_v5_results.py
"""
import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

# Default results dir, can be overridden by --input_dir
NUMERICAL_TASKS = ['object_counting', 'object_size_estimation', 'room_size_estimation', 'object_abs_distance']


def get_results_dir():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='outputs/agentic_pipeline_v5_debias_full')
    args = parser.parse_args()
    return PROJECT_ROOT / args.input_dir


def merge_results(results_dir):
    all_results = []
    gpu_dirs = sorted(results_dir.glob("gpu*"))
    
    if not gpu_dirs:
        print(f"No GPU result directories found in {results_dir}")
        sys.exit(1)
    
    for gpu_dir in gpu_dirs:
        result_file = gpu_dir / "detailed_results.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            print(f"  {gpu_dir.name}: {len(data)} samples")
            all_results.extend(data)
        else:
            print(f"  {gpu_dir.name}: MISSING detailed_results.json")
    
    print(f"\nTotal merged: {len(all_results)} samples from {len(gpu_dirs)} GPUs")
    
    # Deduplicate by (scene_name, question)
    seen = set()
    unique_results = []
    dups = 0
    for r in all_results:
        key = (r.get('scene_name', ''), r.get('question', ''))
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
        else:
            dups += 1
    if dups:
        print(f"  Removed {dups} duplicates → {len(unique_results)} unique")
    all_results = unique_results
    
    return all_results


def print_summary(all_results, results_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "=" * 140)
    print("Agentic Pipeline V5 — FULL TEST RESULTS")
    print(f"Architecture: Manager(VL) → Autonomous Tool Selection → VL Visual Reasoning")
    print(f"Design: No task-type routing. No difficulty-based strategies. Fully autonomous Manager.")
    print(f"Baselines: V7 VL=63.61%, V4 Pipeline=64.38%  |  Samples: {len(all_results)}")
    print("=" * 140)
    
    task_types = sorted(set(r['question_type'] for r in all_results))
    
    print(f"\n{'Task':<35} {'N':>5} {'V7':>6} {'V5':>6} {'Δ':>6}  {'VL#':>4} {'Evo%':>5} {'Cod%':>5} {'t/s':>5}")
    print(f"{'-'*35} {'-'*5} {'-'*6} {'-'*6} {'-'*6}  {'-'*4} {'-'*5} {'-'*5} {'-'*5}")
    
    all_v7, all_v5 = [], []
    by_task_summary = {}
    
    for qt in task_types:
        qr = [r for r in all_results if r['question_type'] == qt]
        v7 = np.mean([r.get('v7_vl_score', 0) for r in qr])
        v5 = np.mean([r['score'] for r in qr])
        d = v5 - v7
        vl = np.mean([r.get('vl_calls', 0) for r in qr])
        evo = np.mean([1 if r.get('grid_modified') else 0 for r in qr]) * 100
        cod = np.mean([1 if r.get('coder_used') else 0 for r in qr]) * 100
        t_avg = np.mean([r.get('elapsed_s', 0) for r in qr])
        mk = "+" if d > 0.01 else ("-" if d < -0.01 else "=")
        print(f"  {qt:<35} {len(qr):>5} {v7:>5.3f} {v5:>5.3f} {d:>+5.3f}{mk} {vl:>4.1f} {evo:>4.0f}% {cod:>4.0f}% {t_avg:>4.0f}s")
        all_v7.extend([r.get('v7_vl_score', 0) for r in qr])
        all_v5.extend([r['score'] for r in qr])
        
        by_task_summary[qt] = {
            'n': len(qr), 'v7': float(v7), 'v5': float(v5), 'delta': float(d),
            'avg_vl': float(vl), 'evo_rate': float(evo/100), 'coder_rate': float(cod/100),
        }
    
    print(f"{'-'*35} {'-'*5} {'-'*6} {'-'*6} {'-'*6}")
    ov7, ov5 = np.mean(all_v7), np.mean(all_v5)
    print(f"  {'Overall':<35} {len(all_results):>5} {ov7:>5.3f} {ov5:>5.3f} {ov5-ov7:>+5.3f}")
    
    total_vl = sum(r.get('vl_calls', 0) for r in all_results)
    avg_vl = total_vl / len(all_results) if all_results else 0
    avg_t = np.mean([r.get('elapsed_s', 0) for r in all_results])
    print(f"\n  VL calls: total={total_vl}, avg={avg_vl:.1f}/sample | Avg time: {avg_t:.0f}s/sample")
    
    # Tool usage
    tc = Counter()
    for r in all_results:
        for e in r.get('tool_trace', []):
            tc[e.get('tool', '?')] += 1
    print(f"  Tool usage: {dict(tc)}")
    
    # Numerical vs Spatial
    num = [r for r in all_results if r['question_type'] in NUMERICAL_TASKS]
    spa = [r for r in all_results if r['question_type'] not in NUMERICAL_TASKS]
    if num:
        nv5 = np.mean([r['score'] for r in num])
        nv7 = np.mean([r.get('v7_vl_score', 0) for r in num])
        print(f"  Numerical: n={len(num)}, V5={nv5:.3f}, V7={nv7:.3f}, Δ={nv5-nv7:+.3f}")
    if spa:
        sv5 = np.mean([r['score'] for r in spa])
        sv7 = np.mean([r.get('v7_vl_score', 0) for r in spa])
        print(f"  Spatial:   n={len(spa)}, V5={sv5:.3f}, V7={sv7:.3f}, Δ={sv5-sv7:+.3f}")
    
    # Evolution analysis
    evo_samples = [r for r in all_results if r.get('grid_modified')]
    if evo_samples:
        evo_v5 = np.mean([r['score'] for r in evo_samples])
        evo_v7 = np.mean([r.get('v7_vl_score', 0) for r in evo_samples])
        print(f"  Evolved:   n={len(evo_samples)} ({len(evo_samples)/len(all_results)*100:.1f}%), "
              f"V5={evo_v5:.3f}, V7={evo_v7:.3f}, Δ={evo_v5-evo_v7:+.3f}")
    
    print(f"\n{'='*60}")
    print(f"  V5 Overall = {ov5:.4f}  vs  V7 VL = {ov7:.4f}  (Δ = {ov5-ov7:+.4f})")
    print(f"  V4 Overall = 0.6438  (previous best)")
    print(f"{'='*60}")
    
    # Save merged results + summary
    merged_dir = results_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean numpy types
    clean_results = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                cr[k] = float(v)
            elif isinstance(v, np.ndarray):
                cr[k] = v.tolist()
            else:
                cr[k] = v
        clean_results.append(cr)
    
    with open(merged_dir / "detailed_results.json", 'w') as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    
    summary = {
        'timestamp': timestamp,
        'version': 'v5_autonomous_vl_reasoner',
        'architecture': 'Manager(VL) → Autonomous Tool Selection → VL Visual Reasoning',
        'design': 'No task-type routing, no difficulty strategies, fully autonomous',
        'n_samples': len(all_results),
        'overall': {'v7_vl': float(ov7), 'v5': float(ov5), 'delta': float(ov5 - ov7)},
        'v4_overall': 0.6438,
        'avg_vl_calls': float(avg_vl),
        'avg_time_s': float(avg_t),
        'tool_usage': dict(tc),
        'by_task': by_task_summary,
    }
    with open(merged_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nMerged results saved to: {merged_dir}")


if __name__ == "__main__":
    results_dir = get_results_dir()
    results = merge_results(results_dir)
    print_summary(results, results_dir)
