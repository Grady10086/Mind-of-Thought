#!/usr/bin/env python3
"""Merge V21 8-GPU results and compare with V20."""

import json, os, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent

def load_v20_results():
    """Load V20 full results for comparison."""
    v20_dir = PROJECT_ROOT / "outputs" / "agentic_pipeline_v20_ref"
    all_results = []
    for gpu_dir in sorted(v20_dir.glob("gpu*")):
        f = gpu_dir / "detailed_results.json"
        if f.exists():
            all_results.extend(json.load(open(f)))
    return all_results

def main():
    v21_dir = PROJECT_ROOT / "outputs" / "agentic_pipeline_v21_ref"
    
    # Merge V21 results from 8 GPUs
    all_results = []
    for gpu_dir in sorted(v21_dir.glob("gpu*")):
        f = gpu_dir / "detailed_results.json"
        if f.exists():
            data = json.load(open(f))
            all_results.extend(data)
            print(f"  {gpu_dir.name}: {len(data)} samples")
    
    print(f"\nV21 Total: {len(all_results)} samples")
    
    if not all_results:
        print("ERROR: No results found!")
        return
    
    # Load V20 for comparison
    v20_results = load_v20_results()
    v20_by_key = {}
    for r in v20_results:
        key = (r['scene_name'], r['question'])
        v20_by_key[key] = r
    
    print(f"V20 comparison: {len(v20_results)} samples\n")
    
    # Per-task analysis
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r['question_type']].append(r)
    
    print("=" * 130)
    print("V21 Confidence-Aware Consensus — Full-Scale Results")
    print("=" * 130)
    print(f"  {'Task':<35} {'N':>4} {'V21':>6} {'V20':>6} {'Δ':>6}  {'AvgR':>4} {'Conv%':>5} {'AvgVL':>5} {'T':>5}")
    print("-" * 110)
    
    tts = sorted(by_type.keys())
    for qt in tts:
        qr = by_type[qt]
        v21_score = np.mean([r['score'] for r in qr])
        
        # V20 comparison
        v20_scores = []
        for r in qr:
            key = (r['scene_name'], r['question'])
            if key in v20_by_key:
                v20_scores.append(v20_by_key[key]['score'])
        v20_score = np.mean(v20_scores) if v20_scores else 0
        delta = v21_score - v20_score
        
        avg_r = np.mean([r.get('converged_phase', 1) for r in qr])
        conv = np.mean([1 if r.get('converged') else 0 for r in qr]) * 100
        avg_vl = np.mean([r.get('vl_calls', 0) for r in qr])
        avg_t = np.mean([r.get('elapsed_s', 0) for r in qr])
        
        mk = "+" if delta > 0.005 else ("-" if delta < -0.005 else "=")
        print(f"  {qt:<35} {len(qr):>4} {v21_score:>5.3f} {v20_score:>5.3f} {delta:>+5.3f}{mk} {avg_r:>3.1f} {conv:>4.0f}% {avg_vl:>4.1f} {avg_t:>4.0f}s")
    
    ov21 = np.mean([r['score'] for r in all_results])
    ov20 = np.mean([r['score'] for r in v20_results]) if v20_results else 0
    print("-" * 110)
    print(f"  {'Overall':<35} {len(all_results):>4} {ov21:>5.3f} {ov20:>5.3f} {ov21-ov20:>+5.3f}")
    
    # Choice vs Numerical breakdown
    choice_types = {'obj_appearance_order', 'object_rel_direction_easy', 'object_rel_direction_medium',
                    'object_rel_direction_hard', 'object_rel_distance', 'route_planning'}
    
    choice_r = [r for r in all_results if r['question_type'] in choice_types]
    num_r = [r for r in all_results if r['question_type'] not in choice_types]
    choice_v20 = [r for r in v20_results if r['question_type'] in choice_types]
    num_v20 = [r for r in v20_results if r['question_type'] not in choice_types]
    
    if choice_r:
        c21 = np.mean([r['score'] for r in choice_r])
        c20 = np.mean([r['score'] for r in choice_v20]) if choice_v20 else 0
        print(f"\n  Choice tasks:    V21={c21:.4f}  V20={c20:.4f}  Δ={c21-c20:+.4f}")
    if num_r:
        n21 = np.mean([r['score'] for r in num_r])
        n20 = np.mean([r['score'] for r in num_v20]) if num_v20 else 0
        print(f"  Numerical tasks: V21={n21:.4f}  V20={n20:.4f}  Δ={n21-n20:+.4f}")
    
    # Convergence analysis
    conv_types = defaultdict(int)
    for r in all_results:
        ct = r.get('converge_type', 'unknown')
        conv_types[ct] += 1
    print(f"\n  Convergence types:")
    for ct, count in sorted(conv_types.items(), key=lambda x: -x[1]):
        print(f"    {ct}: {count} ({100*count/len(all_results):.1f}%)")
    
    # Weak consensus stats (from reasoning)
    weak_count = sum(1 for r in all_results if 'weak_consensus_skip' in r.get('reasoning', ''))
    print(f"\n  Weak consensus rejected: {weak_count} ({100*weak_count/len(all_results):.1f}%)")
    
    # Per-sample delta analysis (V21 vs V20)
    if v20_by_key:
        better = worse = same = 0
        for r in all_results:
            key = (r['scene_name'], r['question'])
            if key in v20_by_key:
                d = r['score'] - v20_by_key[key]['score']
                if d > 0.01: better += 1
                elif d < -0.01: worse += 1
                else: same += 1
        print(f"\n  Per-sample: +{better} better, -{worse} worse, ={same} same (net {better-worse:+d})")
    
    # Save merged results
    merged_path = v21_dir / "all_results.json"
    clean = []
    for r in all_results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)): cr[k] = float(v)
            elif isinstance(v, np.ndarray): cr[k] = v.tolist()
            else: cr[k] = v
        clean.append(cr)
    with open(merged_path, 'w') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\n  Merged results saved: {merged_path}")
    
    # Save summary
    summary = {
        'version': 'v21_confidence_aware',
        'n_samples': len(all_results),
        'overall_v21': float(ov21),
        'overall_v20': float(ov20),
        'delta': float(ov21 - ov20),
        'by_task': {qt: {'n': len(qr), 'v21': float(np.mean([r['score'] for r in qr]))}
                    for qt, qr in by_type.items()},
        'convergence_types': dict(conv_types),
        'weak_consensus_rejected': weak_count,
    }
    with open(v21_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
