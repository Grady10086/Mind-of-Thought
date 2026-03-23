#!/usr/bin/env python3
"""合并8卡并行 Pipeline V2 结果并输出统计"""
import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
SHARD_DIR = PROJECT_ROOT / "outputs" / "agentic_pipeline_v2_full"


def main():
    all_results = []
    for gpu_id in range(8):
        shard_path = SHARD_DIR / f"gpu{gpu_id}" / "detailed_results.json"
        if shard_path.exists():
            with open(shard_path) as f:
                shard = json.load(f)
            all_results.extend(shard)
            print(f"GPU {gpu_id}: {len(shard)} samples loaded")
        else:
            print(f"GPU {gpu_id}: NOT FOUND ({shard_path})")

    if not all_results:
        print("No results found!")
        sys.exit(1)

    print(f"\nTotal merged: {len(all_results)} samples")

    task_types = sorted(set(r['question_type'] for r in all_results))

    print("\n" + "=" * 130)
    print("Agentic Pipeline V2 FULL TEST (VL-Centric: Manager → Critic → Evolver → Reasoner)")
    print(f"基准: V7 VL Overall = 63.61%")
    print(f"测试样本: {len(all_results)}")
    print("=" * 130)

    print(f"\n{'Task':<35} {'N':>5} {'V7_VL':>7} {'V2':>7} {'Delta':>7} {'Critic%':>8} {'Evolve%':>8}")
    print(f"{'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

    all_v7, all_pipe = [], []

    for qt in task_types:
        qt_r = [r for r in all_results if r['question_type'] == qt]
        v7 = np.mean([r['v7_vl_score'] for r in qt_r])
        pipe = np.mean([r['score'] for r in qt_r])
        delta = pipe - v7
        critic_pct = np.mean([1 if r.get('critic_has_issues') else 0 for r in qt_r]) * 100
        evolve_pct = np.mean([1 if r.get('grid_modified') else 0 for r in qt_r]) * 100
        marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
        print(f"  {qt:<35} {len(qt_r):>5} {v7:>6.3f} {pipe:>6.3f} {delta:>+6.3f} {marker} {critic_pct:>7.1f}% {evolve_pct:>7.1f}%")
        all_v7.extend([r['v7_vl_score'] for r in qt_r])
        all_pipe.extend([r['score'] for r in qt_r])

    print(f"{'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
    ov7, op = np.mean(all_v7), np.mean(all_pipe)
    total_critic = np.mean([1 if r.get('critic_has_issues') else 0 for r in all_results]) * 100
    total_evolve = np.mean([1 if r.get('grid_modified') else 0 for r in all_results]) * 100
    print(f"  {'Overall':<35} {len(all_results):>5} {ov7:>6.3f} {op:>6.3f} {op-ov7:>+6.3f}   {total_critic:>7.1f}% {total_evolve:>7.1f}%")

    # Critic分析
    print(f"\n{'='*60}")
    print("Critic Analysis:")
    critic_total = sum(1 for r in all_results if r.get('critic_has_issues'))
    evolve_total = sum(1 for r in all_results if r.get('grid_modified'))
    print(f"  Critic found issues: {critic_total}/{len(all_results)} ({critic_total/len(all_results)*100:.1f}%)")
    print(f"  Grid evolved: {evolve_total}/{len(all_results)} ({evolve_total/len(all_results)*100:.1f}%)")

    # Critic有issue vs 无issue的准确率对比
    with_issues = [r for r in all_results if r.get('critic_has_issues')]
    no_issues = [r for r in all_results if not r.get('critic_has_issues')]
    if with_issues:
        print(f"\n  With critic issues (n={len(with_issues)}):")
        print(f"    V2={np.mean([r['score'] for r in with_issues]):.3f} V7={np.mean([r['v7_vl_score'] for r in with_issues]):.3f}")
    if no_issues:
        print(f"  No critic issues (n={len(no_issues)}):")
        print(f"    V2={np.mean([r['score'] for r in no_issues]):.3f} V7={np.mean([r['v7_vl_score'] for r in no_issues]):.3f}")

    # Evolution效果
    if evolve_total > 0:
        evo_results = [r for r in all_results if r.get('grid_modified')]
        evo_better = sum(1 for r in evo_results if r['score'] > r['v7_vl_score'] + 0.05)
        evo_worse = sum(1 for r in evo_results if r['score'] < r['v7_vl_score'] - 0.05)
        evo_same = len(evo_results) - evo_better - evo_worse
        print(f"\n  Evolution effect: Better={evo_better}, Worse={evo_worse}, Same={evo_same}")

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_dir = SHARD_DIR / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    with open(merged_dir / "detailed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary = {
        'timestamp': timestamp,
        'version': 'v2_vl_centric',
        'n_samples': len(all_results),
        'overall': {'v7_vl': float(ov7), 'pipeline': float(op), 'delta': float(op - ov7)},
        'by_task': {qt: {
            'n': len([r for r in all_results if r['question_type'] == qt]),
            'v7_vl': float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type'] == qt])),
            'pipeline': float(np.mean([r['score'] for r in all_results if r['question_type'] == qt])),
            'critic_rate': float(np.mean([1 if r.get('critic_has_issues') else 0
                                          for r in all_results if r['question_type'] == qt])),
            'evolve_rate': float(np.mean([1 if r.get('grid_modified') else 0
                                          for r in all_results if r['question_type'] == qt])),
        } for qt in task_types},
        'critic_stats': {
            'total_with_issues': critic_total,
            'total_evolved': evolve_total,
        },
    }

    with open(merged_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nMerged results saved to: {merged_dir}")
    print(f"\n{'='*60}")
    print(f"FINAL: V2 Overall={op:.4f} vs V7 VL={ov7:.4f} (Delta={op-ov7:+.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
