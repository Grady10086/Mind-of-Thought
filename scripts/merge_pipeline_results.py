#!/usr/bin/env python3
"""合并8卡并行结果并输出统计"""
import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
SHARD_DIR = PROJECT_ROOT / "outputs" / "agentic_pipeline_full"

def main():
    # 收集所有shard结果
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
    
    # 统计
    task_types = sorted(set(r['question_type'] for r in all_results))
    
    print("\n" + "=" * 120)
    print("Agentic Pipeline FULL TEST (Manager → Retriever → Evolver → Reasoner)")
    print(f"基准: V7 VL Overall = 63.61%")
    print(f"测试样本: {len(all_results)}")
    print("=" * 120)
    
    print(f"\n{'Task':<35} {'N':>5} {'V7_VL':>7} {'Pipe':>7} {'Delta':>7} {'Mode':>18}")
    print(f"{'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*18}")
    
    all_v7, all_pipe = [], []
    
    for qt in task_types:
        qt_r = [r for r in all_results if r['question_type'] == qt]
        v7 = np.mean([r['v7_vl_score'] for r in qt_r])
        pipe = np.mean([r['score'] for r in qt_r])
        delta = pipe - v7
        modes = Counter(r.get('routing_mode', '?') for r in qt_r)
        mode_str = modes.most_common(1)[0][0] if modes else '?'
        marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
        print(f"  {qt:<35} {len(qt_r):>5} {v7:>6.3f} {pipe:>6.3f} {delta:>+6.3f} {marker} {mode_str:>18}")
        all_v7.extend([r['v7_vl_score'] for r in qt_r])
        all_pipe.extend([r['score'] for r in qt_r])
    
    print(f"{'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*18}")
    ov7, op = np.mean(all_v7), np.mean(all_pipe)
    print(f"  {'Overall':<35} {len(all_results):>5} {ov7:>6.3f} {op:>6.3f} {op-ov7:>+6.3f}")
    
    # Routing分析
    print(f"\n{'='*60}")
    print("Routing Mode Analysis:")
    for mode in sorted(set(r.get('routing_mode', '?') for r in all_results)):
        mr = [r for r in all_results if r.get('routing_mode', '?') == mode]
        mv7 = np.mean([r['v7_vl_score'] for r in mr])
        mp = np.mean([r['score'] for r in mr])
        print(f"  {mode:<18} n={len(mr):>5} V7_VL={mv7:.3f} Pipe={mp:.3f} Delta={mp-mv7:>+.3f}")
    
    # Evolution分析
    evo_count = sum(1 for r in all_results if r.get('grid_modified', False))
    print(f"\nEvolution: {evo_count}/{len(all_results)} samples had grid modifications ({evo_count/len(all_results)*100:.1f}%)")
    
    evo_results = [r for r in all_results if r.get('grid_modified')]
    if evo_results:
        evo_better = sum(1 for r in evo_results if r['score'] > r['v7_vl_score'] + 0.05)
        evo_worse = sum(1 for r in evo_results if r['score'] < r['v7_vl_score'] - 0.05)
        evo_same = len(evo_results) - evo_better - evo_worse
        print(f"  Evolution effect: Better={evo_better}, Worse={evo_worse}, Same={evo_same}")
    
    # 保存合并结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_dir = SHARD_DIR / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    with open(merged_dir / "detailed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    summary = {
        'timestamp': timestamp,
        'n_samples': len(all_results),
        'overall': {'v7_vl': float(ov7), 'pipeline': float(op), 'delta': float(op - ov7)},
        'by_task': {qt: {
            'n': len([r for r in all_results if r['question_type'] == qt]),
            'v7_vl': float(np.mean([r['v7_vl_score'] for r in all_results if r['question_type'] == qt])),
            'pipeline': float(np.mean([r['score'] for r in all_results if r['question_type'] == qt])),
        } for qt in task_types},
        'routing_stats': dict(Counter(r.get('routing_mode', '?') for r in all_results)),
        'evolution_count': evo_count,
    }
    
    with open(merged_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nMerged results saved to: {merged_dir}")
    print(f"\n{'='*60}")
    print(f"FINAL: Pipeline Overall={op:.4f} vs V7 VL={ov7:.4f} (Delta={op-ov7:+.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
