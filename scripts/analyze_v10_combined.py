#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("V10 Combined (Scheme 1+2) - Full Analysis")
print("="*80)

# 收集所有结果
results = []
for gpu_id in range(8):
    f = Path(f"outputs/agentic_pipeline_v10_combined_full/gpu{gpu_id}/detailed_results.json")
    if f.exists():
        with open(f) as fp:
            data = json.load(fp)
            results.extend(data)
            print(f"GPU {gpu_id}: {len(data)} samples")

if not results:
    print("No results found!")
    exit(1)

print(f"\nTotal: {len(results)} samples")

# 保存合并结果
outdir = Path("outputs/agentic_pipeline_v10_combined_full/merged")
outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / "detailed_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to: {outdir}/")

# 分析
overall = np.mean([r['score'] for r in results])
v7_baseline = np.mean([r['v7_vl_score'] for r in results])
delta = overall - v7_baseline

print(f"\n{'='*80}")
print(f"Overall: {overall:.4f} (V7: {v7_baseline:.4f}, Δ: {delta:+.4f})")
print(f"{'='*80}")

# 按任务类型分析
by_type = defaultdict(list)
for r in results:
    by_type[r['question_type']].append(r)

print(f"\n{'Task Type':<35} {'N':>5} {'V10+':>7} {'V7':>7} {'Δ':>7}")
print("-"*60)

for task in sorted(by_type.keys()):
    task_results = by_type[task]
    v10 = np.mean([r['score'] for r in task_results])
    v7 = np.mean([r['v7_vl_score'] for r in task_results])
    d = v10 - v7
    marker = "★" if d > 0.01 else " "
    print(f"{marker} {task:<33} {len(task_results):>5} {v10:>7.3f} {v7:>7.3f} {d:>+7.3f}")

# 统计
print(f"\n{'='*80}")
print("Statistics:")
print(f"{'='*80}")

scale_adj = np.mean([1 if r.get('scale_adjust_triggered') else 0 for r in results]) * 100
evo = np.mean([1 if r.get('grid_modified') else 0 for r in results]) * 100
coder = np.mean([1 if r.get('coder_used') else 0 for r in results]) * 100
verify = np.mean([1 if r.get('verify_triggered') else 0 for r in results]) * 100
vl_calls = np.mean([r.get('vl_calls', 0) for r in results])

print(f"Scale Adjust: {scale_adj:.1f}%")
print(f"Evolution:    {evo:.1f}%")
print(f"Coder Used:   {coder:.1f}%")
print(f"Verify:       {verify:.1f}%")
print(f"Avg VL Calls: {vl_calls:.1f}")

# 目标
print(f"\n{'='*80}")
print(f"Target: Overall >70%")
print(f"Result: {overall:.1%} {'✓ PASS' if overall > 0.70 else '✗ FAIL'}")
print(f"{'='*80}")

# 保存摘要
summary = {
    'total_samples': len(results),
    'overall': {'v10_combined': float(overall), 'v7': float(v7_baseline), 'delta': float(delta)},
    'by_task': {t: {
        'n': len(by_type[t]),
        'v10': float(np.mean([r['score'] for r in by_type[t]])),
        'v7': float(np.mean([r['v7_vl_score'] for r in by_type[t]]))
    } for t in by_type},
    'stats': {
        'scale_adjust_rate': float(scale_adj),
        'evolution_rate': float(evo),
        'coder_rate': float(coder),
        'verify_rate': float(verify),
        'avg_vl_calls': float(vl_calls)
    }
}
with open(outdir / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
