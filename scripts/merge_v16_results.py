#!/usr/bin/env python3
"""Merge V16 multi-GPU results and print summary."""
import json, sys, numpy as np
from pathlib import Path
from collections import defaultdict, Counter

od = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/agentic_pipeline_v16_small_8gpu')
n_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 8

all_results = []
for g in range(n_gpus):
    p = od / f'gpu{g}' / 'detailed_results.json'
    if p.exists():
        with open(p) as f:
            data = json.load(f)
            all_results.extend(data)
            print(f'  GPU {g}: {len(data)} samples')
    else:
        print(f'  GPU {g}: MISSING')

if not all_results:
    print('No results!'); sys.exit(1)

# Save merged
with open(od / 'detailed_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f'\nTotal: {len(all_results)} samples')

# Summary — compare with official baseline
OFFICIAL_BASELINE = {
    'object_counting': 0.6340, 'object_abs_distance': 0.4552,
    'object_size_estimation': 0.7341, 'room_size_estimation': 0.5771,
    'object_rel_distance': 0.5225, 'object_rel_direction': 0.5123,
    'route_planning': 0.3041, 'obj_appearance_order': 0.6100,
}
tts = sorted(set(r['question_type'] for r in all_results))
# Map direction subtypes to aggregated direction
DIR_SUBTYPES = {'object_rel_direction_easy','object_rel_direction_medium','object_rel_direction_hard'}
print(f"\n{'Task':<35} {'N':>4} {'V7':>6} {'V16':>6} {'Δ':>7}  {'Base':>6} {'vsBase':>7}  {'Cod%':>5} {'VL#':>4} {'t/s':>5}")
print("-"*100)
a7, a16 = [], []
all_pass = True
for qt in tts:
    qr = [r for r in all_results if r['question_type']==qt]
    v7 = np.mean([r.get('v7_vl_score',0) for r in qr])
    v16 = np.mean([r['score'] for r in qr])
    d = v16-v7; mk = "↑" if d>0.01 else ("↓" if d<-0.01 else "=")
    cod = np.mean([1 if r.get('coder_used') else 0 for r in qr])*100
    vl = np.mean([r.get('vl_calls',0) for r in qr])
    tavg = np.mean([r.get('elapsed_s',0) for r in qr])
    base = OFFICIAL_BASELINE.get(qt, 0)
    db = v16 - base; bm = "✅" if db > 0 else "❌"
    if base > 0 and db <= 0: all_pass = False
    print(f'  {qt:<35} {len(qr):>4} {v7:>5.3f} {v16:>5.3f} {d:>+6.3f}{mk} {base:>5.3f} {db:>+6.3f}{bm} {cod:>4.0f}% {vl:>4.1f} {tavg:>4.0f}s')
    a7.extend([r.get('v7_vl_score',0) for r in qr])
    a16.extend([r['score'] for r in qr])
ov7, ov16 = np.mean(a7), np.mean(a16)
print("-"*100)
# Aggregated direction score
dir_results = [r for r in all_results if r['question_type'] in DIR_SUBTYPES]
if dir_results:
    dir_v16 = np.mean([r['score'] for r in dir_results])
    dir_v7 = np.mean([r.get('v7_vl_score',0) for r in dir_results])
    dir_base = OFFICIAL_BASELINE.get('object_rel_direction', 0)
    dir_db = dir_v16 - dir_base
    print(f"  {'[AGG] object_rel_direction':<35} {len(dir_results):>4} {dir_v7:>5.3f} {dir_v16:>5.3f} {dir_v16-dir_v7:>+6.3f}  {dir_base:>5.3f} {dir_db:>+6.3f}{'✅' if dir_db>0 else '❌'}")
    if dir_base > 0 and dir_db <= 0: all_pass = False
ov_base = np.mean(list(OFFICIAL_BASELINE.values()))
print(f"  {'Overall':<35} {len(all_results):>4} {ov7:>5.3f} {ov16:>5.3f} {ov16-ov7:>+6.3f}  {ov_base:>5.3f} {ov16-ov_base:>+6.3f}{'✅' if ov16>ov_base else '❌'}")
if all_pass:
    print("\n  🎉 ALL TASK TYPES EXCEED OFFICIAL BASELINE! 🎉")
else:
    print("\n  ⚠ Some task types still below baseline.")

# Tool usage
tc = Counter()
for r in all_results:
    for e in r.get('tool_trace',[]): tc[e.get('tool','?')] += 1
tvl = sum(r.get('vl_calls',0) for r in all_results)
at = np.mean([r.get('elapsed_s',0) for r in all_results])
print(f"\n  VL: total={tvl}, avg={tvl/len(all_results):.1f}/sample | Time: {at:.0f}s/sample")
print(f"  Tools: {dict(tc)}")

# Fast path stats
n_fp = sum(1 for r in all_results if 'num_fastpath' in r.get('reasoning','').lower())
n_num = sum(1 for r in all_results if not r.get('options'))
print(f"  NumFastPath: {n_fp}/{n_num} numerical samples")

print(f"\n{'='*60}")
print(f"  V16 Overall = {ov16:.4f}  vs  V7 = {ov7:.4f}  (Δ = {ov16-ov7:+.4f})")
print(f"  V14 full baseline = 0.6510")
print(f"{'='*60}")

# Save summary
summary = {'n_samples': len(all_results), 'overall_v7': float(ov7), 'overall_v16': float(ov16),
           'delta': float(ov16-ov7), 'num_fastpath': n_fp,
           'by_task': {qt: {'n': len([r for r in all_results if r['question_type']==qt]),
               'v7': float(np.mean([r.get('v7_vl_score',0) for r in all_results if r['question_type']==qt])),
               'v16': float(np.mean([r['score'] for r in all_results if r['question_type']==qt]))}
           for qt in tts}}
with open(od / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
