#!/usr/bin/env python3
import json, numpy as np
from collections import defaultdict

all_r = []
for g in range(8):
    d = json.load(open(f'outputs/agentic_pipeline_v20_ref/gpu{g}/detailed_results.json'))
    all_r.extend(d)
    print(f'GPU{g}: {len(d)} samples')

seen = set(); dd = []
for r in all_r:
    k = (r.get('scene_name'), r.get('question'), r.get('ground_truth'))
    if k not in seen:
        seen.add(k); dd.append(r)

sc = [r['score'] for r in dd]
print(f'\nTotal: {len(dd)} samples (raw {len(all_r)})')
print(f'Overall: {np.mean(sc):.4f}\n')

bt = defaultdict(list)
for r in dd:
    bt[r['question_type']].append(r['score'])

print(f"{'Task':40s} {'N':>4} {'V20':>6}")
print('-' * 55)
for qt in sorted(bt):
    print(f'  {qt:38s} {len(bt[qt]):>4} {np.mean(bt[qt]):.4f}')
print('-' * 55)
print(f'  {"Overall":38s} {len(sc):>4} {np.mean(sc):.4f}')

# V19 comparison
try:
    v19 = json.load(open('outputs/agentic_pipeline_v19_ref/detailed_results_merged.json'))
    v19_sc = np.mean([r['score'] for r in v19])
    print(f'\n  V19 Overall: {v19_sc:.4f}')
    print(f'  V20 - V19:   {np.mean(sc) - v19_sc:+.4f}')
except: pass

# Convergence
conv = defaultdict(int)
for r in dd:
    ct = r.get('convergence_type', '')
    if ct: conv[ct] += 1
if conv:
    total_c = sum(conv.values())
    print(f'\nConvergence ({total_c} choice samples):')
    for k in sorted(conv):
        print(f'  {k:30s}: {conv[k]:>4} ({100*conv[k]/total_c:.1f}%)')

json.dump(dd, open('outputs/agentic_pipeline_v20_ref/detailed_results_merged.json', 'w'))
print(f'\nSaved: detailed_results_merged.json')
