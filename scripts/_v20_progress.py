#!/usr/bin/env python3
"""Monitor V20 full-scale test progress."""
import re, numpy as np, os
from collections import defaultdict

scores = defaultdict(list)

for g in range(8):
    logf = f'outputs/agentic_pipeline_v20_full/gpu{g}.log'
    if not os.path.exists(logf):
        continue
    for line in open(logf):
        # Format 1: "Score=1.000 V7=0.000"
        m1 = re.search(r'Score=([\d.]+)\s+V7=([\d.]+)', line)
        if m1:
            qt_m = re.search(r'(\S+)\s+\[VL:', line)
            qt = qt_m.group(1).strip() if qt_m else '?'
            scores[qt].append((float(m1.group(1)), float(m1.group(2))))
            continue
        # Format 2: "[task_type] ans=X gt=Y score=0.000 vl=N t=Ns"
        m2 = re.search(r'\[(\w+)\]\s+ans=\S+\s+gt=\S+\s+score=([\d.]+)', line)
        if m2:
            qt = m2.group(1)
            sc = float(m2.group(2))
            scores[qt].append((sc, None))  # no V7 baseline in this format

total = sum(len(v) for v in scores.values())
print(f'Parsed: {total}/5130 samples ({100*total/5130:.1f}%)')
print(f"{'Task':<40} {'N':>4} {'V7':>6} {'V20':>6} {'D':>6}")
print('-' * 65)
a7 = []; a20 = []
for qt in sorted(scores.keys()):
    v = scores[qt]
    s20 = np.mean([x[0] for x in v])
    v7_vals = [x[1] for x in v if x[1] is not None]
    s7 = np.mean(v7_vals) if v7_vals else float('nan')
    d_str = f'{s20-s7:+.3f}' if v7_vals else '  N/A'
    s7_str = f'{s7:.3f}' if v7_vals else '  N/A'
    print(f'  {qt:<38} {len(v):>4} {s7_str:>6} {s20:.3f} {d_str:>6}')
    a20.extend([x[0] for x in v])
    a7.extend([x[1] for x in v if x[1] is not None])
print('-' * 65)
if a20:
    overall_20 = np.mean(a20)
    if a7:
        overall_7 = np.mean(a7)
        print(f'  {"Overall":<38} {total:>4} {overall_7:.3f} {overall_20:.3f} {overall_20-overall_7:+.3f}')
    else:
        print(f'  {"Overall":<38} {total:>4}   N/A {overall_20:.3f}   N/A')
    print(f'\n  V20 Overall = {overall_20:.4f}')

# Per-GPU progress
print(f'\n  Per-GPU:')
for g in range(8):
    logf = f'outputs/agentic_pipeline_v20_full/gpu{g}.log'
    if not os.path.exists(logf):
        print(f'    GPU{g}: no log')
        continue
    cnt = 0
    for line in open(logf):
        if re.search(r'[Ss]core=', line):
            cnt += 1
    print(f'    GPU{g}: {cnt} samples')

# Convergence stats
conv_global = 0; conv_stable = 0; conv_vote = 0; conv_early = 0
for g in range(8):
    logf = f'outputs/agentic_pipeline_v20_full/gpu{g}.log'
    if not os.path.exists(logf): continue
    for line in open(logf):
        if 'global_consensus' in line or 'GLOBAL CONSENSUS' in line:
            if 'Final:' in line or 'final' in line.lower() or 'R1:' in line:
                conv_global += 1
        elif 'evolution_stable' in line or 'EVOLUTION STABLE' in line:
            if 'Final:' in line or 'final' in line.lower():
                conv_stable += 1
        elif 'weighted_vote' in line or 'WEIGHTED VOTE' in line:
            if 'Final:' in line or 'final' in line.lower():
                conv_vote += 1
        elif 'no_new_frames' in line or 'NO NEW FRAMES' in line:
            conv_early += 1
choice_total = conv_global + conv_stable + conv_vote
if choice_total > 0:
    print(f'\n  Convergence ({choice_total} choice samples):')
    print(f'    Global consensus: {conv_global} ({100*conv_global/choice_total:.1f}%)')
    print(f'    Evolution stable: {conv_stable} ({100*conv_stable/choice_total:.1f}%)')
    print(f'    Weighted vote:    {conv_vote} ({100*conv_vote/choice_total:.1f}%)')
    if conv_early:
        print(f'    Early stop (no new frames): {conv_early}')

# V19 comparison
v19_file = 'outputs/agentic_pipeline_v19_ref/detailed_results_merged.json'
if os.path.exists(v19_file):
    import json
    with open(v19_file) as f:
        v19_data = json.load(f)
    v19_scores = defaultdict(list)
    for item in v19_data:
        qt = item.get('question_type', '?')
        sc = item.get('score', 0)
        v19_scores[qt].append(sc)
    v19_overall = np.mean([item['score'] for item in v19_data if 'score' in item])
    print(f'\n  === V19 Comparison (V19 overall = {v19_overall:.4f}) ===')
    v20_overall = np.mean(a20)
    print(f'  V20 current overall = {v20_overall:.4f}  (delta = {v20_overall-v19_overall:+.4f})')
