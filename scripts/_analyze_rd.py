#!/usr/bin/env python3
import json, re, numpy as np
all_r = []
for g in [2,3,6]:
    with open(f'outputs/agentic_pipeline_v16_ta2_8gpu/gpu{g}/detailed_results.json') as f:
        all_r.extend(json.load(f))
rd = [r for r in all_r if r['question_type']=='object_rel_distance']
print(f'Total rel_distance: {len(rd)}, score={np.mean([r["score"] for r in rd]):.3f}')
coder_ok = coder_total = 0
for r in rd:
    m = re.search(r'\[coder\] ans=([A-D])', r.get('reasoning',''))
    if m:
        coder_total += 1
        if m.group(1) == r['ground_truth']: coder_ok += 1
print(f'CODER accuracy: {coder_ok}/{coder_total} ({coder_ok/coder_total*100:.0f}%)' if coder_total else 'CODER: N/A')

condorcet = [r for r in rd if 'reldist_condorcet' in r.get('reasoning','')]
dir_path = [r for r in rd if 'dir_vl_sc4vote' in r.get('reasoning','')]
other = [r for r in rd if r not in condorcet and r not in dir_path]
if condorcet: print(f'Condorcet: N={len(condorcet)} score={np.mean([r["score"] for r in condorcet]):.3f}')
if dir_path: print(f'Dir_path(misrouted): N={len(dir_path)} score={np.mean([r["score"] for r in dir_path]):.3f}')
if other: print(f'Other: N={len(other)} score={np.mean([r["score"] for r in other]):.3f}')

# VL independent accuracy
vi_ok = 0
for r in rd:
    m = re.search(r'\[vl_ind\] ([A-D])', r.get('reasoning',''))
    if m and m.group(1) == r['ground_truth']: vi_ok += 1
print(f'VL_independent accuracy: {vi_ok}/{len(rd)} ({vi_ok/len(rd)*100:.0f}%)')
