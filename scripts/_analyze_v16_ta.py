#!/usr/bin/env python3
"""Analyze V16 type-adaptive results for rel_distance and direction."""
import json, re, numpy as np
from collections import Counter

with open('outputs/agentic_pipeline_v16_typeadapt_8gpu/detailed_results.json') as f:
    results = json.load(f)

# === REL_DISTANCE ===
rd = [r for r in results if r['question_type']=='object_rel_distance']
print('=== REL_DISTANCE (N=%d, score=%.3f) ===' % (len(rd), np.mean([r['score'] for r in rd])))
condorcet = [r for r in rd if 'reldist_condorcet' in r.get('reasoning','')]
vl_vote = [r for r in rd if 'reldist_vl_vote' in r.get('reasoning','')]
other = [r for r in rd if r not in condorcet and r not in vl_vote]
if condorcet: print(f'  Condorcet: {len(condorcet)}, score={np.mean([r["score"] for r in condorcet]):.3f}')
if vl_vote: print(f'  VL_vote: {len(vl_vote)}, score={np.mean([r["score"] for r in vl_vote]):.3f}')
if other: print(f'  Other: {len(other)}, score={np.mean([r["score"] for r in other]):.3f}')

# VL independent accuracy
vl_ind_correct = 0
for r in rd:
    m = re.search(r'\[vl_ind\] ([A-D])', r.get('reasoning',''))
    if m and m.group(1) == r['ground_truth']: vl_ind_correct += 1
print(f'  VL_independent accuracy: {vl_ind_correct}/{len(rd)} ({vl_ind_correct/len(rd)*100:.0f}%)')

# CODER accuracy
coder_correct = 0; coder_total = 0
for r in rd:
    m = re.search(r'\[coder\] ans=([A-D])', r.get('reasoning',''))
    if m:
        coder_total += 1
        if m.group(1) == r['ground_truth']: coder_correct += 1
print(f'  CODER accuracy: {coder_correct}/{coder_total} ({coder_correct/coder_total*100:.0f}%)' if coder_total else '  CODER: N/A')

# Sample wrong condorcets
wrong_c = [r for r in condorcet if r['score'] < 0.5]
print(f'\n  Wrong Condorcet ({len(wrong_c)}):')
for r in wrong_c[:5]:
    print(f'    pred={r["prediction"]} gt={r["ground_truth"]} | {r.get("reasoning","")[:150]}')

# === DIRECTION ===
print('\n=== DIRECTION ===')
for sub in ['object_rel_direction_easy','object_rel_direction_medium','object_rel_direction_hard']:
    dd = [r for r in results if r['question_type']==sub]
    if not dd: continue
    s = np.mean([r['score'] for r in dd])
    print(f'\n{sub}: N={len(dd)}, score={s:.3f}')
    sc = [r for r in dd if 'dir_sc3vote' in r.get('reasoning','')]
    fb = [r for r in dd if 'dir_fallback' in r.get('reasoning','')]
    if sc: print(f'  SC3vote: {len(sc)}, score={np.mean([r["score"] for r in sc]):.3f}')
    if fb: print(f'  Fallback: {len(fb)}, score={np.mean([r["score"] for r in fb]):.3f}')
    
    coder_correct = coder_total = 0
    vl_ind_correct = 0
    for r in dd:
        m = re.search(r'\[coder\] ans=([A-D])', r.get('reasoning',''))
        if m:
            coder_total += 1
            if m.group(1) == r['ground_truth']: coder_correct += 1
        m2 = re.search(r'\[vl_ind\] ([A-D])', r.get('reasoning',''))
        if m2 and m2.group(1) == r['ground_truth']: vl_ind_correct += 1
    if coder_total: print(f'  CODER accuracy: {coder_correct}/{coder_total} ({coder_correct/coder_total*100:.0f}%)')
    print(f'  VL_independent accuracy: {vl_ind_correct}/{len(dd)} ({vl_ind_correct/len(dd)*100:.0f}%)')

# Aggregated direction
all_dir = [r for r in results if r['question_type'].startswith('object_rel_direction')]
print(f'\nAGG direction: N={len(all_dir)}, score={np.mean([r["score"] for r in all_dir]):.3f}')
agg_coder_c = agg_coder_t = agg_vl = 0
for r in all_dir:
    m = re.search(r'\[coder\] ans=([A-D])', r.get('reasoning',''))
    if m:
        agg_coder_t += 1
        if m.group(1) == r['ground_truth']: agg_coder_c += 1
    m2 = re.search(r'\[vl_ind\] ([A-D])', r.get('reasoning',''))
    if m2 and m2.group(1) == r['ground_truth']: agg_vl += 1
if agg_coder_t: print(f'  CODER accuracy: {agg_coder_c}/{agg_coder_t} ({agg_coder_c/agg_coder_t*100:.0f}%)')
print(f'  VL_independent accuracy: {agg_vl}/{len(all_dir)} ({agg_vl/len(all_dir)*100:.0f}%)')

# Check vote distribution for direction
print('\n  Vote distribution for direction SC3vote:')
for r in all_dir[:10]:
    m = re.search(r'\[dir_sc3vote:\d+\] ({.*?}) → ([A-D])', r.get('reasoning',''))
    if m:
        gt = r['ground_truth']
        ok = '✓' if m.group(2) == gt else '✗'
        print(f'    votes={m.group(1)} → {m.group(2)} gt={gt} {ok}')
