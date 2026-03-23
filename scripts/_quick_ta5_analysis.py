#!/usr/bin/env python3
"""Quick analysis of TA5 results to guide TA7 optimization."""
import json, numpy as np, re
from collections import Counter

with open('outputs/agentic_pipeline_v16_ta5_8gpu/detailed_results.json') as f:
    data = json.load(f)

print(f'Total: {len(data)} samples\n')

# Direction analysis
print('=== Direction Analysis ===')
for sub in ['object_rel_direction_easy','object_rel_direction_medium','object_rel_direction_hard']:
    qr = [r for r in data if r['question_type']==sub]
    if not qr: continue
    sc = np.mean([r['score'] for r in qr])
    strats = Counter()
    for r in qr:
        reason = r.get('reasoning','')
        if 'dir_vl_slice' in reason: strats['dir_vl_slice'] += 1
        elif 'dir_slice' in reason: strats['dir_slice'] += 1
        else: strats['other'] += 1
    print(f'{sub}: {sc:.3f} ({len(qr)}), strats={dict(strats)}')
    # CODER vs VL_ind accuracy
    coder_ok = vl_ok = 0
    for r in qr:
        reason = r.get('reasoning','')
        gt = str(r['ground_truth']).strip().upper()
        m = re.search(r'\[coder\] ans=([A-D])', reason)
        if m and m.group(1) == gt: coder_ok += 1
        m2 = re.search(r'\[vl_ind\] ([A-D])', reason)
        if m2 and m2.group(1) == gt: vl_ok += 1
    print(f'  CODER: {coder_ok}/{len(qr)} ({coder_ok/len(qr):.1%}), VL_ind: {vl_ok}/{len(qr)} ({vl_ok/len(qr):.1%})')

# Agg direction
dir_r = [r for r in data if r['question_type'].startswith('object_rel_direction')]
print(f'\n[AGG] direction: {np.mean([r["score"] for r in dir_r]):.3f} ({len(dir_r)})')
coder_ok = vl_ok = both_ok = neither = 0
agree_ok = agree_wrong = disagree_vl_ok = disagree_coder_ok = disagree_both_wrong = 0
for r in dir_r:
    reason = r.get('reasoning','')
    gt = str(r['ground_truth']).strip().upper()
    ca = ''; vi = ''
    m = re.search(r'\[coder\] ans=([A-D])', reason)
    if m: ca = m.group(1)
    m2 = re.search(r'\[vl_ind\] ([A-D])', reason)
    if m2: vi = m2.group(1)
    c_ok = ca == gt; v_ok = vi == gt
    if c_ok: coder_ok += 1
    if v_ok: vl_ok += 1
    if c_ok and v_ok: both_ok += 1
    if not c_ok and not v_ok: neither += 1
    if ca and vi:
        if ca == vi:
            if c_ok: agree_ok += 1
            else: agree_wrong += 1
        else:
            if v_ok: disagree_vl_ok += 1
            elif c_ok: disagree_coder_ok += 1
            else: disagree_both_wrong += 1
print(f'CODER: {coder_ok}/{len(dir_r)} ({coder_ok/len(dir_r):.1%}), VL_ind: {vl_ok}/{len(dir_r)} ({vl_ok/len(dir_r):.1%})')
print(f'Both OK: {both_ok}, Neither: {neither}')
print(f'Agree OK: {agree_ok}, Agree Wrong: {agree_wrong}')
print(f'Disagree VL OK: {disagree_vl_ok}, Disagree CODER OK: {disagree_coder_ok}, Disagree Both Wrong: {disagree_both_wrong}')
total_with_both = agree_ok + agree_wrong + disagree_vl_ok + disagree_coder_ok + disagree_both_wrong
if total_with_both > 0:
    print(f'Oracle(best of CODER,VL): {(agree_ok+disagree_vl_ok+disagree_coder_ok)/total_with_both:.1%}')
    print(f'Trust agree, else VL: {(agree_ok+disagree_vl_ok)/total_with_both:.1%}')
    print(f'Trust agree, else CODER: {(agree_ok+disagree_coder_ok)/total_with_both:.1%}')
    print(f'Always VL: {(agree_ok+agree_wrong-(agree_wrong-0)+disagree_vl_ok)/total_with_both:.1%} = VL_ind rate')

# Rel_distance
print('\n=== Rel_distance Analysis ===')
rd = [r for r in data if r['question_type']=='object_rel_distance']
print(f'Score: {np.mean([r["score"] for r in rd]):.3f} ({len(rd)})')
strats = Counter()
for r in rd:
    reason = r.get('reasoning','')
    if 'reldist_sc3' in reason: strats['reldist_sc3'] += 1
    else: strats['other'] += 1
print(f'Strats: {dict(strats)}')
coder_ok = vl_ok = 0
for r in rd:
    reason = r.get('reasoning','')
    gt = str(r['ground_truth']).strip().upper()
    m = re.search(r'\[coder\] ans=([A-D])', reason)
    if m and m.group(1) == gt: coder_ok += 1
    m2 = re.search(r'\[vl_ind\] ([A-D])', reason)
    if m2 and m2.group(1) == gt: vl_ok += 1
print(f'CODER: {coder_ok}/{len(rd)} ({coder_ok/len(rd):.1%}), VL_ind: {vl_ok}/{len(rd)} ({vl_ok/len(rd):.1%})')

# Route
print('\n=== Route Analysis ===')
rt = [r for r in data if r['question_type']=='route_planning']
print(f'Score: {np.mean([r["score"] for r in rt]):.3f} ({len(rt)})')
strats = Counter()
for r in rt:
    reason = r.get('reasoning','')
    if 'route_slice_vote' in reason: strats['route_slice_vote'] += 1
    elif 'dir_vl_slice' in reason: strats['dir_vl_slice'] += 1
    else: strats['other'] += 1
print(f'Strats: {dict(strats)}')

# Abs_distance
print('\n=== Abs_distance Analysis ===')
ad = [r for r in data if r['question_type']=='object_abs_distance']
print(f'Score: {np.mean([r["score"] for r in ad]):.3f} ({len(ad)})')
strats = Counter()
for r in ad:
    reason = r.get('reasoning','')
    if 'abs_dist_vl_3vote' in reason: strats['abs_dist_vl_3vote'] += 1
    elif 'num_fastpath' in reason: strats['num_fastpath'] += 1
    else: strats['other'] += 1
print(f'Strats: {dict(strats)}')

# Room size
print('\n=== Room Size Analysis ===')
rs = [r for r in data if r['question_type']=='room_size_estimation']
print(f'Score: {np.mean([r["score"] for r in rs]):.3f} ({len(rs)})')
strats = Counter()
for r in rs:
    reason = r.get('reasoning','')
    if 'room_size_vl_3vote' in reason: strats['room_size_vl_3vote'] += 1
    elif 'num_fastpath' in reason: strats['num_fastpath'] += 1
    else: strats['other'] += 1
print(f'Strats: {dict(strats)}')
