#!/usr/bin/env python3
"""Deep analysis of TA5 for TA7 optimization."""
import json, numpy as np, re
from collections import Counter

with open('outputs/agentic_pipeline_v16_ta5_8gpu/detailed_results.json') as f:
    data = json.load(f)

# === Direction: detailed disagree analysis ===
print('=== Direction: Final answer analysis ===')
dir_r = [r for r in data if r['question_type'].startswith('object_rel_direction')]
for r in dir_r:
    reason = r.get('reasoning','')
    gt = str(r['ground_truth']).strip().upper()
    pred = str(r['prediction']).strip().upper()
    ca = ''; vi = ''
    m = re.search(r'\[coder\] ans=([A-D])', reason)
    if m: ca = m.group(1)
    m2 = re.search(r'\[vl_ind\] ([A-D])', reason)
    if m2: vi = m2.group(1)
    # Check: current final answer = dir_vl_slice or dir_slice
    m3 = re.search(r'\[dir_(?:vl_)?slice\] ans=([A-D])', reason)
    final_slice = m3.group(1) if m3 else pred
    r['_ca'] = ca; r['_vi'] = vi; r['_final'] = final_slice; r['_gt'] = gt

# Strategy analysis: what if we use different rules?
strategies = {}

# S1: Always VL_ind (no slice)
s1 = sum(1 for r in dir_r if r['_vi'] == r['_gt']) / len(dir_r)
strategies['S1_always_VL_ind'] = s1

# S2: Always current (VL+slice)
s2 = np.mean([r['score'] for r in dir_r])
strategies['S2_current_vl_slice'] = s2

# S3: When agree → trust agree, when disagree → trust VL_ind
s3_ok = 0
for r in dir_r:
    if r['_ca'] == r['_vi']:
        if r['_ca'] == r['_gt']: s3_ok += 1
    else:
        if r['_vi'] == r['_gt']: s3_ok += 1
strategies['S3_agree_trust_else_VL'] = s3_ok / len(dir_r)

# S4: When agree → trust agree, when disagree → trust current (VL+slice)
s4_ok = 0
for r in dir_r:
    if r['_ca'] and r['_vi'] and r['_ca'] == r['_vi']:
        if r['_ca'] == r['_gt']: s4_ok += 1
    else:
        if r['_final'] == r['_gt']: s4_ok += 1
strategies['S4_agree_trust_else_slice'] = s4_ok / len(dir_r)

# S5: Always CODER
s5 = sum(1 for r in dir_r if r['_ca'] == r['_gt']) / len(dir_r)
strategies['S5_always_CODER'] = s5

# S6: 2-of-3 majority vote (CODER, VL_ind, VL_slice)
s6_ok = 0
for r in dir_r:
    votes = Counter()
    if r['_ca']: votes[r['_ca']] += 1
    if r['_vi']: votes[r['_vi']] += 1
    if r['_final']: votes[r['_final']] += 1
    if votes:
        best = votes.most_common(1)[0][0]
        if best == r['_gt']: s6_ok += 1
strategies['S6_majority_3vote'] = s6_ok / len(dir_r)

# S7: Oracle (best of CODER, VL_ind, VL_slice)
s7_ok = 0
for r in dir_r:
    if r['_gt'] in [r['_ca'], r['_vi'], r['_final']]: s7_ok += 1
strategies['S7_oracle'] = s7_ok / len(dir_r)

print(f'\nDirection strategies (n={len(dir_r)}):')
for k, v in sorted(strategies.items(), key=lambda x: -x[1]):
    marker = '✅' if v > 0.512 else '❌'
    print(f'  {k}: {v:.3f} ({v:.1%}) {marker} (vs official 0.512)')

# === Rel_distance: detailed analysis ===
print('\n\n=== Rel_distance: Strategy Analysis ===')
rd = [r for r in data if r['question_type']=='object_rel_distance']
for r in rd:
    reason = r.get('reasoning','')
    gt = str(r['ground_truth']).strip().upper()
    ca = ''; vi = ''
    m = re.search(r'\[coder\] ans=([A-D])', reason)
    if m: ca = m.group(1)
    m2 = re.search(r'\[vl_ind\] ([A-D])', reason)
    if m2: vi = m2.group(1)
    r['_ca'] = ca; r['_vi'] = vi; r['_gt'] = gt

# Strategy analysis
rd_strategies = {}
rd_strategies['current_sc3'] = np.mean([r['score'] for r in rd])
rd_strategies['always_VL_ind'] = sum(1 for r in rd if r['_vi'] == r['_gt']) / len(rd)
rd_strategies['always_CODER'] = sum(1 for r in rd if r['_ca'] == r['_gt']) / len(rd)

# When agree trust, disagree trust VL
s_ok = 0
for r in rd:
    if r['_ca'] == r['_vi']:
        if r['_ca'] == r['_gt']: s_ok += 1
    else:
        if r['_vi'] == r['_gt']: s_ok += 1
rd_strategies['agree_trust_else_VL'] = s_ok / len(rd)

# When agree trust, disagree trust CODER
s_ok = 0
for r in rd:
    if r['_ca'] == r['_vi']:
        if r['_ca'] == r['_gt']: s_ok += 1
    else:
        if r['_ca'] == r['_gt']: s_ok += 1
rd_strategies['agree_trust_else_CODER'] = s_ok / len(rd)

# Oracle
oracle = sum(1 for r in rd if r['_gt'] in [r['_ca'], r['_vi']]) / len(rd)
rd_strategies['oracle_best_of_2'] = oracle

# Majority 3-vote (CODER, VL_ind, SC3_final)
s_ok = 0
for r in rd:
    votes = Counter()
    if r['_ca']: votes[r['_ca']] += 1
    if r['_vi']: votes[r['_vi']] += 1
    pred = str(r['prediction']).strip().upper()
    if pred: votes[pred] += 1
    best = votes.most_common(1)[0][0] if votes else 'A'
    if best == r['_gt']: s_ok += 1
rd_strategies['majority_3(CODER,VL,SC3)'] = s_ok / len(rd)

print(f'\nRel_distance strategies (n={len(rd)}):')
for k, v in sorted(rd_strategies.items(), key=lambda x: -x[1]):
    marker = '✅' if v > 0.522 else '❌'
    print(f'  {k}: {v:.3f} ({v:.1%}) {marker} (vs official 0.522)')

# Agree/disagree breakdown for rel_distance
agree = [(r['_ca'], r['_gt']) for r in rd if r['_ca'] == r['_vi'] and r['_ca']]
disagree = [(r['_ca'], r['_vi'], r['_gt']) for r in rd if r['_ca'] != r['_vi'] and r['_ca'] and r['_vi']]
no_ca = [r for r in rd if not r['_ca']]
print(f'\nAgree: {len(agree)}, Disagree: {len(disagree)}, No CODER: {len(no_ca)}')
if agree:
    agree_ok = sum(1 for ca, gt in agree if ca == gt)
    print(f'  Agree correct: {agree_ok}/{len(agree)} ({agree_ok/len(agree):.1%})')
if disagree:
    vl_ok = sum(1 for ca, vi, gt in disagree if vi == gt)
    co_ok = sum(1 for ca, vi, gt in disagree if ca == gt)
    print(f'  Disagree VL_ok: {vl_ok}/{len(disagree)}, CODER_ok: {co_ok}/{len(disagree)}, both_wrong: {len(disagree)-vl_ok-co_ok}/{len(disagree)}')

# === Route: why all "other" strategy? ===
print('\n\n=== Route: reasoning patterns ===')
rt = [r for r in data if r['question_type']=='route_planning']
for r in rt[:5]:
    reason = r.get('reasoning','')
    print(f'  pred={r["prediction"]} gt={r["ground_truth"]} score={r["score"]:.1f}')
    print(f'    reason: {reason[:200]}')
    print()

# Count eff_type routing
for r in rt:
    reason = r.get('reasoning','')
    if '[dir_vl_slice]' in reason: r['_route_path'] = 'dir_path'
    elif '[route_slice_vote]' in reason: r['_route_path'] = 'route_path'
    elif '[r1]' in reason: r['_route_path'] = 'default_phaseC'
    else: r['_route_path'] = 'unknown'
paths = Counter(r['_route_path'] for r in rt)
print(f'Route path distribution: {dict(paths)}')
for p in paths:
    filtered = [r for r in rt if r['_route_path'] == p]
    sc = np.mean([r['score'] for r in filtered])
    print(f'  {p}: {sc:.3f} ({len(filtered)})')

# Room size: why ta5=0.860 but ta6=0.974?
print('\n\n=== Room Size: FastPath problem ===')
rs = [r for r in data if r['question_type']=='room_size_estimation']
fp = [r for r in rs if 'num_fastpath' in r.get('reasoning','').lower()]
vl = [r for r in rs if 'room_size_vl_3vote' in r.get('reasoning','')]
print(f'FastPath: {len(fp)} samples, score={np.mean([r["score"] for r in fp]):.3f}')
print(f'VL 3-vote: {len(vl)} samples, score={np.mean([r["score"] for r in vl]):.3f}')
for r in fp:
    print(f'  FP: pred={r["prediction"]} gt={r["ground_truth"]} score={r["score"]:.3f} reason={r["reasoning"][:100]}')
