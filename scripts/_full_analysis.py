#!/usr/bin/env python3
"""Full analysis across all TA versions + V14 to plan optimization."""
import json, re, numpy as np, glob
from pathlib import Path
from collections import Counter

# V14 baseline (from memory: V14 full 5130 samples)
V14 = {'object_counting': 0.823, 'object_abs_distance': 0.796, 'object_size_estimation': 0.878,
       'room_size_estimation': 0.915, 'object_rel_distance': 0.363,
       'object_rel_direction_easy': 0.571, 'object_rel_direction_medium': 0.481,
       'object_rel_direction_hard': 0.346, 'route_planning': 0.380,
       'obj_appearance_order': 0.640}

OFFICIAL = {'object_counting': 0.634, 'object_abs_distance': 0.455, 'object_size_estimation': 0.734,
            'room_size_estimation': 0.577, 'object_rel_distance': 0.522,
            'object_rel_direction': 0.512, 'route_planning': 0.304, 'obj_appearance_order': 0.610}

# Load TA4
ta4 = []
for g in range(8):
    p = Path(f'outputs/agentic_pipeline_v16_ta4_8gpu/gpu{g}/detailed_results.json')
    if p.exists():
        with open(p) as f: ta4.extend(json.load(f))

print(f'=== TA4 vs V14 vs Official ===')
print(f'TA4: {len(ta4)} samples\n')
print(f'{"Task":<35} {"TA4":>6} {"V14":>6} {"Off":>6}  {"vsV14":>7} {"vsOff":>7}')
print("-"*80)

tts = sorted(set(r['question_type'] for r in ta4))
all_scores = []
for qt in tts:
    qr = [r for r in ta4 if r['question_type'] == qt]
    s = np.mean([r['score'] for r in qr])
    all_scores.extend([r['score'] for r in qr])
    v14 = V14.get(qt, 0)
    off = OFFICIAL.get(qt, 0)
    d14 = s - v14 if v14 else 0
    doff = s - off if off else 0
    m14 = '✅' if d14 > 0 else '❌' if v14 > 0 else '  '
    moff = '✅' if doff > 0 else '❌' if off > 0 else '  '
    print(f'  {qt:<35} {s:>5.3f} {v14:>5.3f} {off:>5.3f}  {d14:>+6.3f}{m14} {doff:>+6.3f}{moff}')

# AGG direction
d = [r for r in ta4 if r['question_type'].startswith('object_rel_direction')]
if d:
    ds = np.mean([r['score'] for r in d])
    v14d = np.mean([V14.get(f'object_rel_direction_{x}', 0) for x in ['easy','medium','hard']])
    offd = OFFICIAL.get('object_rel_direction', 0.512)
    print(f'  {"[AGG] direction":<35} {ds:>5.3f} {v14d:>5.3f} {offd:>5.3f}  {ds-v14d:>+6.3f}{"✅" if ds>v14d else "❌"} {ds-offd:>+6.3f}{"✅" if ds>offd else "❌"}')

ov = np.mean(all_scores)
print(f'  {"Overall":<35} {ov:>5.3f} 0.651 0.544  {ov-0.651:>+6.3f}{"✅" if ov>0.651 else "❌"} {ov-0.544:>+6.3f}✅')

# === Deep per-type analysis ===
print(f'\n=== PER-TYPE STRATEGY ANALYSIS ===')
for qt in tts:
    qr = [r for r in ta4 if r['question_type'] == qt]
    if not qr: continue
    s = np.mean([r['score'] for r in qr])
    
    # Strategy distribution
    strategies = Counter()
    for r in qr:
        reason = r.get('reasoning', '')
        if 'num_fastpath' in reason: strategies['fastpath'] += 1
        elif 'room_size_vl' in reason: strategies['room_vl'] += 1
        elif 'num_low_vl' in reason: strategies['num_low_vl'] += 1
        elif 'reldist_' in reason: strategies['reldist_phc'] += 1
        elif 'dir_vl_slice' in reason: strategies['dir_slice'] += 1
        elif 'route_vl_vote' in reason: strategies['route_sc'] += 1
        elif 'appear_sc3vote' in reason: strategies['appear_sc'] += 1
        else: strategies['default'] += 1
    
    # CODER accuracy
    c_ok = c_t = 0
    for r in qr:
        m = re.search(r'\[coder\] ans=([A-D\d.]+)', r.get('reasoning',''))
        if m:
            c_t += 1
            ca = m.group(1)
            if ca == r['ground_truth']: c_ok += 1
            elif not r.get('options'):
                try:
                    if abs(float(ca) - float(r['ground_truth'])) / max(float(r['ground_truth']), 0.01) < 0.3:
                        c_ok += 1
                except: pass
    
    # VL independent accuracy
    vi_ok = 0
    for r in qr:
        m = re.search(r'\[vl_ind\] ([A-D])', r.get('reasoning',''))
        if m and m.group(1) == r['ground_truth']: vi_ok += 1
    
    v14 = V14.get(qt, 0)
    gap = s - v14
    print(f'\n  {qt}: score={s:.3f} V14={v14:.3f} gap={gap:+.3f}')
    print(f'    strategies: {dict(strategies)}')
    if c_t > 0: print(f'    CODER: {c_ok}/{c_t} ({c_ok/c_t*100:.0f}%)')
    if vi_ok > 0 or any(r.get('options') for r in qr): print(f'    VL_ind: {vi_ok}/{len(qr)} ({vi_ok/len(qr)*100:.0f}%)')
    
    # For tasks below V14, show wrong sample patterns
    if gap < -0.05:
        wrong = [r for r in qr if r['score'] < 0.5]
        print(f'    WRONG: {len(wrong)}/{len(qr)}')
        for r in wrong[:3]:
            print(f'      pred={r["prediction"][:10]} gt={r["ground_truth"][:10]} | {r.get("reasoning","")[:120]}')
