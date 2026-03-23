#!/usr/bin/env python3
import json, re, numpy as np
from collections import Counter

with open('outputs/agentic_pipeline_v16_ta3_8gpu/detailed_results.json') as f:
    results = json.load(f)

# === REL_DISTANCE ===
rd = [r for r in results if r['question_type']=='object_rel_distance']
print(f'=== REL_DISTANCE: N={len(rd)}, score={np.mean([r["score"] for r in rd]):.3f} ===')

# Strategy breakdown
phc_sc = [r for r in rd if 'reldist_phc_sc' in r.get('reasoning','')]
other = [r for r in rd if r not in phc_sc]
if phc_sc: print(f'  PhaseC_SC: N={len(phc_sc)}, score={np.mean([r["score"] for r in phc_sc]):.3f}')
if other: print(f'  Other: N={len(other)}, score={np.mean([r["score"] for r in other]):.3f}')

# CODER accuracy
coder_ok = coder_total = 0
for r in rd:
    m = re.search(r'\[coder\] ans=([A-D])', r.get('reasoning',''))
    if m:
        coder_total += 1
        if m.group(1) == r['ground_truth']: coder_ok += 1
if coder_total: print(f'  CODER accuracy: {coder_ok}/{coder_total} ({coder_ok/coder_total*100:.0f}%)')

# VL independent accuracy
vi_ok = 0
for r in rd:
    m = re.search(r'\[vl_ind\] ([A-D])', r.get('reasoning',''))
    if m and m.group(1) == r['ground_truth']: vi_ok += 1
print(f'  VL_independent: {vi_ok}/{len(rd)} ({vi_ok/len(rd)*100:.0f}%)')

# PhaseC SC vote analysis: does SC improve or hurt?
for r in phc_sc[:5]:
    reason = r.get('reasoning','')
    m_vi = re.search(r'\[vl_ind\] ([A-D])', reason)
    m_sc = re.search(r'→ ([A-D])', reason)
    vi = m_vi.group(1) if m_vi else '?'
    sc = m_sc.group(1) if m_sc else '?'
    gt = r['ground_truth']
    print(f'    vl_ind={vi}{"✓" if vi==gt else "✗"} sc_final={sc}{"✓" if sc==gt else "✗"} gt={gt} | {reason[reason.find("[reldist"):reason.find("[reldist")+120]}')

# === DIRECTION ===
print(f'\n=== DIRECTION ===')
all_dir = [r for r in results if r['question_type'].startswith('object_rel_direction')]
print(f'AGG: N={len(all_dir)}, score={np.mean([r["score"] for r in all_dir]):.3f}')

for sub in ['object_rel_direction_easy','object_rel_direction_medium','object_rel_direction_hard']:
    dd = [r for r in results if r['question_type']==sub]
    if not dd: continue
    s = np.mean([r['score'] for r in dd])
    
    # Strategy breakdown
    vl_sc = [r for r in dd if 'dir_vl_sc4vote' in r.get('reasoning','')]
    fb = [r for r in dd if 'dir_fallback' in r.get('reasoning','')]
    oth = [r for r in dd if r not in vl_sc and r not in fb]
    
    print(f'\n  {sub}: N={len(dd)}, score={s:.3f}')
    if vl_sc: print(f'    VL_SC4vote: N={len(vl_sc)}, score={np.mean([r["score"] for r in vl_sc]):.3f}')
    if fb: print(f'    Fallback: N={len(fb)}, score={np.mean([r["score"] for r in fb]):.3f}')
    if oth: print(f'    Other: N={len(oth)}, score={np.mean([r["score"] for r in oth]):.3f}')
    
    # CODER vs VL independent
    c_ok = c_t = vi_ok2 = 0
    for r in dd:
        m = re.search(r'\[coder\] ans=([A-D])', r.get('reasoning',''))
        if m:
            c_t += 1
            if m.group(1) == r['ground_truth']: c_ok += 1
        m2 = re.search(r'\[vl_ind\] ([A-D])', r.get('reasoning',''))
        if m2 and m2.group(1) == r['ground_truth']: vi_ok2 += 1
    if c_t: print(f'    CODER: {c_ok}/{c_t} ({c_ok/c_t*100:.0f}%)')
    print(f'    VL_ind: {vi_ok2}/{len(dd)} ({vi_ok2/len(dd)*100:.0f}%)')

    # SC vote: did voting help or hurt vs VL_ind?
    helped = hurt = same = 0
    for r in vl_sc:
        m_vi = re.search(r'\[vl_ind\] ([A-D])', r.get('reasoning',''))
        m_sc = re.search(r'→ ([A-D])', r.get('reasoning',''))
        if not m_vi or not m_sc: continue
        vi, sc, gt = m_vi.group(1), m_sc.group(1), r['ground_truth']
        if vi == sc: same += 1
        elif sc == gt: helped += 1
        else: hurt += 1
    print(f'    SC effect: helped={helped}, hurt={hurt}, same={same}')
