#!/usr/bin/env python3
"""Per-sample comparison: V17 vs V18"""
import json, numpy as np
from collections import Counter, defaultdict

v18 = json.load(open('outputs/agentic_pipeline_v18_mot/detailed_results.json'))
v17 = json.load(open('outputs/agentic_pipeline_v17_coevo/detailed_results.json'))
print(f'V18: {len(v18)} samples, V17: {len(v17)} samples')

v17_map = {}; v18_map = {}
for r in v17:
    key = (r['scene_name'], r['question'][:80])
    v17_map[key] = r
for r in v18:
    key = (r['scene_name'], r['question'][:80])
    v18_map[key] = r

common = set(v17_map.keys()) & set(v18_map.keys())
print(f'Common: {len(common)}')

by_type = defaultdict(list)
for k in common:
    r17 = v17_map[k]; r18 = v18_map[k]
    qt = r17['question_type']
    s17 = r17['score']; s18 = r18['score']
    by_type[qt].append({
        'key': k, 's17': s17, 's18': s18,
        'pred17': r17.get('prediction',''), 'pred18': r18.get('prediction',''),
        'gt': r17['ground_truth'],
        'reasoning_18': r18.get('reasoning',''),
        'reasoning_17': r17.get('reasoning',''),
        'converged_phase_18': r18.get('converged_phase', 3),
        'vl_calls_18': r18.get('vl_calls', 0),
        'belief_modified_18': r18.get('belief_modified', False),
    })

print('\n' + '='*110)
print('Per-Task: V17 vs V18')
print('='*110)
for qt in sorted(by_type.keys()):
    items = by_type[qt]; n = len(items)
    s17_avg = np.mean([x['s17'] for x in items])
    s18_avg = np.mean([x['s18'] for x in items])
    helped = [x for x in items if x['s18'] > x['s17'] + 0.001]
    hurt = [x for x in items if x['s18'] < x['s17'] - 0.001]
    same = [x for x in items if abs(x['s18'] - x['s17']) < 0.001]
    print(f'\n{qt:35s} n={n:3d} V17={s17_avg:.3f} V18={s18_avg:.3f} d={s18_avg-s17_avg:+.3f} | helped={len(helped)} hurt={len(hurt)} same={len(same)}')
    
    # Show helped samples
    if helped:
        print(f'  HELPED ({len(helped)}):')
        for x in helped[:5]:
            print(f'    V17={x["s17"]:.3f}→V18={x["s18"]:.3f} pred17={x["pred17"]} pred18={x["pred18"]} gt={str(x["gt"])[:20]}')
            print(f'      R18: {x["reasoning_18"][:120]}')
    
    # Show hurt samples  
    if hurt:
        print(f'  HURT ({len(hurt)}):')
        for x in hurt[:5]:
            print(f'    V17={x["s17"]:.3f}→V18={x["s18"]:.3f} pred17={x["pred17"]} pred18={x["pred18"]} gt={str(x["gt"])[:20]}')
            print(f'      R17: {x["reasoning_17"][:120]}')
            print(f'      R18: {x["reasoning_18"][:120]}')

print('\n' + '='*110)
ov17 = np.mean([v17_map[k]['score'] for k in common])
ov18 = np.mean([v18_map[k]['score'] for k in common])
print(f'Overall: V17={ov17:.4f} V18={ov18:.4f} delta={ov18-ov17:+.4f}')

# Convergence analysis for V18
print('\n=== V18 Convergence Analysis ===')
for qt in sorted(by_type.keys()):
    items = by_type[qt]
    phases = Counter(x['converged_phase_18'] for x in items)
    vl_avg = np.mean([x['vl_calls_18'] for x in items])
    bm = sum(1 for x in items if x['belief_modified_18'])
    print(f'  {qt:35s} P1={phases.get(1,0)} P2={phases.get(2,0)} P3={phases.get(3,0)} VL_avg={vl_avg:.1f} bel_mod={bm}')

# Key diagnosis: For choice tasks where V18 hurt, what's the pattern?
print('\n=== Diagnosis: Choice tasks where V18 hurt ===')
choice_types = ['object_rel_direction_easy','object_rel_direction_hard','object_rel_direction_med',
                'object_rel_distance','route_planning','obj_appearance_order']
all_hurt = []
for qt in choice_types:
    for x in by_type.get(qt, []):
        if x['s18'] < x['s17'] - 0.001:
            all_hurt.append((qt, x))
print(f'Total hurt choice samples: {len(all_hurt)}')
for qt, x in all_hurt:
    print(f'\n  [{qt}] V17={x["s17"]:.3f}→V18={x["s18"]:.3f} gt={x["gt"]}')
    print(f'    pred17={x["pred17"]}  pred18={x["pred18"]}')
    print(f'    R17: {x["reasoning_17"][:150]}')
    print(f'    R18: {x["reasoning_18"][:150]}')
    # Check if hypothesis anchoring is the issue
    r18 = x['reasoning_18']
    if 'coder=' in r18:
        import re
        m = re.search(r'coder=([A-D])', r18)
        coder_ans = m.group(1) if m else '?'
        # Check if V18's answer follows CODER despite it being wrong
        if x['pred18'] == coder_ans and coder_ans != x['gt']:
            print(f'    *** HYPOTHESIS ANCHORING: VL followed wrong CODER answer {coder_ans} ***')

print('\n=== Hypothesis Anchoring Analysis (all choice tasks) ===')
anchor_count = 0; anchor_hurt = 0; total_choice = 0
for qt in choice_types:
    for x in by_type.get(qt, []):
        total_choice += 1
        import re
        m = re.search(r'coder=([A-D])', x['reasoning_18'])
        coder_ans = m.group(1) if m else ''
        if coder_ans and x['pred18'] == coder_ans:
            anchor_count += 1
            if coder_ans != x['gt']:
                anchor_hurt += 1
                
coder_right = 0; coder_total = 0
for qt in choice_types:
    for x in by_type.get(qt, []):
        import re
        m = re.search(r'coder=([A-D])', x['reasoning_18'])
        if m:
            coder_total += 1
            if m.group(1) == x['gt']:
                coder_right += 1

print(f'Total choice: {total_choice}')
print(f'CODER accuracy: {coder_right}/{coder_total} = {coder_right/max(1,coder_total):.1%}')
print(f'VL followed CODER: {anchor_count}/{total_choice} = {anchor_count/max(1,total_choice):.1%}')
print(f'VL anchored on WRONG CODER: {anchor_hurt}/{total_choice} = {anchor_hurt/max(1,total_choice):.1%}')

# V17 VL independence check 
print('\n=== V17 vs V18: VL independence from CODER ===')
for qt in choice_types:
    items = by_type.get(qt, [])
    if not items: continue
    followed_17 = 0; followed_18 = 0
    for x in items:
        import re
        # V18 coder
        m18 = re.search(r'coder=([A-D])', x['reasoning_18'])
        c18 = m18.group(1) if m18 else ''
        if c18 and x['pred18'] == c18: followed_18 += 1
        # V17 doesn't directly expose coder in same way, skip
    if items:
        print(f'  {qt:35s} V18_followed_coder={followed_18}/{len(items)} ({followed_18/len(items):.0%})')
