#!/usr/bin/env python3
"""Three-way comparison: V17 vs V18 vs V19"""
import json, numpy as np, re
from collections import Counter, defaultdict

v17 = json.load(open('outputs/agentic_pipeline_v17_coevo/detailed_results.json'))
v18 = json.load(open('outputs/agentic_pipeline_v18_mot/detailed_results.json'))
v19 = json.load(open('outputs/agentic_pipeline_v19_ref/detailed_results.json'))

def build_map(results):
    m = {}
    for r in results:
        key = (r['scene_name'], r['question'][:80])
        m[key] = r
    return m

m17, m18, m19 = build_map(v17), build_map(v18), build_map(v19)
common = set(m17.keys()) & set(m18.keys()) & set(m19.keys())
print(f'V17={len(v17)}, V18={len(v18)}, V19={len(v19)}, Common={len(common)}')

# Per-task comparison
by_type = defaultdict(list)
for k in common:
    r17, r18, r19 = m17[k], m18[k], m19[k]
    by_type[r17['question_type']].append({
        's17': r17['score'], 's18': r18['score'], 's19': r19['score'],
        'pred17': r17.get('prediction',''), 'pred18': r18.get('prediction',''),
        'pred19': r19.get('prediction',''),
        'gt': r17['ground_truth'],
        'reasoning_19': r19.get('reasoning',''),
        'converged_phase_19': r19.get('converged_phase', 3),
    })

print('\n' + '='*120)
print(f'{"Task":<35} {"N":>4} {"V17":>6} {"V18":>6} {"V19":>6} {"V19-V17":>8} {"V19-V18":>8}')
print('-'*90)
for qt in sorted(by_type.keys()):
    items = by_type[qt]; n = len(items)
    s17 = np.mean([x['s17'] for x in items])
    s18 = np.mean([x['s18'] for x in items])
    s19 = np.mean([x['s19'] for x in items])
    print(f'  {qt:<35} {n:>3} {s17:>5.3f} {s18:>5.3f} {s19:>5.3f} {s19-s17:>+7.3f} {s19-s18:>+7.3f}')

ov17 = np.mean([m17[k]['score'] for k in common])
ov18 = np.mean([m18[k]['score'] for k in common])
ov19 = np.mean([m19[k]['score'] for k in common])
print('-'*90)
print(f'  {"Overall":<35} {len(common):>3} {ov17:>5.3f} {ov18:>5.3f} {ov19:>5.3f} {ov19-ov17:>+7.3f} {ov19-ov18:>+7.3f}')

# V19 helped/hurt vs V17
print('\n=== V19 vs V17: helped/hurt per task ===')
choice_types = ['object_rel_direction_easy','object_rel_direction_hard','object_rel_direction_medium',
                'object_rel_distance','route_planning','obj_appearance_order']
for qt in sorted(by_type.keys()):
    items = by_type[qt]
    h19 = sum(1 for x in items if x['s19'] > x['s17'] + 0.001)
    d19 = sum(1 for x in items if x['s19'] < x['s17'] - 0.001)
    s19 = sum(1 for x in items if abs(x['s19'] - x['s17']) < 0.001)
    h18 = sum(1 for x in items if x['s18'] > x['s17'] + 0.001)
    d18 = sum(1 for x in items if x['s18'] < x['s17'] - 0.001)
    print(f'  {qt:<35} V19_vs_V17: helped={h19} hurt={d19} same={s19} | V18_vs_V17: helped={h18} hurt={d18}')

# V19 convergence analysis
print('\n=== V19 Convergence ===')
for qt in sorted(by_type.keys()):
    items = by_type[qt]
    phases = Counter(x['converged_phase_19'] for x in items)
    print(f'  {qt:<35} P2_converge={phases.get(2,0)} P3_referee={phases.get(3,0)}')

# V19 hypothesis anchoring check (should be 0 since P1/P2 don't see hypothesis)
print('\n=== V19 CODER anchoring check (choice tasks only) ===')
for qt in choice_types:
    items = by_type.get(qt, [])
    if not items: continue
    followed = 0; total = 0
    for x in items:
        total += 1
        r19 = m19[(k for k in common if m19[k]['question_type'] == qt and 
                    m19[k]['ground_truth'] == x['gt']).__next__() if False else None]
    # Read from reasoning
    for k in common:
        r19 = m19[k]
        if r19['question_type'] != qt: continue
        reasoning = r19.get('reasoning', '')
        m_coder = re.search(r'\[coder\]\s*([A-D])', reasoning)
        if not m_coder: continue
        coder_ans = m_coder.group(1)
        pred = r19.get('prediction', '')
        if pred == coder_ans:
            followed += 1
    print(f'  {qt:<35} VL_followed_coder={followed}/{len(items)}')

# Key: V19 hurt cases vs V17
print('\n=== V19 hurt cases (vs V17) ===')
all_hurt = []
for qt in sorted(by_type.keys()):
    for k in common:
        r17, r19 = m17[k], m19[k]
        if r17['question_type'] != qt: continue
        if r19['score'] < r17['score'] - 0.001:
            all_hurt.append((qt, k, r17, r19))

print(f'Total V19 hurt (vs V17): {len(all_hurt)}')
for qt, k, r17, r19 in all_hurt[:15]:
    print(f'\n  [{qt}] V17={r17["score"]:.3f}→V19={r19["score"]:.3f} gt={r17["ground_truth"]}')
    print(f'    pred17={r17.get("prediction","")}  pred19={r19.get("prediction","")}')
    print(f'    R19: {r19.get("reasoning","")[:150]}')
