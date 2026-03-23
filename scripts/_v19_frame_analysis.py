#!/usr/bin/env python3
"""Analyze: does Active Focusing actually help? Does belief correction change frame selection?"""
import json, numpy as np, re
from collections import Counter, defaultdict

v19 = json.load(open('outputs/agentic_pipeline_v19_ref/detailed_results.json'))

choice_types = {'object_rel_direction_easy','object_rel_direction_hard','object_rel_direction_medium',
                'object_rel_distance','route_planning','obj_appearance_order'}

# Q1: Among P2-converge samples, was VL_P1 already correct?
print('=== Q1: P2 converge — was P1 already correct? ===')
for qt in sorted(choice_types):
    converge_correct = 0; converge_total = 0
    diverge_total = 0; diverge_p1_correct = 0; diverge_p2_correct = 0
    for r in v19:
        if r['question_type'] != qt: continue
        if not r.get('options'): continue
        reasoning = r.get('reasoning', '')
        phase = r.get('converged_phase', 3)
        
        m1 = re.search(r'\[vl_full\]\s*([A-D])', reasoning)
        m2 = re.search(r'\[vl_focused_\w+\]\s*([A-D])', reasoning) or re.search(r'\[vl_slice\]\s*([A-D])', reasoning)
        vl1 = m1.group(1) if m1 else '?'
        vl2 = m2.group(1) if m2 else '?'
        gt = str(r['ground_truth'])
        
        if phase == 2:
            converge_total += 1
            if r['score'] > 0.5: converge_correct += 1
        else:
            diverge_total += 1
            if vl1 == gt: diverge_p1_correct += 1
            if vl2 == gt: diverge_p2_correct += 1
    
    ca = converge_correct/max(1,converge_total)
    dp1 = diverge_p1_correct/max(1,diverge_total)
    dp2 = diverge_p2_correct/max(1,diverge_total)
    print(f'  {qt:<35} converge: {converge_correct}/{converge_total} ({ca:.0%}) | '
          f'diverge: P1={diverge_p1_correct}/{diverge_total} ({dp1:.0%}) P2={diverge_p2_correct}/{diverge_total} ({dp2:.0%})')

# Q2: Among diverge samples, P1 vs P2 — which is more often correct?
print('\n=== Q2: Diverge samples — P1 vs P2 accuracy ===')
all_p1_correct = 0; all_p2_correct = 0; all_div = 0
for r in v19:
    if r['question_type'] not in choice_types: continue
    if not r.get('options'): continue
    if r.get('converged_phase', 3) != 3: continue
    
    reasoning = r.get('reasoning', '')
    m1 = re.search(r'\[vl_full\]\s*([A-D])', reasoning)
    m2 = re.search(r'\[vl_focused_\w+\]\s*([A-D])', reasoning) or re.search(r'\[vl_slice\]\s*([A-D])', reasoning)
    vl1 = m1.group(1) if m1 else '?'
    vl2 = m2.group(1) if m2 else '?'
    gt = str(r['ground_truth'])
    
    all_div += 1
    if vl1 == gt: all_p1_correct += 1
    if vl2 == gt: all_p2_correct += 1

print(f'  Total diverge: {all_div}')
print(f'  P1 (full) correct: {all_p1_correct}/{all_div} ({all_p1_correct/max(1,all_div):.0%})')
print(f'  P2 (focused) correct: {all_p2_correct}/{all_div} ({all_p2_correct/max(1,all_div):.0%})')

# Q3: What about CODER accuracy on diverge samples?
print('\n=== Q3: CODER accuracy on diverge samples ===')
coder_correct = 0; coder_total = 0
for r in v19:
    if r['question_type'] not in choice_types: continue
    if not r.get('options'): continue
    if r.get('converged_phase', 3) != 3: continue
    
    reasoning = r.get('reasoning', '')
    mc = re.search(r'\[coder\]\s*([A-D])', reasoning)
    if not mc: continue
    coder_total += 1
    if mc.group(1) == str(r['ground_truth']): coder_correct += 1

print(f'  CODER on diverge: {coder_correct}/{coder_total} ({coder_correct/max(1,coder_total):.0%})')

# Q4: Oracle of P1, P2, CODER on diverge samples
print('\n=== Q4: Oracle P1∪P2∪CODER on diverge ===')
oracle = 0; total = 0
for r in v19:
    if r['question_type'] not in choice_types: continue
    if not r.get('options'): continue
    if r.get('converged_phase', 3) != 3: continue
    
    reasoning = r.get('reasoning', '')
    m1 = re.search(r'\[vl_full\]\s*([A-D])', reasoning)
    m2 = re.search(r'\[vl_focused_\w+\]\s*([A-D])', reasoning) or re.search(r'\[vl_slice\]\s*([A-D])', reasoning)
    mc = re.search(r'\[coder\]\s*([A-D])', reasoning)
    gt = str(r['ground_truth'])
    
    candidates = []
    if m1: candidates.append(m1.group(1))
    if m2: candidates.append(m2.group(1))
    if mc: candidates.append(mc.group(1))
    
    total += 1
    if gt in candidates: oracle += 1

print(f'  Oracle(P1∪P2∪CODER) on diverge: {oracle}/{total} ({oracle/max(1,total):.0%})')

# Q5: What if we simply trust P2 (focused) on all diverge?
print('\n=== Q5: What-if: always trust P2 on diverge ===')
p2_scores = []
for r in v19:
    if r['question_type'] not in choice_types: continue
    if not r.get('options'): continue
    if r.get('converged_phase', 3) != 3: continue
    
    reasoning = r.get('reasoning', '')
    m2 = re.search(r'\[vl_focused_\w+\]\s*([A-D])', reasoning) or re.search(r'\[vl_slice\]\s*([A-D])', reasoning)
    vl2 = m2.group(1) if m2 else '?'
    gt = str(r['ground_truth'])
    p2_scores.append(1.0 if vl2 == gt else 0.0)

print(f'  P2-only on diverge: {sum(p2_scores):.0f}/{len(p2_scores)} ({np.mean(p2_scores):.0%})')
