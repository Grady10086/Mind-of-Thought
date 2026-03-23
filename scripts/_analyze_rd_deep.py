#!/usr/bin/env python3
"""Deep analysis: can GRID distance data solve rel_distance better?"""
import json, re, sys, numpy as np
sys.path.insert(0, '.')
from scripts.grid64_agentic_pipeline_v16 import _auto_coder_type

with open('outputs/agentic_pipeline_v16_ta3_8gpu/detailed_results.json') as f:
    results = json.load(f)

rd = [r for r in results if r['question_type']=='object_rel_distance']
print(f'=== REL_DISTANCE deep analysis: N={len(rd)} ===\n')

# For each sample, check if CODER answer matches GT
# Also check: does the Phase C prompt contain useful distance info?
for r in rd[:10]:
    q = r['question']
    gt = r['ground_truth']
    pred = r['prediction']
    reason = r.get('reasoning','')
    
    # Extract CODER answer
    m = re.search(r'\[coder\] ans=([A-D])', reason)
    ca = m.group(1) if m else '?'
    
    # Extract VL independent
    m2 = re.search(r'\[vl_ind\] ([A-D])', reason)
    vi = m2.group(1) if m2 else '?'
    
    # Extract final answer
    m3 = re.search(r'→ ([A-D])', reason)
    final = m3.group(1) if m3 else pred
    
    print(f'GT={gt} CODER={ca}{"✓" if ca==gt else "✗"} VL_ind={vi}{"✓" if vi==gt else "✗"} Final={final}{"✓" if final==gt else "✗"}')
    print(f'  Q: {q[:120]}')
    print(f'  Opts: {r.get("options",[])}')
    print()

# Overall: what if we just trust CODER for rel_distance?
print('\n=== Strategy comparison ===')
coder_correct = vl_correct = final_correct = 0
both_wrong = 0
for r in rd:
    gt = r['ground_truth']
    reason = r.get('reasoning','')
    m = re.search(r'\[coder\] ans=([A-D])', reason)
    ca = m.group(1) if m else None
    m2 = re.search(r'\[vl_ind\] ([A-D])', reason)
    vi = m2.group(1) if m2 else None
    
    if ca == gt: coder_correct += 1
    if vi == gt: vl_correct += 1
    if r['prediction'] == gt: final_correct += 1
    if ca and ca != gt and vi and vi != gt: both_wrong += 1

print(f'  CODER only: {coder_correct}/{len(rd)} ({coder_correct/len(rd)*100:.0f}%)')
print(f'  VL_ind only: {vl_correct}/{len(rd)} ({vl_correct/len(rd)*100:.0f}%)')
print(f'  Final (SC): {final_correct}/{len(rd)} ({final_correct/len(rd)*100:.0f}%)')
print(f'  Both wrong: {both_wrong}/{len(rd)} ({both_wrong/len(rd)*100:.0f}%)')

# Oracle: if we pick the right one between CODER and VL_ind
oracle = 0
for r in rd:
    gt = r['ground_truth']
    reason = r.get('reasoning','')
    m = re.search(r'\[coder\] ans=([A-D])', reason)
    ca = m.group(1) if m else None
    m2 = re.search(r'\[vl_ind\] ([A-D])', reason)
    vi = m2.group(1) if m2 else None
    if (ca and ca == gt) or (vi and vi == gt): oracle += 1
print(f'  Oracle(best of CODER,VL): {oracle}/{len(rd)} ({oracle/len(rd)*100:.0f}%)')
