#!/usr/bin/env python3
import json, re, numpy as np

with open('outputs/agentic_pipeline_v16_ta3_8gpu/detailed_results.json') as f:
    results = json.load(f)

all_dir = [r for r in results if r['question_type'].startswith('object_rel_direction')]
print(f'Direction total: N={len(all_dir)}, score={np.mean([r["score"] for r in all_dir]):.3f}')

# What if we just use VL_ind directly?
vi_correct = 0
for r in all_dir:
    m = re.search(r'\[vl_ind\] ([A-D])', r.get('reasoning',''))
    if m and m.group(1) == r['ground_truth']: vi_correct += 1
print(f'VL_ind only: {vi_correct}/{len(all_dir)} = {vi_correct/len(all_dir)*100:.1f}%')

# What if CODER agrees with VL → trust, disagrees → trust VL?
agree_right = agree_wrong = disagree_vl_right = disagree_coder_right = disagree_both_wrong = no_coder = 0
for r in all_dir:
    gt = r['ground_truth']
    reason = r.get('reasoning','')
    m_vi = re.search(r'\[vl_ind\] ([A-D])', reason)
    m_ca = re.search(r'\[coder\] ans=([A-D])', reason)
    vi = m_vi.group(1) if m_vi else None
    ca = m_ca.group(1) if m_ca else None
    if not ca:
        no_coder += 1; continue
    if vi == ca:  # Agree
        if vi == gt: agree_right += 1
        else: agree_wrong += 1
    else:  # Disagree
        if vi == gt: disagree_vl_right += 1
        elif ca == gt: disagree_coder_right += 1
        else: disagree_both_wrong += 1

print(f'\nCODER+VL agreement analysis (N={len(all_dir)-no_coder}, no_coder={no_coder}):')
print(f'  Agree & right: {agree_right}')
print(f'  Agree & wrong: {agree_wrong}')
print(f'  Disagree, VL right: {disagree_vl_right}')
print(f'  Disagree, CODER right: {disagree_coder_right}')
print(f'  Disagree, both wrong: {disagree_both_wrong}')
agree_total = agree_right + agree_wrong
disagree_total = disagree_vl_right + disagree_coder_right + disagree_both_wrong
print(f'\n  If always trust VL: {agree_right + disagree_vl_right}/{agree_total+disagree_total} = {(agree_right+disagree_vl_right)/(agree_total+disagree_total)*100:.1f}%')
print(f'  If trust agreement, else VL: {agree_right + disagree_vl_right}/{agree_total+disagree_total} = same')
print(f'  If trust agreement, else CODER: {agree_right + disagree_coder_right}/{agree_total+disagree_total} = {(agree_right+disagree_coder_right)/(agree_total+disagree_total)*100:.1f}%')
print(f'  Oracle: {agree_right + disagree_vl_right + disagree_coder_right}/{agree_total+disagree_total} = {(agree_right+disagree_vl_right+disagree_coder_right)/(agree_total+disagree_total)*100:.1f}%')
