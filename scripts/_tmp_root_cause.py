#!/usr/bin/env python3
"""Deep root cause analysis for rel_distance, route, direction, appearance_order
Focus: what would give the HIGHEST uplift with minimal risk"""
import json, numpy as np, re
from collections import Counter, defaultdict

with open('outputs/agentic_pipeline_v7_full/merged/detailed_results.json') as f:
    results = json.load(f)

with open('outputs/agentic_pipeline_v6_full/merged/detailed_results.json') as f:
    v6_results = json.load(f)

# ====================================================================
# 1. rel_distance: CODER accuracy breakdown
# ====================================================================
print("="*100)
print("1. REL_DISTANCE: CODER准确率分解")
print("="*100)

rd = [r for r in results if r['question_type'] == 'object_rel_distance']
# 从reasoning提取coder answer和gt
coder_correct = coder_wrong = coder_noanswer = 0
coder_correct_decide_correct = coder_correct_decide_wrong = 0
coder_wrong_decide_correct = coder_wrong_decide_wrong = 0
for r in rd:
    reasoning = r.get('reasoning', '')
    m = re.search(r'answer=([A-D])', reasoning)
    gt = str(r['ground_truth']).strip().upper()
    if m:
        coder_ans = m.group(1)
        if coder_ans == gt:
            coder_correct += 1
            if r['score'] > 0:
                coder_correct_decide_correct += 1
            else:
                coder_correct_decide_wrong += 1
        else:
            coder_wrong += 1
            if r['score'] > 0:
                coder_wrong_decide_correct += 1
            else:
                coder_wrong_decide_wrong += 1
    else:
        coder_noanswer += 1

print(f"  Total: {len(rd)}")
print(f"  CODER correct: {coder_correct} ({100*coder_correct/len(rd):.1f}%)")
print(f"    → Decide also correct: {coder_correct_decide_correct}")
print(f"    → Decide overturned (LOSS): {coder_correct_decide_wrong}")
print(f"  CODER wrong: {coder_wrong} ({100*coder_wrong/len(rd):.1f}%)")
print(f"    → Decide rescued: {coder_wrong_decide_correct}")
print(f"    → Decide also wrong: {coder_wrong_decide_wrong}")
print(f"  CODER no answer: {coder_noanswer}")

# Oracle: 如果Decide不推翻正确CODER，分数是多少
oracle_score = (coder_correct + coder_wrong_decide_correct) / len(rd)
print(f"\n  Oracle (never overturn correct CODER): {oracle_score:.3f} vs actual {np.mean([r['score'] for r in rd]):.3f}")
# 只修复Decide推翻正确CODER的损失
loss_from_overturn = coder_correct_decide_wrong / len(rd)
print(f"  Loss from Decide overturning correct CODER: {loss_from_overturn:.3f} ({coder_correct_decide_wrong} samples)")

# ====================================================================
# 2. DIRECTION: 同样分析
# ====================================================================
print("\n" + "="*100)
print("2. DIRECTION (all): CODER准确率分解")
print("="*100)

for qt in ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard']:
    dr = [r for r in results if r['question_type'] == qt]
    cc = cw = cn = ccd = ccw = cwc = cww = 0
    for r in dr:
        reasoning = r.get('reasoning', '')
        m = re.search(r'answer=([A-D])', reasoning)
        gt = str(r['ground_truth']).strip().upper()
        if m:
            coder_ans = m.group(1)
            if coder_ans == gt:
                cc += 1
                if r['score'] > 0: ccd += 1
                else: ccw += 1
            else:
                cw += 1
                if r['score'] > 0: cwc += 1
                else: cww += 1
        else:
            cn += 1
    
    actual = np.mean([r['score'] for r in dr])
    oracle = (cc + cwc) / len(dr) if dr else 0
    print(f"\n  {qt}:")
    print(f"    CODER correct: {cc}/{len(dr)} ({100*cc/len(dr):.1f}%) | Decide kept: {ccd}, Decide overturned: {ccw} (LOSS)")
    print(f"    CODER wrong: {cw}/{len(dr)} ({100*cw/len(dr):.1f}%) | Decide rescued: {cwc}, Decide also wrong: {cww}")
    print(f"    Oracle: {oracle:.3f} vs actual: {actual:.3f} | Loss from overturn: {ccw}/{len(dr)}={100*ccw/len(dr):.1f}%")

# ====================================================================
# 3. ROUTE: 分析
# ====================================================================
print("\n" + "="*100)
print("3. ROUTE: CODER准确率分解")
print("="*100)

rt = [r for r in results if r['question_type'] == 'route_planning']
cc = cw = cn = ccd = ccw = cwc = cww = 0
not_found_cnt = 0
for r in rt:
    reasoning = r.get('reasoning', '')
    m = re.search(r'answer=([A-D])', reasoning)
    gt = str(r['ground_truth']).strip().upper()
    if 'not found' in reasoning.lower():
        not_found_cnt += 1
    if m:
        coder_ans = m.group(1)
        if coder_ans == gt:
            cc += 1
            if r['score'] > 0: ccd += 1
            else: ccw += 1
        else:
            cw += 1
            if r['score'] > 0: cwc += 1
            else: cww += 1
    else:
        cn += 1

actual = np.mean([r['score'] for r in rt])
oracle = (cc + cwc) / len(rt) if rt else 0
print(f"  Total: {len(rt)}, not_found: {not_found_cnt} ({100*not_found_cnt/len(rt):.1f}%)")
print(f"  CODER correct: {cc}/{len(rt)} ({100*cc/len(rt):.1f}%)")
print(f"    → Decide kept: {ccd}, Decide overturned: {ccw} (LOSS)")
print(f"  CODER wrong: {cw}/{len(rt)} ({100*cw/len(rt):.1f}%)")
print(f"    → Decide rescued: {cwc}, Decide also wrong: {cww}")
print(f"  Oracle: {oracle:.3f} vs actual: {actual:.3f}")

# ====================================================================
# 4. APPEARANCE_ORDER: 分析
# ====================================================================
print("\n" + "="*100)
print("4. APPEARANCE_ORDER: CODER准确率分解")
print("="*100)

ao = [r for r in results if r['question_type'] == 'obj_appearance_order']
cc = cw = cn = ccd = ccw = cwc = cww = 0
for r in ao:
    reasoning = r.get('reasoning', '')
    m = re.search(r'answer=([A-D])', reasoning)
    gt = str(r['ground_truth']).strip().upper()
    if m:
        coder_ans = m.group(1)
        if coder_ans == gt:
            cc += 1
            if r['score'] > 0: ccd += 1
            else: ccw += 1
        else:
            cw += 1
            if r['score'] > 0: cwc += 1
            else: cww += 1
    else:
        cn += 1

actual = np.mean([r['score'] for r in ao])
oracle = (cc + cwc) / len(ao) if ao else 0
print(f"  Total: {len(ao)}")
print(f"  CODER correct: {cc}/{len(ao)} ({100*cc/len(ao):.1f}%)")
print(f"    → Decide kept: {ccd}, Decide overturned: {ccw} (LOSS)")
print(f"  CODER wrong: {cw}/{len(ao)} ({100*cw/len(ao):.1f}%)")
print(f"    → Decide rescued: {cwc}, Decide also wrong: {cww}")
print(f"  Oracle: {oracle:.3f} vs actual: {actual:.3f}")

# ====================================================================
# 5. 全局: Decide推翻正确CODER导致的总损失
# ====================================================================
print("\n" + "="*100)
print("5. 全局: Decide推翻正确CODER的总损失")
print("="*100)

for qt in sorted(set(r['question_type'] for r in results)):
    qr = [r for r in results if r['question_type'] == qt]
    overturn_loss = 0
    coder_total = 0
    coder_correct_total = 0
    for r in qr:
        reasoning = r.get('reasoning', '')
        m = re.search(r'answer=([A-D])', reasoning)
        gt = str(r['ground_truth']).strip().upper()
        if m:
            coder_total += 1
            if m.group(1) == gt:
                coder_correct_total += 1
                if r['score'] == 0:
                    overturn_loss += 1
    actual = np.mean([r['score'] for r in qr])
    coder_acc = coder_correct_total / len(qr) if qr else 0
    print(f"  {qt:<35} CODER_acc={coder_acc:.1%} Decide_overturn_loss={overturn_loss:>3} ({100*overturn_loss/len(qr):.1f}%) actual={actual:.3f}")

# ====================================================================
# 6. 如果完全禁止Decide推翻CODER的选择题
# ====================================================================
print("\n" + "="*100)
print("6. CODER Trust Oracle: 选择题用CODER答案，其他用Decide")
print("="*100)

all_scores_coder_trust = []
all_scores_actual = []
for r in results:
    reasoning = r.get('reasoning', '')
    gt = str(r['ground_truth']).strip().upper()
    m = re.search(r'answer=([A-D])', reasoning)
    
    if m and r.get('options'):
        # 选择题: 用CODER答案
        coder_ans = m.group(1)
        score = 1.0 if coder_ans == gt else 0.0
        all_scores_coder_trust.append(score)
    else:
        all_scores_coder_trust.append(r['score'])
    all_scores_actual.append(r['score'])

print(f"  CODER Trust Overall: {np.mean(all_scores_coder_trust):.4f}")
print(f"  Actual Overall:      {np.mean(all_scores_actual):.4f}")
print(f"  Delta: {np.mean(all_scores_coder_trust)-np.mean(all_scores_actual):+.4f}")

# Per task
for qt in sorted(set(r['question_type'] for r in results)):
    qr = [r for r in results if r['question_type'] == qt]
    ct_scores = []
    ac_scores = []
    for r in qr:
        reasoning = r.get('reasoning', '')
        gt = str(r['ground_truth']).strip().upper()
        m = re.search(r'answer=([A-D])', reasoning)
        if m and r.get('options'):
            ct_scores.append(1.0 if m.group(1) == gt else 0.0)
        else:
            ct_scores.append(r['score'])
        ac_scores.append(r['score'])
    d = np.mean(ct_scores) - np.mean(ac_scores)
    mk = '+' if d > 0.005 else ('-' if d < -0.005 else '=')
    print(f"    {qt:<35} CODER={np.mean(ct_scores):.3f} Actual={np.mean(ac_scores):.3f} {d:+.3f}{mk}")
