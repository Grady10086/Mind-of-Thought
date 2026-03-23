"""Analyze evolving (AUTO_ADD) effect on V16 full results.
Questions: 
1. Does AUTO_ADD actually improve scores for affected samples?
2. What task types benefit?
3. What would scores be WITHOUT evolving?
"""
import json, re, glob, numpy as np
from collections import defaultdict, Counter

v16_all = []
for f in glob.glob('outputs/agentic_pipeline_v16_full/gpu*/detailed_results.json'):
    v16_all.extend(json.load(open(f)))

print(f"Total samples: {len(v16_all)}")

# Identify samples where AUTO_ADD was triggered
evolved = []
not_evolved = []
for r in v16_all:
    reasoning = r.get('reasoning', '')
    if '[AUTO_ADD]' in reasoning or 'auto_add' in reasoning.lower():
        evolved.append(r)
    else:
        not_evolved.append(r)

print(f"\nSamples with AUTO_ADD: {len(evolved)} ({100*len(evolved)/len(v16_all):.1f}%)")
print(f"Samples without:      {len(not_evolved)}")

# 1. Per-task breakdown of evolved samples
print(f"\n=== 1. EVOLVED SAMPLES BY TASK TYPE ===")
ev_by_type = defaultdict(list)
for r in evolved: ev_by_type[r['question_type']].append(r)
ne_by_type = defaultdict(list)
for r in not_evolved: ne_by_type[r['question_type']].append(r)

print(f"{'Task':<38} {'Evolved':>8} {'EV_score':>9} {'NoEV_score':>10} {'All':>8}")
print("-" * 80)
for qt in sorted(set(r['question_type'] for r in v16_all)):
    ev = ev_by_type.get(qt, [])
    ne = ne_by_type.get(qt, [])
    all_s = ev + ne
    ev_score = np.mean([r['score'] for r in ev]) if ev else 0
    ne_score = np.mean([r['score'] for r in ne]) if ne else 0
    all_score = np.mean([r['score'] for r in all_s])
    print(f"  {qt:<36} {len(ev):>4}/{len(all_s):<4} {ev_score:>8.3f}  {ne_score:>9.3f}  {all_score:>7.3f}")

# 2. For evolved samples: did CODER succeed after ADD?
print(f"\n=== 2. CODER RESULT AFTER AUTO_ADD ===")
coder_after_add = {'found': 0, 'still_notfound': 0, 'no_coder': 0}
for r in evolved:
    reasoning = r.get('reasoning', '')
    # Check if there's a [coder] result after [auto_add]
    add_pos = reasoning.find('[AUTO_ADD]')
    if add_pos >= 0:
        after = reasoning[add_pos:]
        if '[coder]' in after or 'ans=' in after:
            if 'not found' in after.lower():
                coder_after_add['still_notfound'] += 1
            else:
                coder_after_add['found'] += 1
        else:
            coder_after_add['no_coder'] += 1
    else:
        coder_after_add['no_coder'] += 1

for k, v in coder_after_add.items():
    print(f"  {k}: {v} ({100*v/len(evolved):.0f}%)" if evolved else "")

# 3. Score comparison: evolved vs V7 baseline (to measure pipeline + evolving contribution)
print(f"\n=== 3. EVOLVED SAMPLES: V16 vs V7 (pipeline contribution) ===")
ev_v16 = np.mean([r['score'] for r in evolved])
ev_v7 = np.mean([r.get('v7_vl_score', 0) for r in evolved])
print(f"  V7 (no pipeline): {ev_v7:.3f}")
print(f"  V16 (with evolving): {ev_v16:.3f}")
print(f"  Pipeline + evolving contribution: {ev_v16 - ev_v7:+.3f}")

ne_v16 = np.mean([r['score'] for r in not_evolved])
ne_v7 = np.mean([r.get('v7_vl_score', 0) for r in not_evolved])
print(f"\n  Non-evolved V7: {ne_v7:.3f} → V16: {ne_v16:.3f} (delta {ne_v16-ne_v7:+.3f})")
print(f"  Evolved     V7: {ev_v7:.3f} → V16: {ev_v16:.3f} (delta {ev_v16-ev_v7:+.3f})")

# 4. Counterfactual: what if AUTO_ADD failed (use V7 score as proxy for "no evolving")
print(f"\n=== 4. COUNTERFACTUAL: NO EVOLVING ===")
print("If evolved samples fell back to V7 score (no pipeline help):")
total_with = np.mean([r['score'] for r in v16_all])
# Replace evolved scores with v7 scores
counterfactual = []
for r in v16_all:
    reasoning = r.get('reasoning', '')
    if '[AUTO_ADD]' in reasoning:
        counterfactual.append(r.get('v7_vl_score', 0))
    else:
        counterfactual.append(r['score'])
total_without = np.mean(counterfactual)
print(f"  Overall WITH evolving:    {total_with:.4f}")
print(f"  Overall WITHOUT evolving: {total_without:.4f}")
print(f"  Evolving contribution:    {total_with - total_without:+.4f}")

# 5. Per-task evolving contribution
print(f"\n=== 5. PER-TASK EVOLVING CONTRIBUTION ===")
print(f"{'Task':<38} {'With':>7} {'Without':>8} {'Delta':>7}")
print("-" * 65)
by_type_all = defaultdict(list)
by_type_cf = defaultdict(list)
for r in v16_all:
    qt = r['question_type']
    by_type_all[qt].append(r['score'])
    reasoning = r.get('reasoning', '')
    if '[AUTO_ADD]' in reasoning:
        by_type_cf[qt].append(r.get('v7_vl_score', 0))
    else:
        by_type_cf[qt].append(r['score'])

for qt in sorted(by_type_all):
    w = np.mean(by_type_all[qt])
    wo = np.mean(by_type_cf[qt])
    d = w - wo
    print(f"  {qt:<36} {w:>6.3f} {wo:>7.3f} {d:>+6.3f} {'←' if abs(d) > 0.003 else ''}")

# 6. CODER confidence distribution for evolved vs non-evolved
print(f"\n=== 6. WHAT ENTITIES WERE ADDED? ===")
added_entities = Counter()
for r in evolved:
    for m in re.finditer(r"ADD '([^']+)'", r.get('reasoning', '')):
        added_entities[m.group(1)] += 1
print("Top 20 added entities:")
for name, cnt in added_entities.most_common(20):
    print(f"  {name:<25} {cnt:>4}")
