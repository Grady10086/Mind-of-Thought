#!/usr/bin/env python3
"""Deep analysis of CODER rel_distance: confusion patterns, top-k, pass-k"""
import json, glob, re, numpy as np
from collections import Counter, defaultdict

# Load TA7 (latest) + V10_G256 full (largest)
def load(pat):
    d = []
    for f in sorted(glob.glob(pat)):
        d.extend(json.load(open(f)))
    return [r for r in d if r['question_type'] == 'object_rel_distance']

ta7 = load('outputs/agentic_pipeline_v16_ta7_8gpu/gpu*/detailed_results.json')
v10 = load('outputs/agentic_pipeline_v10_g256_full/gpu*/detailed_results.json')
v14 = load('outputs/agentic_pipeline_v14_full/gpu*/detailed_results.json')

def parse_coder_answer(reasoning):
    m = re.search(r'\[coder\] ans=([A-D])', reasoning)
    return m.group(1) if m else None

def parse_vl_ind(reasoning):
    m = re.search(r'\[vl_ind\] ([A-D])', reasoning)
    return m.group(1) if m else None

def parse_coder_distances(reasoning):
    """Extract distance values from CODER result like 'chair=2.31m, table=3.45m'"""
    dists = {}
    for m in re.finditer(r'(\w[\w\s]*)=([\d.]+)m', reasoning):
        name = m.group(1).strip().lower()
        dist = float(m.group(2))
        if name != 'answer':
            dists[name] = dist
    return dists

# ============================================================
print("=" * 80)
print("REL_DISTANCE DEEP ANALYSIS")
print("=" * 80)

# 1. CODER accuracy across versions
print("\n=== 1. CODER ACCURACY ===")
for name, data in [('TA7', ta7), ('V10_G256', v10), ('V14', v14)]:
    total = 0; correct = 0
    for r in data:
        gt = str(r['ground_truth']).strip().upper()
        ca = parse_coder_answer(r.get('reasoning', ''))
        if ca:
            total += 1
            if ca == gt: correct += 1
    if total:
        print(f"  {name:12s}: CODER correct {correct}/{total} = {100*correct/total:.1f}%")
    else:
        print(f"  {name:12s}: no CODER data")

# 2. TA7: Confusion matrix (CODER answer vs GT)
print("\n=== 2. TA7 CODER CONFUSION MATRIX ===")
conf_matrix = defaultdict(Counter)
for r in ta7:
    gt = str(r['ground_truth']).strip().upper()
    ca = parse_coder_answer(r.get('reasoning', ''))
    if ca:
        conf_matrix[gt][ca] += 1

hdr = 'GT\\Pred'
print(f"  {hdr:>8s}", end='')
for p in 'ABCD':
    print(f"  {p:>4s}", end='')
print("  Total")
for gt in 'ABCD':
    row = conf_matrix[gt]
    total = sum(row.values())
    if total == 0: continue
    print(f"  {gt:>8s}", end='')
    for p in 'ABCD':
        v = row.get(p, 0)
        print(f"  {v:>4d}", end='')
    correct = row.get(gt, 0)
    print(f"  {total:>5d}  ({100*correct/total:.0f}% correct)")

# 3. CODER top-k analysis: how often is GT in top-1, top-2, top-3?
print("\n=== 3. CODER TOP-K ANALYSIS (GT rank in CODER's distance ranking) ===")
top_counts = Counter()  # rank -> count
total_with_dists = 0
for r in ta7:
    rs = r.get('reasoning', '')
    gt = str(r['ground_truth']).strip().upper()
    # Parse distances from CODER output
    m = re.search(r'RelDist:.*?detail=(.*)', rs)
    if not m:
        # Try parsing from [coder] section
        coder_section = re.search(r'RelDist: answer=([A-D]),\s*detail=ref=\w+,\s*(.*?)$', rs)
        continue
    detail = m.group(1)
    # Extract letter=distance pairs
    pairs = re.findall(r'(\w[\w\s]*)=([\d.]+)m', detail)
    if len(pairs) < 2:
        continue
    
    # Get the option letters from the question options
    opts = r.get('options', [])
    if not opts: continue
    
    # Map option names to letters
    opt_map = {}
    for opt in opts:
        letter = opt[0].upper()
        name = re.sub(r'^[A-D]\.\s*', '', opt).strip().lower()
        opt_map[name] = letter
    
    # Build distance ranking
    dist_ranking = []
    for name, dist_str in pairs:
        name = name.strip().lower()
        dist = float(dist_str)
        letter = opt_map.get(name)
        if not letter:
            # Try fuzzy match
            for on, ol in opt_map.items():
                if name in on or on in name:
                    letter = ol
                    break
        if letter:
            dist_ranking.append((letter, dist))
    
    if not dist_ranking: continue
    
    # Check if "closest" or "farthest"
    q = r['question'].lower()
    is_farthest = 'farthest' in q or 'furthest' in q
    
    if is_farthest:
        dist_ranking.sort(key=lambda x: -x[1])
    else:
        dist_ranking.sort(key=lambda x: x[1])
    
    # Find GT rank
    total_with_dists += 1
    gt_rank = None
    for i, (letter, _) in enumerate(dist_ranking):
        if letter == gt:
            gt_rank = i + 1
            break
    
    if gt_rank:
        top_counts[gt_rank] += 1
    else:
        top_counts['not_found'] += 1

if total_with_dists > 0:
    cum = 0
    for k in [1, 2, 3, 4]:
        cum += top_counts.get(k, 0)
        print(f"  Top-{k}: {cum}/{total_with_dists} = {100*cum/total_with_dists:.1f}% (pass@{k})")
    nf = top_counts.get('not_found', 0)
    if nf:
        print(f"  Not found in ranking: {nf}")
else:
    print("  No distance data available for top-k analysis")

# 4. V10_G256 full: same top-k analysis
print("\n=== 4. V10_G256 FULL TOP-K (710 samples) ===")
top_counts2 = Counter()
total2 = 0
for r in v10:
    rs = r.get('reasoning', '')
    gt = str(r['ground_truth']).strip().upper()
    ca = parse_coder_answer(rs)
    if ca:
        total2 += 1
        if ca == gt:
            top_counts2[1] += 1

# Can't do full top-k without distance values in V10 reasoning
# Just report CODER accuracy
print(f"  CODER top-1 (=accuracy): {top_counts2.get(1,0)}/{total2} = {100*top_counts2.get(1,0)/total2:.1f}%" if total2 else "  No data")

# 5. Error pattern: Is it top-1 vs top-2 confusion or random?
print("\n=== 5. TA7: VL_ind vs CODER vs Final — who gets it right? ===")
vl_correct = coder_correct = final_correct = 0
both_wrong = vl_only = coder_only = agree_right = agree_wrong = 0
total = 0
for r in ta7:
    rs = r.get('reasoning', '')
    gt = str(r['ground_truth']).strip().upper()
    ca = parse_coder_answer(rs)
    vi = parse_vl_ind(rs)
    pred = str(r['prediction']).strip().upper()
    
    if not ca or not vi: continue
    total += 1
    
    cr = (ca == gt)
    vr = (vi == gt)
    fr = (pred == gt)
    
    if cr: coder_correct += 1
    if vr: vl_correct += 1
    if fr: final_correct += 1
    
    if cr and vr: agree_right += 1
    elif not cr and not vr: both_wrong += 1
    elif vr and not cr: vl_only += 1
    elif cr and not vr: coder_only += 1

print(f"  Total: {total}")
print(f"  CODER correct: {coder_correct} ({100*coder_correct/total:.0f}%)")
print(f"  VL_ind correct: {vl_correct} ({100*vl_correct/total:.0f}%)")
print(f"  Final correct:  {final_correct} ({100*final_correct/total:.0f}%)")
print(f"  Both right:     {agree_right} ({100*agree_right/total:.0f}%)")
print(f"  Only VL right:  {vl_only} ({100*vl_only/total:.0f}%)")
print(f"  Only CODER right: {coder_only} ({100*coder_only/total:.0f}%)")
print(f"  Both wrong:     {both_wrong} ({100*both_wrong/total:.0f}%)")
print(f"  Oracle(best-of): {agree_right+vl_only+coder_only} ({100*(agree_right+vl_only+coder_only)/total:.0f}%)")

# 6. Distance margin analysis: when CODER is wrong, how close are the top-2 distances?
print("\n=== 6. DISTANCE MARGIN ANALYSIS ===")
print("  (When CODER ranks options by distance, how close is top-1 to top-2?)")
# We need the raw CODER output with distances
margins_right = []
margins_wrong = []
for r in ta7:
    rs = r.get('reasoning', '')
    gt = str(r['ground_truth']).strip().upper()
    ca = parse_coder_answer(rs)
    if not ca: continue
    
    # Look for distance values in reasoning
    # Pattern: "name=X.XXm" 
    dist_matches = re.findall(r'[A-D]\.\s*(\w[\w\s]*).*?=([\d.]+)m', rs)
    if not dist_matches:
        # Try from detail section
        detail = re.search(r'detail=ref=\w+,\s*(.*?)(?:\||\s*$)', rs)
        if detail:
            dist_matches = re.findall(r'(\w[\w\s]*)=([\d.]+)m', detail.group(1))
    
    if len(dist_matches) >= 2:
        dists = sorted([float(d) for _, d in dist_matches])
        margin = dists[1] - dists[0] if len(dists) >= 2 else 0
        if ca == gt:
            margins_right.append(margin)
        else:
            margins_wrong.append(margin)

if margins_right:
    print(f"  When CODER correct:  avg margin = {np.mean(margins_right):.2f}m, median = {np.median(margins_right):.2f}m (n={len(margins_right)})")
if margins_wrong:
    print(f"  When CODER wrong:    avg margin = {np.mean(margins_wrong):.2f}m, median = {np.median(margins_wrong):.2f}m (n={len(margins_wrong)})")
if margins_right and margins_wrong:
    all_margins = margins_right + margins_wrong
    threshold = np.median(all_margins)
    n_small_right = sum(1 for m in margins_right if m < threshold)
    n_small_wrong = sum(1 for m in margins_wrong if m < threshold)
    print(f"  Small margin (<{threshold:.2f}m): correct={n_small_right}, wrong={n_small_wrong}")
    n_large_right = sum(1 for m in margins_right if m >= threshold)
    n_large_wrong = sum(1 for m in margins_wrong if m >= threshold)
    print(f"  Large margin (>={threshold:.2f}m): correct={n_large_right}, wrong={n_large_wrong}")

print("\n=== SUMMARY ===")
print("  rel_distance is fundamentally harder because:")
print("  1. It's a 4-way choice (25% random), not numerical estimation")
print("  2. 3D calibration errors corrupt distance RANKING (not just magnitude)")
print("  3. Indoor objects often at similar distances → small errors flip rankings")
print("  4. VL struggles to compare relative distances from video frames")
