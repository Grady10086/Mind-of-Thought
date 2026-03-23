#!/usr/bin/env python3
"""V17 ablation: Compare VL_r1 (bare VL) vs CODER_r1 vs VL_r2 vs Final across 100 samples."""
import json, re, sys, os, numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.grid64_real_test import evaluate_sample

data = json.load(open('outputs/agentic_pipeline_v17_coevo/detailed_results.json'))
print(f'Total: {len(data)} samples\n')

# Extract per-sample: VL_r1, CODER_r1, VL_r2, Final
records = []
for r in data:
    reas = r['reasoning']
    gt = r['ground_truth']
    qt = r['question_type']
    opts = r.get('options', [])
    is_choice = bool(opts)
    final = r['prediction']

    # === Choice tasks: extract VL_r1, CODER, VL_r2 from reasoning tags ===
    if is_choice:
        vr1_m = re.search(r'\[vl_r1\] (\w)', reas)
        coder_m = re.search(r'\[coder_r1\] ans=(\w?)', reas)
        vr2_m = re.search(r'\[vl_r2_(?:focused|temporal|slice)\] (\w)', reas)

        vr1 = vr1_m.group(1) if vr1_m else ''
        coder = coder_m.group(1) if coder_m and coder_m.group(1) else ''
        vr2 = vr2_m.group(1) if vr2_m else ''

        s_vl_r1 = evaluate_sample(qt, vr1, gt) if vr1 else None
        s_coder = evaluate_sample(qt, coder, gt) if coder else None
        s_vl_r2 = evaluate_sample(qt, vr2, gt) if vr2 else None
        s_final = r['score']
    else:
        # === Numerical tasks: extract CODER answer, VL numerical answers ===
        # Numerical path: [numerical] | [coder] ans=X ... | [room_vl_3vote] / [abs_dist_vl] / etc
        coder_m = re.search(r'\[coder\] ans=([\d.]+)', reas)
        # For numerical, there's no separate VL_r1/VL_r2; the VL call is different per task type
        # We'll use coder answer vs final to measure coder vs VL contribution
        coder = coder_m.group(1) if coder_m else ''

        vr1 = ''  # No VL_r1 for numerical
        vr2 = ''  # No VL_r2 for numerical
        s_vl_r1 = None
        s_coder = evaluate_sample(qt, coder, gt) if coder else None
        s_vl_r2 = None
        s_final = r['score']

    records.append({
        'qt': qt, 'gt': gt, 'opts': opts, 'is_choice': is_choice,
        'vl_r1': vr1, 'coder_r1': coder, 'vl_r2': vr2, 'final': final,
        's_vl_r1': s_vl_r1,
        's_coder': s_coder,
        's_vl_r2': s_vl_r2,
        's_final': s_final,
        's_v7': r['v7_vl_score'],
        'reasoning': reas,
    })


def safe_mean(vals):
    """Mean of non-None values."""
    valid = [v for v in vals if v is not None]
    return np.mean(valid) if valid else None


def fmt(val, width=6):
    """Format a float or None."""
    return f'{val:>{width}.3f}' if val is not None else f'{"N/A":>{width}}'


# ========================================================
# Per-task comparison
# ========================================================
tts = sorted(set(r['qt'] for r in records))

print("=" * 95)
print("  CHOICE TASKS (direction, appearance, route, rel_distance)")
print("=" * 95)
print(f"  {'Task':<35} {'N':>3} {'V7':>6} {'VL_r1':>6} {'CODER':>6} {'VL_r2':>6} {'Final':>6}  {'F-V7':>6}")
print('-' * 95)

choice_records = [r for r in records if r['is_choice']]
num_records = [r for r in records if not r['is_choice']]

for qt in tts:
    qr = [r for r in choice_records if r['qt'] == qt]
    if not qr:
        continue
    n = len(qr)
    v7 = safe_mean([r['s_v7'] for r in qr])
    vl1 = safe_mean([r['s_vl_r1'] for r in qr])
    cod = safe_mean([r['s_coder'] for r in qr])
    vl2 = safe_mean([r['s_vl_r2'] for r in qr])
    fin = safe_mean([r['s_final'] for r in qr])
    diff = (fin - v7) if fin is not None and v7 is not None else None
    diff_vl1 = (fin - vl1) if fin is not None and vl1 is not None else None
    print(f'  {qt:<35} {n:>3} {fmt(v7)} {fmt(vl1)} {fmt(cod)} {fmt(vl2)} {fmt(fin)}  {fmt(diff)}')

# Choice totals
c_v7 = safe_mean([r['s_v7'] for r in choice_records])
c_vl1 = safe_mean([r['s_vl_r1'] for r in choice_records])
c_cod = safe_mean([r['s_coder'] for r in choice_records])
c_vl2 = safe_mean([r['s_vl_r2'] for r in choice_records])
c_fin = safe_mean([r['s_final'] for r in choice_records])
print('-' * 95)
print(f'  {"Choice Overall":<35} {len(choice_records):>3} {fmt(c_v7)} {fmt(c_vl1)} {fmt(c_cod)} {fmt(c_vl2)} {fmt(c_fin)}  {fmt(c_fin - c_v7 if c_fin and c_v7 else None)}')

print()
print("=" * 95)
print("  NUMERICAL TASKS (counting, size_est, room_size, abs_distance)")
print("=" * 95)
print(f"  {'Task':<35} {'N':>3} {'V7':>6} {'CODER':>6} {'Final':>6}  {'F-V7':>6}")
print('-' * 95)

for qt in tts:
    qr = [r for r in num_records if r['qt'] == qt]
    if not qr:
        continue
    n = len(qr)
    v7 = safe_mean([r['s_v7'] for r in qr])
    cod = safe_mean([r['s_coder'] for r in qr])
    fin = safe_mean([r['s_final'] for r in qr])
    diff = (fin - v7) if fin is not None and v7 is not None else None
    print(f'  {qt:<35} {n:>3} {fmt(v7)} {fmt(cod)} {fmt(fin)}  {fmt(diff)}')

n_v7 = safe_mean([r['s_v7'] for r in num_records])
n_cod = safe_mean([r['s_coder'] for r in num_records])
n_fin = safe_mean([r['s_final'] for r in num_records])
print('-' * 95)
print(f'  {"Numerical Overall":<35} {len(num_records):>3} {fmt(n_v7)} {fmt(n_cod)} {fmt(n_fin)}  {fmt(n_fin - n_v7 if n_fin and n_v7 else None)}')

# ========================================================
# Overall
# ========================================================
print()
print("=" * 95)
print("  OVERALL SUMMARY")
print("=" * 95)
all_v7 = np.mean([r['s_v7'] for r in records])
all_fin = np.mean([r['s_final'] for r in records])
print(f"  V7 baseline (8f Grid, 360x420):          {all_v7:.4f}")
print(f"  V17 Final (fps=2, 64f Grid, co-evolving): {all_fin:.4f}  (Δ from V7: {all_fin - all_v7:+.4f})")
print()
print(f"  --- Choice tasks breakdown ---")
print(f"  VL_r1 (bare VL, fps=2, no evolving):       {c_vl1:.4f}" if c_vl1 else "  VL_r1: N/A")
print(f"  CODER_r1 (bare Grid, no evolving):          {c_cod:.4f}" if c_cod else "  CODER_r1: N/A")
print(f"  VL_r2 (VL after frame focusing):            {c_vl2:.4f}" if c_vl2 else "  VL_r2: N/A")
print(f"  Final:                                       {c_fin:.4f}" if c_fin else "  Final: N/A")
if c_vl1 and c_fin:
    evol_delta = c_fin - c_vl1
    total_delta = c_fin - c_v7 if c_v7 else None
    print(f"  Evolving contribution (Final-VL_r1):        {evol_delta:+.4f}")
    if total_delta and total_delta > 0.001:
        infra_delta = c_vl1 - c_v7 if c_v7 else None
        print(f"  Infrastructure gain (VL_r1-V7):             {infra_delta:+.4f}" if infra_delta is not None else "")
        print(f"  Total gain (Final-V7):                      {total_delta:+.4f}")
        print(f"  Evolving share: {evol_delta / total_delta * 100:.0f}%, Infra share: {(1 - evol_delta / total_delta) * 100:.0f}%")

# ========================================================
# Per-sample helped/hurt analysis (choice tasks)
# ========================================================
print()
print("=" * 95)
print("  PER-SAMPLE ANALYSIS: Evolving help/hurt (choice tasks)")
print("=" * 95)
helped = []
hurt = []
same = []
for r in choice_records:
    if r['s_vl_r1'] is None:
        same.append(r)
        continue
    if r['s_final'] > r['s_vl_r1'] + 0.001:
        helped.append(r)
    elif r['s_final'] < r['s_vl_r1'] - 0.001:
        hurt.append(r)
    else:
        same.append(r)

print(f"  Helped: {len(helped)}  Hurt: {len(hurt)}  Same: {len(same)}  (net: {len(helped) - len(hurt):+d})")
if helped:
    print(f"\n  --- Helped samples ---")
    for r in helped:
        print(f"    {r['qt']:<30} VL_r1={r['vl_r1']} → Final={r['final']}  (gt={r['gt']})")
if hurt:
    print(f"\n  --- Hurt samples ---")
    for r in hurt:
        print(f"    {r['qt']:<30} VL_r1={r['vl_r1']} → Final={r['final']}  (gt={r['gt']})")

# ========================================================
# Per-sample: Final vs V7 (all tasks)
# ========================================================
print()
print("=" * 95)
print("  PER-SAMPLE ANALYSIS: V17 Final vs V7 (all tasks)")
print("=" * 95)
h2 = sum(1 for r in records if r['s_final'] > r['s_v7'] + 0.001)
u2 = sum(1 for r in records if r['s_final'] < r['s_v7'] - 0.001)
s2 = len(records) - h2 - u2
print(f"  Helped: {h2}  Hurt: {u2}  Same: {s2}  (net: {h2 - u2:+d})")

# By task type
print(f"\n  Per task type:")
for qt in tts:
    qr = [r for r in records if r['qt'] == qt]
    h = sum(1 for r in qr if r['s_final'] > r['s_v7'] + 0.001)
    u = sum(1 for r in qr if r['s_final'] < r['s_v7'] - 0.001)
    s = len(qr) - h - u
    v7m = np.mean([r['s_v7'] for r in qr])
    fm = np.mean([r['s_final'] for r in qr])
    print(f"    {qt:<35} V7={v7m:.3f} F={fm:.3f} Δ={fm-v7m:+.3f}  (H={h} U={u} S={s})")
