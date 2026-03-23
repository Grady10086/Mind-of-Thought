#!/usr/bin/env python3
"""Merge 8-GPU results for each V21 variant and print comparison table."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT / "outputs" / "v21_variant_test"

VARIANT_NAMES = {
    'v20': 'V20 Baseline',
    '1': 'V21-1 CODER Vote',
    '3': 'V21-3 Disagree Evo',
    '4': 'V21-4 Frame Quality',
    '5': 'V21-5 Multi-View',
}

CHOICE_TYPES = [
    'obj_appearance_order',
    'object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard',
    'object_rel_distance', 'route_planning',
]

def load_variant(variant_key):
    vdir = BASE_DIR / variant_key
    all_results = []
    for g in range(8):
        fp = vdir / f"gpu{g}.json"
        if fp.exists():
            data = json.load(open(fp))
            all_results.extend(data)
        else:
            fp2 = vdir / "results.json"
            if fp2.exists():
                all_results.extend(json.load(open(fp2)))
                break
    # Deduplicate
    seen = set()
    deduped = []
    for r in all_results:
        key = (r.get('scene_name',''), r.get('question',''), str(r.get('ground_truth','')))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped

def main():
    results_all = {}
    for vkey, vname in VARIANT_NAMES.items():
        vdir = BASE_DIR / vkey
        if not vdir.exists():
            continue
        results = load_variant(vkey)
        if results:
            results_all[vname] = results
            print(f"Loaded {vname}: {len(results)} samples")

    if not results_all:
        print("No results found!")
        return

    # Print comparison table
    print("\n" + "=" * 130)
    print("V21 Variant Comparison — Choice Questions (100 samples)")
    print("=" * 130)

    variant_names = list(results_all.keys())

    # Header
    header = f"  {'Task':<28} {'N':>3}"
    for vn in variant_names:
        short = vn.split(' ', 1)[-1] if ' ' in vn else vn
        header += f" {short:>14}"
    print(header)
    print("-" * 130)

    # Per type
    for qt in CHOICE_TYPES:
        # Get N from first variant
        first_results = list(results_all.values())[0]
        qr0 = [r for r in first_results if r['question_type'] == qt]
        row = f"  {qt:<28} {len(qr0):>3}"
        
        scores = []
        for vn in variant_names:
            results = results_all[vn]
            qr = [r for r in results if r['question_type'] == qt]
            if qr:
                s = np.mean([r['score'] for r in qr])
                scores.append(s)
                row += f" {s:>14.4f}"
            else:
                scores.append(None)
                row += f" {'—':>14}"
        print(row)

    # Overall
    print("-" * 130)
    first_results = list(results_all.values())[0]
    choice_n = len([r for r in first_results if r['question_type'] in CHOICE_TYPES])
    row = f"  {'CHOICE OVERALL':<28} {choice_n:>3}"
    best_score = -1
    best_name = ''
    for vn in variant_names:
        results = results_all[vn]
        qr = [r for r in results if r['question_type'] in CHOICE_TYPES]
        if qr:
            s = np.mean([r['score'] for r in qr])
            row += f" {s:>14.4f}"
            if s > best_score:
                best_score = s
                best_name = vn
        else:
            row += f" {'—':>14}"
    print(row)

    # Stats
    print("-" * 130)
    row = f"  {'Avg VL Calls':<28} {'':>3}"
    for vn in variant_names:
        results = results_all[vn]
        avg_vl = np.mean([r.get('vl_calls', 0) for r in results])
        row += f" {avg_vl:>14.1f}"
    print(row)

    row = f"  {'Avg Time (s)':<28} {'':>3}"
    for vn in variant_names:
        results = results_all[vn]
        avg_t = np.mean([r.get('elapsed_s', 0) for r in results])
        row += f" {avg_t:>14.1f}"
    print(row)

    print("=" * 130)
    if best_name:
        print(f"\n  Best variant: {best_name} = {best_score:.4f}")

    # Per-sample delta analysis (vs V20)
    if 'V20 Baseline' in results_all and len(results_all) > 1:
        v20 = results_all['V20 Baseline']
        v20_map = {}
        for r in v20:
            key = (r.get('scene_name',''), r.get('question',''))
            v20_map[key] = r.get('score', 0)

        print(f"\n  Per-sample delta vs V20 Baseline:")
        for vn in variant_names:
            if vn == 'V20 Baseline': continue
            results = results_all[vn]
            better = worse = same = 0
            for r in results:
                if r['question_type'] not in CHOICE_TYPES: continue
                key = (r.get('scene_name',''), r.get('question',''))
                v20_s = v20_map.get(key, 0)
                if r['score'] > v20_s + 0.001: better += 1
                elif r['score'] < v20_s - 0.001: worse += 1
                else: same += 1
            print(f"    {vn}: better={better}, same={same}, worse={worse}")

    # Save merged
    for vkey, vname in VARIANT_NAMES.items():
        if vname in results_all:
            merged_path = BASE_DIR / vkey / "merged.json"
            with open(merged_path, 'w') as f:
                json.dump(results_all[vname], f, indent=2, ensure_ascii=False)
    print(f"\nMerged results saved to {BASE_DIR}/*/merged.json")


if __name__ == '__main__':
    main()
