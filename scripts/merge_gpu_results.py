#!/usr/bin/env python3
"""
Merge multi-GPU results from ablation experiments.

Usage:
    python3 scripts/merge_gpu_results.py outputs/ablation_attention_dilution
    python3 scripts/merge_gpu_results.py outputs/ablation_metadata_noise_B
    python3 scripts/merge_gpu_results.py outputs/ablation_metadata_noise_D
"""
import json, sys, os
import numpy as np
from collections import defaultdict
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 merge_gpu_results.py <output_dir>")
        sys.exit(1)

    base = Path(sys.argv[1])
    all_results = []

    for g in range(16):  # Support up to 16 GPUs
        p = base / f"gpu{g}" / "detailed_results.json"
        if p.exists():
            d = json.load(open(p))
            all_results.extend(d)
            print(f"  gpu{g}: {len(d)} samples")

    if not all_results:
        print(f"No results found in {base}")
        sys.exit(1)

    # Deduplicate by (scene_name, question, ground_truth)
    seen = set()
    deduped = []
    for r in all_results:
        key = (r.get('scene_name', ''), r.get('question', ''), r.get('ground_truth', ''))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f"\nTotal: {len(deduped)} unique samples (from {len(all_results)} raw)")

    # Check if this is attention dilution (has 'levels' field)
    is_dilution = 'levels' in deduped[0] if deduped else False

    if is_dilution:
        # Attention Dilution format
        levels = ['oracle', '16', '32', 'full']
        qtypes = sorted(set(r['question_type'] for r in deduped))

        print(f"\n  {'Task':<35}", end="")
        for lv in levels:
            print(f" {lv:>8}", end="")
        print(f"  {'N':>5}")
        print("-" * 90)

        for qt in qtypes:
            qr = [r for r in deduped if r['question_type'] == qt]
            print(f"  {qt:<35}", end="")
            for lv in levels:
                scores = [r['scores'].get(lv, 0) for r in qr if lv in r.get('scores', {})]
                avg = np.mean(scores) if scores else 0
                print(f" {avg:>7.4f}", end="")
            print(f"  {len(qr):>5}")

        print("-" * 90)
        print(f"  {'Overall':<35}", end="")
        for lv in levels:
            scores = [r['scores'].get(lv, 0) for r in deduped if lv in r.get('scores', {})]
            avg = np.mean(scores) if scores else 0
            print(f" {avg:>7.4f}", end="")
        print(f"  {len(deduped):>5}")
    else:
        # Standard format (metadata noise, etc.)
        qtypes = sorted(set(r['question_type'] for r in deduped))
        mca_types = {'object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard',
                     'object_rel_distance', 'route_planning', 'obj_appearance_order'}

        print(f"\n  {'Task':<40} {'Score':>8} {'N':>5}")
        print("-" * 60)

        for qt in qtypes:
            qr = [r for r in deduped if r['question_type'] == qt]
            avg = np.mean([r['score'] for r in qr])
            tag = 'MCA' if qt in mca_types else 'NA'
            print(f"  [{tag}] {qt:<35} {avg:>7.4f} {len(qr):>5}")

        mca = [r for r in deduped if r['question_type'] in mca_types]
        na = [r for r in deduped if r['question_type'] not in mca_types]
        if mca:
            print(f"\n  {'MCA Overall':<40} {np.mean([r['score'] for r in mca]):>7.4f} {len(mca):>5}")
        if na:
            print(f"  {'NA Overall':<40} {np.mean([r['score'] for r in na]):>7.4f} {len(na):>5}")

        overall = np.mean([r['score'] for r in deduped])
        print(f"\n  {'OVERALL':<40} {overall:>7.4f} {len(deduped):>5}")

    # Save merged
    out_path = base / "detailed_results_merged.json"
    with open(out_path, 'w') as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    print(f"\nSaved merged results to: {out_path}")


if __name__ == '__main__':
    main()
