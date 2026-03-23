#!/usr/bin/env python3
"""Check video availability for V7 samples"""
import json, os
from collections import Counter

VIDEO_DIRS = [
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
]

with open('outputs/evolving_agent_v7_20260203_134612/detailed_results.json') as f:
    results = json.load(f)

found = 0
qt_counter = Counter()
for r in results:
    sn = r['scene_name']
    for d in VIDEO_DIRS:
        if os.path.exists(os.path.join(d, f'{sn}.mp4')):
            found += 1
            qt_counter[r['question_type']] += 1
            break

print(f'Videos found: {found}/{len(results)}')
for qt, n in sorted(qt_counter.items()):
    print(f'  {qt}: {n}')
