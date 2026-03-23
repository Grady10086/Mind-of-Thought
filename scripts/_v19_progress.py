#!/usr/bin/env python3
import re, numpy as np
from collections import defaultdict
scores = defaultdict(list)
for g in range(8):
    try:
        for line in open(f'outputs/agentic_pipeline_v19_full/gpu{g}.log'):
            m = re.search(r'Score=([\d.]+)\s+V7=([\d.]+)', line)
            if not m: continue
            qt_m = re.search(r'(\S+)\s+\[VL:', line)
            qt = qt_m.group(1).strip() if qt_m else '?'
            scores[qt].append((float(m.group(1)), float(m.group(2))))
    except: pass
total = sum(len(v) for v in scores.values())
print(f'Parsed: {total} samples')
print(f"{'Task':<35} {'N':>4} {'V7':>6} {'V19':>6} {'D':>6}")
print('-'*60)
a7=[]; a19=[]
for qt in sorted(scores.keys()):
    v = scores[qt]
    s19 = np.mean([x[0] for x in v]); s7 = np.mean([x[1] for x in v])
    print(f'  {qt:<33} {len(v):>4} {s7:.3f} {s19:.3f} {s19-s7:+.3f}')
    a19.extend([x[0] for x in v]); a7.extend([x[1] for x in v])
print('-'*60)
print(f'  {"Overall":<33} {total:>4} {np.mean(a7):.3f} {np.mean(a19):.3f} {np.mean(a19)-np.mean(a7):+.3f}')
