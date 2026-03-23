#!/usr/bin/env python3
import json, numpy as np, sys
from pathlib import Path
od = Path(sys.argv[1]) if len(sys.argv)>1 else Path('outputs/agentic_pipeline_v16_ta4_8gpu')
a=[]
for g in range(8):
    p=od/f'gpu{g}'/'detailed_results.json'
    if p.exists():
        with open(p) as f: a.extend(json.load(f))
print(f'Total: {len(a)} samples')
B={'object_counting':.634,'object_abs_distance':.455,'object_size_estimation':.734,
   'room_size_estimation':.577,'object_rel_distance':.522,'object_rel_direction':.512,
   'route_planning':.304,'obj_appearance_order':.610}
tts=sorted(set(r['question_type'] for r in a))
for qt in tts:
    qr=[r for r in a if r['question_type']==qt]; s=np.mean([r['score'] for r in qr])
    b=B.get(qt,0); mk='✅' if b>0 and s>b else ('❌' if b>0 else '  ')
    print(f'  {qt:<35} N={len(qr):>3} {s:.3f} base={b:.3f} {mk}')
d=[r for r in a if r['question_type'].startswith('object_rel_direction')]
if d:
    ds=np.mean([r['score'] for r in d])
    print(f'  {"[AGG] direction":<35} N={len(d):>3} {ds:.3f} base=0.512 {"✅" if ds>0.512 else "❌"}')
print(f'  Overall: N={len(a)} {np.mean([r["score"] for r in a]):.3f}')
