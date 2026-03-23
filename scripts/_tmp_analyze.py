#!/usr/bin/env python3
import json, re
d = json.load(open('outputs/agentic_pipeline_v6_full/merged/detailed_results.json'))
count = [r for r in d if r['question_type'] == 'object_counting']
over = [r for r in count if r['score'] < r['v7_vl_score'] - 0.05 and float(r.get('prediction','0')) > float(r['ground_truth'])]
print(f'Over-count cases: {len(over)}')
for r in over[:15]:
    reas = r.get('reasoning','')
    coder_val = ''
    for part in reas.split('|'):
        if 'coder' in part.lower() and 'count' in part.lower():
            m = re.search(r'answer=(\d+)', part)
            if m: coder_val = m.group(1)
    conf = 'normal'
    for part in reas.split('|'):
        if 'coder_conf' in part and 'low' in part: conf = 'low'
    print(f'  pred={r["prediction"]:>3} gt={r["ground_truth"]:>3} coder={coder_val:>3} conf={conf}')
