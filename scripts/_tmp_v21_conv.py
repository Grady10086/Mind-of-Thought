import re
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

samples = []
for g in range(8):
    with open(f'{base}/gpu{g}.log') as f:
        lines = f.readlines()
    
    for line in lines:
        m = re.search(r'\[(\w+)\] ans=([A-D]) gt=([A-D]) score=([\d.]+)', line)
        if m:
            task = m.group(1)
            score = float(m.group(4))
            
            if 'confident_consensus' in line:
                conv = 'confident_consensus'
            elif 'evolution_stable' in line:
                conv = 'evolution_stable'
            elif 'conf_weighted_vote' in line:
                conv = 'conf_weighted_vote'
            else:
                conv = 'early'
            
            rounds = 0
            if '[R3]' in line:
                rounds = 3
            elif '[R2]' in line:
                rounds = 2
            elif '[R1]' in line:
                rounds = 1
            
            samples.append({'task': task, 'score': score, 'conv': conv, 'rounds': rounds})

print(f'Total: {len(samples)}')
print(f'Overall: {np.mean([s[\"score\"] for s in samples]):.4f}')

print(f'\nBy converge type:')
by_conv = defaultdict(list)
for s in samples:
    by_conv[s['conv']].append(s['score'])
for conv in sorted(by_conv.keys()):
    print(f'  {conv:25s}: {len(by_conv[conv]):4d} samples, {np.mean(by_conv[conv]):.4f}')

print(f'\nBy rounds:')
by_rounds = defaultdict(list)
for s in samples:
    by_rounds[s['rounds']].append(s['score'])
for r in sorted(by_rounds.keys()):
    print(f'  R{r}: {len(by_rounds[r]):4d} samples, {np.mean(by_rounds[r]):.4f}')
