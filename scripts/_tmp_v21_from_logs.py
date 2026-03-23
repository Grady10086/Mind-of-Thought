import re
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

# Parse each sample from log lines
# Sample format: [task] ans=X gt=Y score=Z ... | [P1:vl_global] | [vl_full] PRED conf=CF | [R1] | [R1:vl_focused_TYPE] PRED conf=CF (Nf) ...

samples = []

for g in range(8):
    with open(f'{base}/gpu{g}.log') as f:
        content = f.read()
    
    # Find all sample result lines
    # Pattern: [task] ans=X gt=Y score=Z ... | [vl_full] PRED conf=CF | ... | [R1:vl_focused_TYPE] PRED conf=CF
    lines = content.split('\n')
    
    for line in lines:
        if 'ans=' in line and 'gt=' in line and 'score=' in line:
            m = re.search(r'\[(\w+)\] ans=([A-D]) gt=([A-D]) score=([\d.]+)', line)
            if m:
                task = m.group(1)
                final_pred = m.group(2)
                gt = m.group(3)
                score = float(m.group(4))
                
                rounds = {}
                
                # Global: [vl_full] X conf=Y.YZ
                m_g = re.search(r'\[vl_full\] ([A-D]) conf=([\d.]+)', line)
                if m_g:
                    rounds['global'] = {'pred': m_g.group(1), 'conf': float(m_g.group(2))}
                
                # R1: [R1:vl_focused_TYPE] X conf=Y.YZ (Nf)
                m_r1 = re.search(r'\[R1:vl_focused_\w+\] ([A-D]) conf=([\d.]+)', line)
                if m_r1:
                    rounds['R1'] = {'pred': m_r1.group(1), 'conf': float(m_r1.group(2))}
                
                # R2
                m_r2 = re.search(r'\[R2:vl_focused_\w+\] ([A-D]) conf=([\d.]+)', line)
                if m_r2:
                    rounds['R2'] = {'pred': m_r2.group(1), 'conf': float(m_r2.group(2))}
                
                # R3
                m_r3 = re.search(r'\[R3:vl_focused_\w+\] ([A-D]) conf=([\d.]+)', line)
                if m_r3:
                    rounds['R3'] = {'pred': m_r3.group(1), 'conf': float(m_r3.group(2))}
                
                samples.append({
                    'task': task,
                    'gt': gt,
                    'final_pred': final_pred,
                    'score': score,
                    'rounds': rounds
                })

print(f'=== V21 All Rounds (from logs) ===')
print(f'Total: {len(samples)}')

# Score by round
print('\nScore by round:')
for rname in ['global', 'R1', 'R2', 'R3']:
    scores = []
    for s in samples:
        if rname in s['rounds']:
            pred = s['rounds'][rname]['pred']
            scores.append(1.0 if pred == s['gt'] else 0.0)
    if scores:
        print(f'  {rname:10s}: {np.mean(scores):.4f} (n={len(scores)})')

# Final
print(f'  Final:      {np.mean([s["score"] for s in samples]):.4f}')

# By converge pattern
print('\nBy converge pattern:')
early = [s for s in samples if 'R1' in s['rounds'] and 'R2' not in s['rounds']]
multi = [s for s in samples if 'R2' in s['rounds']]
print(f'  Early (R1 only): {len(early)} samples, score={np.mean([s["score"] for s in early]):.4f}')
print(f'  Multi (R2+):     {len(multi)} samples, score={np.mean([s["score"] for s in multi]):.4f}')

# By task
print('\nBy task (final score):')
by_task = defaultdict(list)
for s in samples:
    by_task[s['task']].append(s['score'])
for task in sorted(by_task.keys()):
    print(f'  {task:30s}: {len(by_task[task]):4d} samples, {np.mean(by_task[task]):.4f}')
