import re
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

all_results = []
for g in range(8):
    try:
        with open(f'{base}/gpu{g}.log') as f:
            content = f.read()
        
        # Match Global VL: [vl_full] X conf=Y.ZZ ... [R1]
        # But need to find the corresponding task line
        # Pattern: [task] ans=X gt=Y score=Z ... [vl_full] ANS conf=W.WW
        pattern = r'\[(\w+)\] ans=([A-D]) gt=([A-D]) score=([\d.]+)[^\n]*\[vl_full\] ([A-D]) conf=[\d.]+'
        matches = re.findall(pattern, content)
        
        for m in matches:
            all_results.append({
                'task': m[0],
                'pred': m[4],
                'gt': m[2],
                'score': float(m[3])
            })
    except Exception as e:
        print(f'GPU{g}: {e}')

print(f'=== V21 P1 Global (Full Video) Results ===')
print(f'Total samples: {len(all_results)}')

if all_results:
    scores = [r['score'] for r in all_results]
    print(f'Overall Score: {np.mean(scores):.4f}')
    
    by_task = defaultdict(list)
    for r in all_results:
        by_task[r['task']].append(r['score'])
    
    print(f'\nBy task:')
    for task in sorted(by_task.keys()):
        s = by_task[task]
        print(f'  {task:30s}: {len(s):4d} samples, score={np.mean(s):.4f}')
