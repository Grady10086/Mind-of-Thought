import re, glob
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

all_results = []
for g in range(8):
    try:
        with open(f'{base}/gpu{g}.log') as f:
            content = f.read()
        
        # Match: [task] ans=X gt=Y score=Z ... [R1:vl_focused_TYPE] ANS
        pattern = r'\[(\w+)\] ans=([A-D]) gt=([A-D]) score=([\d.]+)[^\n]*\[R1:vl_focused_(\w+)\] ([A-D])'
        matches = re.findall(pattern, content)
        
        for m in matches:
            all_results.append({
                'task': m[0],
                'pred': m[5],
                'gt': m[2],
                'score': float(m[3]),
                'focus_type': m[4]
            })
    except Exception as e:
        print(f'GPU{g}: {e}')

print(f'=== V21 R1 Focused (Oracle) Results ===')
print(f'Total samples: {len(all_results)}')

if all_results:
    scores = [r['score'] for r in all_results]
    print(f'Overall Score: {np.mean(scores):.4f}')
    
    # By task
    by_task = defaultdict(list)
    for r in all_results:
        by_task[r['task']].append(r['score'])
    
    print(f'\nBy task:')
    for task in sorted(by_task.keys()):
        s = by_task[task]
        print(f'  {task:30s}: {len(s):4d} samples, score={np.mean(s):.4f}')
    
    # By focus type
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r['focus_type']].append(r['score'])
    
    print(f'\nBy focus type:')
    for ft in sorted(by_type.keys()):
        s = by_type[ft]
        print(f'  {ft:15s}: {len(s):4d} samples, score={np.mean(s):.4f}')
