import re, glob
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'
all_results = []

for g in range(8):
    try:
        with open(f'{base}/gpu{g}.log') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            m1 = re.search(r'R1: VL\(focused\)=([A-D]) conf=[\d.]+ \((\d+) frames\)', line)
            if m1 and i + 1 < len(lines):
                pred = m1.group(1)
                n_frames = int(m1.group(2))
                
                next_line = lines[i + 1]
                m2 = re.search(r'\[(\w+)\] ans=\S+ gt=([A-D]) score=([\d.]+)', next_line)
                if m2:
                    task = m2.group(1)
                    gt = m2.group(2)
                    score = float(m2.group(3))
                    
                    all_results.append({
                        'task': task,
                        'pred': pred,
                        'gt': gt,
                        'score': score,
                        'n_frames': n_frames
                    })
    except Exception as e:
        print(f'GPU{g} error: {e}')

print(f'=== V21 R1 Focused (Oracle) Results ===')
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
    
    frames = [r['n_frames'] for r in all_results]
    print(f'\nFrame count: mean={np.mean(frames):.1f}, median={np.median(frames):.0f}')
    vals, cnts = np.unique(frames, return_counts=True)
    print(f'Distribution: {dict(zip(vals.tolist(), cnts.tolist()))}')
