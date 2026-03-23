import re
import numpy as np

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

na_tasks = {'object_abs_distance', 'object_counting', 'object_size_estimation', 'room_size_estimation'}

na_samples = []
for g in range(8):
    with open(f'{base}/gpu{g}.log') as f:
        for line in f:
            m = re.search(r'\[(\w+)\] ans=(\S+) gt=(\S+) score=([\d.]+)', line)
            if m:
                task = m.group(1)
                if task in na_tasks:
                    score = float(m.group(4))
                    na_samples.append({'task': task, 'score': score})

print('=== NA Tasks (with CODER) ===')
na_scores = [s['score'] for s in na_samples]
print(f'N={len(na_samples)}, Score={np.mean(na_scores):.4f}')

# By task
from collections import defaultdict
by_task = defaultdict(list)
for s in na_samples:
    by_task[s['task']].append(s['score'])

for task in sorted(by_task.keys()):
    print(f'  {task:30s}: {len(by_task[task]):4d} samples, {np.mean(by_task[task]):.4f}')

# Compare to what NA would be with Pure VL (estimate from MCA Global)
print('\n=== Comparison ===')
print(f'NA with CODER:     {np.mean(na_scores):.4f}')
print(f'Estimated Pure VL: ~0.47 (from MCA Global)')
print(f'Improvement:       +{np.mean(na_scores) - 0.4712:.4f} ({(np.mean(na_scores)/0.4712 - 1)*100:.1f}%)')
