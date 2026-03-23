import re
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

all_samples = []
mca_samples = []
na_samples = []

mca_tasks = {'obj_appearance_order', 'object_rel_direction_easy', 'object_rel_direction_medium',
             'object_rel_direction_hard', 'object_rel_distance', 'route_planning'}
na_tasks = {'object_abs_distance', 'object_counting', 'object_size_estimation', 'room_size_estimation'}

for g in range(8):
    with open(f'{base}/gpu{g}.log') as f:
        for line in f:
            m = re.search(r'\[(\w+)\] ans=(\S+) gt=(\S+) score=([\d.]+)', line)
            if m:
                task = m.group(1)
                score = float(m.group(4))
                sample = {'task': task, 'score': score}
                all_samples.append(sample)
                if task in mca_tasks:
                    mca_samples.append(sample)
                elif task in na_tasks:
                    na_samples.append(sample)

print('=== V21 Overall Results ===')
all_scores = [s['score'] for s in all_samples]
mca_scores = [s['score'] for s in mca_samples]
na_scores = [s['score'] for s in na_samples]

print(f'Total: {len(all_samples)} samples, score={np.mean(all_scores):.4f}')
print(f'MCA:   {len(mca_samples)} samples, score={np.mean(mca_scores):.4f}')
print(f'NA:    {len(na_samples)} samples, score={np.mean(na_scores):.4f}')

by_task = defaultdict(list)
for s in all_samples:
    by_task[s['task']].append(s['score'])

print('\nBy task:')
for task in sorted(by_task.keys()):
    scores = by_task[task]
    task_type = 'MCA' if task in mca_tasks else 'NA'
    print(f'  {task:35s} ({task_type}): {len(scores):4d} samples, {np.mean(scores):.4f}')
