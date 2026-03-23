import re
import numpy as np

# M2-B early samples
m2b_scores = []
for g in range(8):
    try:
        with open(f'outputs/ablation_metadata_noise_B/gpu{g}.log') as f:
            for line in f:
                m = re.search(r'score=([\d.]+)', line)
                if m:
                    m2b_scores.append(float(m.group(1)))
    except: pass

m2b_early = m2b_scores[:324]
print(f'M2-B first 324: {np.mean(m2b_early):.4f}')

# V21 first 324
v21_scores = []
for g in range(8):
    try:
        with open(f'outputs/agentic_pipeline_v21_ref/gpu{g}.log') as f:
            for line in f:
                m = re.search(r'score=([\d.]+)', line)
                if m:
                    v21_scores.append(float(m.group(1)))
    except: pass

v21_early = v21_scores[:324]
print(f'V21 first 324:  {np.mean(v21_early):.4f}')

ratio = np.mean(m2b_early) / np.mean(v21_early)
print(f'\nRatio (M2B/V21): {ratio:.4f}')

v21_final = np.mean(v21_scores)
m2b_proj = v21_final * ratio
print(f'\nV21 final:      {v21_final:.4f}')
print(f'M2-B projected: {m2b_proj:.4f}')

print('\n=== M2-B by Task (early) ===')
m2b_tasks = {}
for g in range(8):
    try:
        with open(f'outputs/ablation_metadata_noise_B/gpu{g}.log') as f:
            for line in f:
                m = re.search(r'\[(\w+)\].*score=([\d.]+)', line)
                if m:
                    task = m.group(1)
                    score = float(m.group(2))
                    if task not in m2b_tasks:
                        m2b_tasks[task] = []
                    m2b_tasks[task].append(score)
    except: pass

for task in sorted(m2b_tasks.keys()):
    scores = m2b_tasks[task]
    print(f'{task:35s}: {len(scores):3d} samples, {np.mean(scores):.4f}')
