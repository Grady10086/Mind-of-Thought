import re
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

all_choice = []
mca = []
na_choice = []

mca_tasks = {'obj_appearance_order', 'object_rel_direction_easy', 'object_rel_direction_medium', 
             'object_rel_direction_hard', 'object_rel_distance', 'route_planning'}

for g in range(8):
    with open(f'{base}/gpu{g}.log') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = re.search(r'\[(\w+)\] ans=([A-D]) gt=([A-D]) score=([\d.]+)', line)
        if m:
            task = m.group(1)
            score = float(m.group(4))
            sample = {'task': task, 'gt': m.group(3), 'score': score, 'rounds': {}}
            m_global = re.search(r'\[vl_full\] ([A-D]) conf=([\d.]+)', line)
            if m_global:
                sample['rounds']['global'] = {'pred': m_global.group(1), 'conf': float(m_global.group(2))}
            for j in range(max(0, i-15), i):
                prev = lines[j]
                for ri, rn in [(1, 'R1'), (2, 'R2'), (3, 'R3')]:
                    m_r = re.search(rf'R{ri}: VL\(focused\)=([A-D]) conf=([\d.]+) \((\d+) frames\)', prev)
                    if m_r and rn not in sample['rounds']:
                        sample['rounds'][rn] = {'pred': m_r.group(1), 'conf': float(m_r.group(2)), 'n_frames': int(m_r.group(3))}
            all_choice.append(sample)
            if task in mca_tasks:
                mca.append(sample)
            else:
                na_choice.append(sample)

print('=== V21 Multi-Round Ablation ===')
print(f'Total choice tasks: {len(all_choice)} (MCA: {len(mca)}, NA: {len(na_choice)})')

for name, samples in [('ALL CHOICE', all_choice), ('MCA ONLY', mca), ('NA CHOICE', na_choice)]:
    if not samples:
        continue
    print(f'\n=== {name} (N={len(samples)}) ===')
    for rn in ['global', 'R1', 'R2', 'R3']:
        scores = []
        frames = []
        for s in samples:
            if rn in s['rounds']:
                scores.append(1.0 if s['rounds'][rn]['pred'] == s['gt'] else 0.0)
                if 'n_frames' in s['rounds'][rn]:
                    frames.append(s['rounds'][rn]['n_frames'])
        if scores:
            print(f'  {rn:8s}: {np.mean(scores):.4f} (n={len(scores)})', end='')
            if frames:
                print(f' | frames={np.mean(frames):.1f}')
            else:
                print()
    final = [s['score'] for s in samples]
    print(f'  Final:   {np.mean(final):.4f}')
    
    early = [s for s in samples if 'R1' in s['rounds'] and 'R2' not in s['rounds']]
    multi = [s for s in samples if 'R2' in s['rounds']]
    gonly = [s for s in samples if 'global' in s['rounds'] and 'R1' not in s['rounds']]
    if gonly:
        gs = [s['score'] for s in gonly]
        print(f'  Global:  {len(gonly)} samples, {np.mean(gs):.4f}')
    if early:
        es = [s['score'] for s in early]
        print(f'  Early:   {len(early)} samples, {np.mean(es):.4f}')
    if multi:
        ms = [s['score'] for s in multi]
        print(f'  Multi:   {len(multi)} samples, {np.mean(ms):.4f}')
