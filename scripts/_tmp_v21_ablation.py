import re
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'
samples = {}

for g in range(8):
    with open(f'{base}/gpu{g}.log') as f:
        lines = f.readlines()
    current_sample = None
    for i, line in enumerate(lines):
        m_sample = re.search(r'\[(\w+)\] ans=([A-D]) gt=([A-D]) score=([\d.]+)', line)
        if m_sample:
            task = m_sample.group(1)
            final_pred = m_sample.group(2)
            gt = m_sample.group(3)
            score = float(m_sample.group(4))
            current_sample = {'task': task, 'gt': gt, 'final_pred': final_pred, 'score': score, 'rounds': {}}
            m_global = re.search(r'\[vl_full\] ([A-D]) conf=([\d.]+)', line)
            if m_global:
                current_sample['rounds']['global'] = {'pred': m_global.group(1), 'conf': float(m_global.group(2))}
            for j in range(max(0, i-15), i):
                prev = lines[j]
                m_r1 = re.search(r'R1: VL\(focused\)=([A-D]) conf=([\d.]+) \((\d+) frames\)', prev)
                if m_r1 and 'R1' not in current_sample['rounds']:
                    current_sample['rounds']['R1'] = {'pred': m_r1.group(1), 'conf': float(m_r1.group(2)), 'n_frames': int(m_r1.group(3))}
                m_r2 = re.search(r'R2: VL\(focused\)=([A-D]) conf=([\d.]+) \((\d+) frames\)', prev)
                if m_r2 and 'R2' not in current_sample['rounds']:
                    current_sample['rounds']['R2'] = {'pred': m_r2.group(1), 'conf': float(m_r2.group(2)), 'n_frames': int(m_r2.group(3))}
                m_r3 = re.search(r'R3: VL\(focused\)=([A-D]) conf=([\d.]+) \((\d+) frames\)', prev)
                if m_r3 and 'R3' not in current_sample['rounds']:
                    current_sample['rounds']['R3'] = {'pred': m_r3.group(1), 'conf': float(m_r3.group(2)), 'n_frames': int(m_r3.group(3))}
            sample_id = f'{g}_{len(samples)}'
            samples[sample_id] = current_sample

print('=== V21 Multi-Round Ablation ===')
print(f'Total: {len(samples)}')

print('\n=== Score by Round ===')
for round_name in ['global', 'R1', 'R2', 'R3']:
    scores = []
    frame_counts = []
    for s in samples.values():
        if round_name in s['rounds']:
            pred = s['rounds'][round_name]['pred']
            gt = s['gt']
            scores.append(1.0 if pred == gt else 0.0)
            if 'n_frames' in s['rounds'][round_name]:
                frame_counts.append(s['rounds'][round_name]['n_frames'])
    if scores:
        print(f'{round_name:10s}: {np.mean(scores):.4f} (n={len(scores)})', end='')
        if frame_counts:
            print(f' | avg_frames={np.mean(frame_counts):.1f}')
        else:
            print()

final_scores = [s['score'] for s in samples.values()]
print(f'Final:      {np.mean(final_scores):.4f}')

print('\n=== By Converge Pattern ===')
early = [s for s in samples.values() if 'R1' in s['rounds'] and 'R2' not in s['rounds']]
multi_r2 = [s for s in samples.values() if 'R2' in s['rounds'] and 'R3' not in s['rounds']]
multi_r3 = [s for s in samples.values() if 'R3' in s['rounds']]
global_only = [s for s in samples.values() if 'global' in s['rounds'] and 'R1' not in s['rounds']]

go_scores = [s['score'] for s in global_only]
early_scores = [s['score'] for s in early]
r2_scores = [s['score'] for s in multi_r2]
r3_scores = [s['score'] for s in multi_r3]

print(f'Global only: {len(global_only)} samples, score={np.mean(go_scores):.4f}')
print(f'Early (R1):  {len(early)} samples, score={np.mean(early_scores):.4f}')
print(f'Multi (R2):  {len(multi_r2)} samples, score={np.mean(r2_scores):.4f}')
print(f'Multi (R3):  {len(multi_r3)} samples, score={np.mean(r3_scores):.4f}')

print('\n=== By Task ===')
by_task = defaultdict(list)
for s in samples.values():
    by_task[s['task']].append(s['score'])
for task in sorted(by_task.keys()):
    print(f'{task:35s}: {len(by_task[task]):4d} samples, {np.mean(by_task[task]):.4f}')

print('\n=== Evolution: R1→R2 Answer Change ===')
changed_r1_r2 = []
stable_r1_r2 = []
for s in samples.values():
    if 'R1' in s['rounds'] and 'R2' in s['rounds']:
        if s['rounds']['R1']['pred'] != s['rounds']['R2']['pred']:
            changed_r1_r2.append(s)
        else:
            stable_r1_r2.append(s)

print(f'Samples with R1→R2: {len(changed_r1_r2) + len(stable_r1_r2)}')
if changed_r1_r2:
    ch_scores = [s['score'] for s in changed_r1_r2]
    print(f'  Answer changed:   {len(changed_r1_r2)} samples, final score={np.mean(ch_scores):.4f}')
if stable_r1_r2:
    st_scores = [s['score'] for s in stable_r1_r2]
    print(f'  Answer stable:    {len(stable_r1_r2)} samples, final score={np.mean(st_scores):.4f}')
