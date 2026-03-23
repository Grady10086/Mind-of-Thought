import json, glob, re
import numpy as np

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

all_results = []
for g in range(8):
    files = glob.glob(f'{base}/gpu{g}/*.json')
    for f in files:
        data = json.load(open(f))
        if isinstance(data, list):
            all_results.extend(data)

# Filter only MCA tasks (choice questions with options)
mcas = [r for r in all_results if r.get('options')]
print(f'Total MCA samples: {len(mcas)}')

# Parse each sample
samples = []
for r in mcas:
    reason = r.get('reasoning', '')
    gt = str(r.get('ground_truth', ''))
    
    rounds = {}
    
    # Global: [vl_full] X conf=Y.YZ
    m = re.search(r'\[vl_full\] ([A-D]) conf=([\d.]+)', reason)
    if m:
        rounds['global'] = {'pred': m.group(1), 'conf': float(m.group(2))}
    
    # R1 focused: [R1:vl_focused_TYPE] X conf=Y.YZ (Nf)
    m = re.search(r'\[R1:vl_focused_\w+\] ([A-D]) conf=([\d.]+)', reason)
    if m:
        rounds['R1'] = {'pred': m.group(1), 'conf': float(m.group(2))}
    
    # R2 focused
    m = re.search(r'\[R2:vl_focused_\w+\] ([A-D]) conf=([\d.]+)', reason)
    if m:
        rounds['R2'] = {'pred': m.group(1), 'conf': float(m.group(2))}
    
    # R3 focused
    m = re.search(r'\[R3:vl_focused_\w+\] ([A-D]) conf=([\d.]+)', reason)
    if m:
        rounds['R3'] = {'pred': m.group(1), 'conf': float(m.group(2))}
    
    samples.append({
        'gt': gt,
        'score': r.get('score', 0),
        'rounds': rounds,
        'reason': reason
    })

print(f'Parsed MCA: {len(samples)}')

# Score by round
print('\n=== Score by Round ===')
for rname in ['global', 'R1', 'R2', 'R3']:
    scores = []
    for s in samples:
        if rname in s['rounds']:
            pred = s['rounds'][rname]['pred']
            scores.append(1.0 if pred == s['gt'] else 0.0)
    if scores:
        print(f'{rname:10s}: {np.mean(scores):.4f} (n={len(scores)})')

# Final
print(f'Final:      {np.mean([s["score"] for s in samples]):.4f}')

# By converge pattern
print('\n=== By Converge Pattern ===')
early = [s for s in samples if 'R1' in s['rounds'] and 'R2' not in s['rounds']]
multi = [s for s in samples if 'R2' in s['rounds']]
only_global = [s for s in samples if 'global' in s['rounds'] and 'R1' not in s['rounds']]

print(f'Global only:  {len(only_global)} samples, score={np.mean([s["score"] for s in only_global]):.4f}')
print(f'Early (R1):   {len(early)} samples, score={np.mean([s["score"] for s in early]):.4f}')
print(f'Multi (R2+):  {len(multi)} samples, score={np.mean([s["score"] for s in multi]):.4f}')
