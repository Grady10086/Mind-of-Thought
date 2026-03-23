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

print(f'Total from JSON: {len(all_results)}')

# Parse reasoning to extract per-round predictions
samples = []
for r in all_results:
    reason = r.get('reasoning', '')
    gt = str(r.get('ground_truth', ''))
    
    rounds = {}
    
    # Global
    m = re.search(r'\[vl_full\] ([A-D]) conf=([\d.]+)', reason)
    if m:
        rounds['global'] = {'pred': m.group(1), 'conf': float(m.group(2))}
    
    # R1, R2, R3
    for ri in [1, 2, 3]:
        m = re.search(rf'\[R{ri}:vl_focused_\w+\] ([A-D])', reason)
        if m:
            rounds[f'R{ri}'] = {'pred': m.group(1)}
    
    samples.append({
        'gt': gt,
        'score': r.get('score', 0),
        'rounds': rounds
    })

print(f'Parsed: {len(samples)}')

# Score by round
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

# Early vs multi
early = [s for s in samples if 'R1' in s['rounds'] and 'R2' not in s['rounds']]
multi = [s for s in samples if 'R2' in s['rounds']]
print(f'\nEarly (R1 only): {len(early)} samples, score={np.mean([s["score"] for s in early]):.4f}')
print(f'Multi (R2+):     {len(multi)} samples, score={np.mean([s["score"] for s in multi]):.4f}')
