import re, numpy as np
base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'
cooccur_nf, temporal_nf = [], []
for g in range(8):
    with open(f'{base}/gpu{g}.log') as f:
        lines = f.readlines()
    pending_nf = []
    for line in lines:
        m = re.search(r'VL\(focused\)=\S+ conf=[\d.]+ \((\d+) frames\)', line)
        if m:
            pending_nf.append(int(m.group(1)))
        if 'vl_focused_cooccur' in line:
            cooccur_nf.extend(pending_nf)
            pending_nf = []
        elif 'vl_focused_temporal' in line:
            temporal_nf.extend(pending_nf)
            pending_nf = []
        elif re.search(r'ans=\S+ gt=\S+ score=', line) and 'vl_focused' not in line:
            pending_nf = []

c, t = np.array(cooccur_nf), np.array(temporal_nf)
print(f'Cooccur: n={len(c)}, mean={c.mean():.1f}')
v, n = np.unique(c, return_counts=True)
print(f'  dist: {dict(zip(v.tolist(),n.tolist()))}')
print(f'Temporal: n={len(t)}, mean={t.mean():.1f}')
v, n = np.unique(t, return_counts=True)
print(f'  dist: {dict(zip(v.tolist(),n.tolist()))}')
