import re, numpy as np

oracle_scores = []
uniform_scores = []
K_values = []

for g in range(8):
    try:
        with open(f'/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/ablation_attention_dilution/gpu{g}.log') as f:
            for line in f:
                m = re.search(r'oracle=([\d.]+).*uniform-K=([\d.]+).*K=(\d+)', line)
                if m:
                    oracle_scores.append(float(m.group(1)))
                    uniform_scores.append(float(m.group(2)))
                    K_values.append(int(m.group(3)))
    except: pass

if oracle_scores:
    print(f'=== M1-MCA Final Results ===')
    print(f'Total samples: {len(oracle_scores)}')
    print(f'')
    print(f'Oracle (Grid-guided + entity guidance): {np.mean(oracle_scores):.4f}')
    print(f'Uniform-8 (Random, no guidance):        {np.mean(uniform_scores):.4f}')
    print(f'Delta:                                  {np.mean(oracle_scores) - np.mean(uniform_scores):+.4f}')
    print(f'')
    print(f'K (frames used): mean={np.mean(K_values):.1f}, median={np.median(K_values):.0f}')
    vals, cnts = np.unique(K_values, return_counts=True)
    print(f'Distribution: {dict(zip(vals.tolist(), cnts.tolist()))}')
