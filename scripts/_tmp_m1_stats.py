import re
oracle_scores = []
uniform_scores = []
for g in range(8):
    try:
        with open(f'/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/ablation_attention_dilution/gpu{g}.log') as f:
            for line in f:
                m = re.search(r'oracle=([\d.]+).*uniform-K=([\d.]+)', line)
                if m:
                    oracle_scores.append(float(m.group(1)))
                    uniform_scores.append(float(m.group(2)))
    except: pass

if oracle_scores:
    print(f'Total: {len(oracle_scores)} samples')
    print(f'Oracle:    {sum(oracle_scores)/len(oracle_scores):.4f}')
    print(f'Uniform-8: {sum(uniform_scores)/len(uniform_scores):.4f}')
    print(f'Delta:     {sum(oracle_scores)/len(oracle_scores) - sum(uniform_scores)/len(uniform_scores):+.4f}')
