import json, glob, os

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'
all_results = []
for g in range(8):
    files = glob.glob(f'{base}/gpu{g}/*.json')
    for f in files:
        data = json.load(open(f))
        if isinstance(data, list):
            all_results.extend(data)

print(f'Total: {len(all_results)} samples')

# Converge type
conv_types = {}
for r in all_results:
    ct = r.get('converge_type', 'unknown')
    conv_types[ct] = conv_types.get(ct, 0) + 1

print(f'\nConverge types:')
for ct, n in sorted(conv_types.items(), key=lambda x: -x[1]):
    pct = n / len(all_results) * 100
    subset = [r for r in all_results if r.get('converge_type') == ct]
    scores = [r.get('score', 0) for r in subset]
    avg = sum(scores) / len(scores)
    print(f'  {ct}: {n} ({pct:.1f}%) | score={avg:.4f}')

# VL calls
vl_calls = [r.get('vl_calls', 0) for r in all_results]
print(f'\nVL calls: avg={sum(vl_calls)/len(vl_calls):.1f}')
