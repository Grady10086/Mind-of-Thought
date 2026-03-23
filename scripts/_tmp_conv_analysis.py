import json, glob
from collections import Counter

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'
all_results = []
for g in range(8):
    files = glob.glob(f'{base}/gpu{g}/*.json')
    for f in files:
        data = json.load(open(f))
        if isinstance(data, list):
            all_results.extend(data)

type_by_conv = {}
for r in all_results:
    ct = r.get('converge_type', 'unknown')
    qt = r['question_type']
    if ct not in type_by_conv:
        type_by_conv[ct] = Counter()
    type_by_conv[ct][qt] += 1

print('Question types by converge type:')
for ct in ['early', 'global_consensus_confident', 'evolution_stable', 'conf_weighted_vote']:
    if ct in type_by_conv:
        print(f'\n{ct}:')
        for qt, n in type_by_conv[ct].most_common():
            print(f'  {qt}: {n}')
