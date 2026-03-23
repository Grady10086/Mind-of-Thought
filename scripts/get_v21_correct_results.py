import json
import os
import numpy as np
from collections import defaultdict

def load_all_results(base_path):
    """从所有GPU目录加载结果"""
    all_results = []
    for g in range(8):
        filepath = f'{base_path}/gpu{g}/detailed_results.json'
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                    print(f'GPU {g}: {len(data)} samples')
                elif isinstance(data, dict) and 'results' in data:
                    all_results.extend(data['results'])
                    print(f'GPU {g}: {len(data["results"])} samples')
    return all_results

# 读取V21结果
print("=== Loading V21 Results ===")
v21_results = load_all_results('/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref')
print(f'\nTotal V21 samples: {len(v21_results)}')

if not v21_results:
    print("No results found!")
    exit(1)

# 计算每个类型的结果
type_stats = defaultdict(lambda: {'scores': [], 'count': 0})
for r in v21_results:
    qtype = r['question_type']
    score = r['score']
    type_stats[qtype]['scores'].append(score)
    type_stats[qtype]['count'] += 1

# 打印每个类型的结果
print("\n=== Per Type Results (V21) ===")
for qtype in sorted(type_stats.keys()):
    stats = type_stats[qtype]
    scores = np.array(stats['scores'])
    weighted = scores.mean()
    binary = (scores >= 0.999).mean()
    correct = (scores >= 0.999).sum()
    print(f"{qtype}: N={stats['count']}, Weighted={weighted:.4f}, Binary={binary:.4f}, Correct={correct}/{stats['count']}")

# 计算Overall
all_scores = np.array([r['score'] for r in v21_results])
overall_weighted = all_scores.mean()
overall_binary = (all_scores >= 0.999).mean()
overall_correct = (all_scores >= 0.999).sum()

print(f"\n=== Overall Results (V21) ===")
print(f"Total N: {len(v21_results)}")
print(f"Weighted Accuracy: {overall_weighted:.4f}")
print(f"Binary Accuracy: {overall_binary:.4f}")
print(f"Correct/Total: {overall_correct}/{len(v21_results)}")

# 区分MCA和NA
mca_scores = []
na_scores = []
for r in v21_results:
    if r.get('options') and len(r.get('options', [])) > 0:
        mca_scores.append(r['score'])
    else:
        na_scores.append(r['score'])

mca_arr = np.array(mca_scores)
na_arr = np.array(na_scores)

print(f"\n=== MCA vs NA ===")
print(f"MCA: N={len(mca_scores)}, Weighted={mca_arr.mean():.4f}, Binary={(mca_arr >= 0.999).mean():.4f}")
print(f"NA:  N={len(na_scores)}, Weighted={na_arr.mean():.4f}, Binary={(na_arr >= 0.999).mean():.4f}")
