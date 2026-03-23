#!/usr/bin/env python3
import json
import re
from datasets import load_dataset

ds = load_dataset('nyu-visionx/VSI-Bench', split='test', 
                  cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench')

with open('/home/tione/notebook/tianjungu/projects/Spatial-MLLM/evaluate/annotation/eval_vsibench.json') as f:
    local = json.load(f)

# 创建映射
hf_map = {}
for d in ds:
    key = (d['scene_name'], d['question_type'], d['question'])
    hf_map[key] = d['ground_truth']

local_map = {}
for d in local:
    scene = d['path'].split('/')[-1].replace('.mp4', '')
    sol = d.get('solution', '')
    m = re.search(r'<answer>(.*?)</answer>', sol)
    gt = m.group(1) if m else ''
    key = (scene, d.get('original_question_type'), d.get('problem'))
    local_map[key] = gt

# 比较
match = mismatch = only_hf = only_local = 0
mismatches = []

for key in hf_map:
    if key in local_map:
        if str(hf_map[key]) == str(local_map[key]):
            match += 1
        else:
            mismatch += 1
            mismatches.append((key, hf_map[key], local_map[key]))
    else:
        only_hf += 1

for key in local_map:
    if key not in hf_map:
        only_local += 1

print(f'Match: {match}')
print(f'Mismatch: {mismatch}')
print(f'Only in HF: {only_hf}')
print(f'Only in Local: {only_local}')

if mismatches:
    print('\nFirst 5 mismatches:')
    for k, hf_gt, local_gt in mismatches[:5]:
        print(f'  Scene: {k[0]}, Type: {k[1]}')
        print(f'    HF GT: {hf_gt}, Local GT: {local_gt}')
