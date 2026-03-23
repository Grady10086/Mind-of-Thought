#!/usr/bin/env python3
"""Analyze V21 frame statistics from logs."""
import json, os, glob, re
import numpy as np

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

print("=" * 60)
print("V21 Focused VL Frame Statistics (from logs)")
print("=" * 60)

# VL(focused)=X conf=Y.YY (N frames)
focused_entries = []  # (n_frames, conf)
# Also track per-sample: how many focused VL calls, which types
sample_focused_calls = []
cur_sample_calls = []

for g in range(8):
    logfile = f'{base}/gpu{g}.log'
    if not os.path.exists(logfile):
        continue
    with open(logfile) as f:
        for line in f:
            # Focused VL call
            m = re.search(r'VL\(focused\)=\S+ conf=([\d.]+) \((\d+) frames\)', line)
            if m:
                conf = float(m.group(1))
                nf = int(m.group(2))
                focused_entries.append((nf, conf))
                cur_sample_calls.append(nf)
            
            # Sample boundary: score line
            if re.search(r'ans=\S+ gt=\S+ score=', line):
                if cur_sample_calls:
                    sample_focused_calls.append(cur_sample_calls[:])
                cur_sample_calls = []

    if cur_sample_calls:
        sample_focused_calls.append(cur_sample_calls[:])
        cur_sample_calls = []

print(f'Total focused VL calls: {len(focused_entries)}')
print(f'Samples with >=1 focused call: {len(sample_focused_calls)}')

if focused_entries:
    all_nf = np.array([e[0] for e in focused_entries])
    all_conf = np.array([e[1] for e in focused_entries])
    print(f'\nFrame counts:')
    print(f'  mean={all_nf.mean():.1f}, median={np.median(all_nf):.0f}, '
          f'min={all_nf.min()}, max={all_nf.max()}')
    vals, cnts = np.unique(all_nf, return_counts=True)
    print(f'  Distribution: {dict(zip(vals.tolist(), cnts.tolist()))}')
    print(f'\nConfidence:')
    print(f'  mean={all_conf.mean():.2f}, median={np.median(all_conf):.2f}')

if sample_focused_calls:
    calls_per_sample = [len(c) for c in sample_focused_calls]
    frames_per_sample = [sum(c) for c in sample_focused_calls]
    print(f'\nPer sample (with focused VL):')
    print(f'  Focused calls: mean={np.mean(calls_per_sample):.1f}, '
          f'median={np.median(calls_per_sample):.0f}')
    print(f'  Total focused frames: mean={np.mean(frames_per_sample):.1f}')
    vals2, cnts2 = np.unique(calls_per_sample, return_counts=True)
    print(f'  Calls distribution: {dict(zip(vals2.tolist(), cnts2.tolist()))}')

# Also check cooccur vs temporal from the summary line
print("\n" + "=" * 60)
print("Cooccur vs Temporal breakdown")
print("=" * 60)

cooccur_frames = []
temporal_frames = []

for g in range(8):
    logfile = f'{base}/gpu{g}.log'
    if not os.path.exists(logfile):
        continue
    with open(logfile) as f:
        for line in f:
            m_co = re.search(r'vl_focused_cooccur.*?(\d+)f?\b', line)
            m_te = re.search(r'vl_focused_temporal.*?(\d+)f?\b', line)
            
            # Alternative: look for the R* lines
            m2 = re.search(r'\[R\d+:vl_focused_(\w+)\]', line)
            if m2:
                ftype = m2.group(1)
                # Find frame count from nearby VL(focused) line
                m3 = re.search(r'\((\d+) frames\)', line)
                # Actually the frame count is in the preceding log line, not this one
                # Let's use a different approach

# Better approach: parse pairs of consecutive lines
for g in range(8):
    logfile = f'{base}/gpu{g}.log'
    if not os.path.exists(logfile):
        continue
    with open(logfile) as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        m = re.search(r'VL\(focused\)=\S+ conf=[\d.]+ \((\d+) frames\)', line)
        if m and i + 1 < len(lines):
            nf = int(m.group(1))
            next_line = lines[i + 1]
            if 'vl_focused_cooccur' in next_line:
                cooccur_frames.append(nf)
            elif 'vl_focused_temporal' in next_line:
                temporal_frames.append(nf)

print(f'Cooccur focused calls: {len(cooccur_frames)}')
if cooccur_frames:
    arr = np.array(cooccur_frames)
    print(f'  frames: mean={arr.mean():.1f}, median={np.median(arr):.0f}')
    vals, cnts = np.unique(arr, return_counts=True)
    print(f'  Distribution: {dict(zip(vals.tolist(), cnts.tolist()))}')

print(f'Temporal focused calls: {len(temporal_frames)}')
if temporal_frames:
    arr = np.array(temporal_frames)
    print(f'  frames: mean={arr.mean():.1f}, median={np.median(arr):.0f}')
    vals, cnts = np.unique(arr, return_counts=True)
    print(f'  Distribution: {dict(zip(vals.tolist(), cnts.tolist()))}')

# Grid frames
print("\n" + "=" * 60)
print("Grid construction stats")
print("=" * 60)

grid_frames = []
grid_entities = []
for g in range(8):
    logfile = f'{base}/gpu{g}.log'
    if not os.path.exists(logfile):
        continue
    with open(logfile) as f:
        content = f.read()
    matches = re.findall(r'Grid fps=[\d.]+: (\d+) frames', content)
    for m in matches:
        grid_frames.append(int(m))
    matches2 = re.findall(r'Grid256: (\d+) ents', content)
    for m in matches2:
        grid_entities.append(int(m))

if grid_frames:
    arr = np.array(grid_frames)
    print(f'Grid frames: n={len(arr)}, mean={arr.mean():.1f}, '
          f'median={np.median(arr):.0f}, min={arr.min()}, max={arr.max()}')
if grid_entities:
    arr = np.array(grid_entities)
    print(f'Grid entities: n={len(arr)}, mean={arr.mean():.1f}, '
          f'median={np.median(arr):.0f}, min={arr.min()}, max={arr.max()}')

# V21 overall
print("\n" + "=" * 60)
print("V21 Overall Results")
print("=" * 60)

all_results = []
for g in range(8):
    files = glob.glob(f'{base}/gpu{g}/*.json')
    for f in files:
        data = json.load(open(f))
        if isinstance(data, list):
            all_results.extend(data)

scores = [r.get('score', 0) for r in all_results]
print(f'Total: {len(all_results)}, Overall: {np.mean(scores):.4f}')

conv_types = {}
for r in all_results:
    ct = r.get('converge_type', 'unknown')
    conv_types[ct] = conv_types.get(ct, 0) + 1
print(f'Converge: {conv_types}')
