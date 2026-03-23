import re, sys
from datetime import datetime

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/ablation_attention_dilution'
for g in range(8):
    try:
        with open(f'{base}/gpu{g}.log') as f:
            lines = f.readlines()
    except: continue
    scenes = []
    samples = sum(1 for l in lines if 'oracle=' in l)
    for line in lines:
        m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[(\d+)/(\d+)\]', line)
        if m:
            ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            cur, total = int(m.group(2)), int(m.group(3))
            scenes.append((ts, cur, total))
    if len(scenes) >= 2:
        elapsed = (scenes[-1][0] - scenes[0][0]).total_seconds()
        done = scenes[-1][1] - 1
        total = scenes[-1][2]
        if done > 0:
            per_scene = elapsed / done
            remaining = (total - scenes[-1][1] + 1) * per_scene
            print(f'GPU{g}: {scenes[-1][1]}/{total} | {done} done in {elapsed/60:.0f}min | '
                  f'{per_scene/60:.1f}min/scene | samples={samples} | ETA {remaining/60:.0f}min ({remaining/3600:.1f}h)')
