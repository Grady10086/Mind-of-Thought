import re
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

# Parse each sample's full trajectory
all_samples = []

for g in range(8):
    try:
        with open(f'{base}/gpu{g}.log') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Find sample result line: [task] ans=X gt=Y score=Z ...
            m = re.search(r'\[(\w+)\] ans=([A-D]) gt=([A-D]) score=([\d.]+)', line)
            if m:
                task = m.group(1)
                final_pred = m.group(2)
                gt = m.group(3)
                score = float(m.group(4))
                
                # Look backwards up to 20 lines to find all VL calls
                rounds_info = {}
                start = max(0, i - 20)
                
                for j in range(start, i):
                    prev = lines[j]
                    
                    # Global VL (P1): [vl_full] X conf=Y.YZ
                    if '[vl_full]' in prev and 'vl_full' not in rounds_info:
                        m_g = re.search(r'\[vl_full\] ([A-D]) conf=([\d.]+)', prev)
                        if m_g:
                            rounds_info['global'] = {
                                'pred': m_g.group(1),
                                'conf': float(m_g.group(2))
                            }
                    
                    # R1 Focused: R1: VL(focused)=X conf=Y.YZ (N frames)
                    if 'R1: VL(focused)' in prev and 'R1' not in rounds_info:
                        m_r1 = re.search(r'R1: VL\(focused\)=([A-D]) conf=([\d.]+)', prev)
                        if m_r1:
                            rounds_info['R1'] = {
                                'pred': m_r1.group(1),
                                'conf': float(m_r1.group(2))
                            }
                    
                    # R2 Focused
                    if 'R2: VL(focused)' in prev and 'R2' not in rounds_info:
                        m_r2 = re.search(r'R2: VL\(focused\)=([A-D]) conf=([\d.]+)', prev)
                        if m_r2:
                            rounds_info['R2'] = {
                                'pred': m_r2.group(1),
                                'conf': float(m_r2.group(2))
                            }
                    
                    # R3 Focused
                    if 'R3: VL(focused)' in prev and 'R3' not in rounds_info:
                        m_r3 = re.search(r'R3: VL\(focused\)=([A-D]) conf=([\d.]+)', prev)
                        if m_r3:
                            rounds_info['R3'] = {
                                'pred': m_r3.group(1),
                                'conf': float(m_r3.group(2))
                            }
                
                all_samples.append({
                    'gpu': g,
                    'task': task,
                    'gt': gt,
                    'final_pred': final_pred,
                    'score': score,
                    'rounds': rounds_info
                })
            
            i += 1
    except Exception as e:
        print(f'GPU{g}: {e}')

print(f'=== V21 All Rounds (v2) ===')
print(f'Total samples: {len(all_samples)}')

if all_samples:
    # Round availability
    print(f'\nRound availability:')
    for rname in ['global', 'R1', 'R2', 'R3']:
        n = sum(1 for s in all_samples if rname in s['rounds'])
        print(f'  {rname:10s}: {n} samples')
    
    # Score by round
    print(f'\nScore by round:')
    for rname in ['global', 'R1', 'R2', 'R3']:
        scores = []
        for s in all_samples:
            if rname in s['rounds']:
                pred = s['rounds'][rname]['pred']
                gt = s['gt']
                scores.append(1.0 if pred == gt else 0.0)
        if scores:
            print(f'  {rname:10s}: {np.mean(scores):.4f} (n={len(scores)})')
    
    # Final
    final_scores = [s['score'] for s in all_samples]
    print(f'\nFinal:        {np.mean(final_scores):.4f} (n={len(final_scores)})')
    
    # Converge type analysis
    print(f'\nBy converge type (from final score pattern):')
    # Early stop (R1 only): score matches R1
    early = [s for s in all_samples if 'R1' in s['rounds'] and 'R2' not in s['rounds']]
    early_scores = [s['score'] for s in early]
    print(f'  Early stop (R1 only):  {len(early)} samples, score={np.mean(early_scores):.4f}')
    
    # Multi-round
    multi = [s for s in all_samples if 'R2' in s['rounds']]
    multi_scores = [s['score'] for s in multi]
    print(f'  Multi-round (R2+):     {len(multi)} samples, score={np.mean(multi_scores):.4f}')
