import re
import numpy as np
from collections import defaultdict

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'

# Structure: sample_key -> {round: prediction}
# sample_key = (gpu_id, line_number_approx)

all_samples = []

for g in range(8):
    try:
        with open(f'{base}/gpu{g}.log') as f:
            lines = f.readlines()
        
        # Find all sample result lines and extract per-round info
        for i, line in enumerate(lines):
            # Match sample summary line
            m = re.search(r'\[(\w+)\] ans=([A-D]) gt=([A-D]) score=([\d.]+)', line)
            if m:
                task = m.group(1)
                final_pred = m.group(2)
                gt = m.group(3)
                score = float(m.group(4))
                
                # Look backwards to find all rounds
                rounds_info = {}
                
                # Check for global (P1)
                if '[vl_full]' in line:
                    m_global = re.search(r'\[vl_full\] ([A-D]) conf=([\d.]+)', line)
                    if m_global:
                        rounds_info['global'] = {
                            'pred': m_global.group(1),
                            'conf': float(m_global.group(2))
                        }
                
                # Check for R1, R2, R3 focused
                for r in range(1, 4):
                    marker = f'[R{r}:vl_focused'
                    if marker in line:
                        # Extract from the same line or nearby
                        pattern = rf'\[R{r}:vl_focused_\w+\] ([A-D])'
                        m_r = re.search(pattern, line)
                        if m_r:
                            rounds_info[f'R{r}'] = {'pred': m_r.group(1)}
                
                # Also check previous lines for round info
                for j in range(max(0, i-10), i):
                    prev = lines[j]
                    # R1 focused
                    m_r1 = re.search(r'R1: VL\(focused\)=([A-D]) conf=([\d.]+)', prev)
                    if m_r1:
                        rounds_info['R1'] = {
                            'pred': m_r1.group(1),
                            'conf': float(m_r1.group(2))
                        }
                    # R2 focused
                    m_r2 = re.search(r'R2: VL\(focused\)=([A-D]) conf=([\d.]+)', prev)
                    if m_r2:
                        rounds_info['R2'] = {
                            'pred': m_r2.group(1),
                            'conf': float(m_r2.group(2))
                        }
                    # R3 focused
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
    except Exception as e:
        print(f'GPU{g}: {e}')

print(f'=== V21 All Rounds Analysis ===')
print(f'Total samples: {len(all_samples)}')

if all_samples:
    # Analyze round availability
    has_global = sum(1 for s in all_samples if 'global' in s['rounds'])
    has_r1 = sum(1 for s in all_samples if 'R1' in s['rounds'])
    has_r2 = sum(1 for s in all_samples if 'R2' in s['rounds'])
    has_r3 = sum(1 for s in all_samples if 'R3' in s['rounds'])
    
    print(f'\nRound availability:')
    print(f'  Global (P1): {has_global} samples')
    print(f'  R1:          {has_r1} samples')
    print(f'  R2:          {has_r2} samples')
    print(f'  R3:          {has_r3} samples')
    
    # Score by round
    print(f'\nScore by round (when available):')
    for round_name in ['global', 'R1', 'R2', 'R3']:
        scores = []
        for s in all_samples:
            if round_name in s['rounds']:
                pred = s['rounds'][round_name]['pred']
                gt = s['gt']
                scores.append(1.0 if pred == gt else 0.0)
        if scores:
            print(f'  {round_name:10s}: {np.mean(scores):.4f} (n={len(scores)})')
    
    # Final score
    final_scores = [s['score'] for s in all_samples]
    print(f'\nFinal score: {np.mean(final_scores):.4f}')
