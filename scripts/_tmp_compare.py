import re
import numpy as np

def parse_scores(filepath, is_v21=True):
    """按顺序提取所有score"""
    scores = []
    try:
        with open(filepath) as f:
            for line in f:
                if is_v21:
                    m = re.search(r'\[\w+\]\s+ans=\S+\s+gt=\S+\s+score=([\d.]+)', line)
                else:
                    m = re.search(r'\[\w+\]\s+score=([\d.]+)\s+pred=\S+\s+gt=\S+', line)
                if m:
                    scores.append(float(m.group(1)))
    except Exception as e:
        pass
    return scores

# 按GPU顺序收集（gpu0 -> gpu1 -> ...）
m2b_scores = []
v21_scores = []

for g in range(8):
    m2b_scores.extend(parse_scores(f'outputs/ablation_metadata_noise_B/gpu{g}.log', is_v21=False))
    v21_scores.extend(parse_scores(f'outputs/agentic_pipeline_v21_ref/gpu{g}.log', is_v21=True))

print(f'M2-B样本数: {len(m2b_scores)}')
print(f'V21样本数: {len(v21_scores)}')

# 直接用索引一一对比（相同shard分片，相同位置 = 相同样本）
n_common = min(len(m2b_scores), len(v21_scores))
print(f'\n一一对比前{n_common}个样本（按索引位置）:')

m2b_arr = np.array(m2b_scores[:n_common])
v21_arr = np.array(v21_scores[:n_common])

print(f'  M2-B (Raw Grid Dump): {m2b_arr.mean():.4f}')
print(f'  V21 (Full Framework): {v21_arr.mean():.4f}')
print(f'  差值: {m2b_arr.mean() - v21_arr.mean():.4f}')

# Binary accuracy (>=0.999认为完全正确)
m2b_binary = (m2b_arr >= 0.999).astype(float)
v21_binary = (v21_arr >= 0.999).astype(float)

print(f'\nBinary Accuracy (完全正确率):')
print(f'  M2-B: {m2b_binary.mean():.4f}')
print(f'  V21:  {v21_binary.mean():.4f}')

# 胜负分析
m2b_win = ((m2b_arr >= 0.999) & (v21_arr < 0.999)).sum()
v21_win = ((v21_arr >= 0.999) & (m2b_arr < 0.999)).sum()
both_wrong = ((m2b_arr < 0.999) & (v21_arr < 0.999)).sum()
both_correct = ((m2b_arr >= 0.999) & (v21_arr >= 0.999)).sum()

print(f'\n胜负分析 ({n_common}个样本):')
print(f'  M2-B赢 (M2-B对, V21错): {m2b_win} ({m2b_win/n_common*100:.1f}%)')
print(f'  V21赢  (V21对, M2-B错): {v21_win} ({v21_win/n_common*100:.1f}%)')
print(f'  都打错: {both_wrong} ({both_wrong/n_common*100:.1f}%)')
print(f'  都对:   {both_correct} ({both_correct/n_common*100:.1f}%)')

# 按样本索引分组统计
print(f'\n样本分布:')
print(f'  仅M2-B对: {m2b_win}')
print(f'  仅V21对:  {v21_win}')
print(f'  都对:     {both_correct}')
print(f'  都错:     {both_wrong}')
