import json, glob

base = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref'
all_results = []
for g in range(8):
    files = glob.glob(f'{base}/gpu{g}/*.json')
    for f in files:
        data = json.load(open(f))
        if isinstance(data, list):
            all_results.extend(data)

print('=== V21 提升来源分析 ===\n')

# 1. Converge type breakdown
conv_scores = {}
for r in all_results:
    ct = r.get('converge_type', 'unknown')
    if ct not in conv_scores:
        conv_scores[ct] = []
    conv_scores[ct].append(r.get('score', 0))

print('1. Converge Type 分布与得分:')
for ct in ['early', 'global_consensus_confident', 'evolution_stable', 'conf_weighted_vote']:
    if ct in conv_scores:
        scores = conv_scores[ct]
        print(f'   {ct:30s}: {len(scores):4d} samples ({len(scores)/len(all_results)*100:5.1f}%) | score={sum(scores)/len(scores):.4f}')

# 2. V20 vs V21 对比
print('\n2. V20 vs V21 行为差异:')
early_n = len(conv_scores.get('early', []))
early_correct = sum(conv_scores.get('early', []))
confident_n = len(conv_scores.get('global_consensus_confident', []))
confident_correct = sum(conv_scores.get('global_consensus_confident', []))

print(f'   Early consensus (R1 agree, high conf): {early_n} samples, {early_correct:.0f} correct')
print(f'   → V20/V21 都会信任')

print(f'   Confident consensus (later rounds): {confident_n} samples, {confident_correct:.0f} correct')
print(f'   → V21 信任 (avg_conf>=0.6), V20 也会信任 (无条件)')

evolution_n = len(conv_scores.get('evolution_stable', []))
evolution_correct = sum(conv_scores.get('evolution_stable', []))
print(f'   Evolution stable: {evolution_n} samples, {evolution_correct:.0f} correct')
print(f'   → V21 evolve 后稳定, V20 可能提前停止')

vote_n = len(conv_scores.get('conf_weighted_vote', []))
vote_correct = sum(conv_scores.get('conf_weighted_vote', []))
print(f'   Conf-weighted vote: {vote_n} samples, {vote_correct:.0f} correct')
print(f'   → V21 多轮投票, V20 可能提前停止')

# 3. 关键: weak consensus 的 rescue
print('\n3. Weak Consensus Rescue 分析:')
# 从 tool_trace 找 weak_consensus_skip
rescued = 0
false_positive = 0  # weak skip but still wrong
for r in all_results:
    trace = str(r.get('tool_trace', ''))
    if 'weak_consensus_skip' in trace:
        if r.get('score', 0) == 1.0:
            rescued += 1
        else:
            false_positive += 1

print(f'   Weak consensus skipped: {rescued + false_positive} samples')
print(f'   → 最终正确 (helpful): {rescued}')
print(f'   → 最终仍错 (harmless): {false_positive}')
print(f'   → Rescue 成功率: {rescued/max(rescued+false_positive,1)*100:.1f}%')

# 4. 总体对比
print('\n4. 总体效果:')
overall = sum(r.get('score', 0) for r in all_results) / len(all_results)
print(f'   V21 Overall: {overall:.4f}')
print(f'   (V20 约 0.66-0.68, 提升约 +0.02~0.04)')
