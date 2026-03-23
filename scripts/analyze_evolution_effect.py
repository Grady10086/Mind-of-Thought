#!/usr/bin/env python3
"""
分析V7中Evolution的实际效果
"""
import json
from collections import defaultdict

# 读取V7详细结果
with open('outputs/evolving_agent_v7_20260203_134612/detailed_results.json', 'r') as f:
    results = json.load(f)

print("=" * 80)
print("V7 Evolution效果分析")
print("=" * 80)

# 分析Evolution动作
evolution_samples = [r for r in results if r.get('evolution_actions')]
no_evolution_samples = [r for r in results if not r.get('evolution_actions')]

print(f"\n样本分布:")
print(f"  有Evolution动作: {len(evolution_samples)} ({len(evolution_samples)/len(results)*100:.1f}%)")
print(f"  无Evolution动作: {len(no_evolution_samples)} ({len(no_evolution_samples)/len(results)*100:.1f}%)")

# 对比有/无Evolution动作的样本表现
def calc_avg_score(samples, key='vl_score'):
    scores = [r.get(key, 0) for r in samples if key in r]
    return sum(scores) / len(scores) if scores else 0

print(f"\n准确率对比 (VL):")
print(f"  有Evolution动作的样本: {calc_avg_score(evolution_samples)*100:.2f}%")
print(f"  无Evolution动作的样本: {calc_avg_score(no_evolution_samples)*100:.2f}%")

# 按任务类型分析
print(f"\n按任务类型分析:")
print("-" * 80)
print(f"{'任务类型':<35} {'有Evo样本数':>12} {'有Evo VL准确率':>15} {'无Evo VL准确率':>15}")
print("-" * 80)

type_stats = defaultdict(lambda: {'evo': [], 'no_evo': []})
for r in results:
    qtype = r.get('question_type', 'unknown')
    if r.get('evolution_actions'):
        type_stats[qtype]['evo'].append(r.get('vl_score', 0))
    else:
        type_stats[qtype]['no_evo'].append(r.get('vl_score', 0))

for qtype in sorted(type_stats.keys()):
    stats = type_stats[qtype]
    evo_count = len(stats['evo'])
    evo_acc = sum(stats['evo']) / len(stats['evo']) * 100 if stats['evo'] else 0
    no_evo_acc = sum(stats['no_evo']) / len(stats['no_evo']) * 100 if stats['no_evo'] else 0
    print(f"{qtype:<35} {evo_count:>12} {evo_acc:>14.2f}% {no_evo_acc:>14.2f}%")

# 分析Evolution动作类型
print(f"\n\nEvolution动作类型统计:")
print("-" * 80)
action_types = defaultdict(int)

for r in evolution_samples:
    for action in r.get('evolution_actions', []):
        action_type = action.get('type', 'unknown')
        action_types[action_type] += 1

for atype, count in sorted(action_types.items(), key=lambda x: -x[1]):
    print(f"  {atype}: {count}")

print(f"\n结论:")
print("-" * 80)
evo_vl_acc = calc_avg_score(evolution_samples)
no_evo_vl_acc = calc_avg_score(no_evolution_samples)
diff = evo_vl_acc - no_evo_vl_acc
print(f"有Evolution动作的样本VL准确率: {evo_vl_acc*100:.2f}%")
print(f"无Evolution动作的样本VL准确率: {no_evo_vl_acc*100:.2f}%")
print(f"差异: {diff*100:+.2f}%")

if diff > 0.01:
    print("\n→ Evolution动作对被选中的样本有正向效果")
elif diff < -0.01:
    print("\n→ Evolution动作对被选中的样本有负面效果")
else:
    print("\n→ Evolution动作对样本的效果不明显")

# 更深入分析：Evolution修正前后的对比
print("\n" + "=" * 80)
print("Evolution修正效果深入分析")
print("=" * 80)

# 检查evolution动作中的具体修正
counting_corrections = []
for r in evolution_samples:
    if r.get('question_type') == 'object_counting':
        for action in r.get('evolution_actions', []):
            if action.get('type') == 'correct_count':
                counting_corrections.append({
                    'old': action.get('old'),
                    'new': action.get('new'),
                    'rule_score': r.get('rule_score', 0),
                    'vl_score': r.get('vl_score', 0),
                })

if counting_corrections:
    print(f"\nCounting任务Evolution修正分析 ({len(counting_corrections)}个修正):")
    improved = sum(1 for c in counting_corrections if c['vl_score'] > 0.5)
    print(f"  修正后VL正确的比例: {improved/len(counting_corrections)*100:.1f}%")
