#!/usr/bin/env python3
"""生成最终实验对比报告 - 修复版"""
import json
import os
from collections import defaultdict

def load_results(result_dir):
    """加载测试结果"""
    all_results = []
    for i in range(8):
        fpath = os.path.join(result_dir, f"results_gpu{i}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                results = json.load(f)
                all_results.extend(results)
    return all_results

def normalize_answer(answer):
    """标准化答案用于比较"""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    # 移除单位和标点
    answer = answer.replace('meters', '').replace('meter', '').replace('m', '')
    answer = answer.replace('feet', '').replace('foot', '').replace('ft', '')
    answer = answer.strip()
    return answer

def analyze_vl_accuracy(results):
    """分析VL准确率"""
    by_task = defaultdict(lambda: {'total': 0, 'vl_correct': 0})
    
    for r in results:
        task = r.get('question_type', 'unknown')
        by_task[task]['total'] += 1
        
        # 获取VL预测和真实答案
        vl_pred = normalize_answer(r.get('vl_prediction', ''))
        gt = normalize_answer(r.get('ground_truth', ''))
        
        # 判断是否正确
        vl_correct = (vl_pred == gt) if vl_pred and gt else False
        if vl_correct:
            by_task[task]['vl_correct'] += 1
    
    return by_task

# 实验1: 无Evolution
exp1_dir = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/exp1_no_evolution_test_full'
exp1_results = load_results(exp1_dir)
exp1_stats = analyze_vl_accuracy(exp1_results)

# 实验2: 带Evolution (3%)
exp2_dir = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/exp2_with_evolution_test'
exp2_results = load_results(exp2_dir)
exp2_stats = analyze_vl_accuracy(exp2_results)

print("=" * 90)
print("🔬 Evolution对比实验 - 最终结果报告")
print("=" * 90)
print()

print("📊 实验1 vs 实验2 VL准确率对比")
print("-" * 90)
print(f"{'任务类型':<35} | {'实验1(无Evo)':<18} | {'实验2(Evo 3%)':<18} | {'差异':<12}")
print("-" * 90)

all_tasks = set(exp1_stats.keys()) | set(exp2_stats.keys())
exp1_total_correct = 0
exp1_total_all = 0
exp2_total_correct = 0
exp2_total_all = 0

for task in sorted(all_tasks):
    s1 = exp1_stats[task]
    s2 = exp2_stats[task]
    
    acc1 = s1['vl_correct'] / s1['total'] * 100 if s1['total'] > 0 else 0
    acc2 = s2['vl_correct'] / s2['total'] * 100 if s2['total'] > 0 else 0
    diff = acc2 - acc1
    diff_str = f"{diff:+.2f}%" if diff != 0 else "0.00%"
    
    print(f"{task:<35} | {acc1:>6.2f}% ({s1['total']:>4}) | {acc2:>6.2f}% ({s2['total']:>4}) | {diff_str:<12}")
    
    exp1_total_correct += s1['vl_correct']
    exp1_total_all += s1['total']
    exp2_total_correct += s2['vl_correct']
    exp2_total_all += s2['total']

print("-" * 90)
exp1_overall = exp1_total_correct / exp1_total_all * 100
exp2_overall = exp2_total_correct / exp2_total_all * 100
overall_diff = exp2_overall - exp1_overall
print(f"{'Overall':<35} | {exp1_overall:>6.2f}% ({exp1_total_all:>4}) | {exp2_overall:>6.2f}% ({exp2_total_all:>4}) | {overall_diff:+.2f}%")
print("=" * 90)

print()
print("📈 结论:")
if overall_diff > 0:
    print(f"   ✅ Evolution训练提升了VL准确率 {overall_diff:.2f}%")
elif overall_diff < 0:
    print(f"   ⚠️ Evolution训练(3%覆盖率)降低了VL准确率 {abs(overall_diff):.2f}%")
else:
    print(f"   ➡️ Evolution训练对VL准确率无明显影响")
print()

# 阶段2数据状态
print("=" * 90)
print("📦 阶段2数据生成状态 (扩大Evolution覆盖)")
print("-" * 90)
evo_data_file = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/mindmap_expanded_evolution_v4.jsonl'
if os.path.exists(evo_data_file):
    with open(evo_data_file) as f:
        lines = f.readlines()
    print(f"   总样本数: {len(lines)}")
    
    task_counts = defaultdict(lambda: {'total': 0, 'evo': 0})
    for line in lines:
        data = json.loads(line)
        meta = data.get('metadata', {})
        task = meta.get('question_type', 'unknown')
        evo = meta.get('evolution_applied', False)
        task_counts[task]['total'] += 1
        if evo:
            task_counts[task]['evo'] += 1
    
    total_evo = sum(c['evo'] for c in task_counts.values())
    print(f"   Evolution应用数: {total_evo} ({total_evo/len(lines)*100:.1f}%)")
    print()
    print("   按任务类型:")
    for task, c in sorted(task_counts.items()):
        pct = c['evo']/c['total']*100 if c['total'] > 0 else 0
        status = "✅" if pct > 50 else "❌" if pct == 0 else "⚠️"
        print(f"     {status} {task}: {c['evo']}/{c['total']} ({pct:.1f}%)")
else:
    print("   ❌ 数据文件不存在")
print("=" * 90)

print()
print("🔄 下一步行动:")
print("   1. 阶段2数据已生成完成，但Evolution覆盖率只有20.2%")
print("   2. 只有direction类任务应用了Evolution，其他8类任务需要扩展")
print("   3. 需要修复脚本以支持更多任务类型的Evolution")
