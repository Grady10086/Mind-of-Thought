#!/usr/bin/env python3
"""生成正确的实验对比报告 - 使用summary.json数据"""
import json

# 实验1: 无Evolution训练
exp1_summary = {
    "name": "实验1: 无Evolution训练",
    "adapter": "qwen3vl_mindmap_no_evo_9908/checkpoint-155",
    "train_data": "mindmap_9908_no_evolution.jsonl (9908条)",
    "results": {
        "object_counting": 0.8794690265486734,
        "object_size_estimation": 0.85099685204617,
        "room_size_estimation": 0.9319444444444446,
        "object_abs_distance": 0.7706235011990401,
        "object_rel_direction_hard": 0.30563002680965146,
        "object_rel_direction_medium": 0.4656084656084656,
        "object_rel_direction_easy": 0.5023041474654378,
        "object_rel_distance": 0.3591549295774648,
        "obj_appearance_order": 0.5598705501618123,
        "route_planning": 0.30412371134020616,
    },
    "overall_vl": 0.6389863547758284,
    "samples": 5130,
}

# 实验2: 带Evolution训练 (3%覆盖率)
exp2_summary = {
    "name": "实验2: 带Evolution训练 (3%覆盖)",
    "adapter": "qwen3vl_mindmap_with_evo_9908/checkpoint-140",
    "train_data": "mindmap_9908_with_evolution.jsonl (8908条)",
    "results": {
        "object_counting": 0.8506265664160403,
        "object_size_estimation": 0.8174576271186441,
        "room_size_estimation": 0.9045454545454543,
        "object_abs_distance": 0.7630695443645081,
        "object_rel_direction_hard": 0.30294906166219837,
        "object_rel_direction_medium": 0.4470899470899471,
        "object_rel_direction_easy": 0.5069124423963134,
        "object_rel_distance": 0.35070422535211265,
        "obj_appearance_order": 0.5631067961165048,
        "route_planning": 0.30927835051546393,
    },
    "overall_vl": 0.5939630207173089,
    "samples": 4489,  # 注意: 样本数不同是因为部分数据被Evolution过滤
}

# V7基线 (未微调)
v7_baseline = {
    "name": "V7基线 (未微调)",
    "results": {
        "object_counting": 0.8594690265486729,
        "object_size_estimation": 0.878488982161594,
        "room_size_estimation": 0.905555555555556,
        "object_abs_distance": 0.7561151079136685,
        "object_rel_direction_hard": 0.32707774798927614,
        "object_rel_direction_medium": 0.4656084656084656,
        "object_rel_direction_easy": 0.48847926267281105,
        "object_rel_distance": 0.34507042253521125,
        "obj_appearance_order": 0.5566343042071198,
        "route_planning": 0.28865979381443296,
    },
    "overall_vl": 0.636101364522417,
}

print("=" * 100)
print("🔬 Evolution对比实验 - 最终结果报告 (正确版)")
print("=" * 100)
print()

# 打印详细对比
print("📊 VL准确率详细对比")
print("-" * 100)
print(f"{'任务类型':<35} | {'V7基线':>12} | {'实验1(无Evo)':>12} | {'实验2(Evo 3%)':>12} | {'差异':>10}")
print("-" * 100)

tasks = list(exp1_summary["results"].keys())
for task in tasks:
    v7 = v7_baseline["results"].get(task, 0) * 100
    e1 = exp1_summary["results"].get(task, 0) * 100
    e2 = exp2_summary["results"].get(task, 0) * 100
    diff = e1 - e2
    
    print(f"{task:<35} | {v7:>10.2f}% | {e1:>10.2f}% | {e2:>10.2f}% | {diff:>+8.2f}%")

print("-" * 100)
v7_overall = v7_baseline["overall_vl"] * 100
e1_overall = exp1_summary["overall_vl"] * 100
e2_overall = exp2_summary["overall_vl"] * 100
diff_overall = e1_overall - e2_overall

print(f"{'Overall VL':<35} | {v7_overall:>10.2f}% | {e1_overall:>10.2f}% | {e2_overall:>10.2f}% | {diff_overall:>+8.2f}%")
print("=" * 100)

print()
print("📈 关键发现:")
print(f"   1. V7基线VL准确率: {v7_overall:.2f}%")
print(f"   2. 无Evolution微调后: {e1_overall:.2f}% ({e1_overall - v7_overall:+.2f}%)")
print(f"   3. 带Evolution(3%)微调后: {e2_overall:.2f}% ({e2_overall - v7_overall:+.2f}%)")
print()
print(f"   ⚠️ Evolution训练(3%覆盖率)比无Evolution训练低 {diff_overall:.2f}%")
print()

# 分析改进和退步的任务
print("📋 任务级别分析:")
print("   改进的任务 (实验1 > V7):")
for task in tasks:
    e1 = exp1_summary["results"][task] * 100
    v7 = v7_baseline["results"].get(task, 0) * 100
    if e1 > v7:
        print(f"     ✅ {task}: +{e1-v7:.2f}%")

print()
print("   退步的任务 (实验1 < V7):")
for task in tasks:
    e1 = exp1_summary["results"][task] * 100
    v7 = v7_baseline["results"].get(task, 0) * 100
    if e1 < v7:
        print(f"     ❌ {task}: {e1-v7:.2f}%")

print()
print("=" * 100)
print("🔍 结论:")
print("   1. 无Evolution的Mind Map微调比V7基线略有提升 (+0.3%)")
print("   2. 带Evolution(3%覆盖)的微调反而降低了性能 (-4.2%)")
print("   3. Evolution覆盖率太低(仅3%)可能导致模型学习到不一致的模式")
print()
print("🚀 下一步建议:")
print("   1. 扩大Evolution覆盖率到80%+ (所有任务)")
print("   2. 确保Evolution策略对所有任务类型都适用")
print("   3. 重新训练并测试")
print("=" * 100)
