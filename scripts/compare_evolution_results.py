#!/usr/bin/env python3
"""对比三种模式的实验结果"""
import json

# 加载三组结果
results = {}

# 1. V7原始 (仅Counting Evolution)
with open('outputs/evolving_agent_v7_20260203_134612/summary.json') as f:
    results['v7_original'] = json.load(f)

# 2. V7 完整规则Evolution 
with open('outputs/evolving_agent_v7_20260220_140422/summary.json') as f:
    results['v7_rule_evo'] = json.load(f)

# 3. V7 VL-Driven Evolution (两阶段推理)
with open('outputs/evolving_agent_v7_20260220_201744/summary.json') as f:
    results['v7_vl_evo'] = json.load(f)

print("=" * 120)
print("V7 Evolution消融实验 - 三组对比")
print("=" * 120)

print(f"\n{'任务类型':<35} {'V7原始(仅CountEvo)':>20} {'V7完整规则Evo':>18} {'V7 VL-Driven Evo':>20}")
print("-" * 120)

all_types = set()
for r in results.values():
    all_types.update(r['results_by_type'].keys())

for qtype in sorted(all_types):
    orig = results['v7_original']['results_by_type'].get(qtype, {}).get('vl', 0) * 100
    rule = results['v7_rule_evo']['results_by_type'].get(qtype, {}).get('vl', 0) * 100
    vl = results['v7_vl_evo']['results_by_type'].get(qtype, {}).get('vl', 0) * 100
    
    # 标记最好的
    best = max(orig, rule, vl)
    orig_mark = " *" if orig == best else ""
    rule_mark = " *" if rule == best else ""
    vl_mark = " *" if vl == best else ""
    
    print(f"{qtype:<35} {orig:>18.2f}%{orig_mark} {rule:>16.2f}%{rule_mark} {vl:>18.2f}%{vl_mark}")

print("-" * 120)
orig_overall = results['v7_original']['overall']['vl'] * 100
rule_overall = results['v7_rule_evo']['overall']['vl'] * 100  
vl_overall = results['v7_vl_evo']['overall']['vl'] * 100

print(f"{'Overall VL':<35} {orig_overall:>18.2f}%   {rule_overall:>16.2f}%   {vl_overall:>18.2f}%")
print("=" * 120)

print(f"\n对比基准 (V7原始 = {orig_overall:.2f}%):")
print(f"  完整规则Evolution:  {rule_overall - orig_overall:+.2f}%")
print(f"  VL-Driven Evolution: {vl_overall - orig_overall:+.2f}%")

print(f"\n各任务变化 (相对V7原始):")
print(f"{'任务类型':<35} {'规则Evo变化':>15} {'VL Evo变化':>15}")
print("-" * 70)
for qtype in sorted(all_types):
    orig = results['v7_original']['results_by_type'].get(qtype, {}).get('vl', 0) * 100
    rule = results['v7_rule_evo']['results_by_type'].get(qtype, {}).get('vl', 0) * 100
    vl = results['v7_vl_evo']['results_by_type'].get(qtype, {}).get('vl', 0) * 100
    
    rule_diff = rule - orig
    vl_diff = vl - orig
    
    rule_str = f"{rule_diff:+.2f}%" 
    vl_str = f"{vl_diff:+.2f}%"
    
    print(f"{qtype:<35} {rule_str:>15} {vl_str:>15}")

rule_diff_total = rule_overall - orig_overall
vl_diff_total = vl_overall - orig_overall
print("-" * 70)
print(f"{'Overall':<35} {rule_diff_total:+.2f}%{'':>8} {vl_diff_total:+.2f}%")
