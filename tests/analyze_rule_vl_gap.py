#!/usr/bin/env python3
import json

# 读取详细结果
with open('outputs/evolving_agent_v72_da3_11_20260208_213704/detailed_results.json') as f:
    results = json.load(f)

for qtype in ['object_rel_direction_easy', 'object_rel_direction_medium', 'route_planning']:
    samples = [r for r in results if r.get('question_type') == qtype]
    
    rule_correct_vl_wrong = 0
    vl_correct_rule_wrong = 0
    both_correct = 0
    both_wrong = 0
    
    for r in samples:
        rule_ok = r.get('rule_score', 0) > 0.5
        vl_ok = r.get('vl_score', 0) > 0.5
        
        if rule_ok and vl_ok:
            both_correct += 1
        elif rule_ok and not vl_ok:
            rule_correct_vl_wrong += 1
        elif vl_ok and not rule_ok:
            vl_correct_rule_wrong += 1
        else:
            both_wrong += 1
    
    total = len(samples)
    print(f'=== {qtype} ({total} 样本) ===')
    print(f'Rule对 VL对: {both_correct} ({both_correct/total*100:.1f}%)')
    print(f'Rule对 VL错: {rule_correct_vl_wrong} ({rule_correct_vl_wrong/total*100:.1f}%)')
    print(f'Rule错 VL对: {vl_correct_rule_wrong} ({vl_correct_rule_wrong/total*100:.1f}%)')
    print(f'都错:        {both_wrong} ({both_wrong/total*100:.1f}%)')
    print(f'Best = {(both_correct+rule_correct_vl_wrong+vl_correct_rule_wrong)/total*100:.1f}%')
    print()
