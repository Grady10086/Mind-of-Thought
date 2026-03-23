#!/usr/bin/env python3
import json

with open('outputs/v710_cross_product_20260206_142754/results_neg_y_axis.json') as f:
    data = json.load(f)

found_wrong = [r for r in data if r.get('objects_found',{}).get('standing') and 
               r.get('objects_found',{}).get('facing') and r.get('objects_found',{}).get('target')
               and not r.get('correct')]

print('错误案例详细分析:')
print('='*60)

for r in found_wrong:
    di = r.get('debug_info', {})
    pf = di.get('proj_forward', 0)
    pr = di.get('proj_right', 0)
    
    opts = r.get('options', [])
    gt_dir = ''
    for opt in opts:
        if opt.startswith(r['ground_truth'] + '.'):
            gt_dir = opt.split('.')[1].strip().lower()
    
    pred_dir = r.get('predicted_direction', '')
    
    gt_fb = 'front' if 'front' in gt_dir else ('back' if 'back' in gt_dir else '')
    gt_lr = 'left' if 'left' in gt_dir else ('right' if 'right' in gt_dir else '')
    pred_fb = 'front' if 'front' in pred_dir else ('back' if 'back' in pred_dir else '')
    pred_lr = 'left' if 'left' in pred_dir else ('right' if 'right' in pred_dir else '')
    
    fb_wrong = gt_fb and pred_fb and gt_fb != pred_fb
    lr_wrong = gt_lr and pred_lr and gt_lr != pred_lr
    
    # 根据 GT 判断 proj 应该的符号
    expected_pf_sign = '+' if gt_fb == 'front' else ('-' if gt_fb == 'back' else '?')
    expected_pr_sign = '+' if gt_lr == 'right' else ('-' if gt_lr == 'left' else '?')
    actual_pf_sign = '+' if pf > 0 else '-'
    actual_pr_sign = '+' if pr > 0 else '-'
    
    print(f'Scene: {r["scene_name"]}')
    print(f'  GT: {gt_dir}, Pred: {pred_dir}')
    print(f'  proj_forward: {pf:+.2f} (期望{expected_pf_sign}, 实际{actual_pf_sign})')
    print(f'  proj_right:   {pr:+.2f} (期望{expected_pr_sign}, 实际{actual_pr_sign})')
    print()
