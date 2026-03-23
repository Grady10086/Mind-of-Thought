#!/usr/bin/env python3
"""
V15 Phase 1 结果分析: 汇总4个配置的参数搜索结果，自动选出最优配置

用法: python scripts/v15_analyze_search.py
"""

import json, os, sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEARCH_DIR = PROJECT_ROOT / "outputs" / "v15_param_search"

SEARCH_CONFIGS = [
    {'id': 0, 'name': 'C0_32f_640x480', 'nframes': 32, 'max_pixels': 640 * 480},
    {'id': 1, 'name': 'C1_48f_480x560', 'nframes': 48, 'max_pixels': 480 * 560},
    {'id': 2, 'name': 'C2_64f_480x360', 'nframes': 64, 'max_pixels': 480 * 360},
    {'id': 3, 'name': 'C3_32f_360x420', 'nframes': 32, 'max_pixels': 360 * 420},
]

QWEN3_OFFICIAL = {
    'object_counting': 0.6340,
    'object_abs_distance': 0.4552,
    'object_size_estimation': 0.7341,
    'room_size_estimation': 0.5771,
    'object_rel_distance': 0.5225,
    'object_rel_direction': 0.5123,
    'route_planning': 0.3041,
    'obj_appearance_order': 0.6100,
}

# V14 数值题成绩 (不变)
V14_NUMERICAL = {
    'object_counting': 0.9823,
    'object_abs_distance': 0.8945,
    'object_size_estimation': 0.9843,
    'room_size_estimation': 0.9583,
}

def to_official_category(qt):
    qt = qt.lower().strip()
    if 'appearance' in qt or 'appear' in qt:
        return 'obj_appearance_order'
    if 'direction' in qt:
        return 'object_rel_direction'
    if 'rel_dist' in qt or 'rel_distance' in qt:
        return 'object_rel_distance'
    if 'route' in qt:
        return 'route_planning'
    return qt


def load_config_results(config_name):
    """加载一个配置的所有GPU分片结果"""
    config_dir = SEARCH_DIR / config_name
    all_results = []

    # Try direct results.json
    direct = config_dir / "results.json"
    if direct.exists():
        with open(direct) as f:
            all_results.extend(json.load(f))

    # Try GPU shards
    for gpu_dir in sorted(config_dir.glob("gpu*")):
        fp = gpu_dir / "results.json"
        if fp.exists():
            with open(fp) as f:
                all_results.extend(json.load(f))

    return all_results


def analyze_config(results, config):
    """分析一个配置的per-category结果"""
    by_off = defaultdict(lambda: {'n': 0, 'correct': 0, 'v7_correct': 0})

    for r in results:
        off = to_official_category(r['question_type'])
        by_off[off]['n'] += 1
        if r.get('score', 0) > 0:
            by_off[off]['correct'] += 1
        if r.get('v7_vl_score', 0) > 0:
            by_off[off]['v7_correct'] += 1

    choice_cats = ['obj_appearance_order', 'object_rel_distance', 'object_rel_direction', 'route_planning']
    cat_results = {}
    all_above = True
    for cat in choice_cats:
        s = by_off[cat]
        if s['n'] == 0:
            cat_results[cat] = {'n': 0, 'acc': 0, 'above_qwen': False}
            all_above = False
            continue
        acc = s['correct'] / s['n']
        v7_acc = s['v7_correct'] / s['n']
        above = acc > QWEN3_OFFICIAL[cat]
        if not above:
            all_above = False
        cat_results[cat] = {
            'n': s['n'],
            'correct': s['correct'],
            'acc': acc,
            'v7_acc': v7_acc,
            'above_qwen': above,
        }

    total_n = sum(by_off[c]['n'] for c in choice_cats)
    total_c = sum(by_off[c]['correct'] for c in choice_cats)
    choice_overall = total_c / total_n if total_n > 0 else 0

    # 计算包含数值题的Overall (macro avg of 8 categories)
    all_8 = {}
    for cat in choice_cats:
        all_8[cat] = cat_results[cat]['acc']
    for cat, score in V14_NUMERICAL.items():
        all_8[cat] = score
    overall_8 = sum(all_8.values()) / 8

    return {
        'config_name': config['name'],
        'nframes': config['nframes'],
        'max_pixels': config['max_pixels'],
        'n_samples': total_n,
        'choice_overall': choice_overall,
        'overall_8cat': overall_8,
        'all_above_qwen': all_above,
        'categories': cat_results,
    }


def main():
    print("=" * 90)
    print("V15 Phase 1: 参数搜索结果汇总")
    print("=" * 90)

    all_analyses = []
    for config in SEARCH_CONFIGS:
        results = load_config_results(config['name'])
        if not results:
            print(f"\n  ⚠ Config {config['name']}: 没有结果文件")
            continue

        analysis = analyze_config(results, config)
        all_analyses.append(analysis)

        tokens_est = (config['nframes'] / 2) * (config['max_pixels'] / 1024)
        print(f"\n{'─' * 70}")
        print(f"Config: {config['name']}  (nframes={config['nframes']}, "
              f"max_pixels={config['max_pixels']}, ~{tokens_est:.0f} tokens)")
        print(f"{'─' * 70}")
        print(f"  {'Category':>25} {'N':>5} {'Acc':>7} {'V7':>7} {'Qwen3':>7} {'Status':>8}")
        print(f"  {'-' * 63}")

        choice_cats = ['obj_appearance_order', 'object_rel_distance', 'object_rel_direction', 'route_planning']
        for cat in choice_cats:
            cr = analysis['categories'].get(cat, {})
            n = cr.get('n', 0)
            acc = cr.get('acc', 0) * 100
            v7 = cr.get('v7_acc', 0) * 100
            q = QWEN3_OFFICIAL[cat] * 100
            above = cr.get('above_qwen', False)
            print(f"  {cat:>25} {n:>5} {acc:>6.1f}% {v7:>6.1f}% {q:>6.1f}% {'  ✓' if above else '  ✗'}")

        print(f"  {'-' * 63}")
        print(f"  {'Choice Overall':>25} {analysis['n_samples']:>5} {analysis['choice_overall']*100:>6.1f}%")
        print(f"  {'Overall (8-cat macro)':>25}       {analysis['overall_8cat']*100:>6.1f}%")
        print(f"  All above Qwen3-VL? {'✓ YES' if analysis['all_above_qwen'] else '✗ NO'}")

    if not all_analyses:
        print("\n没有找到任何结果。请先运行参数搜索。")
        return

    # ============================================================
    # 自动选择最优配置
    # ============================================================
    print("\n" + "=" * 90)
    print("最优配置选择")
    print("=" * 90)

    # 优先级: 1) all_above_qwen=True 2) overall_8cat最高
    passed = [a for a in all_analyses if a['all_above_qwen']]
    if passed:
        best = max(passed, key=lambda a: a['overall_8cat'])
        print(f"\n  ✓ 找到 {len(passed)} 个配置满足「所有类别超过Qwen3-VL」")
        print(f"  ★ 最优配置: {best['config_name']}")
        print(f"    Overall(8-cat): {best['overall_8cat']*100:.1f}%")
        print(f"    nframes={best['nframes']}, max_pixels={best['max_pixels']}")
    else:
        print(f"\n  ✗ 没有配置能让所有类别都超过Qwen3-VL")
        # 选缺口最小的
        def gap_score(a):
            """缺口总和越小越好"""
            total_gap = 0
            for cat, cr in a['categories'].items():
                q = QWEN3_OFFICIAL.get(cat, 0)
                if cr['acc'] < q:
                    total_gap += q - cr['acc']
            return total_gap
        best = min(all_analyses, key=gap_score)
        print(f"  ★ 缺口最小的配置: {best['config_name']}")
        print(f"    Overall(8-cat): {best['overall_8cat']*100:.1f}%")
        for cat, cr in best['categories'].items():
            q = QWEN3_OFFICIAL.get(cat, 0)
            if cr['acc'] < q:
                print(f"    ✗ {cat}: {cr['acc']*100:.1f}% < {q*100:.1f}% (gap={q*100-cr['acc']*100:.1f}pp)")

    # 排行榜
    print(f"\n{'─' * 70}")
    print(f"  {'Rank':>4} {'Config':>20} {'Choice%':>8} {'8cat%':>8} {'AllAbove':>10}")
    print(f"  {'─' * 54}")
    for i, a in enumerate(sorted(all_analyses, key=lambda x: x['overall_8cat'], reverse=True)):
        mark = "★" if a == best else " "
        print(f"  {mark}{i+1:>3} {a['config_name']:>20} {a['choice_overall']*100:>7.1f}% "
              f"{a['overall_8cat']*100:>7.1f}% {'  ✓' if a['all_above_qwen'] else '  ✗'}")

    # Save summary
    summary = {
        'analyses': all_analyses,
        'best_config': best,
        'qwen3_baseline': QWEN3_OFFICIAL,
        'v14_numerical': V14_NUMERICAL,
    }
    summary_path = SEARCH_DIR / "search_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary saved: {summary_path}")

    # 输出推荐的V15 pipeline参数
    print(f"\n{'=' * 70}")
    print(f"V15 Pipeline 推荐参数:")
    print(f"  选择题: nframes={best['nframes']}, max_pixels={best['max_pixels']}")
    print(f"  数值题: nframes=32, max_pixels={360*420} (保持V14不变)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
