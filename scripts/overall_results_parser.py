#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Tuple

TASK_RE = re.compile(r'\[(?P<task>[^\]]+)\]\s+ans=(?P<ans>\S+)\s+gt=(?P<gt>\S+)\s+score=(?P<score>[\d.]+)(?:.*?V7=(?P<v7>[\d.]+))?')

CHOICE_TASKS = {
    'obj_appearance_order',
    'object_rel_direction_easy',
    'object_rel_direction_medium',
    'object_rel_direction_hard',
    'object_rel_distance',
    'route_planning',
}
NUMERIC_TASKS = {
    'object_abs_distance',
    'object_counting',
    'object_size_estimation',
    'room_size_estimation',
}


def _has_log_sources(path: Path) -> bool:
    return path.is_dir() and any(path.glob('gpu*.log'))


def _has_json_sources(path: Path) -> bool:
    if path.is_file() and path.name == 'detailed_results.json':
        return True
    if not path.is_dir():
        return False
    if (path / 'detailed_results.json').exists():
        return True
    return any(path.glob('gpu*/detailed_results.json'))


def discover_default_base() -> Path:
    outputs_root = Path(__file__).resolve().parent.parent / 'outputs'
    candidates = [
        path for path in outputs_root.iterdir()
        if path.is_dir() and (_has_log_sources(path) or _has_json_sources(path))
    ]
    if not candidates:
        return outputs_root
    return max(candidates, key=lambda item: item.stat().st_mtime)


def resolve_base(base_arg: str | None) -> Path:
    candidate = Path(base_arg).expanduser() if base_arg else discover_default_base()
    if _has_log_sources(candidate) or _has_json_sources(candidate):
        return candidate.resolve()
    raise FileNotFoundError(
        f'Could not find gpu*.log or detailed_results.json under: {candidate}'
    )


def collect_logs(base: Path, num_gpus: int | None) -> List[Path]:
    if not base.is_dir():
        return []
    if num_gpus is None:
        return sorted(base.glob('gpu*.log'))
    logs: List[Path] = []
    for gpu_id in range(num_gpus):
        path = base / f'gpu{gpu_id}.log'
        if path.exists():
            logs.append(path)
    return logs


def collect_json_results(base: Path, num_gpus: int | None) -> List[Path]:
    if base.is_file() and base.name == 'detailed_results.json':
        return [base]
    if not base.is_dir():
        return []
    direct = base / 'detailed_results.json'
    if direct.exists():
        return [direct]
    if num_gpus is None:
        return sorted(base.glob('gpu*/detailed_results.json'))
    results: List[Path] = []
    for gpu_id in range(num_gpus):
        path = base / f'gpu{gpu_id}' / 'detailed_results.json'
        if path.exists():
            results.append(path)
    return results


def parse_logs(log_paths: Iterable[Path]) -> list[dict]:
    samples = []
    for log_path in log_paths:
        with log_path.open(encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                match = TASK_RE.search(line)
                if not match:
                    continue
                samples.append({
                    'task': match.group('task'),
                    'answer': match.group('ans'),
                    'ground_truth': match.group('gt'),
                    'score': float(match.group('score')),
                    'v7': float(match.group('v7')) if match.group('v7') else None,
                    'source': str(log_path),
                })
    return samples


def parse_json_results(result_paths: Iterable[Path]) -> list[dict]:
    samples = []
    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding='utf-8'))
        for item in payload:
            samples.append({
                'task': item.get('question_type'),
                'answer': item.get('prediction'),
                'ground_truth': item.get('ground_truth'),
                'score': float(item.get('score', 0.0)),
                'v7': item.get('v7_vl_score', item.get('vl_score')),
                'source': str(result_path),
            })
    return samples


def load_samples(base: Path, num_gpus: int | None) -> Tuple[list[dict], str]:
    logs = collect_logs(base, num_gpus)
    if logs:
        return parse_logs(logs), 'logs'
    json_results = collect_json_results(base, num_gpus)
    if json_results:
        return parse_json_results(json_results), 'detailed_results'
    raise FileNotFoundError(
        f'No gpu*.log or detailed_results.json found under: {base}'
    )


def _task_kind(task: str) -> str:
    if task in CHOICE_TASKS:
        return 'choice'
    if task in NUMERIC_TASKS:
        return 'numeric'
    return 'other'


def summarize(samples: list[dict]) -> dict:
    if not samples:
        raise SystemExit('No samples parsed from the provided sources.')

    by_task = defaultdict(list)
    for sample in samples:
        by_task[sample['task']].append(sample)

    v7_scores = [float(sample['v7']) for sample in samples if sample['v7'] is not None]
    summary = {
        'n_samples': len(samples),
        'overall': {
            'score': mean(sample['score'] for sample in samples),
            'v7': mean(v7_scores) if v7_scores else None,
        },
        'choice': {
            'n': sum(1 for sample in samples if sample['task'] in CHOICE_TASKS),
            'score': mean(sample['score'] for sample in samples if sample['task'] in CHOICE_TASKS)
            if any(sample['task'] in CHOICE_TASKS for sample in samples)
            else None,
        },
        'numeric': {
            'n': sum(1 for sample in samples if sample['task'] in NUMERIC_TASKS),
            'score': mean(sample['score'] for sample in samples if sample['task'] in NUMERIC_TASKS)
            if any(sample['task'] in NUMERIC_TASKS for sample in samples)
            else None,
        },
        'by_task': {},
    }
    if summary['overall']['v7'] is not None:
        summary['overall']['delta'] = summary['overall']['score'] - summary['overall']['v7']
    for task, task_samples in sorted(by_task.items()):
        task_v7 = [float(sample['v7']) for sample in task_samples if sample['v7'] is not None]
        summary['by_task'][task] = {
            'n': len(task_samples),
            'score': mean(sample['score'] for sample in task_samples),
            'v7': mean(task_v7) if task_v7 else None,
            'kind': _task_kind(task),
        }
    return summary


def print_summary(base: Path, source_kind: str, summary: dict) -> None:
    print(f'=== Parsed Results: {base} ({source_kind}) ===')
    overall = summary['overall']
    if overall.get('v7') is None:
        print(f"Total:  {summary['n_samples']} samples, score={overall['score']:.4f}")
    else:
        print(
            f"Total:  {summary['n_samples']} samples, score={overall['score']:.4f}, "
            f"v7={overall['v7']:.4f}, delta={overall['delta']:+.4f}"
        )
    choice = summary['choice']
    numeric = summary['numeric']
    print(f"Choice: {choice['n']} samples, score={choice['score']:.4f}" if choice['score'] is not None else f"Choice: {choice['n']} samples")
    print(f"Numeric:{numeric['n']} samples, score={numeric['score']:.4f}" if numeric['score'] is not None else f"Numeric:{numeric['n']} samples")
    print('\nBy task:')
    for task, task_summary in summary['by_task'].items():
        line = f"  {task:35s} {task_summary['kind']:>7s}  n={task_summary['n']:4d}  score={task_summary['score']:.4f}"
        if task_summary['v7'] is not None:
            line += f"  v7={task_summary['v7']:.4f}  delta={task_summary['score'] - task_summary['v7']:+.4f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description='Parse Mind_of_Thought multi-GPU logs or detailed_results.json into summary metrics.')
    parser.add_argument('--base', type=str, default=None, help='Directory containing gpu*.log, gpu*/detailed_results.json, or a detailed_results.json file. Defaults to the latest outputs run.')
    parser.add_argument('--num_gpus', type=int, default=None, help='Expected number of GPU workers. Defaults to auto-detect from available sources.')
    args = parser.parse_args()

    base = resolve_base(args.base)
    samples, source_kind = load_samples(base, args.num_gpus)
    summary = summarize(samples)
    print_summary(base, source_kind, summary)
    target_dir = base if base.is_dir() else base.parent
    (target_dir / 'parsed_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
