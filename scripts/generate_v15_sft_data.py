#!/usr/bin/env python3
"""
V15 SFT 训练数据生成

核心策略: 方向C — VL-driven Evolution  
让模型在 V14 pipeline 下做得更好:
  - Choice tasks: 学会更准确的空间判断 (V14 bypass 41.2%, oracle 78.1%)
  - Numerical tasks: 维持已有的 92.2% 高准确率

训练数据来源: VSI-590K 训练集 (118K 样本)
  - relative_direction_object (38K) → 选择题
  - relative_distance_object (16K) → 选择题
  - relative_size_object (19K) → 选择题
  - relative_count (0.8K) → 选择题
  - absolute_* → 数值题

数据格式: 100% 统一的官方 prompt 格式 (之前实验证明混合格式有害)
  - Choice: "These are frames of a video.\n{question}\nOptions:\n{options}\n
            Answer with the option's letter from the given choices directly."
  - Numerical: "These are frames of a video.\n{question}\n
               Please answer the question using a single word or phrase."
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict


# 选择题类型 (答案是 ABCD 字母)
LETTER_ANSWER_TYPES = {
    'relative_count',
    'relative_direction_object',
    'relative_distance_object', 
    'relative_size_object',
}

# 数值题类型 (答案是数字)
NUMBER_ANSWER_TYPES = {
    'absolute_count',
    'absolute_direction_object',
    'absolute_distance_object',
    'absolute_size_object',
    'absolute_size_room',
}


def is_choice_task(qtype: str) -> bool:
    return qtype in LETTER_ANSWER_TYPES


def extract_question_and_options(human_msg: str):
    """从原始 human message 中提取问题和选项
    
    Returns: (question, options_text) or (question, None) for numerical
    """
    # 去掉 <image> tag
    msg = human_msg.replace('<image>\n', '').replace('<image>', '')
    # 去掉 "These are frames of a video.\n"
    msg = msg.replace('These are frames of a video.\n', '')
    
    # 对于选择题: 提取 question + options
    # 格式通常是: question\nA. xxx\nB. xxx\nC. xxx\nD. xxx\nAnswer with...
    # 或: question\nOptions:\nA. xxx\n...
    
    # 尝试找选项
    lines = msg.split('\n')
    opt_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('A.') or stripped.startswith('A ') or stripped == 'Options:':
            opt_start = i
            break
    
    if opt_start >= 0:
        question_lines = lines[:opt_start]
        opt_lines = lines[opt_start:]
        
        question = '\n'.join(question_lines).strip()
        # 清理 question 末尾的 "Please answer..." 等
        for suffix in ['Please answer the question using a single word or phrase.',
                       'Answer with the option\'s letter from the given choices directly.']:
            question = question.replace(suffix, '').strip()
        
        # 提取选项 (去掉 "Options:" 头和 "Answer with..." 尾)
        opts = []
        for line in opt_lines:
            stripped = line.strip()
            if stripped == 'Options:':
                continue
            if stripped.startswith('Answer with'):
                continue
            if stripped.startswith('Please answer'):
                continue
            if stripped:
                opts.append(stripped)
        
        options_text = '\n'.join(opts) if opts else None
        return question, options_text
    else:
        # 数值题: 整段都是 question
        question = msg.strip()
        for suffix in ['Please answer the question using a single word or phrase.',
                       'Answer with the option\'s letter from the given choices directly.']:
            question = question.replace(suffix, '').strip()
        return question, None


def build_choice_prompt(question: str, options_text: str) -> str:
    """V14 bypass 使用的官方 VLMEvalKit MCA prompt"""
    return f"""These are frames of a video.
{question}
Options:
{options_text}
Answer with the option's letter from the given choices directly."""


def build_numerical_prompt(question: str) -> str:
    """数值题 prompt — 保持原始格式"""
    return f"""These are frames of a video.
{question}
Please answer the question using a single word or phrase."""


def process_dataset(train_path: str, output_path: str, 
                    max_choice: int = 2000, max_numerical: int = 2000,
                    seed: int = 42):
    """处理训练集，生成 V15 SFT 数据"""
    random.seed(seed)
    
    print(f"读取训练集: {train_path}")
    all_samples = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))
    print(f"总样本数: {len(all_samples)}")
    
    # 按 question_type 分组
    by_type = defaultdict(list)
    for s in all_samples:
        by_type[s.get('question_type', 'unknown')].append(s)
    
    print(f"\n原始数据分布:")
    total_choice = total_numerical = 0
    for qt in sorted(by_type.keys()):
        n = len(by_type[qt])
        is_c = is_choice_task(qt)
        if is_c: total_choice += n
        else: total_numerical += n
        print(f"  {qt:<40} n={n:>6}  {'CHOICE' if is_c else 'NUMER'}")
    print(f"  Total: choice={total_choice}, numerical={total_numerical}")
    
    # 生成训练数据
    output_samples = []
    stats = Counter()
    
    for qt, samples in by_type.items():
        is_c = is_choice_task(qt)
        max_n = max_choice if is_c else max_numerical
        
        # 随机采样
        if len(samples) > max_n:
            selected = random.sample(samples, max_n)
        else:
            selected = samples
        
        for s in selected:
            convs = s.get('conversations', [])
            if len(convs) < 2:
                stats['skip_bad_conv'] += 1
                continue
            
            video = s.get('video', '')
            gt_answer = convs[1]['value'].strip()
            human_msg = convs[0]['value']
            
            question, options_text = extract_question_and_options(human_msg)
            
            if is_c:
                if not options_text:
                    stats['skip_no_options'] += 1
                    continue
                
                # 验证答案是字母
                if gt_answer not in 'ABCD':
                    stats['skip_bad_answer'] += 1
                    continue
                
                prompt = build_choice_prompt(question, options_text)
                response = gt_answer
                stats['choice'] += 1
            else:
                prompt = build_numerical_prompt(question)
                response = gt_answer
                stats['numerical'] += 1
            
            output_sample = {
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt}"},
                    {"from": "gpt", "value": response}
                ],
                "video": video,
                "question_type": qt,
            }
            output_samples.append(output_sample)
    
    # 打乱顺序
    random.shuffle(output_samples)
    
    # 写出
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for s in output_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    print(f"\n=== 生成结果 ===")
    print(f"总样本: {len(output_samples)}")
    print(f"  choice:    {stats['choice']}")
    print(f"  numerical: {stats['numerical']}")
    print(f"  skip_no_options: {stats['skip_no_options']}")
    print(f"  skip_bad_answer: {stats['skip_bad_answer']}")
    print(f"  skip_bad_conv:   {stats['skip_bad_conv']}")
    print(f"\n输出: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    # 验证
    print(f"\n=== 样本验证 ===")
    choice_samples = [s for s in output_samples if is_choice_task(s['question_type'])]
    numer_samples = [s for s in output_samples if not is_choice_task(s['question_type'])]
    print(f"Choice samples: {len(choice_samples)}")
    print(f"Numerical samples: {len(numer_samples)}")
    
    if choice_samples:
        s = choice_samples[0]
        print(f"\n--- Choice 样本 ({s['question_type']}) ---")
        print(f"Human: {s['conversations'][0]['value'][:400]}")
        print(f"GPT: {s['conversations'][1]['value']}")
    
    if numer_samples:
        s = numer_samples[0]
        print(f"\n--- Numerical 样本 ({s['question_type']}) ---")
        print(f"Human: {s['conversations'][0]['value'][:400]}")
        print(f"GPT: {s['conversations'][1]['value']}")
    
    return output_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate V15 SFT training data')
    parser.add_argument('--train-path', type=str,
                        default='/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/data/vsi590k_subset/train.jsonl')
    parser.add_argument('--output', type=str,
                        default='/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/v15_sft_training.jsonl')
    parser.add_argument('--max-choice', type=int, default=2000,
                        help='Max samples per choice question type')
    parser.add_argument('--max-numerical', type=int, default=2000,
                        help='Max samples per numerical question type')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    process_dataset(args.train_path, args.output, args.max_choice, args.max_numerical, args.seed)
