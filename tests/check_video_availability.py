#!/usr/bin/env python3
"""检查 VSIBench 视频可用性"""
import os
from datasets import load_dataset

dataset = load_dataset('nyu-visionx/VSI-Bench', split='test', 
                       cache_dir='/home/tione/notebook/tianjungu/hf_cache/vsibench')

video_dirs = {
    'ARKitScenes': '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes',
    'ScanNet': '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet',
    'ScanNetPP': '/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp',
}

# 统计
found = {'ARKitScenes': 0, 'ScanNet': 0, 'HM3D_in_ScanNetPP': 0, 'missing': 0}
counting_found = {'ARKitScenes': 0, 'ScanNet': 0, 'HM3D_in_ScanNetPP': 0, 'missing': 0}
missing_scenes = []

for item in dataset:
    scene = item['scene_name']
    is_counting = item['question_type'] == 'object_counting'
    
    ark_path = os.path.join(video_dirs['ARKitScenes'], f'{scene}.mp4')
    scan_path = os.path.join(video_dirs['ScanNet'], f'{scene}.mp4')
    scanpp_path = os.path.join(video_dirs['ScanNetPP'], f'{scene}.mp4')
    
    if os.path.exists(ark_path):
        found['ARKitScenes'] += 1
        if is_counting: counting_found['ARKitScenes'] += 1
    elif os.path.exists(scan_path):
        found['ScanNet'] += 1
        if is_counting: counting_found['ScanNet'] += 1
    elif os.path.exists(scanpp_path):
        found['HM3D_in_ScanNetPP'] += 1
        if is_counting: counting_found['HM3D_in_ScanNetPP'] += 1
    else:
        found['missing'] += 1
        if is_counting: 
            counting_found['missing'] += 1
            if scene not in missing_scenes:
                missing_scenes.append(scene)

print('所有样本视频可用性:')
for k, v in found.items():
    print(f'  {k}: {v}')

print('\nCounting样本视频可用性:')
for k, v in counting_found.items():
    print(f'  {k}: {v}')

print(f'\nCounting总样本: {sum(counting_found.values())}')
print(f'Counting可用样本: {sum(counting_found.values()) - counting_found["missing"]}')
print(f'\n缺失的场景数: {len(missing_scenes)}')
if missing_scenes[:5]:
    print(f'缺失场景示例: {missing_scenes[:5]}')
