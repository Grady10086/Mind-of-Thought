#!/usr/bin/env python3
import numpy as np
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class MindMapEntity3D:
    entity_id: str = ''
    label: str = ''
    count: int = 0
    avg_confidence: float = 0.0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    position_3d: Optional[np.ndarray] = None
    size_3d: Optional[np.ndarray] = None
    depth_median: float = 0.0

# 创建测试心智地图
mind_map = {
    'table': MindMapEntity3D(label='table', count=2, size_3d=np.array([2.4, 1.5, 1.0]), 
                              position_3d=np.array([0, 0, 2.5]), depth_median=2.5),
    'chair': MindMapEntity3D(label='chair', count=3, size_3d=np.array([0.6, 1.2, 0.5]),
                              position_3d=np.array([1, 0, 2.0]), depth_median=2.0),
}

def v3_answer_counting(mind_map, question):
    match = re.search(r'How many (\w+)\(s\)', question)
    if not match:
        match = re.search(r'How many (\w+)', question)
    if not match:
        return '0'
    target = match.group(1).lower()
    for label, entity in mind_map.items():
        if target in label.lower() or label.lower() in target:
            return str(entity.count)
    return '0'

questions = [
    'How many table(s) are in this room?',
    'How many chair(s) are in this room?',
    'How many Table(s) are in this room?',  # 大写
    'How many CHAIR(s) are in this room?',
]

for q in questions:
    result = v3_answer_counting(mind_map, q)
    print(f'Q: {q}')
    print(f'Result: {result}\n')
