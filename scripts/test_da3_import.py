#!/usr/bin/env python3
"""测试DA3在multiprocessing中的导入"""
import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3')
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Depth-Anything-3/src')

import os
import multiprocessing as mp

def test_worker(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        from depth_anything_3.api import DepthAnything3
        print(f"[GPU {gpu_id}] ✓ DA3 导入成功")
        return True
    except Exception as e:
        print(f"[GPU {gpu_id}] ❌ DA3 导入失败: {e}")
        return False

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print("测试8个GPU的DA3导入...")
    with mp.Pool(8) as pool:
        results = pool.map(test_worker, range(8))
    
    success_count = sum(results)
    print(f"\n结果: {success_count}/8 成功")
