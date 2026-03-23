#!/usr/bin/env python3
"""
快速验证ablation脚本的核心组件在spawn子进程中能正常工作
"""
import os
import sys
import multiprocessing as mp
import torch
import numpy as np

os.environ['HF_HOME'] = '/home/tione/notebook/tianjungu/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '1'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_worker(gpu_id, result_queue):
    """在子进程中测试所有组件"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    results = {}
    
    # 测试1: 深度估计模型
    try:
        from core.perception import DepthEstimator
        estimator = DepthEstimator(
            model_name="depth-anything/DA3-Large",
            device='cuda',
            half_precision=True,
        )
        
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth, conf = estimator.infer_single(dummy)
        results['depth'] = f"OK: shape={depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]"
        del estimator
    except Exception as e:
        results['depth'] = f"FAIL: {e}"
    
    # 测试2: GroundingDINO
    try:
        from core.semantic_labeler import GroundingDINOLabeler
        labeler = GroundingDINOLabeler(
            model_id="IDEA-Research/grounding-dino-base",
            device='cuda',
            box_threshold=0.25,
            text_threshold=0.25,
        )
        labeler.load_model()
        
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = labeler.detect(dummy, "chair . table . door .")
        results['dino'] = f"OK: detected {len(detections)} objects"
        del labeler
    except Exception as e:
        results['dino'] = f"FAIL: {e}"
    
    # 测试3: VL模型
    try:
        vl_model_path = '/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct'
        
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            processor = AutoProcessor.from_pretrained(vl_model_path, trust_remote_code=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                vl_model_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True
            )
            results['vl'] = f"OK: Qwen3VL loaded"
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            processor = AutoProcessor.from_pretrained(vl_model_path, trust_remote_code=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vl_model_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True
            )
            results['vl'] = f"OK: Qwen2.5VL loaded (fallback)"
        
        del model, processor
    except Exception as e:
        results['vl'] = f"FAIL: {e}"
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    result_queue.put(results)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print("=" * 60)
    print("验证ablation脚本组件在spawn子进程中的兼容性")
    print("=" * 60)
    
    result_queue = mp.Queue()
    p = mp.Process(target=test_worker, args=(0, result_queue))
    p.start()
    p.join(timeout=300)  # 5分钟超时
    
    if p.is_alive():
        p.terminate()
        print("TIMEOUT: 子进程超时")
    else:
        results = result_queue.get()
        all_ok = True
        for component, status in results.items():
            ok = status.startswith("OK")
            mark = "✓" if ok else "✗"
            print(f"  [{mark}] {component}: {status}")
            if not ok:
                all_ok = False
        
        print()
        if all_ok:
            print("全部通过! 可以启动完整实验。")
        else:
            print("存在失败项，需要修复。")
