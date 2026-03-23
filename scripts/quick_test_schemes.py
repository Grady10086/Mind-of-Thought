#!/usr/bin/env python3
"""
快速验证三个V10优化方案的核心功能
"""

import sys
sys.path.insert(0, '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube')

print("="*60)
print("Quick Verification of V10 Schemes")
print("="*60)

# 验证 Scheme 1: Route Optimization
print("\n[Scheme 1: Route Optimization]")
try:
    import scripts.grid64_agentic_pipeline_v10_route as scheme1
    # 检查是否使用了优化后的路由函数
    if hasattr(scheme1, 'grid_answer_route_v2'):
        print("  ✓ grid_answer_route_v2 function exists")
    else:
        print("  ✗ grid_answer_route_v2 not found")
    
    # 检查grid_answer_route是否指向v2
    import inspect
    if 'v2' in str(scheme1.grid_answer_route.__code__.co_filename):
        print("  ✓ grid_answer_route is using v2 implementation")
    else:
        print("  ? grid_answer_route implementation check needed")
    print("  ✓ Scheme 1 loaded successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 验证 Scheme 2: Scale Calibration
print("\n[Scheme 2: Scale Calibration]")
try:
    import scripts.grid64_agentic_pipeline_v10_scale as scheme2
    # 检查是否包含大距离检测逻辑
    import inspect
    source = inspect.getsource(scheme2._coder_result_confidence)
    if 'low_distance_large' in source:
        print("  ✓ large distance detection (low_distance_large) added")
    else:
        print("  ✗ large distance detection not found")
    
    if 'computed_dist > 5.0' in source:
        print("  ✓ 5m threshold check added")
    else:
        print("  ✗ 5m threshold not found")
    print("  ✓ Scheme 2 loaded successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 验证 Scheme 3: Route Verify Tool
print("\n[Scheme 3: Route Verify Tool]")
try:
    import scripts.grid64_agentic_pipeline_v10_verify as scheme3
    # 检查route_verify_tool是否存在
    if hasattr(scheme3, 'route_verify_tool'):
        print("  ✓ route_verify_tool function exists")
    else:
        print("  ✗ route_verify_tool not found")
    
    # 检查manager_code_agent_loop是否包含route_verify调用
    import inspect
    source = inspect.getsource(scheme3.manager_code_agent_loop)
    if 'route_verify_tool' in source:
        print("  ✓ route_verify_tool is called in manager loop")
    else:
        print("  ✗ route_verify_tool not called in manager loop")
    print("  ✓ Scheme 3 loaded successfully")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("Verification Complete!")
print("="*60)
print("\nTo run full comparison test:")
print("  bash scripts/run_schemes_sequential.sh")
print("\nTo run a quick 1-sample test on each scheme:")
print("  CUDA_VISIBLE_DEVICES=0 python scripts/grid64_agentic_pipeline_v10_route.py --n_per_type 1")
print("  CUDA_VISIBLE_DEVICES=1 python scripts/grid64_agentic_pipeline_v10_scale.py --n_per_type 1")
print("  CUDA_VISIBLE_DEVICES=2 python scripts/grid64_agentic_pipeline_v10_verify.py --n_per_type 1")
