#!/usr/bin/env python3
import json

# 加载 V3 结果
f = '/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/mindmap_depth_test_20260126_080121/detailed_results.json'

# 尝试读取整个文件
with open(f, 'r') as fp:
    content = fp.read()

# 找到所有完整的 JSON 对象
import re

# 使用更复杂的模式匹配完整的嵌套 JSON
results = []
in_object = False
brace_count = 0
current = ""

for char in content:
    if char == '{':
        brace_count += 1
        in_object = True
    if in_object:
        current += char
    if char == '}':
        brace_count -= 1
        if brace_count == 0 and in_object:
            try:
                obj = json.loads(current)
                results.append(obj)
            except:
                pass
            current = ""
            in_object = False

print(f"解析到 {len(results)} 个结果")

# 统计 object_counting
counting = [r for r in results if r.get('question_type') == 'object_counting']
print(f"\nobject_counting: {len(counting)} 样本")
for c in counting[:5]:
    print(f"  Pred: {c.get('prediction')}, GT: {c.get('ground_truth')}, Score: {c.get('score'):.3f}")

# 统计 object_size_estimation
size = [r for r in results if r.get('question_type') == 'object_size_estimation']
print(f"\nobject_size_estimation: {len(size)} 样本")
for s in size[:5]:
    print(f"  Pred: {s.get('prediction')}, GT: {s.get('ground_truth')}, Score: {s.get('score'):.3f}")
