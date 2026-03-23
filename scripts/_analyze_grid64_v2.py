import json
r = json.load(open('outputs/grid64_test_20260221_024655/detailed_results.json'))

# Room size analysis - why 0.1?
print("=== Room Size Estimation ===")
for x in r:
    if x['question_type'] == 'room_size_estimation':
        print(f"  GT={x['ground_truth']:>6s} Pred={x['grid_prediction']:>6s} Score={x['grid_score']:.3f} | {x['grid_reasoning']}")

# Size estimation - why 0.34?
print("\n=== Size Estimation ===")
for x in r:
    if x['question_type'] == 'object_size_estimation':
        print(f"  GT={x['ground_truth']:>6s} Pred={x['grid_prediction']:>6s} Score={x['grid_score']:.3f} | {x['grid_reasoning'][:80]}")

# Direction detailed
print("\n=== Direction Results ===")
for x in r:
    if 'direction' in x['question_type']:
        print(f"  [{x['question_type'][-6:]}] GT={x['ground_truth']} Pred={x['grid_prediction']} Score={x['grid_score']:.0f} | {x['grid_reasoning'][:80]}")

# Counting
print("\n=== Counting ===")
for x in r:
    if x['question_type'] == 'object_counting':
        print(f"  GT={x['ground_truth']:>3s} Pred={x['grid_prediction']:>3s} Score={x['grid_score']:.3f} | {x['grid_reasoning'][:60]}")

# Rel distance
print("\n=== Rel Distance ===")
for x in r:
    if x['question_type'] == 'object_rel_distance':
        print(f"  GT={x['ground_truth']} Pred={x['grid_prediction']} Score={x['grid_score']:.0f} | {x['grid_reasoning'][:80]}")
