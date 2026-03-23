import json
from collections import Counter
r = json.load(open('outputs/grid64_test_20260221_023512/detailed_results.json'))
# Parse failures
reasons = Counter()
for x in r:
    reason = x.get('grid_reasoning','')
    if 'cannot parse' in reason or 'not found' in reason or 'default' in reason or 'insufficient' in reason:
        reasons[reason[:60]] += 1
print("=== Parse Failures ===")
for k,v in reasons.most_common():
    print(f'  {v}x {k}')

# Direction samples
print("\n=== Direction Samples ===")
for x in r:
    if 'direction' in x['question_type']:
        print(f"\nQ: {x['question'][:150]}")
        print(f"  Reason: {x['grid_reasoning']}")
        print(f"  GT={x.get('ground_truth','')} Pred={x.get('grid_prediction','')} Score={x['grid_score']}")

# rel_distance samples
print("\n=== Rel Distance Samples ===")
for x in r:
    if x['question_type'] == 'object_rel_distance':
        print(f"\nQ: {x['question'][:150]}")
        print(f"  Reason: {x['grid_reasoning']}")
        print(f"  GT={x.get('ground_truth','')} Pred={x.get('grid_prediction','')} Score={x['grid_score']}")

# abs_distance
print("\n=== Abs Distance Samples ===")
for x in r:
    if x['question_type'] == 'object_abs_distance':
        print(f"\nQ: {x['question'][:150]}")
        print(f"  Reason: {x['grid_reasoning']}")
        print(f"  GT={x.get('ground_truth','')} Pred={x.get('grid_prediction','')} Score={x['grid_score']}")

# room size
print("\n=== Room Size Samples ===")
for x in r:
    if x['question_type'] == 'room_size_estimation':
        print(f"\nQ: {x['question'][:150]}")
        print(f"  Reason: {x['grid_reasoning']}")
        print(f"  GT={x.get('ground_truth','')} Pred={x.get('grid_prediction','')} Score={x['grid_score']}")
