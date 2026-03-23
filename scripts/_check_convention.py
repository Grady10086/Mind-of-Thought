"""Check DA3 extrinsics convention"""
import json
r = json.load(open('outputs/grid64_test_20260221_024655/detailed_results.json'))

# Print some entity sizes to understand scale
for x in r:
    if x['question_type'] == 'room_size_estimation':
        print(f"Scene: {x['scene_name']}, Pred area: {x['grid_prediction']}m², Reason: {x['grid_reasoning']}")

# The ranges being 50+m means DA3 absolute depth is huge
# Could be the w2c/c2w convention is wrong
print()
print("If ranges are 50+m, the world coordinate scale is wrong.")
print("V1 voxel_mental_map.py uses: world = R @ cam_point + t")
print("perception_da3_full.py uses: world = R^T @ cam_point - R^T @ t")
print("These are DIFFERENT if extrinsics is c2w vs w2c")
print()
print("Need to check: if extrinsics = c2w, then V1 is correct")
print("If extrinsics = w2c, then perception_da3_full is correct")
print()
# Hint: DA3 docs and V1 both use R@cam+t
# perception_da3_full uses R^T@cam - R^T@t and also R^T@t for camera center
# If extrinsics is c2w: world = R@cam + t (V1), camera_center = t (not -R^T@t)
# If extrinsics is w2c: cam = R@world + t -> world = R^T@(cam-t) (da3_full)
print("The massive scales suggest the SIGN is wrong in the transform")
