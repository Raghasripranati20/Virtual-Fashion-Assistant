def generate_counterfactuals(original_pose):
    poses = {}
    poses["original"] = original_pose
    poses["wide_shoulders"] = widen_shoulders(original_pose.copy(), 1.15)
    poses["narrow_shoulders"] = widen_shoulders(original_pose.copy(), 0.9)
    return poses
