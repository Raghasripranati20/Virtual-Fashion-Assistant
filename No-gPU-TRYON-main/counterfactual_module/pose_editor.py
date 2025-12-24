def widen_shoulders(pose_keypoints, scale=1.1):
    left = pose_keypoints["left_shoulder"]
    right = pose_keypoints["right_shoulder"]

    center_x = (left[0] + right[0]) / 2

    left[0] = center_x + (left[0] - center_x) * scale
    right[0] = center_x + (right[0] - center_x) * scale

    pose_keypoints["left_shoulder"] = left
    pose_keypoints["right_shoulder"] = right

    return pose_keypoints
