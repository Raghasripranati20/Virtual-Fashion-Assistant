import numpy as np

def sleeve_arm_alignment(pose_keypoints):
    # Example: shoulder → elbow → wrist
    shoulder = np.array(pose_keypoints["right_shoulder"])
    elbow = np.array(pose_keypoints["right_elbow"])
    wrist = np.array(pose_keypoints["right_wrist"])

    arm_vec = wrist - shoulder
    elbow_vec = elbow - shoulder

    cosine = np.dot(arm_vec, elbow_vec) / (
        np.linalg.norm(arm_vec) * np.linalg.norm(elbow_vec) + 1e-6
    )

    return 1 - cosine  # higher = more violation
