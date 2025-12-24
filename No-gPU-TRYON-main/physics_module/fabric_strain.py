import numpy as np
import cv2

def compute_fabric_strain(original_mask, warped_mask):
    orig_area = np.sum(original_mask > 0)
    warped_area = np.sum(warped_mask > 0)

    if orig_area == 0:
        return 0.0

    strain = warped_area / orig_area
    return abs(strain - 1.0)  # deviation from natural fabric size
