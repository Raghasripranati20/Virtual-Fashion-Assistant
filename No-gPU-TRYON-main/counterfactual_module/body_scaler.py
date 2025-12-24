import cv2

def scale_waist(mask, scale=1.1):
    h, w = mask.shape
    scaled = cv2.resize(mask, None, fx=scale, fy=1.0)
    cropped = scaled[:, :w]
    return cropped
