import numpy as np

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
