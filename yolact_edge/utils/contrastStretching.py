import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to map each intensity level to output intensity level.
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


def contrast_stretching(image_array):
    # Define parameters.
    r1 = 30
    s1 = 0
    r2 = 225
    s2 = 255

    # Vectorize the function to apply it to each value in the Numpy array.
    pixelVal_vec = np.vectorize(pixelVal)

    # Apply contrast stretching.
    contrast_stretched = pixelVal_vec(image_array, r1, s1, r2, s2)
    return contrast_stretched.astype(np.uint8)