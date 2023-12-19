import cv2
import numpy as np

def edge_amplify(image, thickness, lt, ut):
    # Adjust parameters for Canny edge detection
    edgecolor = [0, 0, 0]
    edges = cv2.Canny(image, lt, ut)

    # Edge dilation for a thicker edge
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Apply the edge on the original image
    image[dilated > 0] = edgecolor

    return image