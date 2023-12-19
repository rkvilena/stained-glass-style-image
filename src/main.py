import cv2
from matplotlib import pyplot as plt
import numpy as np
from cutout import cutout_filter
from colorburn import color_burn
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize
from edge import edge_amplify

EDGE_LOWER_THRESHOLD = 40
EDGE_UPPER_THRESHOLD = 300
EDGE_THICKNESS = 2
COLOR_BURN_INTENSITY = 1.4

if __name__ == '__main__':
    texture = imread("texture.jpg")

    # Get cutout filtered image
    # Using TACIXAT's code
    canvas, original = cutout_filter()

    # Amplify the edge for stained-glass effect
    # Tin-like effect (timah in indonesian)
    cp = canvas.copy()
    artresult = edge_amplify(
        cp, 
        EDGE_THICKNESS,
        EDGE_LOWER_THRESHOLD,
        EDGE_UPPER_THRESHOLD
    )

    # Adjust texture size to canvas
    # For color burn size match
    texture_res = resize(texture, artresult.shape, mode='constant', anti_aliasing=False)

    # Convert canvas result to RGB
    # cv2 treated image in BGR in default
    art_rgb = cv2.cvtColor(artresult, cv2.COLOR_BGR2RGB)
    
    # Convert to floating point value
    # For color burn division calculation
    art_float = img_as_float(art_rgb)
    
    # Color burn effect (make it looks like glass texture)
    # Heavily dependant on its texture
    result = color_burn(art_float, texture_res, COLOR_BURN_INTENSITY)
    
    result_uint8 = (result * 255).astype(np.uint8)
    cv2.imwrite("output2.png", cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR))
    
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original RGB Image')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title('Cutout Filter - result')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(artresult, cv2.COLOR_BGR2RGB))
    plt.title('Edge Amplification - result')

    plt.subplot(2, 2, 4)
    plt.imshow(result_uint8)
    plt.title('Final Result (after color burn)')

    plt.show()
