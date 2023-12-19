import numpy as np

def color_burn(main, texture, intensity):
    # Color burn formula
    adjusted = np.clip(texture * intensity, 0, 1)
    result = 1 - (1 - main) / np.maximum(adjusted, 1e-10)
    return np.clip(result, 0, 1)