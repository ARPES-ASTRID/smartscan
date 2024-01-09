from typing import Sequence
import numpy as np

def reshape(
        x: np.ndarray, 
        shape: tuple
    ) -> np.ndarray:
    """Reshape an array to a given shape

    Args:
        x (np.ndarray): input array
        shape (tuple): desired shape
    
    Returns:
        np.ndarray: reshaped array
    """
    return np.array(x).reshape(shape)

def roi(
        x: np.ndarray, 
        x_lim: Sequence[int], 
        y_lim: Sequence[int]
    ) -> np.ndarray:
    """Return the region of interest of a 2D array
    Args:
        x (np.ndarray): input array
        x_lim (Sequence[int]): x limits
        y_lim (Sequence[int]): y limits
    Returns:
        np.ndarray: region of interest
    """
    return x[x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]]