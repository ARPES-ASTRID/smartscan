from typing import Callable

import numpy as np
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter

def mean(x : np.ndarray) -> float:
    """Compute the mean of an array"""
    return np.mean(x)

def laplace_filter(
        x: np.ndarray, 
        sigma: float = 10, 
        norm: bool = True,
        reduction: Callable = np.mean
    ) -> float:
    """ Laplacian filter

    Args:
        x (np.ndarray): input array
        sigma (float, optional): sigma of the gaussian filter. Defaults to 10.
        norm (bool, optional): normalize the output. Defaults to False.
    
    Returns:
        float: mean absolute value of the laplacian
    """
    filt = gaussian_filter(x.astype(np.float64), sigma=sigma)
    if norm:
        filt /= np.mean(filt)
    if reduction is not None:
        return reduction(np.abs(laplace(filt)))
    else:
        return np.abs(laplace(filt))