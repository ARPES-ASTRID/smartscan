from typing import Tuple, Callable
from scipy.ndimage import gaussian_filter
import numpy as np

def mean_std(sample):
    return sample.mean(), sample.std()

def compose(
        sample: np.ndarray, 
        func_a: Callable,
        func_b: Callable, 
        func_a_kwargs: dict = None, 
        func_b_kwargs: dict = None,
    ) -> Tuple[float]:
    """ Apply two functions to a sample and return the results as a tuple

    Args:
        sample: data to be processed
        func_a: _description_
        func_b: _description_
        func_a_kwargs: _description_. Defaults to None.
        func_b_kwargs: _description_. Defaults to None.

    Returns:
        _description_
    """
    if func_a_kwargs is None:
        func_a_kwargs = {}
    if func_b_kwargs is None:
        func_b_kwargs = {}
    return np.asarray([
        func_a(sample, **func_a_kwargs), 
        func_b(sample, **func_b_kwargs),
    ])

def sharpness(
        data:np.ndarray, 
        sigma:float = 2.,
        r: float = 1., 
        reduce: callable = np.mean
    ) -> float:
    """Compute the sharpness of an image

    Args:
        data (np.ndarray): 2D array
        sigma (float): sigma for the gaussian filter. Defaults to 2..
        r (float): ratio for the derivative. Defaults to 1..
        reduce (callable): function to reduce the sharpness map. Defaults to np.mean

    Returns:
        float: sharpness
    """
    dx,dy = np.gradient(gaussian_filter(data,sigma=sigma))
    ddx,_ = np.gradient(dx)
    _,ddy = np.gradient(dy)
    ddxddy = np.sqrt(ddx**2 + r*ddy**2)
    return reduce(ddxddy)

def select_roi(
        data: np.ndarray,
        x_lim: Tuple[int,int] = None,
        y_lim: Tuple[int,int] = None,
    ) -> np.ndarray:
    """Select a region of interest from a 2D array

    Args:
        data (np.ndarray): 2D array
        x_lim (Tuple[int,int]): x limits of the ROI. Defaults to None.
        y_lim (Tuple[int,int]): y limits of the ROI. Defaults to None.

    Returns:
        np.ndarray: ROI
    """
    if x_lim is None:
        x_lim = (0,data.shape[0])
    if y_lim is None:
        y_lim = (0,data.shape[1])
    return data[x_lim[0]:x_lim[1],y_lim[0]:y_lim[1]]

