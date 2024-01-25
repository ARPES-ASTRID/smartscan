from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import laplace, sobel
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy


def mean(
        x : np.ndarray, 
        roi:Sequence[Sequence[int]] = None
    ) -> float:
    """Compute the mean of an array"""
    if roi is not None:
        roi = np.array(roi)
        x = x[roi[0,0]:roi[0,1], roi[1,0]:roi[1,1]]
    return np.mean(x)


def std(
        x : np.ndarray, 
        roi:Sequence[Sequence[int]] = None
    ) -> float:
    """Compute the standard deviation of an array"""
    if roi is not None:
        roi = np.array(roi)
        x = x[roi[0,0]:roi[0,1], roi[1,0]:roi[1,1]]
    return np.std(x)


def laplace_filter(
        x: np.ndarray, 
        sigma: float = 10, 
        norm: bool = True,
        reduction: Callable = np.mean,
        roi:Sequence[Sequence[int]] = None,
    ) -> float:
    """ Laplacian filter

    Args:
        x (np.ndarray): input array
        sigma (float, optional): sigma of the gaussian filter. Defaults to 10.
        norm (bool, optional): normalize the output. Defaults to False.
    
    Returns:
        float: mean absolute value of the laplacian
    """
    if roi is not None:
        roi = np.array(roi)
        x = x[roi[0,0]:roi[0,1], roi[1,0]:roi[1,1]]
    filt = gaussian_filter(x.astype(np.float64), sigma=sigma)
    if norm:
        filt /= np.mean(filt)
    if reduction is not None:
        return reduction(np.abs(laplace(filt)))
    else:
        return laplace(filt)
    

def contrast_noise_ratio(
        x:np.ndarray, 
        signal_roi: Sequence[Sequence[int]],
        bg_roi: Sequence[Sequence[int]],
 
    ) -> float:
    """Compute the contrast to noise ratio of an image

    Args:
        x (np.ndarray): input array
        sig_roi (Sequence[Sequence[int]]): signal region of interest
        bg_roi (Sequence[Sequence[int]]): background region of interest
    
    Returns:
        float: contrast to noise ratio
    """
    signal_roi = np.array(signal_roi)
    bg_roi = np.array(bg_roi)
    x = np.array(x).squeeze()
    assert x.ndim == 2, "x must be 2D"
    signal = x[signal_roi[0,0]:signal_roi[0,1], signal_roi[1,0]:signal_roi[1,1]]
    background = x[bg_roi[0,0]:bg_roi[0,1], bg_roi[1,0]:bg_roi[1,1]]
    mu_signal = np.mean(signal)
    mu_background = np.mean(background)
    sigma_noise = np.std(background)
    # Calculate CNR
    return np.abs(mu_signal - mu_background) / sigma_noise


def edge_density(
        x: np.ndarray,
        roi: Sequence[Sequence[int]] = None,
    ) -> float:
    """ Calculates the edge density of an image

    Args:
        x (np.ndarray): input array
      
    Returns:
        float: edge density
    """
    if roi is not None:
        roi = np.array(roi)
        x = x[roi[0,0]:roi[0,1], roi[1,0]:roi[1,1]]
    edges = sobel(x)
    return np.sum(edges) / edges.size


def image_entropy(
        x: np.ndarray, 
        roi: Sequence[Sequence[int]] = None,
    ) -> float:
    """ Calculates the (negative) entropy of an image

    Args:
        x (np.ndarray): input array
        roi (Sequence[Sequence[int]], optional): region of interest. Defaults to None.

    Returns:
        float: negative entropy
    """
    if roi is not None:
        roi = np.array(roi)
        x = x[roi[0,0]:roi[0,1], roi[1,0]:roi[1,1]]
    hist, _ = np.histogram(x, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    return entropy(hist)

def curvature(
        arpesmap: np.ndarray, 
        bw: float = 5, 
        c1: float = 0.001,
        c2: float = 0.001,
        w: float = 1,
        roi: Sequence[Sequence[int]] = None,
        reduction: Callable = np.mean,
    ) -> float:
    """ Calculations curvature array

    Calculates the curvature of an array using a 2D gaussian kernel
    defined in https://doi.org/10.1063/1.3585113

    Args:
        x (np.ndarray): input array
        bw (float, optional): bandwidth of the smoothing kernel. Defaults to 5.
        c1 (float, optional): curvature parameter. Defaults to 0.001.
        c2 (float, optional): curvature parameter. Defaults to 0.001.
        w (float, optional): aspect ratio. Defaults to 1.
        roi (Sequence[Sequence[int]], optional): region of interest. Defaults to None.
    
    Returns:
        cv2d(np.array): curvature
    """
    from astropy.convolution import convolve, Box2DKernel

    # https://docs.astropy.org/en/latest/api/astropy.convolution.Box2DKernel.html
    # https://docs.astropy.org/en/latest/api/astropy.convolution.convolve.html

    if roi is not None:
        roi = np.array(roi)
        arpesmap = arpesmap[roi[0,0]:roi[0,1], roi[1,0]:roi[1,1]]
    
    x = np.arange(arpesmap.shape[0]) 
    y = np.arange(arpesmap.shape[1])
    
    data_smth = convolve(arpesmap, Box2DKernel(bw), boundary='extend')

    dx = np.gradient(data_smth, axis=0)
    dy = np.gradient(data_smth, axis=1) * w
    d2x = np.gradient(np.gradient(data_smth, x, axis=0), x, axis=0)
    d2y = np.gradient(np.gradient(data_smth, y, axis=1), y, axis=1) * w * w
    dxdy = np.gradient(np.gradient(data_smth, y, axis=1), x, axis=0) * w

    # 2D curvature 
    cv2d = ((1 + c1*dx**2)*c2*d2y - 2*c1*c2*dx*dy*dxdy +
            (1 + c2*dy**2)*c1*d2x) / (1 + c1*dx**2 + c2*dy**2)**1.5
    if reduction is not None:
        return reduction(np.abs(cv2d))
    else:
        return cv2d

