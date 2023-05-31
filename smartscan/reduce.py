import numpy as np


def mean_std(data):
    """Get mean and standard deviation of data."""
    return (data.mean(), data.std())


def mean_std_err(data):
    """Get mean and standard error of data."""
    return (data.mean(), data.std()/np.sqrt(len(data)))


def get_quadrant(data):
    """Get the 4 quadrants of a 2D array."""
    x, y = data.shape
    x = x // 2
    y = y // 2
    return data[:x, :y], data[:x, y:], data[x:, :y], data[x:, y:]


def get_5_quadrant(data):
    """Get the 5 quadrants of a 2D array.
    
    The 5th quadrant is the center of the array.
    
    Args:
        data: 2D array
        
    Returns:
        5 quadrants of the array"""
    x, y = data.shape
    x = x // 2
    y = y // 2
    return data[:x, :y], data[:x, y:], data[x:, :y], data[x:, y:], data[x//2:3*x//2, y//2:3*y//2]
    

def reduce_quadrant(data,func):
    """Sum the 4 quadrants of a 2D array.
    
    Args:
        data: 2D array
        func: function to reduce quadrants with
        
    Returns:
        out: list of reduced quadrants
    """
    out = []
    for q in get_quadrant(data):
        out.append(func(q))
    return np.array(out)


def reduce_5_quadrant(data,func):
    """Sum the 5 quadrants of a 2D array.
    
    Args:
        data: 2D array
        func: function to reduce quadrants with
        
    Returns:
        out: list of reduced quadrants
    """
    out = []
    for q in get_5_quadrant(data):
        out.append(func(q))
    return np.array(out)

