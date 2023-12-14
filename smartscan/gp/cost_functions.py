from typing import Callable, Sequence, Tuple
import logging
import numpy as np

from ..utils import manhattan_distance

def movement_cost(
        origin: tuple, 
        x: tuple, 
        cost_func_params: dict = None, 
        verbose=False,
        logger=None,
    ) -> float:
    """Compute the cost of moving from x to y
    
    Args:
        x (tuple): starting position
        y (tuple): ending position
        cost_func_params (dict): dictionary of parameters for the cost function
            - speed (float): speed of the scanner, expressed in mm/s
            - dwell_time (float): dwell time, expressed in s
            - dead_time (float): dead time, expressed in s
            - point_to_um (float): conversion factor from mm to seconds
    
    Returns:
        float: cost of moving from x to y, expressed in seconds

    """
    if logger is None:
        logger = logging.getLogger('movement_cost')
    x = np.array(x)
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    if cost_func_params is None:
        cost_func_params = {}
    else:
        cost_func_params = cost_func_params.copy()
    speed = cost_func_params.pop('speed',250)
    dwell_time = cost_func_params.pop('dwell_time',0.5)
    dead_time = cost_func_params.pop('dead_time',0.6)
    point_to_um = cost_func_params.pop('point_to_um',1.0)
    weight = cost_func_params.pop('weight',1.0)
    min_distance = cost_func_params.pop('min_distance',1.0)

    gp_x_data:np.ndarray = cost_func_params.pop('prev_points',np.empty((0,2)))
    prev_points = gp_x_data[:,:2]
    # all_distances = np.array([manhattan_distance(x,p) for p in prev_points])
    # print(x.shape,prev_points.shape)
    if prev_points.shape[0] > 0:
        all_distances = np.linalg.norm(x-prev_points)
        if np.any(all_distances < min_distance):
            idx = np.argmin(all_distances - min_distance)
            if verbose:
                logger.debug(f"Point {np.asarray(x).ravel()} close to previous point {prev_points[idx]}: d={all_distances[idx]}")
            return [1_000_000]
    # if any([manhattan_distance(x,p) < min_distance for p in prev_points]):
    #     print(f"Point {x} close to previous points")
    #     if verbose:
    #         print(f"Point already visited, returning {1_000_000}")
    #     return [1_000_000]

    if len(cost_func_params) > 0:
        raise ValueError(f"Unrecognized parameters: {cost_func_params.keys()}")
    distance = manhattan_distance(origin,x) * point_to_um
    time: float = weight * distance / speed  + dwell_time + dead_time
    if verbose:
        logger.debug(f"Distance: {distance:.2f} um, Time: {time:.2f} s"
            f" (dwell: {dwell_time:.2f} s, dead: {dead_time:.2f} s)" 
        )
    return time

def compute_costs(
        origin: Sequence[tuple], 
        x: Sequence[tuple], 
        cost_func: Callable = movement_cost,
        cost_func_params: dict = None, 
        verbose=False,
        logger=None,
    ) -> np.ndarray[float]:
    """ Compute the cost of moving from origin to each point in x
    
    Args:
        origin (tuple): starting position
        x (tuple): list of N ending positions
        cost_func_params (dict): dictionary of parameters for the cost function
            - speed (float): speed of the scanner, expressed in mm/s
            - dwell_time (float): dwell time, expressed in s
            - dead_time (float): dead time, expressed in s
            - point_to_um (float): conversion factor from mm to seconds#
    
    Returns:
        Sequence[float]: cost of moving from origin to each point in x, expressed in seconds
    """
    if logger is None:
        logger = logging.getLogger('compute_costs')
    cost = []
    for xx in x:
        movcost: float = movement_cost(origin,xx,cost_func_params,verbose=verbose,logger=logger)
        cost.append(movcost)
    logger.debug(f"Costs computed: {cost}")
    return np.asarray(cost).T

