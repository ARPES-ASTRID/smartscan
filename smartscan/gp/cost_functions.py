from typing import Any, Callable, Sequence, Tuple
import logging
from networkx import selfloop_edges
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

def weighted_manhattan_distance(
        a:np.ndarray,
        b:np.ndarray,
        weights:np.ndarray=None
    ) -> float:
    if weights is None:
        weights = np.ones(len(a))
    assert len(a) == len(b) == len(weights), "a, b and weights must have the same length"
    return np.sum(weights * np.abs(a-b))

def manhattan_cost_function(
        origin: np.ndarray[float],
        x: Sequence[Tuple[float,float]],
        cost_func_params: dict[str, Any] = None,
        logger=None,
    ) -> float:
    """Compute the movement cost between two points

    Args:
        origin (Sequence[float]): origin point
        x (Sequence[float]): destination point
        cost_func_params (dict[str, Any], optional): cost function parameters. 
            All parameters need to be passed in this dictionary because of the structure
            imposed by gpCAM. Defaults to None, which means that the default parameters are:
            - speed (float): 250 um/s
            - dwell_time (float): 0.5 s
            - dead_time (float): 0.6 s
            - point_to_um (float): 1.0 um/point unit conversion factor
    
    Returns:
        float: movement cost
    """
    if logger is None:
        logger = logging.getLogger('manhattan_cost_function')

    origin = np.array(origin)
    # assert origin.shape[0] == 2, "origin must be a 2D point in the cost function"

    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(-1,1) # make sure x is a 2D array
    # assert x.shape[1] == 2, "dest.pt. x must be a 2D point or a list of 2D points in the cost function"

    # gather parameters
    if cost_func_params is None:
        cost_func_params = {}
    else:
        cost_func_params = cost_func_params.copy()
    dwell_time = cost_func_params.get('dwell_time',0.5)
    dead_time = cost_func_params.get('dead_time',0.6)
    point_to_um = cost_func_params.get('point_to_um',1.0)

    speed = cost_func_params.get('speed',250)
    if np.array(speed).size == 1:
        speed = np.ones_like(origin) * speed
    elif np.array(speed).shape != origin.shape:
        raise ValueError(f"speed must be a scalar or an array of size {origin.shape}")
    
    weight = cost_func_params.get('weight',np.ones_like(origin))
    if np.array(weight).size == 1:
        weight = np.ones_like(origin) * weight
    elif np.array(weight).shape != origin.shape:
        raise ValueError(f"weight must be a scalar or an array of size {origin.shape}")
    unrecognized = [k for k in cost_func_params.keys() if k not in ['speed','dwell_time','dead_time','point_to_um','weight']]

    if len(unrecognized) > 0:
        logger.warning(f"Unrecognized parameters: {unrecognized}")

    # logger.debug(f"Parameters: speed={speed}, dwell_time={dwell_time}, dead_time={dead_time}, point_to_um={point_to_um}, weight={weight}")
    times = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        distance = weighted_manhattan_distance(origin, x[i,:], weight/speed) * point_to_um
        times[i] = distance / speed  + dwell_time + dead_time
    logger.debug(f"Times: {min(times):.3f} - {max(times):.3f} ")
    return times

def manhattan_avoid_repetition(
        origin: np.ndarray[float],
        x: Sequence[Tuple[float,float]],
        cost_func_params: dict[str, Any] = {},
    ) -> float:
    """Avoid repeating the same point twice and compute the movement cost between two points"""
    logger = logging.getLogger('manhattan_avoid_repetition')
    logger.debug("cost func params:")
    for k,v in cost_func_params.items():
        if isinstance(v,np.ndarray):
            logger.debug(f'\t{k}: {v.shape}')
        else:
            logger.debug(f'\t{k}: {v}')
    cfp = cost_func_params.copy()
    min_distance = cfp.pop('min_distance')
    gp_x_data:np.ndarray = cfp.pop('prev_points')
    n_tasks = cfp.pop('n_tasks')
    n_dim = cfp.pop('n_dim')
    
    
    prev_points = np.array(gp_x_data)[::n_tasks,:-1]
    logger.debug(f"some prev points: {prev_points[-3:]}")

    if prev_points.shape[0] > 0 and n_dim == 2:
        min_dist = np.inf
        x_ = None
        for xx in x:
            all_distances = np.linalg.norm(xx-prev_points, axis=1)
            cur_min = np.min(all_distances)
            if cur_min < min_dist:
                x_ = xx
                min_dist = cur_min
            min_dist = min(min_dist,np.min(all_distances))
            if np.any(all_distances < min_distance):
                idx = np.argmin(all_distances - min_distance)
                logger.warning(f"Point {np.asarray(xx).ravel()} close to previous point {prev_points[idx]}: d={all_distances[idx]}")
                return [1_000_000_000]
        logger.debug(f"closest point to {x_} is {min_dist} away")
    return manhattan_cost_function(origin,x,cfp)

def cost_per_axis(        
        origin: np.ndarray[float],
        x: Sequence[Tuple[float,float]],
        cost_func_params: dict[str, Any] = {},
    ) -> Sequence[float]:
    """Compute the movement cost between an origin and a list of points"""
    logger = logging.getLogger('cost_per_axis')
    logger.debug("cost func params:")
    for k,v in cost_func_params.items():
        if isinstance(v,np.ndarray):
            logger.debug(f'\t{k}: {v.shape}')
        else:
            logger.debug(f'\t{k}: {v}')

    cfp = cost_func_params.copy()
    speed = cfp.get('speed',250)
    if np.array(speed).size == 1:
        speed = np.ones_like(origin) * speed
    elif np.array(speed).shape != origin.shape:
        raise ValueError(f"speed must be a scalar or an array of size {origin.shape}")
    speed[speed == 0] = np.inf # avoid division by zero
    weight = cfp.get('weight',np.ones_like(origin))
    if np.array(weight).size == 1:
        weight = np.ones_like(origin) * weight
    elif np.array(weight).shape != origin.shape:
        raise ValueError(f"weight must be a scalar or an array of size {origin.shape}")
    
    n_tasks = cfp.get('n_tasks')
    n_dim = cfp.get('n_dim')
    axes = cfp.get('axes')
    prev_points = np.array(cfp.get('prev_points'))[::n_tasks,:-1]
    
    distances = np.zeros((len(x),n_dim))
    out = np.zeros((len(x)))
    for i,xx in enumerate(x):
        rounded = round_to_axes(xx,axes)
        if np.any(np.all(np.isclose(rounded, prev_points), axis=1)):
            # idx = np.argmin(np.linalg.norm(xx-prev_points, axis=1))
            logger.warning(f"Point {np.asarray(xx).ravel()} rounds to {rounded} which was already measured")
            out[i] = 1_000_000_000
        else:
            distances[i,:] = np.abs(xx - origin)
            out[i] = np.sum(1 + weight * distances / speed)
    assert len(out) == len(x)
    return out

def round_to_axes(
        x: np.ndarray[float],
        axes: Sequence[Sequence[float]],
    ) -> np.ndarray[float]:
    """Round a point to the closest point on a grid"""
    new = np.empty_like(x)
    for i in range(len(x)):
        new[i] = np.argmin(np.abs(axes[i] - x[i]))
    return new
