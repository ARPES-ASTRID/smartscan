from typing import Any, Sequence, Tuple
import logging
import numpy as np


def cost_per_axis(
    origin: np.ndarray[float],
    x: Sequence[Tuple[float, float]],
    cost_func_params: dict[str, Any] = {},
) -> Sequence[float]:
    """Compute the movement cost between an origin and a list of points.

    The cost is computed as the sum of the absolute differences between the origin and the
    destination points (Manhattan distance), divided by the speed of the scanner along each axis.

    Args:
        origin (np.ndarray): origin point where to evaluate the cost from. Shape (n_dim,)
        x (Sequence[Tuple[float, float]]): list of destination points to evaluate the cost to.
            Shape (n_points, n_dim)
        cost_func_params (dict[str, Any], optional): cost function parameters. Defaults to {}.
            Parameters:
            - speed (float): speed of the scanner, expressed in mm/s (default: 250)
            - weight (float): weight for each axis (default: 1)
            - n_tasks (int): number of tasks (default: 1)
            - n_dim (int): number of dimensions (default: 2)
            - axes (Sequence[Sequence[float]]): grid of points to round to
            - prev_points (np.ndarray): previous points measured

    Returns:
        Sequence[float]: cost of moving from origin to each point in x
    """
    logger = logging.getLogger("cost_per_axis")
    logger.debug("cost func params:")
    for k, v in cost_func_params.items():
        if isinstance(v, np.ndarray):
            logger.debug(f"\t{k}: {v.shape}")
        else:
            logger.debug(f"\t{k}: {v}")

    cfp = cost_func_params.copy()
    speed = cfp.get("speed", 250)
    if np.array(speed).size == 1:
        speed = np.ones_like(origin) * speed
    elif np.array(speed).shape != origin.shape:
        raise ValueError(f"speed must be a scalar or an array of size {origin.shape}")
    for i in range(len(speed)):
        if not speed[i] > 0:
            speed[i] = 0
    weight = cfp.get("weight", np.ones_like(origin))
    if np.array(weight).size == 1:
        weight = np.ones_like(origin) * weight
    elif np.array(weight).shape != origin.shape:
        raise ValueError(f"weight must be a scalar or an array of size {origin.shape}")

    n_tasks = cfp.get("n_tasks")
    n_dim = cfp.get("n_dim")
    axes = cfp.get("axes")
    prev_points = np.array(cfp.get("prev_points"))[::n_tasks, :-1]

    distances = np.zeros((len(x), n_dim))
    out = np.zeros((len(x)))
    for i, xx in enumerate(x):
        rounded = round_to_axes(xx, axes)
        if np.any(np.all(np.isclose(rounded, prev_points), axis=1)):
            logger.warning(
                f"Point {np.asarray(xx).ravel()} rounds to {rounded} which was already measured"
            )
            out[i] = 1_000_000_000
        else:
            distances[i, :] = np.abs(xx - origin)
            out[i] = 1 + np.sum(weight * distances / speed)
    assert len(out) == len(x)
    return out


def round_to_axes(
    x: np.ndarray[float],
    axes: Sequence[Sequence[float]],
) -> np.ndarray[float]:
    """Round a point to the closest point on a grid

    Args:
        x (np.ndarray): point to round
        axes (Sequence[Sequence[float]]): grid of points to round to

    Returns:
        np.ndarray: rounded point
    """
    new = np.empty_like(x)
    for i in range(len(x)):
        new[i] = np.argmin(np.abs(axes[i] - x[i]))
    return new
