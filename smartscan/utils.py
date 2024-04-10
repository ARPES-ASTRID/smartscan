import logging
from itertools import product
from typing import Callable, Counter, Sequence, Tuple

import numpy as np


def pretty_print_time(t: float) -> str:
    """Print time as hh:mm:ss"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def manhattan_distance(x: tuple[float], y: tuple[float]) -> float:
    """Compute the Manhattan distance between two points

    Args:
        x (tuple[float]): first point
        y (tuple[float]): second point

    Returns:
        float: Manhattan distance between x and y
    """
    return np.sum(np.abs(np.asarray(x) - np.asarray(y)))


def euclidean_distance(x: tuple[float], y: tuple[float]) -> float:
    """Compute the Euclidean distance between two points

    Args:
        x (tuple[float]): first point
        y (tuple[float]): second point

    Returns:
        float: Euclidean distance between x and y
    """
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def closest_point_on_int_grid(x: Tuple[float], grid_shape: Tuple[int]) -> Tuple[int]:
    """Find the closest point in a grid to a given point

    Args:
        x (Tuple[float]): point to find the closest point to
        grid_shape (Tuple[int]): shape of the grid

    Returns:
        Tuple[int]: closest point in the grid to x
    """
    grid = np.asarray(tuple(product([range(s) for s in grid_shape])))
    return grid[np.linalg.norm(grid - x, axis=1).argmin()]


def closest_point_on_grid(
    x: Tuple[float], axes: Sequence[Sequence[float]]
) -> Tuple[int]:
    """Find the closest point in a grid to a given point

    Args:
        x (Tuple[float]): point to find the closest point to
        axes (Sequence[Sequence[float]]): grid axes

    Returns:
        Tuple[int]: closest point in the grid to x
    """
    grid = np.asarray(tuple(product(*axes)))
    return grid[np.linalg.norm(grid - x, axis=1).argmin()]


def scan_time(
    positions: Tuple[float], movement_cost_func: Callable, cost_func_params: dict = None
) -> float:
    """Compute the time to scan a list of positions based on a movement cost function

    Args:
        positions (Tuple[float]): list of positions to scan
        movement_cost_func (Callable): movement cost function
        cost_func_params (dict, optional): parameters for the movement cost function. Defaults to None.

    Returns:
        float: time to scan the list of positions
    """
    return np.sum(
        [
            movement_cost_func(positions[i - 1], positions[i], cost_func_params)
            for i in range(1, len(positions))
        ]
    )


def how_many_repeated_points(positions: Sequence[Tuple[float]]) -> int:
    """Count how many points are repeated in the list of positions

    Args:
        positions (Sequence[Tuple[float]]): list of positions

    Returns:
        int: number of repeated points
    """
    counts = list(dict(Counter(tuple(map(tuple, positions)))).values())
    return sum(counts) - len(counts)


def duplicate_positions(positions: Sequence[Tuple[float]]) -> dict[tuple[float], int]:
    """a dict with the number of repetitions of each position when the repetition is > 1

    Args:
        positions (Sequence[Tuple[float]]): list of positions

    Returns:
        dict[tuple[float], int]: dict with the number of repetitions of each position
    """
    counts = dict(Counter(tuple(map(tuple, positions))))
    return {k: v for k, v in counts.items() if v > 1}


class ColoredFormatter(logging.Formatter):
    """A colorful formatter for logging messages.

    modified from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

    """

    grey = "\x1b[38;20m"
    white = "\x1b[37;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(
        self,
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
    ) -> None:
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + fmt + self.reset,
            logging.INFO: self.white + fmt + self.reset,
            logging.WARNING: self.yellow + fmt + self.reset,
            logging.ERROR: self.red + fmt + self.reset,
            logging.CRITICAL: self.bold_red + fmt + self.reset,
        }

    def format(self, record) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
