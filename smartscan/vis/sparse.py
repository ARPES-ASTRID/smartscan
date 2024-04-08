from itertools import product
from typing import Any, Callable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import Voronoi


def min_step_size(arr: np.ndarray) -> Any:
    """Returns the minimum step size in a 1D array"""
    b = np.diff(np.sort(arr))
    return b[b > 0].min()


def interpolate_points_to_array(
    positions: np.ndarray,
    values: np.ndarray,
    method: Literal["nearest", "linear", "cubic"] = "nearest",
    dims: Sequence[str] = ["x", "y"],
    coords: dict[np.ndarray | Sequence[float]] = None,
    max_size: int = 1000,
    attrs: dict = None,
    name: str = None,
    border_steps: int = 20,
    **kwargs,
) -> xr.DataArray:
    """Interpolate a set of points to a regular grid.

    Args:
        positions: The positions of the points to interpolate.
        values: The values of the points to interpolate.
        method: The interpolation method to use. One of 'nearest', 'linear', or 'cubic'.
            Nearest produces a pixely version of a vornoi plot.
        dims: The names of the dimensions of the output array.
        coords: The coordinates of the output array. If None, the coordinates are inferred from the positions.
        max_size: The maximum size of the output array. If the inferred size is larger than this, the array is truncated.
        attrs: The attributes of the output array.
        name: The name of the output array.
        border_steps: The number of steps to add to the border of the output array.
        **kwargs: Additional arguments to pass to scipy.interpolate.griddata.

    Returns:
        The interpolated array.
    """
    if coords is None:
        xmax, xmin, xstep = (
            positions[:, 0].max(),
            positions[:, 0].min(),
            min_step_size(positions[:, 0]),
        )
        xmin, xmax = xmin - border_steps * xstep, xmax + border_steps * xstep
        ymax, ymin, ystep = (
            positions[:, 1].max(),
            positions[:, 1].min(),
            min_step_size(positions[:, 1]),
        )
        ymin, ymax = ymin - border_steps * ystep, ymax + border_steps * ystep
        xlen = min(int((xmax - xmin) / xstep), max_size)
        ylen = min(int((ymax - ymin) / ystep), max_size)
        coords = {
            dims[0]: np.linspace(xmin, xmax, xlen),
            dims[1]: np.linspace(ymin, ymax, ylen),
        }
    else:
        xlen, ylen = len(coords[dims[0]]), len(coords[dims[1]])
    if attrs is None:
        attrs = {}
    attrs.update(
        {
            "interp_method": method,
            "truncated": any([xlen == max_size, ylen == max_size]),
        }
    )
    grid_x, grid_y = np.meshgrid(coords[dims[1]], coords[dims[0]])
    sparse_data = griddata(positions, values, (grid_x, grid_y), method=method, **kwargs)
    xarr = xr.DataArray(
        data=sparse_data,
        coords=coords,
        dims=dims,
        attrs=attrs,
        name=name,
    )
    return xarr


def voronoi_polygon_plot(
    points: np.ndarray,
    values: np.ndarray,
    cmap: Callable | str = "viridis",
    draw_points: bool = True,
    draw_lines: bool = True,
    point_color: np.ndarray | str | None = None,
    line_color: np.ndarray | str | None = None,
    alpha: float = 0.4,
    ax: plt.Axes | None = None,
    border: float | None = 0.02,
) -> plt.Axes:
    """
    Plot the Voronoi diagram of a set of points.

    Args:
        points: The points to plot.
        values: The values to use for coloring the Voronoi regions.
        cmap: The colormap to use for the Voronoi regions.
        draw_points: Whether to draw the points.
        draw_lines: Whether to draw the Voronoi lines.
        point_color: The color to use for the points.
        line_color: The color to use for the lines.
        alpha: The alpha value to use for the Voronoi regions.
        ax: The axis to plot on.
        border: The border to add to the x and y limits. If None, no restriction on the
            limits is applied.

    Returns:
        None
    """

    if ax is None:
        ax: plt.Axes = plt.gca()

    # add points outside the box to draw border polygons
    limits = np.asarray(
        [
            np.min(points[:, 0]),
            np.max(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 1]),
        ]
    )
    box_size = limits[1] - limits[0], limits[3] - limits[2]
    double_limits = (
        limits[0] - box_size[0],
        limits[1] + box_size[0],
        limits[2] - box_size[1],
        limits[3] + box_size[1],
    )
    double_corners = np.asarray(list(product(double_limits[:2], double_limits[2:])))
    points = np.append(points, double_corners, axis=0)

    # compute Voronoi tesselation
    vor = Voronoi(points)

    # colorize
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    cmap = plt.get_cmap(cmap)
    colors = cmap(values)

    for point_id, region_id in enumerate(vor.point_region):
        region = vor.regions[region_id]
        if -1 not in region:  # finite region
            polygon = [vor.vertices[i] for i in region]
            if draw_lines:
                edgecolor = colors[point_id] if line_color is None else line_color
            else:
                edgecolor = None
            ax.fill(
                *zip(*polygon), alpha=alpha, color=colors[point_id], edgecolor=edgecolor
            )

    if draw_points:
        point_color = colors if point_color is None else point_color
        scatter = ax.scatter(
            points[:-4, 0], points[:-4, 1], color=point_color, marker="o", s=2
        )
    else:
        scatter = None
    if border is not None:
        ax.set_xlim(limits[0] - box_size[0] * border, limits[1] + box_size[0] * border)
        ax.set_ylim(limits[2] - box_size[1] * border, limits[3] + box_size[1] * border)
    return ax, scatter


if __name__ == "__main__":
    pass
