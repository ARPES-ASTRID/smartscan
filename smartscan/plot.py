from itertools import product
from typing import Any, Callable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import Voronoi


def plot_aqf_panel(
    gp,
    fig: plt.Figure | None,
    pos: np.ndarray,
    val: np.ndarray,
    shape: tuple | None = None,
    old_aqf: np.ndarray | None = None,
    last_spectrum: np.ndarray | None = None,
    settings: dict | None = None,
) -> tuple[plt.Figure | Any, np.ndarray]:
    """Live visualization of the acquisition function of a GP driving data acquisition.

    This is used to visualize the acquisition function and the posterior mean and variance of the GP
    during a measurement.

    Args:
        gp (GaussianProcess): GP object
        fig (plt.Figure): The figure to plot on
        pos (np.ndarray): The positions of the points
        val (np.ndarray): The values of the points
        shape (tuple): The shape of the plot
        old_aqf (np.ndarray): The previous acquisition function
        last_spectrum (np.ndarray): The last spectrum
        settings (dict): The settings of the scan

    Returns:
        fig (plt.Figure): The figure to plot on
        aqf (np.ndarray): The acquisition function
    """

    positions = gp.x_data

    if shape is None:
        shape = settings["plots"]["posterior_map_shape"]
    x_pred_0 = np.empty((np.prod(shape), 3))
    x_pred_1 = np.empty((np.prod(shape), 3))
    counter = 0
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])

    lim_x = gp.input_space_bounds[0]
    lim_y = gp.input_space_bounds[1]

    delta_x = (lim_x[1] - lim_x[0]) / shape[0]
    delta_y = (lim_y[1] - lim_y[0]) / shape[1]

    for i in x:
        for j in y:
            x_pred_0[counter] = np.array(
                [delta_x * i + lim_x[0], delta_y * j + lim_y[0], 0]
            )
            x_pred_1[counter] = np.array(
                [delta_x * i + lim_x[0], delta_y * j + lim_y[0], 1]
            )
            counter += 1

    PM0 = np.reshape(gp.posterior_mean(x_pred_0)["f(x)"], shape)
    PV0 = np.reshape(gp.posterior_covariance(x_pred_0)["v(x)"], shape)
    sPV0 = np.sqrt(PV0)
    PM1 = np.reshape(gp.posterior_mean(x_pred_1)["f(x)"], shape)
    PV1 = np.reshape(gp.posterior_covariance(x_pred_1)["v(x)"], shape)
    sPV1 = np.sqrt(PV1)

    a = settings["acquisition_function"]["params"]["a"]
    norm = settings["acquisition_function"]["params"]["norm"]
    w = settings["acquisition_function"]["params"]["weights"]
    if w is None:
        w = (1, 1)
    aqf = norm * (a * np.sqrt(w[0] * PV0 + w[1] * PV1) + (w[0] * PM0 + w[1] * PM1))
    aqf = np.rot90(aqf, k=-1)[:, ::-1]

    if fig is None:
        fig = plt.figure("ACQ func", figsize=(12, 8), layout="constrained")
    else:
        fig.clear()

    ax = [
        fig.add_subplot(331),
        fig.add_subplot(332),
        fig.add_subplot(333),
        fig.add_subplot(334),
        fig.add_subplot(335),
        fig.add_subplot(336),
        fig.add_subplot(337),
        fig.add_subplot(338),
        fig.add_subplot(339),
    ]

    ax = np.asarray(ax).reshape(3, 3)
    for i, PM, PV in zip(range(2), [PM0, PM1], [sPV0, sPV1]):
        PM = np.rot90(PM, k=-1)[:, ::-1]
        PV = np.rot90(PV, k=-1)[:, ::-1]
        pmmax = PM.max()
        pvmax = PV.max()
        PM /= pmmax
        PV /= pvmax

        ax[i, 0].imshow(
            PM, clim=[0, 1], extent=[*lim_x, *lim_y], origin="lower", aspect="equal"
        )
        ax[i, 0].set_title(f"PM: {pmmax:.3f}")
        ax[i, 1].imshow(
            PV, clim=[0, 1], extent=[*lim_x, *lim_y], origin="lower", aspect="equal"
        )
        ax[i, 1].set_title(f"PV: {a * np.sqrt(pvmax):.3f}")

        ax[i, 0].scatter(positions[:, 0], positions[:, 1], s=15, c="r", alpha=0.5)
        ax[i, 1].scatter(positions[:, 0], positions[:, 1], s=15, c="r", alpha=0.5)
        ax[i, 0].scatter(positions[-1, 0], positions[-1, 1], s=30, c="white")
        ax[i, 1].scatter(positions[-1, 0], positions[-1, 1], s=30, c="white")

    ax[0, 2].imshow(
        np.zeros_like(PM),
        clim=[0, 1],
        extent=[*lim_x, *lim_y],
        origin="lower",
        aspect="equal",
    )
    ax[0, 2].scatter(
        pos[:, 0], pos[:, 1], s=25, c=val[:, 0], cmap="viridis", marker="o"
    )
    ax[1, 2].imshow(
        np.zeros_like(PM),
        clim=[0, 1],
        extent=[*lim_x, *lim_y],
        origin="lower",
        aspect="equal",
    )
    ax[1, 2].scatter(
        pos[:, 0], pos[:, 1], s=25, c=val[:, 1], cmap="viridis", marker="o"
    )
    ax[0, 2].scatter(pos[-1, 0], pos[-1, 1], s=25, c="r", marker="o")
    ax[0, 2].plot(pos[:, 0], pos[:, 1], c="w", alpha=0.5)
    ax[1, 2].scatter(pos[-1, 0], pos[-1, 1], s=25, c="r", marker="o")
    ax[1, 2].plot(pos[:, 0], pos[:, 1], c="w", alpha=0.5)

    ax[2, 0].set_title(f"Aq func {aqf.max():.2f}")
    ax[2, 0].imshow(
        aqf,
        extent=[*lim_x, *lim_y],
        origin="lower",
        clim=np.quantile(aqf, (0.01, 0.99)),
        aspect="equal",
    )
    if old_aqf is not None:
        diff = old_aqf - aqf
        ax[2, 1].set_title("aqf changes")
        ax[2, 1].imshow(
            diff, extent=[*lim_x, *lim_y], origin="lower", cmap="bwr", aspect="equal"
        )
    if last_spectrum is not None:
        ax[2, 2].imshow(
            last_spectrum,
            clim=np.quantile(last_spectrum, (0.02, 0.98)),
            origin="lower",
            cmap="terrain",
            aspect="equal",
        )

    plt.pause(0.01)
    return fig, aqf


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
            Nearest produces a pixelated version of a vornoi plot.
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


def vornoi_plot(
    points: np.ndarray,
    values: np.ndarray,
    cmap: Callable | str = "viridis",
    draw_points: bool = True,
    draw_lines: bool = True,
    point_color: np.ndarray | str | None = None,
    point_size: float = 2,
    line_color: np.ndarray | str | None = None,
    line_width: float = 0.5,
    alpha: float = 0.4,
    ax: plt.Axes | None = None,
    border: float | None = 0.02,
) -> plt.Axes:
    """Plot the Voronoi diagram of a set of points.

    Args:
        points (np.ndarray): The points to plot.
        values (np.ndarray): The values of the points.
        cmap (Callable | str): The colormap to use.
        draw_points (bool): Whether to draw the points.
        draw_lines (bool): Whether to draw the Voronoi lines.
        point_color (np.ndarray | str | None): The color of the points.
        point_size (float): The size of the points.
        line_color (np.ndarray | str | None): The color of the lines.
        line_width (float): The width of the lines.
        alpha (float): The transparency of the polygons.
        ax (plt.Axes | None): The axes to plot on.
        border (float | None): The border to add to the plot.

    Returns:
        The axes of the plot.
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
                *zip(*polygon),
                alpha=alpha,
                color=colors[point_id],
                edgecolor=edgecolor,
                linewidth=line_width,
            )

    if draw_points:
        point_color = colors if point_color is None else point_color
        scatter = ax.scatter(
            points[:-4, 0], points[:-4, 1], color=point_color, marker="o", s=point_size
        )
    else:
        scatter = None
    if border is not None:
        ax.set_xlim(limits[0] - box_size[0] * border, limits[1] + box_size[0] * border)
        ax.set_ylim(limits[2] - box_size[1] * border, limits[3] + box_size[1] * border)
    return ax, scatter


if __name__ == "__main__":
    pass
