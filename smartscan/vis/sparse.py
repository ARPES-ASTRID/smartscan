from typing import Callable
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import Voronoi

from typing import Literal, Sequence
from scipy.interpolate import griddata
import xarray as xr
__all__ = [
    'min_step_size',
    'interpolate_sparse_array',
    'voronoi_polygon_plot',
]

def min_step_size(arr):
    """ Returns the minimum step size in a 1D array """
    b = np.diff(np.sort(arr))
    return b[b>0].min()

def interpolate_points_to_array(
        positions: np.ndarray, 
        values: np.ndarray, 
        method: Literal['nearest','linear','cubic'] = 'nearest',
        dims: Sequence[str] = ['x','y'],
        coords: dict[np.ndarray | Sequence[float]] = None,
        max_size: int = 1000,
        attrs: dict = None,
        name: str = None,
        border_steps:int = 20,
        **kwargs
    ) -> xr.DataArray:
    """ Interpolate a set of points to a regular grid.

    Args:
        positions: The positions of the points to interpolate.
        values: The values of the points to interpolate.
        method: The interpolation method to use. One of 'nearest', 'linear', or 'cubic'.
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
        xmax, xmin, xstep = positions[:,0].max(), positions[:,0].min(), min_step_size(positions[:,0])
        xmin, xmax = xmin - border_steps*xstep, xmax + border_steps*xstep
        ymax ,ymin, ystep = positions[:,1].max(), positions[:,1].min(), min_step_size(positions[:,1])
        ymin, ymax = ymin - border_steps*ystep, ymax + border_steps*ystep
        xlen = min(int((xmax-xmin)/xstep), max_size)
        ylen = min(int((ymax-ymin)/ystep), max_size)
        coords = {
            dims[0]:np.linspace(xmin,xmax,xlen), 
            dims[1]:np.linspace(ymin,ymax,ylen), 
        }
    if attrs is None: 
        attrs = {}
    attrs.update({
        'interp_method':method, 
        'truncated':any([xlen==max_size, ylen==max_size])
    })
    grid_x, grid_y = np.meshgrid(coords[dims[1]], coords[dims[0]])
    sparse_data = griddata(positions, values, (grid_x, grid_y), method=method, **kwargs)
    xarr = xr.DataArray(
        data = sparse_data,
        coords = coords,
        dims = dims,
        attrs = attrs,
        name = name,
    )
    return xarr

def voronoi_polygon_plot(
        points: np.ndarray, 
        values: np.ndarray = None, 
        cmap: Callable | str = cm.viridis, 
        draw_points: bool = True,
        draw_lines: bool = True,
        point_color: np.ndarray | str = None,
        line_color: np.ndarray | str = None,
        alpha: float = 0.4,
        ax: plt.Axes = None, 
        border: float | None = 0.02,
    ) -> None:
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
        ax = plt.gca()

    # add points outside the box to draw border polygons
    limits = np.asarray([np.min(points[:,0]), np.max(points[:,0]), np.min(points[:,1]), np.max(points[:,1])])
    box_size = limits[1] - limits[0], limits[3] - limits[2]
    double_limits = limits[0] - box_size[0], limits[1] + box_size[0], limits[2] - box_size[1], limits[3] + box_size[1]
    double_corners = np.asarray(list(product(double_limits[:2], double_limits[2:])))
    points = np.append(points, double_corners, axis = 0)

    # compute Voronoi tesselation
    vor = Voronoi(points)

    # colorize
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    cmap = plt.get_cmap(cmap)
    colors = cmap(values)

    for point_id, region_id in enumerate(vor.point_region):
        region = vor.regions[region_id]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            if draw_lines:
                edgecolor = colors[point_id] if line_color is None else line_color
            else:
                edgecolor = None
            ax.fill(*zip(*polygon), alpha=alpha, color=colors[point_id], edgecolor=edgecolor)

    if draw_points:
        point_color = colors if point_color is None else point_color
        ax.scatter(points[:-4,0], points[:-4,1], color=point_color, marker='o', s=2)

    if border is not None:
        ax.set_xlim(limits[0] - box_size[0] * border, limits[1] +  box_size[0] * border)
        ax.set_ylim(limits[2] - box_size[1] * border, limits[3] +  box_size[1] * border)
    return ax

##############
# DEPRECATED #
##############
def voronoi_finite_polygons_2d(
        vor: Voronoi,
        radius: float = None,
    ) -> tuple:
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Adapted from https://stackoverflow.com/a/20678647/8018502

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_vornoi_diagram(
        points: np.ndarray, 
        values: np.ndarray = None, 
        cmap: Callable | str = cm.viridis, 
        draw_points: bool = True,
        draw_lines: bool = True,
        point_color: np.ndarray | str = None,
        line_color: np.ndarray | str = None,
        ax: plt.Axes = None, 
        border: float | None = 0.02,
    ) -> None:
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
        ax: The axis to plot on.
        border: The border to add to the x and y limits. If None, no restriction on the 
            limits is applied.

    Returns:
        None
    """
    if ax is None:
        ax = plt.gca()
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if values is None:
        values = np.linspace(0, 1, len(points))
    #normalize values to [0,1]
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    colors = cmap(values)

    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # colorize
    for region, color in zip(regions,colors):
        polygon = vertices[region]
        edgecolor = None
        if draw_lines:
            edgecolor = color if line_color is None else line_color
        ax.fill(*zip(*polygon), alpha=0.4, color=color, edgecolor=edgecolor)

    if draw_points:
        point_color = colors if point_color is None else point_color
        ax.scatter(points[:,0], points[:,1], color=point_color, marker='o', s=2)

    if border is not None:
        ptp = vor.max_bound[0] - vor.min_bound[0], vor.max_bound[1] - vor.min_bound[1] 
        ax.set_xlim(vor.min_bound[0] - ptp[0] * border, vor.max_bound[0] + ptp[0] * border)
        ax.set_ylim(vor.min_bound[1] - ptp[0] * border, vor.max_bound[1] + ptp[0] * border)


if __name__ == '__main__':

    points = np.random.rand(500,2)
    values = np.abs(points[:,0]-.5)**0.5 + np.abs(points[:,1]-.5,)**0.5 #np.random.rand(60)
    values += np.random.normal(0, 0.05, len(values))
    vornoi_plot(points, values, cmap='hot')
