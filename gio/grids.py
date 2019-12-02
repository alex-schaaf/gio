import numpy as np
import pandas as pd
from nptyping import Array
from typing import Tuple, Iterable
import pyvista as pv
import mplstereonet


def read_irap_classic_grid(fp: str) -> Array[float, ..., 3]:
    """
    Read Petrel Irap Classic grid files into points array
    Arguments:
        fp (str): Filepath

    Returns:
        (np.ndarray): Points array [n, 3] ([n, (x, y, z)])
    """
    with open(fp, "r") as file:  # opens the file at given filepath
        lines = [line.rstrip("\n") for line in file.readlines()]

    file_header = lines[:4]
    file_grid = lines[4:]

    nx = int(file_header[2].split()[0])
    ny = int(file_header[0].split()[1])
    extent = [float(val) for val in file_header[1].split()]
    gy, gx = [float(val) for val in file_header[0].split(" ")[2:]]
    z_values = []
    for line in file_grid:
        line_values = line.split(" ")
        for z_value in line_values:
            z_values.append(float(z_value))
    z_values = np.array(z_values)

    Z_grid = np.array(
        [z_values[(i - 1) * nx:i * nx] for i in range(1, ny + 1)])
    y_coords = np.arange(ny) * gy + extent[0]
    x_coords = np.arange(nx) * gx + extent[2]

    Y_grid, X_grid = np.meshgrid(x_coords, y_coords)
    stacked_grid = np.stack((X_grid, Y_grid, Z_grid), axis=0)
    points = stacked_grid.reshape((3, ny * nx)).T
    boolean_filter = points[:, 2] != 9999900.000000
    return points[boolean_filter, :]


def read_cps3(fp: str, drop_empty: bool = True,
              return_grid: bool = False) -> pd.DataFrame:
    """Read CPS-3 gridded regular surface files exported by Petrel 2017 and
    returns a Pandas DataFrame.

    Args:
        fp (str): Filepath.
        drop_empty (bool): To drop or not to drop the useless grid points.
            Default: True

    Returns:
        (pandas.DataFrame) Fault stick information stored in dataframe with
        ["X", "Y", "Z"] columns.
    """

    with open(fp, "r") as file:  # open file
        lines = list(map(str.rstrip, file))  # read lines and strip them of \n

    # get extent,
    extent = np.array(lines[2].split(" ")[1:]).astype(float)
    fsnrow = np.array(lines[3].split(" ")[1:]).astype(int)
    # fsxinc = np.array(lines[4].split(" ")[1:]).astype(float)

    grid = []
    for line in lines[6:]:
        grid.append(line.split(" "))

    rows = []
    de = np.arange(0, len(grid) + 1, len(grid) / fsnrow[1]).astype(int)
    for aa, bb in zip(de[:-1], de[1:]):
        row = grid[aa:bb]
        flat_row = np.array([item for sublist in row
                             for item in sublist]).astype(float)
        rows.append((flat_row))

    rows = np.array(rows)

    Xs = np.linspace(extent[0], extent[1], fsnrow[1])
    Ys = np.linspace(extent[3], extent[2], fsnrow[0])

    if return_grid:
        return rows, Xs, Ys

    XY = []
    for x in Xs:
        for y in Ys:
            XY.append([x, y])
    XY = np.array(XY)

    g = np.concatenate((XY, rows.flatten()[:, np.newaxis]), axis=1).T
    if drop_empty:
        g = g[:, g[2, :] != np.max(g)]

    df = pd.DataFrame(columns=["X", "Y", "Z"])
    df["X"] = g[0, :]
    df["Y"] = g[1, :]
    df["Z"] = g[2, :]

    return df


def read_earth_vision_grid(fp: str,
                           surface: str = None,
                           preserve_colrow: bool = False,
                           group: str = None) -> pd.DataFrame:
    """
    Reads Earth Vision Grid files exported by Petrel into GemPy 
    Interfaces-compatible DataFrame.

    Args:
        fp (str): Filepath, e.g. "/surfaces/layer1"
        surface (str): Formation name, Default None
        preserve_colrow (bool): If True preserves row and column values saved
            in the Earth Vision Grid file. Default False
        group (str): If given creates columns with a group name (useful to 
            later identification of formation subsets). Default None

    Returns:
        pandas.DataFrame
    """
    df = pd.read_csv(fp, skiprows=20, header=None, delim_whitespace=True)
    df.columns = "X Y Z col row".split()

    if not surface:
        surface = fp.split("/")[-1]  # take filename

    df["formation"] = surface

    if not preserve_colrow:
        df.drop("col", axis=1, inplace=True)
        df.drop("row", axis=1, inplace=True)

    if group:
        df["group"] = group

    return df


def get_orient(vertices: Array[float, ..., 3],
               simplices: Array[int, ..., 3],
               extent: Iterable[float] = None) -> Tuple[Array, Array]:
    """Get centroids and normal vectors of given triangles.
    
    Args:
        vertices (Array[float, ..., 3]): Vertices of triangular mesh.
        simplices (Array[int, ..., 3]): Simplices of triangular mesh.
        extent (Iterable[float], optional): Dimensional extent for 
            normalization. Defaults to None.
    
    Returns:
        Tuple[Array, Array]: [0] triangle centroid Array (n,3) and [1] triangle
            normal Array (n,3).
    """
    normals = []
    centroids = []

    # normalized vertices along each axis
    vertices_norm = vertices / np.max(vertices, axis=0)

    for tri in simplices:
        # normal vector of triangle
        U = vertices_norm[tri[1]] - vertices_norm[tri[0]]
        V = vertices_norm[tri[2]] - vertices_norm[tri[0]]
        Nx = U[1] * V[2] - U[2] * V[1]
        Ny = U[2] * V[0] - U[0] * V[2]
        Nz = U[0] * V[1] - U[1] * V[0]
        normals.append([Nx, Ny, Nz])
        # centroid of triangle
        centroid = np.mean(vertices[tri, :], axis=0)
        centroids.append(centroid)

    return np.array(centroids), np.array(normals)


def alpha_shape(points: Array[float, ..., 2], alpha: float,
                only_outer=True) -> Array[float, ..., 2]:
    """Compute the alpha shape (concave hull) of a set of points.

    Args:
        points (Array): Array (n, 2) of points.
        alpha (float): Alpha shape value.
        only_outer (bool, optional): Toggle if to keep only outer border or
            also inner edges. Defaults to True.

    Returns:
        Array: set of (i,j) pairs representing edges of the alpha-shape. (i,j)
        are the indices in the points array.

    Source:
        https://stackoverflow.com/a/51082039/8040299
    """
    from scipy.spatial import Delaunay

    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges: set, i: int, j: int) -> None:
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (
                j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
        b = np.sqrt((pb[0] - pc[0])**2 + (pb[1] - pc[1])**2)
        c = np.sqrt((pc[0] - pa[0])**2 + (pc[1] - pa[1])**2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def is_inside(x: float,
              y: float,
              points: Array,
              edges: Array,
              eps: float = 1.0e-10):
    """source: https://stackoverflow.com/a/51082039/8040299"""
    intersection_counter = 0
    for i, j in edges:
        # assert abs((points[i, 1] - y) * (points[j, 1] - y)) > eps, 'Need to handle these end cases separately'
        y_in_edge_domain = (points[i, 1] - y) * (points[j, 1] - y) < 0
        if y_in_edge_domain:
            upper_ind, lower_ind = (i, j) if (points[i, 1] - y) > 0 else (j, i)
            upper_x = points[upper_ind, 0]
            upper_y = points[upper_ind, 1]
            lower_x = points[lower_ind, 0]
            lower_y = points[lower_ind, 1]

            # is_left_turn predicate is evaluated with: sign(cross_product(upper-lower, p-lower))
            cross_prod = (upper_x - lower_x) * (y - lower_y) - (
                upper_y - lower_y) * (x - lower_x)
            assert abs(
                cross_prod) > eps, "Need to handle these end cases separately"
            point_is_left_of_segment = cross_prod > 0.0
            if point_is_left_of_segment:
                intersection_counter = intersection_counter + 1
    return (intersection_counter % 2) != 0


def triangulate_surf(points: Array[float, ..., 3],
                     alpha: float,
                     view: str = "z"):
    """Triangulate surface using Delaunay triangulation

    Args:
        points (Array[float, ..., 3]): Points to be triangulated.
        alpha (float): Alpha shape parameter.

    Returns:
        tuple: Vertices Array[float, ..., 3], Simplices Array[int, ..., 3]
    """
    from scipy.spatial import Delaunay

    slice_, inv_slice = {
        "z": ([0, 1], 2),
        "x": ([1, 2], 0),
        "y": ([0, 2], 1)
    }[view]

    tri = Delaunay(points[:, slice_])
    if alpha is None:
        return tri.points, tri.simplices

    tri.centroids = []
    for triangle in tri.simplices:
        centroid = np.mean(tri.points[triangle], axis=0)
        tri.centroids.append(centroid)

    edges = alpha_shape(points[:, slice_], alpha)
    filter_ = np.array([
        is_inside(c[0], c[1], points[:, slice_], edges) for c in tri.centroids
    ])

    return (
        np.concatenate((tri.points, points[:, inv_slice][:, np.newaxis]),
                       axis=1),
        tri.simplices[filter_],
    )


def get_gempy_data_from_surfpoints(points: np.ndarray,
                                   formation: str = None,
                                   group:str = None,
                                   decimate: float = None,
                                   face_normals=False) -> tuple:
    """

    Args:
        points(np.ndarray): Point x,y,z coordinates of shape (:, 3).
        formation(str, optional): Formation name. Default: None.
        decimate(float, optional: percentage of triangles to decimate from
            original shape. Default: None.

    Returns:
        (pd.DataFrame, pd.DataFrame) Surface Points and Orientations for GemPy.
    """
    surfp_columns = "X Y Z formation".split()
    orient_columns = "X Y Z G_x G_y G_z dip azimuth polarity formation".split()

    surface_points = pd.DataFrame(columns=surfp_columns)
    surface_points["X"] = points[:, 0]
    surface_points["Y"] = points[:, 1]
    surface_points["Z"] = points[:, 2]
    surface_points["formation"] = formation

    orientations = pd.DataFrame(columns=orient_columns)

    pointcloud = pv.PolyData(points)
    trisurf = pointcloud.delaunay_2d()

    if decimate:
        trisurf = trisurf.decimate_pro(decimate)
        pointcloud = pv.PolyData(trisurf.points)

    if face_normals:
        simplices = np.array(
            [trisurf.faces[1 + i * 4:4 + i * 4] \
             for i in range(trisurf.n_faces)])
        points = pv.PolyData(np.mean(trisurf.points[simplices],
                                            axis=1)).points

        normals = trisurf.face_normals
    else:
        points = trisurf.points
        normals = trisurf.point_normals

    orientations["X"] = points[:, 0]
    orientations["Y"] = points[:, 1]
    orientations["Z"] = points[:, 2]
    orientations["G_x"] = normals[:, 0]
    orientations["G_y"] = normals[:, 1]
    orientations["G_z"] = normals[:, 2]

    strike, dip = mplstereonet.vector2pole(
        trisurf.point_normals[:, 0],
        trisurf.point_normals[:, 1],
        trisurf.point_normals[:, 2]
    )

    orientations["dip"] = dip
    orientations["azimuth"] = strike
    orientations["polarity"] = 1
    orientations["formation"] = formation

    if group:
        orientations["group"] = group 
        surface_points["group"] = group 

    return surface_points, orientations
