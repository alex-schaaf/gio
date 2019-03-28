import numpy as np
import pandas as pd


def read_cps3(fp:str, drop_empty:bool=True, return_grid:bool=False):
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
        flat_row = np.array([item for sublist in
                             row for item in sublist]).astype(float)
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

    df = pd.DataFrame(columns=['X', 'Y', 'Z'])
    df["X"] = g[0,:]
    df["Y"] = g[1, :]
    df["Z"] = g[2, :]

    return df


def read_earth_vision_grid(fp:str,
                           surface:str=None,
                           preserve_colrow:bool=False,
                           group:str=None):
    """
    Reads Earth Vision Grid files exported by Petrel into GemPy Interfaces-compatible DataFrame.

    Args:
        fp (str): Filepath, e.g. "/surfaces/layer1"
        surface (str): Formation name, Default None
        preserve_colrow (bool): If True preserves row and column values saved in the Earth Vision Grid file. Default False
        group (str): If given creates columns with a group name (useful to later identification of formation subsets). Default None

    Returns:
        pandas.DataFrame
    """
    df = pd.read_csv(fp, skiprows=20, header=None, delim_whitespace=True)
    df.columns = "X Y Z col row".split()

    if not surface:
        surface = fp.split("/")[-1]  # take filename

    df["surface"] = surface

    if not preserve_colrow:
        df.drop('col', axis=1, inplace=True)
        df.drop('row', axis=1, inplace=True)

    if group:
        df["group"] = group

    return df