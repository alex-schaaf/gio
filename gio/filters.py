from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from nptyping import Array
from typing import Tuple


def below_distance(
        surfpts1: Array[int, ..., 3], 
        surfpts2: Array[int, ..., 3],
        threshold: float
    ) -> Array[int, ...]:
    """Find points in given set of points that are within a threshold
    euclidean distance of a second set of points.
    
    Args:
        surfpts1 (Array[int, ..., 3]): X,Y,Z coordinates of point set 1.
        surfpts2 (Array[int, ..., 3]): X,Y,Z coordinates of point set 2.
        threshold (float): Euclidean distance threshold.
    
    Returns:
        Array[int, ...]: Indices of points within the distance threshold.
    """
    distances = cdist(surfpts1,
                      surfpts2)
    below_threshold = distances <= threshold
    return np.where(np.sum(below_threshold, axis=1) > 0)[0]


def get_filtered_indices(
        df:pd.DataFrame, 
        fmt1:str, 
        fmt2:str, 
        threshold:float, 
        groupby:str="formation"
    ) -> Array[int, ...]:
    """Get the indices of points in formation 1 that are within a distance
    threshold of formation 2 in given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame of input data. GemPy structure, 
            requires the coordinates to be in columns "X", "Y", "Z".
        fmt1 (str): Formation 1 name.
        fmt2 (str): Formation 2 name
        threshold (float): Euclidean distance threshold.
        groupby (str, optional): Column name by which to group the
            formations. Defaults to "formation".
    
    Returns:
        Array[int, ...]: Indices that are within the euclidean
            distance given by the threshold.
    """
    grps = df.groupby(groupby).groups
    i1 = grps[fmt1]
    i2 = grps[fmt2]
    i1_filter = below_distance(
        df.loc[i1][["X", "Y", "Z"]].values, 
        df.loc[i2][["X", "Y", "Z"]].values, threshold)
    return i1[i1_filter]
