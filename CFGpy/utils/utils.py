import numpy as np
import pandas as pd
import os
from CFGpy import NAS_PATH

CFG_RESOURCES_PATH = os.path.join(NAS_PATH, "Projects", "CFG")
VANILLA_FILENAME = r"vanillaMeasures.csv"
ID2COORD = np.load(os.path.join(CFG_RESOURCES_PATH, "grid_coords.npy"))


def get_vanilla():
    """
    Returns the most up-to-date version of the vanilla data.
    :return: pd.DataFrame with shape (614, 36)
    """
    return pd.read_csv(os.path.join(CFG_RESOURCES_PATH, VANILLA_FILENAME))


def get_shape_binary_matrix(shape_id):
    """
    Converts a shape's ID to its binary matrix representation.
    :param shape_id: int
    :return: 2D ndarray with dtype float, all values are binary
    """
    coords = ID2COORD[shape_id]
    binary_mat = np.array([list(np.binary_repr(row, width=10)) for row in coords],
                          dtype=float)
    nrow, ncol = np.max(np.nonzero(binary_mat), axis=1) + 1
    binary_mat = binary_mat[:nrow, :ncol]  # truncate all zeros rows and cols from bottom and right respectively

    return binary_mat


def binary_shape_to_id(binary_shape):
    pad_width = (0, 10 - len(binary_shape))
    padded_shape = np.pad(binary_shape, pad_width=pad_width)
    shape_id = np.flatnonzero(np.all(ID2COORD == padded_shape, axis=1))
    if shape_id.size != 1:
        raise ValueError('Was not able to find the desired shape', binary_shape)

    return shape_id[0]
