import numpy as np
import pandas as pd
import os
import json
import networkx as nx
from CFGpy import NAS_PATH

CFG_RESOURCES_PATH = os.path.join(NAS_PATH, "Projects", "CFG")
VANILLA_FILENAME = r"vanillaMeasures.csv"
ID2COORD = np.load(os.path.join(CFG_RESOURCES_PATH, "grid_coords.npy"))
SHORTEST_PATHS_DICT_PATH = os.path.join(CFG_RESOURCES_PATH, "shortest_path_len.json")


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

    return int(shape_id[0])


def get_shortest_path_len(shape1, shape2):
    shape1, shape2 = min(shape1, shape2), max(shape1, shape2)
    key = str((shape1, shape2))  # JSON doesn't allow tuples as keys, so they're stringified
    with open(SHORTEST_PATHS_DICT_PATH) as shortest_path_len_fp:
        shortest_path_len_dict = json.load(shortest_path_len_fp)

    try:
        shortest_path_len = shortest_path_len_dict[key]
    except KeyError:
        shape_network = nx.read_adjlist(os.path.join(CFG_RESOURCES_PATH, "all_shapes.adjlist"), nodetype=int)
        shortest_path_len = nx.shortest_path_length(shape_network, source=shape1, target=shape2)
        shortest_path_len_dict[key] = shortest_path_len
        with open(SHORTEST_PATHS_DICT_PATH, "w") as shortest_path_len_fp:
            json.dump(shortest_path_len_dict, shortest_path_len_fp)

    return shortest_path_len
