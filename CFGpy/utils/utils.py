import numpy as np
import pandas as pd
import os
import json
import networkx as nx
from collections import Counter, defaultdict
from CFGpy.utils.FilesHandler import FilesHandler

ID2COORD = FilesHandler().id2coord
N_ALL_SHAPES = len(ID2COORD) - 1  # subtract 1 because index 0 in ID2COORD is a placeholder, not a shape

SHORTEST_PATHS_DICT_PATH = FilesHandler().shortest_paths_dict
SHORTEST_PATHS_DICT = {}
NEW_SHORTEST_PATHS_DICT = {}

def get_vanilla():
    """
    Returns the most up-to-date postparsed version of the vanilla data.
    """
    return FilesHandler().vanilla_data


def get_vanilla_features() -> pd.DataFrame:
    """
    Returns the features extracted from the most up-to-date vanilla data.
    """
    return FilesHandler().vanilla_features


def get_vanilla_stats():
    """
    Returns the necessary information for extraction of features relative to vanilla, as required by
    behavioral.FeatureExtractor._extract_relative_features.
    cf. behavioral.data_classes.PostparsedDataset.get_stats
    """
    
    step_counter_dict = FilesHandler().vanilla_step_counter
    step_counter = Counter({tuple(json.loads(key)): orig for key, orig in step_counter_dict.items()})

    covered_steps = set(step_counter.keys())

    gallery_counter_dict = FilesHandler().vanilla_gallery_counter
    gallery_counter = Counter({int(key): orig for key, orig in gallery_counter_dict.items()})

    covered_galleries = set(gallery_counter.keys())

    giant_component = FilesHandler().vanilla_giant_component
    giant_component = {tuple(node) for node in giant_component}

    return covered_steps, step_counter, covered_galleries, gallery_counter, giant_component


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

def is_effective_integer(value: int) -> bool:
    return value == int(value)

def get_shortest_path_len(shape1: int, shape2: int):

    shape1, shape2 = min(shape1, shape2), max(shape1, shape2)

    if not is_effective_integer(shape1) or not is_effective_integer(shape2):
        raise TypeError(f"shape1 is of type {type(shape1)} and shape2 is of type {type(shape2)}. Both must be type int.") 
    
    key = f"({int(shape1)}, {int(shape2)})" # JSON doesn't allow tuples as keys, so they're stringified

    if SHORTEST_PATHS_DICT and key in SHORTEST_PATHS_DICT:
        return SHORTEST_PATHS_DICT[key]

    shape_network = FilesHandler().shape_network
    shortest_path_len = nx.shortest_path_length(shape_network, source=shape1, target=shape2)
    NEW_SHORTEST_PATHS_DICT[key] = shortest_path_len

    return shortest_path_len


def gallery_orig_map_factory(gallery_counter, alpha, d):
    """
    Creates a map from each gallery shape to its Originality, given by -log10(p) where p is the estimated probability.
    :param gallery_counter: a Counter object with the number of times each shape was saved to gallery in the sample.
    :param alpha: pseudocount for Laplace smoothing (see https://en.wikipedia.org/wiki/Additive_smoothing).
    :param d: number of possible shapes to consider (notation follows the wiki article linked above).
    :return: dict
    """
    N = gallery_counter.total()  # notation follows the wiki article linked above
    smoothed_0_prob = alpha / (N + alpha * d)
    orig_of_0_prob = -np.log10(smoothed_0_prob)
    orig_map = defaultdict(lambda: orig_of_0_prob)
    for gallery_shape, count in gallery_counter.items():
        smoothed_prob = (count + alpha) / (N + alpha * d)
        orig_map[gallery_shape] = -np.log10(smoothed_prob)

    return orig_map


def step_orig_map_factory(step_counter, alpha, d):
    """
    Creates a map from each step (s1, s2) to its Originality, given by -log10(p) where p is the estimated probability of
    s2 given s1.
    :param step_counter: a Counter object with the number of times each step was taken in the sample.
    :param alpha: pseudocount for Laplace smoothing (see https://en.wikipedia.org/wiki/Additive_smoothing).
    :param d: number of possible steps from a given shape (notation follows the wiki article linked above).
    :return: dict
    """
    Ns = Counter()
    for step, count in step_counter.items():
        Ns[step[0]] += count

    def orig_of_0_prob(step):
        N = Ns[step[0]]
        if N == alpha == 0:
            smoothed_0_prob = 1 / d
        else:
            smoothed_0_prob = alpha / (N + alpha * d)
        return -np.log10(smoothed_0_prob)

    class DefaultDict(defaultdict):  # Extends defaultdict to allow default_factory to take an arg
        def __missing__(self, key):
            return self.default_factory(key)

    orig_map = DefaultDict(orig_of_0_prob)
    for step, count in step_counter.items():
        smoothed_prob = (count + alpha) / (Ns[step[0]] + alpha * d)
        orig_map[step] = -np.log10(smoothed_prob)

    return orig_map
