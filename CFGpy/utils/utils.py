import numpy as np
import os
import json
import networkx as nx
from collections import Counter, defaultdict
from CFGpy import NAS_PATH

CFG_RESOURCES_PATH = os.path.join(NAS_PATH, "Projects", "CFG")

VANILLA_DATA_DIR = os.path.join(CFG_RESOURCES_PATH, "vanilla_data")
VANILLA_PATH = os.path.join(VANILLA_DATA_DIR, "vanilla.json")
VANILLA_STEP_ORIG_PATH = os.path.join(VANILLA_DATA_DIR, "step_orig.json")
VANILLA_GALLERY_ORIG_PATH = os.path.join(VANILLA_DATA_DIR, "gallery_orig.json")
VANILLA_STEP_COUNTER_PATH = os.path.join(VANILLA_DATA_DIR, "step_counter.json")
VANILLA_GALLERY_COUNTER_PATH = os.path.join(VANILLA_DATA_DIR, "gallery_counter.json")
VANILLA_GC_PATH = os.path.join(VANILLA_DATA_DIR, "giant_component.json")

ID2COORD = np.load(os.path.join(CFG_RESOURCES_PATH, "grid_coords.npy"))
N_ALL_SHAPES = len(ID2COORD) - 1  # subtract 1 because index 0 in ID2COORD is a placeholder, not a shape
SHORTEST_PATHS_DICT_PATH = os.path.join(CFG_RESOURCES_PATH, "shortest_path_len.json")


def get_vanilla():
    """
    Returns the most up-to-date version of the vanilla data.
    """
    with open(VANILLA_PATH) as vanilla_fp:
        return json.load(vanilla_fp)


def dump_vanilla(vanilla):
    with open(VANILLA_PATH, "w") as vanilla_fp:
        return json.dump(vanilla, vanilla_fp)


def get_vanilla_stats():
    """
    Returns the necessary information for extraction of features relative to vanilla, as required by
    behavioral.FeatureExtractor._extract_relative_features.
    cf. behavioral.data_classes.PostparsedDataset.get_stats
    """
    with open(VANILLA_STEP_COUNTER_PATH) as step_counter_fp:
        step_counter_dict = json.load(step_counter_fp)
    step_counter = Counter({tuple(json.loads(key)): orig for key, orig in step_counter_dict.items()})

    covered_steps = set(step_counter.keys())

    with open(VANILLA_GALLERY_COUNTER_PATH) as gallery_counter_fp:
        gallery_counter_dict = json.load(gallery_counter_fp)
    gallery_counter = Counter({int(key): orig for key, orig in gallery_counter_dict.items()})

    covered_galleries = set(gallery_counter.keys())

    with open(VANILLA_GC_PATH) as gc_fp:
        giant_component = json.load(gc_fp)
    giant_component = {tuple(node) for node in giant_component}

    return covered_steps, step_counter, covered_galleries, gallery_counter, giant_component


def dump_vanilla_stats(step_counter, gallery_counter, giant_component):
    # prepare for serialization:
    step_counter = {str(list(step)): orig for step, orig in step_counter.items()}
    giant_component = [list(node) for node in giant_component]

    # serialize:
    with open(VANILLA_STEP_COUNTER_PATH, "w") as step_counter_fp:
        json.dump(step_counter, step_counter_fp)

    with open(VANILLA_GALLERY_COUNTER_PATH, "w") as gallery_counter_fp:
        json.dump(gallery_counter, gallery_counter_fp)

    with open(VANILLA_GC_PATH, "w") as gc_fp:
        json.dump(giant_component, gc_fp)


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

    if key in shortest_path_len_dict:
        return shortest_path_len_dict[key]

    shape_network = nx.read_adjlist(os.path.join(CFG_RESOURCES_PATH, "all_shapes.adjlist"), nodetype=int)
    shortest_path_len = nx.shortest_path_length(shape_network, source=shape1, target=shape2)
    shortest_path_len_dict[key] = shortest_path_len
    with open(SHORTEST_PATHS_DICT_PATH, "w") as shortest_path_len_fp:
        json.dump(shortest_path_len_dict, shortest_path_len_fp)
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
