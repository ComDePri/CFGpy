import numpy as np
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
from CFGpy import NAS_PATH

CFG_RESOURCES_PATH = os.path.join(NAS_PATH, "Projects", "CFG")

VANILLA_DATA_DIR = os.path.join(CFG_RESOURCES_PATH, "vanilla_data")
VANILLA_PATH = os.path.join(VANILLA_DATA_DIR, "vanilla.json")
#VANILLA_STEP_ORIG_PATH = os.path.join(VANILLA_DATA_DIR, "step_orig.json")
#VANILLA_GALLERY_ORIG_PATH = os.path.join(VANILLA_DATA_DIR, "gallery_orig.json")
VANILLA_STEP_ORIG_PATH = os.path.join(VANILLA_DATA_DIR, "step_counter.json")
VANILLA_GALLERY_ORIG_PATH = os.path.join(VANILLA_DATA_DIR, "gallery_counter.json")
VANILLA_GC_PATH = os.path.join(VANILLA_DATA_DIR, "giant_component.json")

ID2COORD = np.load(os.path.join(CFG_RESOURCES_PATH, "grid_coords.npy"))
SHORTEST_PATHS_DICT_PATH = os.path.join(CFG_RESOURCES_PATH, "shortest_path_len.json")
SHAPE_COLOR = "#32CD32"  # CSS "limegreen", as used in the game
SHAPE_BG_COLOR = "k"
GALLERY_BG_COLOR = "r"


def get_vanilla():
    """
    Returns the most up-to-date version of the vanilla data.
    """
    with open(VANILLA_PATH) as vanilla_fp:
        return json.load(vanilla_fp)


def dump_vanilla(vanilla):
    with open(VANILLA_PATH, "w") as vanilla_fp:
        return json.dump(vanilla, vanilla_fp)


def get_vanilla_descriptors():
    with open(VANILLA_STEP_ORIG_PATH) as step_orig_fp:
        step_orig = json.load(step_orig_fp)
    step_orig_map = {tuple(json.loads(key)): orig for key, orig in step_orig.items()}

    covered_steps = set(step_orig_map.keys())

    with open(VANILLA_GALLERY_ORIG_PATH) as gallery_orig_fp:
        gallery_orig = json.load(gallery_orig_fp)
    gallery_orig_map = {int(key): orig for key, orig in gallery_orig.items()}

    covered_galleries = set(gallery_orig_map.keys())

    with open(VANILLA_GC_PATH) as gc_fp:
        giant_component = json.load(gc_fp)
    giant_component = {tuple(node) for node in giant_component}

    return covered_steps, step_orig_map, covered_galleries, gallery_orig_map, giant_component


def dump_vanilla_descriptors(step_orig_map, gallery_orig_map, giant_component):
    # prepare to serialization:
    step_orig_map = {str(list(step)): orig for step, orig in step_orig_map.items()}
    giant_component = [list(node) for node in giant_component]

    # serialize:
    with open(VANILLA_STEP_ORIG_PATH, "w") as step_orig_fp:
        json.dump(step_orig_map, step_orig_fp)

    with open(VANILLA_GALLERY_ORIG_PATH, "w") as gallery_orig_fp:
        json.dump(gallery_orig_map, gallery_orig_fp)

    with open(VANILLA_GC_PATH, "w") as gc_fp:
        json.dump(giant_component, gc_fp)


def shape_id_to_binary_matrix(shape_id):
    """
    Converts a shape's ID to its binary matrix representation.
    :param shape_id: int
    :return: 2D ndarray with dtype float, all values are binary
    """
    coords = ID2COORD[shape_id]
    binary_mat = np.array([list(np.binary_repr(row, width=10)) for row in coords], dtype=float)
    nrow, ncol = np.max(np.nonzero(binary_mat), axis=1) + 1
    binary_mat = binary_mat[:nrow, :ncol]  # truncate all zeros rows and cols from bottom and right

    return binary_mat


def binary_matrix_to_shape_id(binary_matrix):
    pad_width = (0, 10 - len(binary_matrix))
    padded_shape = np.pad(binary_matrix, pad_width=pad_width)
    shape_id = np.flatnonzero(np.all(ID2COORD == padded_shape, axis=1))
    if shape_id.size != 1:
        raise ValueError('Was not able to find the desired shape', binary_matrix)

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


def get_orig_map(counter, alpha=0.5, group_func=None):
    """
    Returns a mapping from each object to its Originality, given by -log10(p) where p is the estimated probability of
    the object in its group.
    :param counter: a Counter of all objects in the sample.
    :param alpha: pseudocount for Laplace smoothing (see https://en.wikipedia.org/wiki/Additive_smoothing).
    :param group_func: a Callable that returns an object's group. If None (default), objects are not grouped.
    :return: dict
    """
    total = counter.total()
    group_totals = counter
    if group_func is not None:
        group_totals = Counter()
        for obj, count in counter.items():
            group_totals[group_func(obj)] += count
    n_groups = len(group_totals)

    # transform counts to orig values
    orig_map = {}
    for obj, count in counter.items():
        if group_func is not None:
            total = group_totals[group_func(obj)]
        smoothed_step_probability = (count + alpha) / (total + n_groups * alpha)
        orig_map[obj] = -np.log10(smoothed_step_probability)

    return orig_map


def plot_binary_matrix(binary_matrix, ax=None, is_gallery=False, save_filename=None, title='', color=SHAPE_COLOR):
    """
    Plots the binary matrix representation of a shape.
    :param color: colors of colored squares, default- yellow
    :param binary_matrix: a binary matrix representation of a shape.
    :param ax: plt Axes object. If None, a new one is created and eventually displayed.
    :param is_gallery: True iff this a gallery shape. affects background color.
    :param save_filename: a filename to save the image, or None (to avoid saving).
    :param title: plot title.
    """
    bg_color = GALLERY_BG_COLOR if is_gallery else SHAPE_BG_COLOR
    nrow, ncol = binary_matrix.shape
    pad_rows = (10 - nrow) / 2
    pad_rows = (np.ceil(pad_rows).astype('int'), np.floor(pad_rows).astype('int'))
    pad_cols = (10 - ncol) / 2
    pad_cols = (np.ceil(pad_cols).astype('int'), np.floor(pad_cols).astype('int'))
    binary_matrix = np.pad(binary_matrix, (pad_rows, pad_cols))

    show = False
    if ax is None:
        show = True
        dpi = 100
        res = (750 / dpi, 750 / dpi)
        fig, ax = plt.subplots(figsize=res, dpi=dpi)
    ax.matshow(binary_matrix, cmap=ListedColormap([bg_color, color]))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which='minor', color=bg_color, linestyle='-', linewidth=3)
    ax.tick_params(which="both", bottom=False, top=False, left=False, right=False)
    plt.title(title)

    if save_filename:
        plt.savefig(save_filename)

    if show:
        plt.show()


def plot_shape(shape_id, *args, **kwargs):
    """
    Plots a shape.
    :param shape_id: int
    All other params (positional and keyword) are as in plot_binary_matrix
    """
    binary_matrix = shape_id_to_binary_matrix(shape_id)
    plot_binary_matrix(binary_matrix, *args, **kwargs)
