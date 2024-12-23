import numpy as np
import pandas as pd
import json
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from CFGpy.behavioral._consts import (SERVER_COORDS_TYPE_ERROR, EXPLORE_KEY, PRETTIFY_WARNING, PARSED_ALL_SHAPES_KEY,
                                      PARSED_CHOSEN_SHAPES_KEY, EXPLOIT_KEY)
from _ctypes import PyObj_FromPtr


class CFGPipelineException(Exception):
    pass


def load_json(json_path):
    with open(json_path) as json_fp:
        j = json.load(json_fp)
    return j


def server_coords_to_binary_shape(coords):
    """
    Wraps _server_coords_to_binary_shape to handle edge cases
    :param coords: as found in raw data from server, in customData.newShape col
    :return: list of ints if input is valid
    """
    if coords == "" or coords is np.nan:
        return None

    if type(coords) is str:
        coords = json.loads(coords)

    if type(coords) is not list:
        raise TypeError(SERVER_COORDS_TYPE_ERROR.format(type(coords)))

    return _server_coords_to_binary_shape(coords)


def _server_coords_to_binary_shape(coords):
    """
    Converts shape coordinates from str as saved on the server to a binary number list representation,
    We get a list of coordinates representing sqaures in a 10x10 matrix, we take each row of the matrix, and encode it as a binary sequence.
    So for example if I have 3 squares on the first row and 3 squares on the second row I'll look at them as [[1011000000],[1110000000]]
    The actual squares would look like this:

        x,o,x,x,o,o,o,o,o,o
        x,x,x,o,o,o,o,o,o,o
        o,o,o,o,o,o,o,o,o,o
        o,o,o,o,o,o,o,o,o,o
        o,o,o,o,o,o,o,o,o,o
        o,o,o,o,o,o,o,o,o,o
        o,o,o,o,o,o,o,o,o,o
        o,o,o,o,o,o,o,o,o,o
        o,o,o,o,o,o,o,o,o,o
        o,o,o,o,o,o,o,o,o,o

    :param coords: as found in raw data from server, in customData.newShape col
    :return: list of ints
    """
    zero_centered = np.array(coords).T[::-1]
    zero_cornered = zero_centered - zero_centered.min(axis=1).reshape(2, -1)

    arr = np.zeros((10, 10))
    arr[zero_cornered[0], zero_cornered[1]] = 1
    shape_id_wth_0 = arr @ 2 ** np.arange(9, -1, -1)

    shape_id = [int(d) for d in shape_id_wth_0 if d != 0]
    return shape_id


def segment_explore_exploit(shapes, shape_move_time_idx, shape_save_time_idx,
                            min_save_for_exploit) -> tuple[list, list]:
    """
    Returns a segmentation of the shapes to explore-exploit.
    :param shapes: 2D array-like
    :param shape_move_time_idx: the column in shapes containing move time. should be relayed from a Configuration object
    :param shape_save_time_idx: the column in shapes containing save time. should be relayed from a Configuration object
    :param min_save_for_exploit: minimal cluster size considered exploit. should be relayed from a Configuration object
    :return: explore_slices, exploit_slices
    """
    n_shapes = len(shapes)
    no_exploit_return_value = [(0, n_shapes)], []
    shapes_df = pd.DataFrame(shapes)

    gallery_saves = shapes_df[shape_save_time_idx]
    gallery_indices = np.flatnonzero(gallery_saves.notna())
    if not any(gallery_indices):
        return no_exploit_return_value

    gallery_times = shapes_df.iloc[gallery_indices][shape_move_time_idx]
    gallery_diffs = np.diff(gallery_times, prepend=gallery_times.iloc[0])

    clusters = []
    if gallery_diffs.size:
        all_monotone_series = pd.Series(group_by_monotone_decreasing(gallery_diffs))
        gallery_diffs_peaks = np.array([gallery_diffs[monotone_series[0]] for monotone_series in all_monotone_series])
        twice_monotone_series = group_by_monotone_decreasing(gallery_diffs_peaks)
        clusters = [np.concatenate(all_monotone_series[monotone_series].values)
                    for monotone_series in twice_monotone_series]

    exploit_slices = []
    explore_slices = []
    prev_exploit_end = 0
    for cluster in clusters:
        start = gallery_indices[cluster][0]
        end = gallery_indices[cluster][-1] + 1
        if cluster.size >= min_save_for_exploit:
            exploit_slices.append((int(start), int(end)))
            if prev_exploit_end != start:
                explore_slices.append((int(prev_exploit_end), int(start)))

            prev_exploit_end = end

    if not exploit_slices:
        return no_exploit_return_value

    exploit_end = exploit_slices[-1][1]
    explore_end = explore_slices[-1][1]
    if explore_end < exploit_end < n_shapes:
        explore_slices.append((exploit_end, n_shapes))
    elif exploit_end < explore_end < n_shapes:
        last_explore_slice = (explore_slices[-1][0], n_shapes)
        explore_slices[-1] = last_explore_slice

    return explore_slices, exploit_slices


def group_by_monotone_decreasing(sequence):
    monotone_sequences = []
    current_sequence = [0]
    for i in range(1, sequence.size):
        if sequence[i - 1] <= sequence[i]:
            monotone_sequences.append(current_sequence)
            current_sequence = [i]
        else:
            current_sequence.append(i)

    if current_sequence not in monotone_sequences:
        monotone_sequences.append(current_sequence)

    return monotone_sequences


def is_semantic_connection(cluster1, cluster2, min_overlap_for_semantic_connection):
    return len(set(cluster1) & set(cluster2)) >= min_overlap_for_semantic_connection


def plot_gallery_dt(postparsed_player_data, shape_move_time_idx):
    """
    Visualize explore-exploit segmentation, by plotting time difference between galleries as a function of time.
    :param postparsed_player_data: behavioral.data_classes.PostparsedPlayerData. No type hinting here, to avoid
    having to import that class in this file.
    :param shape_move_time_idx: the column in shapes containing move time. should be relayed from a Configuration object
    """
    is_gallery = postparsed_player_data.get_gallery_mask()
    gallery_times = postparsed_player_data.shapes_df[is_gallery].iloc[:, shape_move_time_idx]
    gallery_diffs = np.diff(gallery_times, prepend=gallery_times.iloc[0])

    cluster_label = np.empty(len(postparsed_player_data.shapes_df), dtype=object)
    phase_type = np.empty(len(postparsed_player_data.shapes_df), dtype=object)
    for phase_name, phase_slices in zip(("explore", "exploit"),
                                        (postparsed_player_data.explore_slices, postparsed_player_data.exploit_slices)):
        for i, (start, end) in enumerate(phase_slices):
            cluster_label[start:end] = f"{phase_name}{i}"
            phase_type[start:end] = phase_name

    df = pd.DataFrame({"gallery_time": gallery_times, "gallery_diff": gallery_diffs,
                       "cluster": cluster_label[is_gallery], "phase_type": phase_type[is_gallery]})
    fig, ax = plt.subplots(figsize=(10, 5))
    marker_dict = {"explore": "o", "exploit": "X"}
    sns.lineplot(data=df, ax=ax, x="gallery_time", y="gallery_diff", hue="cluster", style="phase_type",
                 markers=marker_dict, markersize=10)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_markers = dict(zip(labels, handles))
    plt.legend((unique_markers["explore"], unique_markers["exploit"]), ("explore", "exploit"))
    plt.xlabel(r"$t$ (s)", fontsize=14)
    plt.ylabel(r"$\Delta t$ (s)", fontsize=14)
    plt.suptitle("Gallery shapes creation time, segmented by clusters", fontsize=16)
    plt.title(f"Player ID: {postparsed_player_data.id}")
    plt.grid()
    plt.show()


#########################
# json formatting utils #
#########################
def prettify_games_json(parsed_games):
    warnings.warn(PRETTIFY_WARNING)
    prettified_games = []
    for game in parsed_games:
        game[PARSED_ALL_SHAPES_KEY] = [NoIndent(action) for action in game[PARSED_ALL_SHAPES_KEY]]
        chosen_shapes = game.get(PARSED_CHOSEN_SHAPES_KEY, None)
        if chosen_shapes is not None:
            game[PARSED_CHOSEN_SHAPES_KEY] = [NoIndent(chosen_shape) for chosen_shape in chosen_shapes]

        explore = game.get(EXPLORE_KEY, None)
        if explore:
            game[EXPLORE_KEY] = NoIndent(explore)

        exploit = game.get(EXPLOIT_KEY, None)
        if exploit:
            game[EXPLOIT_KEY] = NoIndent(exploit)

        prettified_games.append(game)

    return json.dumps(prettified_games, cls=CustomIndentEncoder, sort_keys=True, indent=4)


# Using the answer from here https://stackoverflow.com/a/13252112 to make a prettier json file
class NoIndent(object):
    """ Value wrapper. """

    def __init__(self, value):
        self.value = value


class CustomIndentEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(CustomIndentEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(CustomIndentEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(CustomIndentEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr
