import numpy as np
import pandas as pd
import json
import re
from _ctypes import PyObj_FromPtr
from ._consts import *


class CFGPipelineException(Exception):
    pass


def load_json(json_path):
    if not str.endswith(json_path, ".json"):
        raise CFGPipelineException("Data expected in json format")

    with open(json_path) as json_fp:
        j = json.load(json_fp)

    return j


def csv_coords_to_bin_coords(csv_coords):
    if csv_coords == "" or csv_coords is np.nan:
        return None

    if type(csv_coords) is str:
        csv_coords = json.loads(csv_coords)

    if type(csv_coords) is not list:
        raise TypeError('Received incorrect type as csv_coords, should be str or list, received', type(csv_coords))

    return coords_to_bin_coords(csv_coords)


def coords_to_bin_coords(coords):
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

    :param server_coords_str: as found in raw data from server, in customData.newShape col
    :return: list of ints
    """
    zero_centered = np.array(coords).T[::-1]
    zero_cornered = zero_centered - zero_centered.min(axis=1).reshape(2, -1)

    arr = np.zeros((10, 10))
    arr[zero_cornered[0], zero_cornered[1]] = 1
    shape_id_wth_0 = arr @ 2 ** np.arange(9, -1, -1)

    shape_id = [int(d) for d in shape_id_wth_0 if d != 0]
    return shape_id


def segment_explore_exploit(actions_list, min_save_for_exploit=MIN_SAVE_FOR_EXPLOIT):
    actions_df = pd.DataFrame(actions_list)

    gallery_saves = actions_df[2]
    gallery_indices = np.flatnonzero(gallery_saves.notna())
    gallery_times = actions_df.iloc[gallery_indices][2]
    gallery_diffs = np.diff(gallery_times)

    all_monotone_series = pd.Series(group_by_monotone_decreasing(gallery_diffs))
    gallery_diffs_peaks = np.array([gallery_diffs[monotone_series[0]] for monotone_series in all_monotone_series])
    twice_monotone_series = group_by_monotone_decreasing(gallery_diffs_peaks)
    merged_series = [np.concatenate(all_monotone_series[monotone_series].values) for monotone_series in
                     twice_monotone_series]

    exploit_slices = []
    explore_slices = []
    prev_exploit_end = 0
    for series in merged_series:
        if series.size >= min_save_for_exploit:
            exploit_start = gallery_indices[series][0]
            exploit_end = gallery_indices[series][-1] + 1

            exploit_slices.append((int(exploit_start), int(exploit_end)))
            if prev_exploit_end != exploit_start:
                explore_slices.append((int(prev_exploit_end), int(exploit_start)))

            prev_exploit_end = exploit_end

    if exploit_slices == []:
        return [0, actions_df.shape[0]], []

    if exploit_slices[-1][1] != gallery_saves.size:
        explore_slices.append((exploit_slices[-1][1], len(actions_df)))

    return explore_slices, exploit_slices


def group_by_monotone_decreasing(sequence):
    monotone_sequences = []
    current_sequence = [0]
    for i in range(1, sequence.size):
        if sequence[i - 1] < sequence[i]:
            monotone_sequences.append(current_sequence)
            current_sequence = [i]
        else:
            current_sequence.append(i)

    if current_sequence not in monotone_sequences:
        monotone_sequences.append(current_sequence)

    return monotone_sequences


def get_giant_component(clustered_data, min_shared_shapes_for_edges):
    """
    getGC[clusteredData,minSharedShapesForEdge] returns the Giant \
    Component of the sample, which is the biggest connected component of \
    the network of clusters, where two clusters are connected if they \
    share minSharedShapesForEdge shapes. Running the Girvan-Newman \
    algorithm on the GC finds the meaning categories discovered by this \
    sample.
    """
    return


#########################
# json formatting utils #
#########################
def prettify_games_json(parsed_games):
    prettified_games = []
    for game in parsed_games:
        game['actions'] = [NoIndent(action) for action in game['actions']]
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
