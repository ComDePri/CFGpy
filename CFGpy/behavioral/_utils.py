import copy
from copy import deepcopy

import numpy as np
import pandas as pd
import json
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from CFGpy.behavioral._consts import (SERVER_COORDS_TYPE_ERROR, EXPLORE_KEY, PRETTIFY_WARNING, PARSED_ALL_SHAPES_KEY,
                                      PARSED_CHOSEN_SHAPES_KEY, EXPLOIT_KEY, RM1)
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


def segment_explore_exploit_mri(shapes, min_save_for_exploit, min_efficiency, max_pace,
                            shape_save_time_idx, shape_move_time_idx, shape_max_move_time_idx,
                            shape_id_index):
    """
    This function groups clusters based on efficiency.
    The logic here is this:
        1. Group shapes by decreasing save-time differences (difference from previous gallery shape)
        2. Group these clusters based on efficiency
        (merge consecutive clusters if there is high efficiency between last shape of cluster A and first shape of cluster B)
        Repeat steps 1-2 as follows:
        3. Group the resulting clusters based on save-time differences
        4. Group these clusters based on efficiency
        Determine the number of shapes
        :param shapes: the shapes the user created - still raw (not pandas DataFrame)
        :param min_save_for_exploit: minimal cluster size considered exploit
        :param min_efficiency: minimal efficiency to consider merging two clusters
        :param max_pace: maximal pace (seconds per step) to consider merging two clusters
        :param shape_save_time_idx: the column in shapes data frame containing save time.
        Relayed from a Configuration object
        :param shape_move_time_idx: the column in shapes data frame containing move time.
         It's the first time the shape appeared. Relayed from a Configuration object
         :param shape_max_move_time_idx: the column in shapes data frame containing the last move time
          that resulted in that same shape.
        :return: new list of clusters after grouping by efficiency
    """
    n_shapes = len(shapes)

    # Default return value if no "exploit" clusters are found
    # TODO: Roey, remove the np.nan's and make sure we only try to extract the other outputs when using the pace criterion
    # TODO: done, by Zohar - pace criterion is true always for mri cases
    no_exploit_return_value = [(0, n_shapes)], [], np.nan, np.nan
    # Convert shapes data to a pandas DataFrame
    shapes_df = pd.DataFrame(shapes)

    # Extract the save times of shapes
    gallery_saves = shapes_df[shape_save_time_idx]
    # Find indices where shapes have save times
    gallery_indices = np.flatnonzero(gallery_saves.notna())

    # Return the default value if no shapes were saved
    if not any(gallery_indices):
        return no_exploit_return_value

    # Get the in and out times of shapes based on save indices
    gallery_in_times = shapes_df.iloc[gallery_indices][shape_move_time_idx]
    gallery_out_times = shapes_df.iloc[gallery_indices][shape_max_move_time_idx]

    # Calculate time differences between consecutive shape movements
    gallery_diffs = gallery_in_times - gallery_out_times.shift()
    # Set the first difference to 0 (no previous shape to compare with)
    gallery_diffs.iloc[0] = 0
    gallery_diffs = gallery_diffs.to_numpy()
    # now we remove the empty steps times for all intermediate shapes between each pair of gallery shapes.
    empty_steps_time = shapes_df.iloc[:, shape_max_move_time_idx] - shapes_df.iloc[:, shape_move_time_idx]
    gallery_diffs_fixed = gallery_diffs
    for gI, gallery_idx in enumerate(
            gallery_indices[:-1]):  # [:-1] makes sure we exclude the last element
        start_idx = gallery_idx + 1
        end_idx = gallery_indices[gI + 1] - 1
        gallery_diffs_fixed[gI + 1] = gallery_diffs[gI + 1] - sum(empty_steps_time[start_idx:end_idx])
    gallery_diffs = gallery_diffs_fixed

    # Get the pace, the mean step time between consecutive gallery shapes (in seconds per step)
    gallery_steps_diffs = gallery_indices - np.roll(gallery_indices, 1)
    gallery_steps_diffs[0] = 0  # Set the first difference to 0 (no previous shape to compare with)
    gallery_pace = np.nan_to_num(gallery_diffs / gallery_steps_diffs)

    # Initialize the list of clusters
    clusters = []
    robust_median_value = np.nan
    if gallery_diffs.size:

        # ** Experimental - define subject sepcific threshold with robust medians **
        #  this code runs because pace_criterion is always true for mri cases
        # Get gallery shapes based on the original algorithm
        """
        In the legacy "Aviv" branch code, we were calling this current method recursively, and by that 
        excluding the empty steps and using all the logics above.
        Here, we call the STANDARD segment_explore_exploit, which gives us a better upper-bound on the pace threshold,
        keeps the code cleaner and maintainable, and avoids possible bugs from recursive calls.
    
        AI-generated explanation for this decision:
            If we use the Legacy method, we might calibrate based on segments where the player was actually thinking for a long time (staring),
             just because their hand moved fast. This could pollute the "Pace" baseline with "Thinking" data.
    
            Using the Standard function ensures we only calibrate using segments where the player was fast both physically and mentally.
             This provides a cleaner, higher-confidence signal for what "Maximum Speed" looks like for this subject.
        """

        # def segment_explore_exploit(shapes, shape_move_time_idx, shape_save_time_idx,
        #                             min_save_for_exploit) -> tuple[list, list]:
        explore, exploit = segment_explore_exploit(shapes,
                                                   shape_move_time_idx,
                                                   shape_save_time_idx,
                                                   min_save_for_exploit)

        # Get indices of gallery shapes within each exploit segment, ignoring the first of each exploit segment
        exploit_indices = get_exploit_indices_excluding_first_per_segment(exploit, gallery_indices)

        if len(exploit_indices) == 0:
            max_pace = np.nan
        else:
            robust_median_value, robust_mad_value = robust_median(gallery_pace[exploit_indices])
            max_pace = robust_median_value + 5 * robust_mad_value

        # Group differences into monotone decreasing sequences
        all_monotone_series = pd.Series(group_by_monotone_decreasing(gallery_diffs, gallery_pace, max_pace))

        # Calculate the peaks (first elements) of each monotone sequence
        gallery_diffs_peaks = np.array(
            [gallery_diffs[monotone_series[0]] for monotone_series in all_monotone_series])
        # Get the relevant pace measures for controlling the merge of clusters based on first peaks in time-difference
        gallery_pace_first_shapes = np.array(
            [gallery_pace[monotone_series[0]] for monotone_series in all_monotone_series])

        # Group these peaks into further monotone decreasing sequences
        twice_monotone_series = group_by_monotone_decreasing(gallery_diffs_peaks, gallery_pace_first_shapes,
                                                             max_pace)

        # Form clusters by concatenating the original monotone sequences
        clusters = [np.concatenate(all_monotone_series[monotone_series].values)
                    for monotone_series in twice_monotone_series]

        # Add code to group sequences based on efficiency (unless the pace is too slow)
        clusters = group_by_efficiency(clusters, shapes_df,
                                       gallery_indices, min_efficiency, max_pace,
                                       shape_id_index, shape_move_time_idx,
                                       shape_max_move_time_idx)

    # Initialize lists for exploit and explore slices
    exploit_slices = []
    explore_slices = []
    prev_exploit_end = 0  # Keep track of the end of the previous exploit cluster
    # Process each cluster
    for cluster in clusters:
        # Start index of the current cluster (indices are in terms of all the shapes visited in the game)
        start = gallery_indices[cluster][0]
        # End index of the current cluster (indices are in terms of all the shapes visited in the game)
        end = gallery_indices[cluster][-1] + 1
        # Check if the cluster meets the criteria for "exploit"
        if cluster.size >= min_save_for_exploit:
            exploit_slices.append((int(start), int(end)))

            # If there is a gap between the previous exploit and the current, mark this gap as "explore"
            if prev_exploit_end != start:
                explore_slices.append((int(prev_exploit_end), int(start)))

            # Update the end of the last exploit cluster
            prev_exploit_end = end

    # If no exploit slices were identified, return the default value
    if not exploit_slices:
        return no_exploit_return_value

    # Determine the end indices of the last exploit and explore slices
    exploit_end = exploit_slices[-1][1]
    explore_end = explore_slices[-1][1] if explore_slices else 0

    # Handle the case where there are remaining shapes after the last exploit slice
    if explore_end < exploit_end < n_shapes:
        explore_slices.append((exploit_end, n_shapes))
    elif exploit_end < explore_end < n_shapes:
        # Extend the last explore slice to the end of the shapes
        last_explore_slice = (explore_slices[-1][0], n_shapes)
        explore_slices[-1] = last_explore_slice

    # Return the identified explore and exploit slices
    return explore_slices, exploit_slices, robust_median_value, max_pace

def get_exploit_indices_excluding_first_per_segment(exploit, gallery_indices):
    # This function loops over all exploit segment in "exploit" and returns the indices of the gallery shapes that are
    # part of exploit segments. In also removes the first exploit shape in each segment. This is because we want to
    # use those gallery shapes for calculating the median pace in exploit, and we assume that the pace from explore to
    # the first exploit shape might be different from  the pace strictly within an exploit segment.

    # Convert gallery_indices to a numpy array for easier indexing
    gallery_indices = np.array(gallery_indices)

    all_indices = []

    for (start, end) in exploit:
        # Identify indices within the range
        within_range_indices = np.where((gallery_indices >= start) & (gallery_indices <= end))[0]

        # If there are indices in the range, remove the first one
        if len(within_range_indices) > 0:
            within_range_indices = within_range_indices[1:]

        # Add the remaining indices to the all_indices list
        all_indices.extend(within_range_indices)

    # Flatten the list of indices (if not already flat)
    all_indices = np.array(all_indices).flatten()

    return all_indices

def median_absolute_deviation(data, median):
    return np.median(np.abs(data - median))


def is_outlier(data, median, mad):
    return np.abs(data - median) > (5 * mad)


def robust_median(data):
    data = np.array(data)
    median = np.median(data)
    mad = median_absolute_deviation(data, median)

    while True:
        mask = ~is_outlier(data, median, mad)
        if np.all(mask):
            break
        data = data[mask]
        median = np.median(data)
        mad = median_absolute_deviation(data, median)

    #  mask isn't being used in the original code in aviv branch,
    #  but keeping it here in case it's needed later
    mask = ~is_outlier(data, median, mad)
    # print(data)
    return median, mad


def group_by_efficiency(clusters, shapes_df,
                        gallery_indices, min_efficiency,
                        max_pace, shape_id_index,
                        shape_move_time_idx, shape_max_move_time_idx):
    """
    Groups sequences based on the criterion that the efficiency measure between
    the last shape of the first sequence and the first shape of the following sequence
    is greater than 'min_efficiency'.
    """
    from CFGpy.utils import get_shortest_path_len  # Moved here to avoid circular import

    # Create a list to store the new grouped clusters
    grouped_clusters = []
    current_cluster = clusters[0]

    # Calculate time spent on "empty steps" in each (not necessarily gallery) shape (remove_empty_steps_time)
    empty_steps_time = shapes_df.iloc[:, shape_max_move_time_idx] - shapes_df.iloc[:, shape_move_time_idx]

    for i in range(1, len(clusters)):
        prev_shape_idx = current_cluster[-1]
        next_shape_idx = clusters[i][0]

        prev_shape_id = shapes_df.iloc[gallery_indices[prev_shape_idx], shape_id_index]
        next_shape_id = shapes_df.iloc[gallery_indices[next_shape_idx], shape_id_index]

        # Calculate the shortest path length between shapes
        shortest_path_len = get_shortest_path_len(prev_shape_id, next_shape_id)
        actual_path_len = gallery_indices[next_shape_idx] - gallery_indices[prev_shape_idx]

        # Calculate efficiency
        efficiency = shortest_path_len / actual_path_len

        # Calculate time per step between potential merges
        prev_shape_gallery_out_time = shapes_df.iloc[gallery_indices[prev_shape_idx]][shape_max_move_time_idx]
        next_shape_gallery_in_times = shapes_df.iloc[gallery_indices[next_shape_idx]][shape_move_time_idx]
        # Calculate time differences between consecutive shape movements
        time_diff = next_shape_gallery_in_times - prev_shape_gallery_out_time

        # Reduce the time spent on "empty steps" between each pair of gallery shapes (remove_empty_steps_time)
        start_idx = gallery_indices[prev_shape_idx] + 1
        end_idx = gallery_indices[next_shape_idx] - 1
        time_diff = time_diff - sum(empty_steps_time[start_idx:end_idx])

        # Calculate the steps difference between them
        steps_diff = gallery_indices[next_shape_idx] - gallery_indices[prev_shape_idx]
        # Calculate the average time per step between the two shapes
        pace = time_diff / steps_diff

        # Check if efficiency meets the minimum required efficiency
        if (efficiency >= min_efficiency) and (pace < max_pace):  # and (efficiency < 1):
            # Combine clusters if efficiency is sufficient
            current_cluster = np.concatenate((current_cluster, clusters[i]))
        else:
            # Add current cluster to the grouped clusters list and start a new cluster
            grouped_clusters.append(current_cluster)
            current_cluster = clusters[i]

    # Add the last cluster to the grouped clusters list
    grouped_clusters.append(current_cluster)

    return grouped_clusters

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
    prettified_games = []
    prettified_games = '[\n    '
    for game in parsed_games:
        game = copy.deepcopy(game)
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

        prettified_game = json.dumps(game, cls=CustomIndentEncoder, sort_keys=True, indent=4)
        indented_prettified_game = prettified_game.replace('\n', '\n    ')
        prettified_games += indented_prettified_game
        prettified_games += ',\n    '

    prettified_games = prettified_games[:-6] + '\n]'

    return prettified_games


def _old_prettify_games_json(parsed_games):
    warnings.warn(PRETTIFY_WARNING)
    parsed_games = deepcopy(parsed_games)
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


def version_to_tuple(version_string):
    return tuple(map(int, version_string.split('.')))


def get_default_data_source(cfgpy_version: str | None = None) -> str:
    # retrieve the correct data source based on the given cfgpy version
    if cfgpy_version is None:
        from CFGpy._version import __version__ as cfgpy_version
    cfgpy_version_tuple = version_to_tuple(cfgpy_version)
    if cfgpy_version_tuple < version_to_tuple("1.0.1"):
        return RM1
    else: # should be updated once we migrate to a stable platform for data storage
        raise ValueError(f"Unsupported CFGpy version: {cfgpy_version}. No default data source available.")

def parse_json_column(*, df: pd.DataFrame, column_name: str, prefix: str):
    """Parse JSON columns in the DataFrame"""

    def try_parse(val):
        if pd.isna(val):
            return {}
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return {}

    parsed_df = df[column_name].apply(try_parse).apply(pd.Series)
    parsed_df.columns = [f"{prefix}.{col}" for col in parsed_df.columns]

    return pd.concat([df.drop(columns=[column_name]), parsed_df], axis=1)
