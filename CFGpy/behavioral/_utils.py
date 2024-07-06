import numpy as np
import pandas as pd
import json
import re
from _ctypes import PyObj_FromPtr
from CFGpy.behavioral._consts import *


class CFGPipelineException(Exception):
    pass


def load_json(json_path):
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

def segment_explore_exploit(shapes, min_save_for_exploit=MIN_SAVE_FOR_EXPLOIT, min_efficieny=MIN_EFFICIENCY_FOR_EXPLOIT, max_pace=MAX_PACE_FOR_MERGE):
    # The logic here is this:
    # 1. Group shapes by decreasing save-time differences (difference from previous gallery shape)
    # 2. Group these clusters based on efficiency (merge consecutive clusters if there is high efficiency between last shape of cluster A and first shape of cluster B)
    # Repeat steps 1-2 as follows:
    # 3. Group the resulting clusters based on save-time differences
    # 4. Group these clusters based on efficiency
    # Determine the number of shapes
    n_shapes = len(shapes)
    
    # Default return value if no "exploit" clusters are found
    no_exploit_return_value = [(0, n_shapes)], []
    
    # Convert shapes data to a pandas DataFrame
    shapes_df = pd.DataFrame(shapes)

    # Extract the save times of shapes
    gallery_saves = shapes_df[SHAPE_SAVE_TIME_IDX]
    
    # Find indices where shapes have save times
    gallery_indices = np.flatnonzero(gallery_saves.notna())
    
    # Return the default value if no shapes were saved
    if not any(gallery_indices):
        return no_exploit_return_value

    # Get the in and out times of shapes based on save indices
    gallery_in_times = shapes_df.iloc[gallery_indices][SHAPE_MOVE_TIME_IDX]
    gallery_out_times = shapes_df.iloc[gallery_indices][SHAPE_MAX_MOVE_TIME_IDX]
    
    # Calculate time differences between consecutive shape movements
    gallery_diffs = gallery_in_times - gallery_out_times.shift()
    gallery_diffs.iloc[0] = 0  # Set the first difference to 0 (no previous shape to compare with)
    gallery_diffs = gallery_diffs.to_numpy()

    # Get the pace, the mean step time between consecutive gallery shapes (in seconds per step)
    # TODO: is this really the way to go? Do we need to make sure we use in-steps and out-steps? I don't think so. If a change required 1 step but actually do to duplicate shapes took 4 steps, then we would like efficiency to capture this. If we include the duplicate steps in the calculation, the pace will be a faster one (there is less time for each step) and this won't count as a very slow transition that will undo the efficiency.
    gallery_steps_diffs = gallery_indices - np.roll(gallery_indices,1)
    gallery_steps_diffs[0] = 0  # Set the first difference to 0 (no previous shape to compare with)
    gallery_pace = np.nan_to_num(gallery_diffs/gallery_steps_diffs)

    # Initialize the list of clusters
    clusters = []
    if gallery_diffs.size:

        # Group differences into monotone decreasing sequences
        all_monotone_series = pd.Series(group_by_monotone_decreasing(gallery_diffs,gallery_pace))

        group_by_efficiency_twice = False
        if group_by_efficiency_twice:
            # Group clusters based on efficiency (First time)
            all_monotone_series_list = [all_monotone_series[i] for i in range(len(all_monotone_series))]
            all_monotone_series_list = group_by_efficiency(all_monotone_series_list, shapes_df, gallery_indices, min_efficieny)
            # Calculate the peaks (first elements) of each monotone sequence
            gallery_diffs_peaks = np.array([gallery_diffs[monotone_series[0]] for monotone_series in all_monotone_series_list])
        else:
            # Calculate the peaks (first elements) of each monotone sequence
            gallery_diffs_peaks = np.array([gallery_diffs[monotone_series[0]] for monotone_series in all_monotone_series])
            # Get the relevant pace measures for controlling the merge of clusters based on first peaks in time-difference
            gallery_pace_first_shapes = np.array([gallery_pace[monotone_series[0]] for monotone_series in all_monotone_series])

        # Group these peaks into further monotone decreasing sequences
        twice_monotone_series = group_by_monotone_decreasing(gallery_diffs_peaks, gallery_pace_first_shapes)

        # Form clusters by concatenating the original monotone sequences
        clusters = [np.concatenate(all_monotone_series[monotone_series].values)
                    for monotone_series in twice_monotone_series]

        # Add code to group sequences based on efficiency
        clusters = group_by_efficiency(clusters, shapes_df, gallery_indices, min_efficieny, max_pace)

    # Initialize lists for exploit and explore slices
    exploit_slices = []
    explore_slices = []
    prev_exploit_end = 0  # Keep track of the end of the previous exploit cluster

    # Process each cluster
    for cluster in clusters:
        start = gallery_indices[cluster][0]  # Start index of the current cluster (indices are in terms of all the shapes visited in the game)
        end = gallery_indices[cluster][-1] + 1  # End index of the current cluster (indices are in terms of all the shapes visited in the game)

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
    return explore_slices, exploit_slices

def group_by_efficiency(clusters, shapes_df, gallery_indices, min_efficiency, max_pace):
    """
    Groups sequences based on the criterion that the efficiency measure between 
    the last shape of the first sequence and the first shape of the following sequence 
    is greater than 'min_efficiency'.
    """
    from CFGpy.utils import get_shortest_path_len  # Moved here to avoid circular import
    
    # Create a list to store the new grouped clusters
    grouped_clusters = []
    current_cluster = clusters[0]

    for i in range(1, len(clusters)):
        prev_shape_idx = current_cluster[-1]
        next_shape_idx = clusters[i][0]
        
        prev_shape_id = shapes_df.iloc[gallery_indices[prev_shape_idx], SHAPE_ID_IDX]
        next_shape_id = shapes_df.iloc[gallery_indices[next_shape_idx], SHAPE_ID_IDX]

        # Calculate the shortest path length between shapes
        shortest_path_len = get_shortest_path_len(prev_shape_id, next_shape_id)
        actual_path_len = gallery_indices[next_shape_idx] - gallery_indices[prev_shape_idx]

        # Calculate efficiency
        efficiency = shortest_path_len / actual_path_len

        # Calculate time per step between potential merges
        prev_shape_gallery_out_time = shapes_df.iloc[gallery_indices[prev_shape_idx]][SHAPE_MAX_MOVE_TIME_IDX]
        next_shape_gallery_in_times = shapes_df.iloc[gallery_indices[next_shape_idx]][SHAPE_MOVE_TIME_IDX]
        # Calculate time differences between consecutive shape movements
        time_diff = next_shape_gallery_in_times - prev_shape_gallery_out_time
        # Calculate the steps difference between them
        steps_diff = gallery_indices[next_shape_idx] - gallery_indices[prev_shape_idx]
        # Calculate the average time per step between the two shapes
        pace = time_diff / steps_diff


        # Check if efficiency meets the minimum required efficiency
        if (efficiency >= min_efficiency) and (pace < max_pace):
            # Combine clusters if efficiency is sufficient
            current_cluster = np.concatenate((current_cluster, clusters[i]))
        else:
            # Add current cluster to the grouped clusters list and start a new cluster
            grouped_clusters.append(current_cluster)
            current_cluster = clusters[i]
    
    # Add the last cluster to the grouped clusters list
    grouped_clusters.append(current_cluster)
    
    return grouped_clusters


def group_by_monotone_decreasing(sequence, pace, max_pace=MAX_PACE_FOR_MERGE):
    # pace is the same length as "sequence" but it includes that pace (time per step) between each gallery shape and the previous one
    monotone_sequences = []
    current_sequence = [0]
    for i in range(1, sequence.size):
        if sequence[i - 1] < sequence[i]: # If the current shape breaks the monotone sequence
            monotone_sequences.append(current_sequence)
            current_sequence = [i]
        elif pace[i] > max_pace: # If the current shape continues a monotone sequence, make sure it survives the pace criterion
            monotone_sequences.append(current_sequence)
            current_sequence = [i]
        else: # This shape will be a sequence of its own, as it does not
            current_sequence.append(i)

    if current_sequence not in monotone_sequences:
        monotone_sequences.append(current_sequence)

    return monotone_sequences


def is_semantic_connection(cluster1, cluster2):
    return len(set(cluster1) & set(cluster2)) >= MIN_OVERLAP_FOR_SEMANTIC_CONNECTION


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
