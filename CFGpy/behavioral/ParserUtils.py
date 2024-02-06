import json
import numpy as np
import pandas as pd

MIN_SAVE_FOR_EXPLOIT = 3


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


def check_decimino(coords: np.ndarray):
    """
        Takes a matrix of coordinates, if any row is not length 10, return False
        This is also supposed to check to see that each row is a list of 10 contiguous coordinates in discrete space
        The mathematica code doesn't check for this, I might add this later.
    """
    return coords.shape[1] == 10


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

            exploit_slices.append((exploit_start, exploit_end))
            if prev_exploit_end != exploit_start:
                explore_slices.append((prev_exploit_end, exploit_start))

            prev_exploit_end = exploit_end

    if exploit_slices == []:
        return [0, actions_df.shape[0]], []

    if exploit_slices[-1][1] != gallery_saves.size:
        explore_slices.append((exploit_slices[-1][1], actions_df.shape[0]))

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


def cast_list_of_tuple_to_ints(l):
    return list(map(lambda x: (int(x[0]), int(x[1])), l))


def shapeplot5a(s, backgroundcolor='black'):
    """
    Shows an image of a shape with grid coordinates s.
    Default background color is black
    """
    pass


def alledges():
    pass


def allshapes():
    """
    List containing all possible grid coordinates of shapes in the game
    """
    pass


def R():
    """
    Mapping from every grid coordinates of a shape to its serial number
    """
    pass


def backr():
    """
    backr is a dispatch (?) of a mapping from every shape's serial number
    to its grid coordinates
    """
    return


def g():
    """
    g is the graph of shapes
    """
    return


def symrule():
    """
    ?
    """
    return


def vanilla():
    """
    A parsed dataset of every subject so far that played without any \
    experimental manipulation
    """
    return


def all_gallery_sp():
    """
    AllGallerySP is a dispatch of a mapping from each pair of serial \
    numbers of shapes to the length of the shortest path between them. \
    Accounts only for shapes ever saved to gallery.
    """
    pass


def findclusters2(parsed_game_data):
    """
        findclusters2[parsedGameData] is a list where each element represents \
    an exploitation cluster in this game. In each such list, each element \
    is another list representing a single shape in the cluster. Each \
    shape list includes the following elements (ordered): \[Delta]t, \
    accumulated time, length of shortest path from previous shape, length \
    of actual path form previous shape, accumulated steps, shape id.
    """
    return


def phase_durations(clustered_game_data):
    """
    phaseDurations[clusteredGameData] is a list of two lists: all \
    explorations phase durations, all exploitation phase durations. \
    Expected input clusteredGameData is a single game clustered by \
    findclusters2
    """
    return


def segment_shapes(parsed_data, cluster_finder):
    """
    segmentShapes[parsedData, clusterFinder] returns a partition of the \
    shapes in parsedData to exploration and exploitation phases. The \
    output is of the form {exploreIdx, exploitIdx}. The clusterFinder \
    argument should be a callable wrapping a version of findclusters s.t. \
    it expects just the parsedData argumnt
    """
    return


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


def extract_measures(parsed_game_data):
    """
    extractMeasures[parsedGameData] returns a table where rows are \
    subjects and columns are the 30 standard CFG measures. The output of \
    this function includes a first row of labels for the columns.
    """
    return


def reduce_seqs(parsed_game_data):
    """
    reduceSeqs[parsedGameData] changes the shape series such that \
    sequences of the same shape are reduced to the first instance only. \
    If the sequence included gallery shapes, the reduced shape will be a \
    gallery shape with creation time like the first in the sequence and \
    save time like the first gallery in the sequence
    """
    return


def find_clusters_MRI(parsed_game_data, k, o):
    """
    findclustersMRI[ParsedGameData, k, o] will return a result similar to \
    findclusters2[ParsedGameData], but will nest k times instead of \
    twice, and will then join clusters if the optimality score of the \
    first shape in the latter is higher than o
    """
    return


def extract_measures_mri():
    "like extractMeasures, but uses findclusters2MRI"
    return
