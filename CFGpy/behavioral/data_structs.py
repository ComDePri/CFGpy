import numpy as np
import pandas as pd
from itertools import pairwise, groupby, combinations
from collections import Counter
import networkx as nx
from _consts import *
from _utils import is_semantic_connection


def get_orig_map(counter, alpha=0, group_func=None):
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


class ParsedPlayerData:
    def __init__(self, player_data):
        self.id = player_data[PARSED_PLAYER_ID_KEY]
        self.start_time = player_data[PARSED_TIME_KEY]
        self.shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
        self.chosen_shapes = player_data[PARSED_CHOSEN_SHAPES_KEY]

        self.delta_move_times = np.diff(self.shapes_df.iloc[:, SHAPE_MOVE_TIME_IDX])

    def __len__(self):
        return len(self.shapes_df)

    def get_last_action_time(self):
        last_action_time = self.shapes_df.iloc[-1, SHAPE_SAVE_TIME_IDX]
        if np.isnan(last_action_time) or last_action_time is None:  # is None handles cases of 0-gallery games
            last_action_time = self.shapes_df.iloc[-1, SHAPE_MOVE_TIME_IDX]

        return last_action_time

    def get_max_pause_duration(self):
        return max(self.delta_move_times[3:-4])

    def get_steps(self):
        shape_ids = self.shapes_df.iloc[:, SHAPE_ID_IDX]
        without_empty_moves = [shape_id for shape_id, group in groupby(shape_ids)]
        # TODO: after empty moves are handled by Preprocessor, previous line is redundant and the next one can just run
        #  on shape_ids
        steps = list(pairwise(without_empty_moves))
        return steps

    def get_gallery_mask(self):
        """
        Returns a mask where true means gallery shape.
        :return: ndarray with shape (len(self.shapes_df),) and dtype bool.
        """
        return self.shapes_df.iloc[:, SHAPE_SAVE_TIME_IDX].notna()

    def get_gallery_ids(self):
        is_gallery = self.get_gallery_mask()
        gallery_ids = self.shapes_df[is_gallery].iloc[:, SHAPE_ID_IDX]
        return gallery_ids

    def get_self_avoidance(self):
        shape_ids = self.shapes_df.iloc[:, SHAPE_ID_IDX]
        unique_shapes = np.unique(shape_ids)
        shapes_no_consecutive_duplicates = [k for k, g in groupby(shape_ids)]
        # TODO: in a future version, consecutive duplicates will be handled by Preprocessor. when this is implemented,
        #  there will be no need for shapes_no_consecutive_duplicates, it will be  equivalent to shapes.
        #  Instead of its len, it would then make sense to use n_moves (defined in self.calc)

        return len(unique_shapes) / len(shapes_no_consecutive_duplicates)


class PreprocessedPlayerData(ParsedPlayerData):
    def __init__(self, player_data):
        self.explore_slices = player_data[EXPLORE_KEY]
        self.exploit_slices = player_data[EXPLOIT_KEY]
        super().__init__(player_data)

    def get_explore_mask(self):
        """
        Returns a mask where true means exploration.
        :return: ndarray with shape (len(self.shapes_df),) and dtype bool.
        """
        is_explore = np.zeros(len(self.shapes_df), dtype=bool)
        for start, end in self.explore_slices:
            is_explore[start:end] = True

        return is_explore

    def get_exploit_mask(self):
        """
        Returns a mask where true means exploitation.
        :return: ndarray with shape (len(self.shapes_df),) and dtype bool.
        """
        return ~self.get_explore_mask()

    def total_explore_time(self):
        time_in_explore = 0
        for start, end in self.explore_slices:
            start_time = self.shapes_df.iloc[start, SHAPE_MOVE_TIME_IDX]
            end_time = self.shapes_df.iloc[end - 1, SHAPE_MOVE_TIME_IDX]
            time_in_explore += (end_time - start_time)

        return time_in_explore

    def total_exploit_time(self):
        total_move_time = self.shapes_df.iloc[-1, SHAPE_MOVE_TIME_IDX]
        time_in_exploit = total_move_time - self.total_explore_time()

        return time_in_exploit

    def _phase_name_to_slices(self, phase_name):
        if phase_name == "explore":
            return self.explore_slices
        if phase_name == "exploit":
            return self.exploit_slices

        raise ValueError("phase_name must be 'explore' or 'exploit'")

    def get_phase_efficiency(self, phase_name):
        from CFGpy.utils import get_shortest_path_len  # moved here to avoid circular import

        phase_slices = self._phase_name_to_slices(phase_name)

        actual_paths = {}
        for start, end in phase_slices:
            slice_shape_ids = self.shapes_df.iloc[start:end, SHAPE_ID_IDX]
            slice_is_gallery = self.shapes_df.iloc[start:end, SHAPE_SAVE_TIME_IDX].notna()

            path_start = None
            path_len = 0
            for shape_id, is_gallery in zip(slice_shape_ids, slice_is_gallery):
                path_len += 1
                if is_gallery:
                    if path_start is not None:
                        actual_paths[(path_start, shape_id)] = path_len

                    path_start = shape_id
                    path_len = 0

        paths_efficiency = [actual_path / get_shortest_path_len(path_start, path_end)
                            for (path_start, path_end), actual_path in actual_paths.items() if path_start != path_end]
        # TODO: if path_start == path_end, the shortest path length is 0 and we get a zero division error. this only
        #  happens if the same shape is saved two times in a row, what should be handled as an empty move. when empty moves
        #  handling is implemented, the condition above should always be True and the if statement can be removed.
        return np.median(paths_efficiency)

    def get_clusters_in_phase(self, phase_name):
        phase_slices = self._phase_name_to_slices(phase_name)
        clusters = [tuple(self.shapes_df.iloc[start:end, SHAPE_ID_IDX]) for start, end in phase_slices]
        return clusters


class ParsedDataset:
    def __init__(self, input_data):
        self.input_data = []
        self.players_data = []
        self._reset_state(input_data)

    def _reset_state(self, input_data):
        self.input_data = input_data
        self.players_data = [ParsedPlayerData(player_data) for player_data in self.input_data]

    def __iter__(self):
        yield from self.players_data

    def __len__(self):
        return len(self.players_data)

    def drop_non_first_games(self):
        input_data = (pd.DataFrame(self.input_data).
                      sort_values(by=[PARSED_TIME_KEY], ascending=True).
                      drop_duplicates(subset=[PARSED_PLAYER_ID_KEY], keep="first").
                      to_dict("records"))
        self._reset_state(input_data)

    def filter(self, mask):
        """
        Filters the data.
        :param mask: True in indices to keep, False otherwise
        """
        input_data = (pd.DataFrame(self.input_data).
                      loc[mask].
                      to_dict("records"))
        self._reset_state(input_data)

    def step_counter(self):
        n_times_step_taken = Counter()
        n_players_took_step = Counter()

        for player_data in self.players_data:
            player_steps = player_data.get_steps()
            n_times_step_taken.update(player_steps)
            n_players_took_step.update(set(player_steps))

        return n_times_step_taken, n_players_took_step

    def gallery_counter(self):
        n_times_gallery_saved = Counter()
        n_players_saved_gallery = Counter()

        for player_data in self.players_data:
            player_steps = player_data.get_gallery_ids()
            n_times_gallery_saved.update(player_steps)
            n_players_saved_gallery.update(set(player_steps))

        return n_times_gallery_saved, n_players_saved_gallery

    @staticmethod
    def get_not_uniquely_covered(counter):
        uniquely_covered_galleries = [gallery_id for gallery_id, count in counter.items() if count > 1]
        return uniquely_covered_galleries

    def get_descriptors(self):
        n_times_step_taken, n_players_took_step = self.step_counter()
        n_times_gallery_saved, n_players_saved_gallery = self.gallery_counter()

        steps_not_uniquely_covered = self.get_not_uniquely_covered(n_players_took_step)
        step_orig_map = get_orig_map(n_times_step_taken, group_func=lambda step: step[0])
        galleries_not_uniquely_covered = self.get_not_uniquely_covered(n_players_saved_gallery)
        gallery_orig_map = get_orig_map(n_times_gallery_saved)

        return steps_not_uniquely_covered, step_orig_map, galleries_not_uniquely_covered, gallery_orig_map


class PreprocessedDataset(ParsedDataset):
    def __init__(self, input_data):
        """
        Expects a list of dicts like the output from Preprocessor.preprocess()
        """
        super().__init__(input_data)
        self._reset_state(input_data)

    def _reset_state(self, input_data):
        self.input_data = input_data
        self.players_data = [PreprocessedPlayerData(player_data) for player_data in self.input_data]

    def get_all_exploit_clusters(self):
        all_exploit_clusters = []
        for player_data in self.players_data:
            exploit_clusters = player_data.get_clusters_in_phase("exploit")
            all_exploit_clusters.extend(exploit_clusters)

        return all_exploit_clusters

    def _calc_giant_component(self):
        semantic_network = nx.Graph()
        exploit_clusters = self.get_all_exploit_clusters()
        edges = [(c1, c2) for c1, c2 in combinations(exploit_clusters, 2)
                 if is_semantic_connection(c1, c2)]
        semantic_network.add_edges_from(edges)
        GC = max(nx.connected_components(semantic_network), key=len)

        return GC

    def get_descriptors(self):
        """
        Returns the same descriptors as a ParsedDataset would, plus its GC
        :return: 5-tuple
        """
        giant_component = self._calc_giant_component()
        return super().get_descriptors() + (giant_component,)
