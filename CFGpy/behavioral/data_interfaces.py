import numpy as np
import pandas as pd
from itertools import pairwise, groupby, combinations
from collections import Counter
import networkx as nx
from CFGpy.behavioral._consts import (PARSED_PLAYER_ID_KEY, PARSED_TIME_KEY, PARSED_ALL_SHAPES_KEY,
                                      PARSED_CHOSEN_SHAPES_KEY, EXPLORE_KEY, EXPLOIT_KEY)
from CFGpy.behavioral import Configuration
from CFGpy.behavioral._utils import is_semantic_connection, load_json


# TODO: consider: some methods only serve MeasureCalculator, while other are meant as API for end users (e.g.
#  visualizations). This probably means there's a better way to design these classes

class ParsedPlayerData:
    def __init__(self, player_data, config: Configuration = None):
        self.id = player_data[PARSED_PLAYER_ID_KEY]
        self.start_time = player_data[PARSED_TIME_KEY]
        self.shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
        self.chosen_shapes = player_data.get(PARSED_CHOSEN_SHAPES_KEY)

        self.config = config if config is not None else Configuration.default()
        self.delta_move_times = np.diff(self.shapes_df.iloc[:, self.config.SHAPE_MOVE_TIME_IDX])

    def __len__(self):
        return len(self.shapes_df)

    def get_last_action_time(self):
        last_action_time = self.shapes_df.iloc[-1, self.config.SHAPE_SAVE_TIME_IDX]
        if last_action_time is None or np.isnan(last_action_time):  # is None handles cases of 0-gallery games
            last_action_time = self.shapes_df.iloc[-1, self.config.SHAPE_MOVE_TIME_IDX]

        return last_action_time

    def get_max_pause_duration(self):
        margin = self.config.MARGIN_FOR_PAUSE_DURATION
        truncated_deltas = self.delta_move_times[margin:-margin]
        if len(truncated_deltas):
            return max(truncated_deltas)
        return None

    def get_steps(self):
        shape_ids = self.shapes_df.iloc[:, self.config.SHAPE_ID_IDX]
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
        return self.shapes_df.iloc[:, self.config.SHAPE_SAVE_TIME_IDX].notna()

    def get_gallery_ids(self):
        is_gallery = self.get_gallery_mask()
        gallery_ids = self.shapes_df[is_gallery].iloc[:, self.config.SHAPE_ID_IDX]
        return gallery_ids

    def get_self_avoidance(self):
        shape_ids = self.shapes_df.iloc[:, self.config.SHAPE_ID_IDX]
        unique_shapes = np.unique(shape_ids)
        shapes_no_consecutive_duplicates = [k for k, g in groupby(shape_ids)]
        # TODO: in a future version, consecutive duplicates will be handled by Preprocessor. when this is implemented,
        #  there will be no need for shapes_no_consecutive_duplicates, it will be  equivalent to shapes.
        #  Instead of its len, it would then make sense to use n_moves (defined in self.calc)

        return len(unique_shapes) / len(shapes_no_consecutive_duplicates)


class PostparsedPlayerData(ParsedPlayerData):
    def __init__(self, player_data, config: Configuration = None):
        self.explore_slices = player_data[EXPLORE_KEY]
        self.exploit_slices = player_data[EXPLOIT_KEY]
        super().__init__(player_data, config)

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
        return self.get_last_action_time() - self.total_exploit_time()

    def total_exploit_time(self):
        time_in_exploit = 0
        for start, end in self.exploit_slices:
            start_time = self.shapes_df.iloc[start, self.config.SHAPE_MOVE_TIME_IDX]
            end_time = self.shapes_df.iloc[end - 1, self.config.SHAPE_MOVE_TIME_IDX]
            time_in_exploit += (end_time - start_time)

        return time_in_exploit

    def get_efficiency(self):
        from CFGpy.utils import get_shortest_path_len  # moved here to avoid circular import

        gallery_indices = np.flatnonzero(self.get_gallery_mask())
        if not gallery_indices.size:
            return np.nan, np.nan

        actual_path_lengths = np.diff(gallery_indices, prepend=0)
        gallery_ids = self.shapes_df.iloc[gallery_indices, self.config.SHAPE_ID_IDX]
        shortest_path_lengths = ([get_shortest_path_len(self.config.FIRST_SHAPE_ID, gallery_ids.iloc[0])] +
                                 [get_shortest_path_len(shape1, shape2) for shape1, shape2 in pairwise(gallery_ids)])
        all_efficiency_values = {idx: shortest_len / actual_len for idx, shortest_len, actual_len
                                 in zip(gallery_indices, shortest_path_lengths, actual_path_lengths)
                                 if (shortest_len, actual_len) != (1, 1)}
        # TODO: after empty steps are handled, shortest paths would not be able to be 0 and the condition can be
        #  simplified to actual_len>1

        explore_efficiencies = []
        exploit_efficiencies = []
        is_explore = self.get_explore_mask()
        for idx, efficiency in all_efficiency_values.items():
            if is_explore[idx]:
                explore_efficiencies.append(efficiency)
            else:
                exploit_efficiencies.append(efficiency)

        return np.median(explore_efficiencies), np.median(exploit_efficiencies)

    def get_exploit_clusters(self):
        is_gallery = self.get_gallery_mask()
        clusters = []
        for start, end in self.exploit_slices:
            slice_is_gallery = is_gallery[start:end]
            slice_shape_ids = self.shapes_df.iloc[start:end, self.config.SHAPE_ID_IDX]
            cluster = tuple(slice_shape_ids[slice_is_gallery])
            clusters.append(cluster)

        return clusters


class ParsedDataset:
    def __init__(self, input_data):
        self.input_data = []
        self.players_data = []
        self._reset_state(input_data)

    @classmethod
    def from_json(cls, path: str):
        return cls(load_json(path))

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
        not_uniquely_covered = [key for key, count in counter.items() if count > 1]
        return not_uniquely_covered

    def get_stats(self):
        """
        Returns the necessary information for extraction of features relative to this dataset. Namely:
         steps_not_uniquely_covered: steps that more than one player took.
         n_times_step_taken: a mapping from a step to the number of time it was taken.
         galleries_not_uniquely_covered: shapes that more than one player saved to gallery.
         n_times_gallery_saved: a mapping from a shape to the number of time it was saved to gallery.
        """
        n_times_step_taken, n_players_took_step = self.step_counter()
        n_times_gallery_saved, n_players_saved_gallery = self.gallery_counter()

        steps_not_uniquely_covered = self.get_not_uniquely_covered(n_players_took_step)
        galleries_not_uniquely_covered = self.get_not_uniquely_covered(n_players_saved_gallery)

        return steps_not_uniquely_covered, n_times_step_taken, galleries_not_uniquely_covered, n_times_gallery_saved


class PostparsedDataset(ParsedDataset):
    def __init__(self, input_data, config: Configuration = None):
        """
        Expects a list of dicts like the output from Preprocessor.preprocess()
        """
        self.config = config if config is not None else Configuration.default()
        super().__init__(input_data)  # If this call needs to be removed, call self._reset_state() explicitly

    @classmethod
    def from_json(cls, path: str, config: Configuration = None):
        return cls(load_json(path), config)

    def _reset_state(self, input_data):
        self.input_data = input_data
        self.players_data = [PostparsedPlayerData(player_data, self.config) for player_data in self.input_data]

    def get_all_exploit_clusters(self):
        all_exploit_clusters = []
        for player_data in self.players_data:
            exploit_clusters = player_data.get_exploit_clusters()
            all_exploit_clusters.extend(exploit_clusters)

        return all_exploit_clusters

    def _calc_giant_component(self):
        semantic_network = nx.Graph()
        exploit_clusters = self.get_all_exploit_clusters()
        edges = [(c1, c2) for c1, c2 in combinations(exploit_clusters, 2)
                 if is_semantic_connection(c1, c2, self.config.MIN_OVERLAP_FOR_SEMANTIC_CONNECTION)]
        semantic_network.add_edges_from(edges)
        GC = max(nx.connected_components(semantic_network), key=len)

        return GC

    def get_stats(self):
        """
        Returns the same stats as a ParsedDataset would, plus its GC
        :return: 5-tuple
        """
        giant_component = self._calc_giant_component()
        return super().get_stats() + (giant_component,)
