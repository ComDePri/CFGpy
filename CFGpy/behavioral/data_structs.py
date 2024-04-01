import numpy as np
import pandas as pd
from itertools import pairwise, groupby, combinations
from collections import Counter
import networkx as nx
from CFGpy.behavioral._consts import *
from CFGpy.behavioral._utils import is_semantic_connection
import matplotlib.pyplot as plt
import seaborn as sns


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
            try:
                start_time = self.shapes_df.iloc[start - 1, SHAPE_MOVE_TIME_IDX]  # ending of previous exploit
                # TODO: should this be the save time instead?
            except IndexError:
                start_time = 0
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

    def get_efficiency(self):
        from CFGpy.utils import get_shortest_path_len  # moved here to avoid circular import

        shape_indices = [0] + list(np.flatnonzero(self.get_gallery_mask()))
        actual_path_lengths = np.diff(shape_indices)
        actual_path_lengths[0] += 1  # TODO: for backwards compatibility, waiting for Yuval's answer to drop this
        shape_ids = self.shapes_df.iloc[shape_indices, SHAPE_ID_IDX]
        shortest_path_lengths = [get_shortest_path_len(shape1, shape2) for shape1, shape2 in pairwise(shape_ids)]
        all_efficiency_values = {idx: shortest_len / actual_len for idx, shortest_len, actual_len
                                 in zip(shape_indices[1:], shortest_path_lengths, actual_path_lengths)
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

    def get_clusters_in_phase(self, phase_name):
        phase_slices = self._phase_name_to_slices(phase_name)
        clusters = [tuple(self.shapes_df.iloc[start:end, SHAPE_ID_IDX]) for start, end in phase_slices]
        return clusters

    def plot_gallery_dt(self):
        is_gallery = self.get_gallery_mask()
        gallery_times = self.shapes_df[is_gallery].iloc[:, SHAPE_MOVE_TIME_IDX]
        gallery_diffs = np.diff(gallery_times, prepend=gallery_times.iloc[0])

        cluster_label = np.empty(len(self.shapes_df), dtype=object)
        phase_type = np.empty(len(self.shapes_df), dtype=object)
        for phase_name, phase_slices in zip(("explore", "exploit"), (self.explore_slices, self.exploit_slices)):
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
        plt.title(f"Player ID: {self.id}")
        plt.grid()
        plt.show()


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
        from CFGpy.utils import get_orig_map

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
