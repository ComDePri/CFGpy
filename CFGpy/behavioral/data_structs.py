import numpy as np
import pandas as pd
from itertools import pairwise
from itertools import groupby, combinations
from collections import Counter
import networkx as nx
from CFGpy.behavioral._consts import *
from CFGpy.behavioral._utils import is_semantic_connection
from CFGpy.utils import plot_shape, utils
import matplotlib.pyplot as plt
import seaborn as sns

PATH_FROM_REP_ROOT="output/"

class ParsedPlayerData:
    def __init__(self, player_data):
        self.id = player_data[PARSED_PLAYER_ID_KEY]
        self.start_time = player_data[PARSED_TIME_KEY]
        self.shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
        self.shapes_df[SHAPE_ID_IDX] = self.shapes_df[SHAPE_ID_IDX].astype(int)
        # self.chosen_shapes = player_data[PARSED_CHOSEN_SHAPES_KEY]

        self.delta_move_times = np.diff(self.shapes_df.iloc[:, SHAPE_MOVE_TIME_IDX])

        # Roey's additions for the subject-specific pace threshold
        self.robust_median_exploit_pace = player_data['robust_median_exploit_pace']
        self.robust_threshold_exploit_pace = player_data['robust_threshold_exploit_pace']

    def __len__(self):
        return len(self.shapes_df)

    def get_last_action_time(self):
        last_action_time = self.shapes_df.iloc[-1, SHAPE_SAVE_TIME_IDX]
        if last_action_time is None or np.isnan(last_action_time):  # is None handles cases of 0-gallery games
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

    def plot_shapes(self, ncols=10, path=PATH_FROM_REP_ROOT):
        is_galleries = self.get_gallery_mask()
        shape_ids = self.shapes_df.iloc[:, SHAPE_ID_IDX]

        nrows = int(np.ceil(len(shape_ids) / ncols))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows), dpi=400)
        plt.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0)
        for i, ax in enumerate(axs.flat):
            if i < len(shape_ids):
                # TODO: adjust grid linewidth to be the correct fraction of the bbox width instead of always 3
                plot_shape(shape_ids[i], ax=ax, is_gallery=is_galleries[i])
            else:
                ax.axis("off")

        # plt.show()
        imagePath = f"{path}ShapesPlayer{self.id}.png"
        plt.savefig(imagePath)
        plt.close()

        return imagePath

    def plot_clusters(self, path=PATH_FROM_REP_ROOT):
        is_galleries = self.get_gallery_mask()
        shape_ids = self.shapes_df.iloc[:, SHAPE_ID_IDX]

        # Combine exploit and explore stages with labels
        stages = [(start, end, 'exploit') for start, end in self.exploit_slices] + \
                 [(start, end, 'explore') for start, end in self.explore_slices]
        stages.sort(key=lambda x: x[0])  # Sort by start time

        # Group shapes by stages
        clusters = []
        for start, end, label in stages:
            cluster_ids = self.shapes_df.iloc[start:end][is_galleries[start:end]][SHAPE_ID_IDX].tolist()
            if cluster_ids:  # only add non-empty clusters
                clusters.append((cluster_ids, label))

        if len(clusters) < 2:
            return -1
        # Calculate number of rows and cols needed
        ncols = max(len(cluster[0]) for cluster in clusters)
        # total_shapes = sum(len(cluster[0]) for cluster in clusters)
        # nrows = int(np.ceil(total_shapes / ncols))
        nrows = len(clusters)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows), dpi=400)
        plt.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0)

        ax_iter = iter(axs.flat)
        for cluster, label in clusters:
            for shape_id in cluster:
                try:
                    ax = next(ax_iter)
                    is_exploit = (label == 'exploit')
                    color = utils.SHAPE_COLOR if is_exploit else 'pink'
                    plot_shape(shape_id, ax=ax, color=color, is_gallery=False)
                except StopIteration:
                    print("Ran out of axes for shapes!")
                    break
            # Fill the remaining columns in the current row with empty plots
            if len(cluster) % ncols > 0:
                for _ in range(len(cluster) % ncols, ncols):
                    try:
                        ax = next(ax_iter)
                        ax.axis("off")
                    except StopIteration:
                        print("Ran out of axes for whites!")
                        break

        # Turn off any remaining unused axes
        for ax in ax_iter:
            ax.axis("off")

        imagePath = f"{path}shapes_clusters/ShapesPlayer{self.id}.png"
        plt.savefig(imagePath)
        plt.close()

        return imagePath

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
        return self.get_last_action_time() - self.total_exploit_time()

    def total_exploit_time(self):
        time_in_exploit = 0
        for start, end in self.exploit_slices:
            start_time = self.shapes_df.iloc[start, SHAPE_MOVE_TIME_IDX]
            end_time = self.shapes_df.iloc[end - 1, SHAPE_MOVE_TIME_IDX]
            time_in_exploit += (end_time - start_time)

        return time_in_exploit

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


    def get_efficiency_all(self):
        from CFGpy.utils import get_shortest_path_len  # moved here to avoid circular import

        shape_indices = [0] + list(np.flatnonzero(self.get_gallery_mask()))
        actual_path_lengths = np.diff(shape_indices)
        actual_path_lengths[0] += 1  # TODO: for backwards compatibility, waiting for Yuval's answer to drop this
        shape_ids = self.shapes_df.iloc[shape_indices, SHAPE_ID_IDX]
        shortest_path_lengths = [get_shortest_path_len(shape1, shape2) for shape1, shape2 in pairwise(shape_ids)]
        all_efficiency_values = {idx: shortest_len / actual_len for idx, shortest_len, actual_len
                                 in zip(shape_indices[1:], shortest_path_lengths, actual_path_lengths)
                                 if (shortest_len, actual_len) != (1, 1)}

        return all_efficiency_values


    def get_exploit_clusters(self):
        is_gallery = self.get_gallery_mask()
        clusters = []
        for start, end in self.exploit_slices:
            slice_is_gallery = is_gallery[start:end]
            slice_shape_ids = self.shapes_df.iloc[start:end, SHAPE_ID_IDX]
            cluster = tuple(slice_shape_ids[slice_is_gallery])
            clusters.append(cluster)

        return clusters

    def get_clusters_times(self):
        explore_clusters_times = []

    def get_cluster_times(self):

        def get_times_for_slices(slices):
            cluster_times = []
            for start, end in slices:
                first_out_time = self.shapes_df.iloc[start, self.shapes_df.columns.get_loc(SHAPE_MAX_MOVE_TIME_IDX)]
                last_in_time = self.shapes_df.iloc[end - 1, self.shapes_df.columns.get_loc(SHAPE_MOVE_TIME_IDX)]
                cluster_times.append([first_out_time, last_in_time])
            return cluster_times

        explore_times = get_times_for_slices(self.explore_slices)
        exploit_times = get_times_for_slices(self.exploit_slices)

        return explore_times, exploit_times

    def plot_gallery_dt(self, path=PATH_FROM_REP_ROOT):
        # if there is no explore / exploit --> don't plot
        if self.exploit_slices == [] or self.explore_slices == []:
            return -1

        is_gallery = self.get_gallery_mask()
        gallery_in_times = self.shapes_df[is_gallery].iloc[:, SHAPE_MOVE_TIME_IDX]
        gallery_out_times = self.shapes_df[is_gallery].iloc[:, SHAPE_MAX_MOVE_TIME_IDX]
        gallery_diffs = gallery_in_times - gallery_out_times.shift()
        # print(self.id) # --> for testing
        gallery_diffs.iloc[0] = 0
        gallery_diffs = gallery_diffs.to_numpy()

        remove_empty_steps_time = True # &&&
        if remove_empty_steps_time:
            gallery_indices =  np.where(is_gallery)[0].tolist()
            empty_steps_time = self.shapes_df.iloc[:, SHAPE_MAX_MOVE_TIME_IDX] - self.shapes_df.iloc[:, SHAPE_MOVE_TIME_IDX]
            gallery_diffs_fixed = gallery_diffs
            for gI, gallery_idx in enumerate(gallery_indices[:-1]):  # [:-1] makes sure we exclude the last element
                start_idx = gallery_idx + 1
                end_idx = gallery_indices[gI + 1] - 1
                gallery_diffs_fixed[gI + 1] = gallery_diffs[gI + 1] - sum(empty_steps_time[start_idx:end_idx])
            gallery_diffs = gallery_diffs_fixed


        cluster_label = np.empty(len(self.shapes_df), dtype=object)
        phase_type = np.empty(len(self.shapes_df), dtype=object)
        for phase_name, phase_slices in zip(("explore", "exploit"), (self.explore_slices, self.exploit_slices)):
            for i, (start, end) in enumerate(phase_slices):
                cluster_label[start:end] = f"{phase_name}{i}"
                phase_type[start:end] = phase_name

        df = pd.DataFrame({"gallery_time": gallery_in_times,
                           "gallery_diff": gallery_diffs,
                           "cluster": cluster_label[is_gallery],
                           "phase_type": phase_type[is_gallery]})

        fig, ax = plt.subplots(figsize=(10, 5))
        marker_dict = {"explore": "o", "exploit": "X"}
        sns.lineplot(data=df, ax=ax, x="gallery_time", y="gallery_diff", hue="cluster", style="phase_type",
                     markers=marker_dict, markersize=10)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_markers = dict(zip(labels, handles))

        #ROEY COMMENTED TO AVOID ISSUES WITH GAMES THAT ARE ONLY EXPLOIT WHEN USING EFFICIENY
        #plt.legend((unique_markers["explore"], unique_markers["exploit"]), ("explore", "exploit"))

        for ax in plt.gcf().get_axes():  # Get the current figure and all its axes
            legend = ax.get_legend()  # Get the legend from the current axis
            if legend:  # If there is a legend
                legend.remove()  # Remove the legend

        plt.xlabel(r"$t$ (s)", fontsize=14)
        plt.ylabel(r"$\Delta t$ (s)", fontsize=14)
        plt.suptitle("Gallery shapes creation time, segmented by clusters", fontsize=16)
        plt.title(f"Player ID: {self.id}")
        plt.grid()

        # Show
        # plt.show()

        # Save the plot as an image file
        imagePath = f"{path}GalleryPlayer{self.id}.png"
        plt.savefig(imagePath)
        plt.close()

        return imagePath

    def plot_gallery_steps(self, path=PATH_FROM_REP_ROOT):
        # if there is no explore / exploit --> don't plot
        if self.exploit_slices == [] or self.explore_slices == []:
            return -1

        is_gallery = self.get_gallery_mask()
        gallery_step_idx = self.shapes_df[is_gallery].index
        gallery_diffs = gallery_step_idx.diff().fillna(0).to_numpy()

        cluster_label = np.empty(len(self.shapes_df), dtype=object)
        phase_type = np.empty(len(self.shapes_df), dtype=object)
        for phase_name, phase_slices in zip(("explore", "exploit"), (self.explore_slices, self.exploit_slices)):
            for i, (start, end) in enumerate(phase_slices):
                cluster_label[start:end] = f"{phase_name}{i}"
                phase_type[start:end] = phase_name

        df = pd.DataFrame({
            "gallery_steps": gallery_step_idx,
            "gallery_diff": gallery_diffs,
            "cluster": cluster_label[is_gallery],
            "phase_type": phase_type[is_gallery]
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        marker_dict = {"explore": "o", "exploit": "X"}
        sns.lineplot(data=df, ax=ax, x="gallery_steps", y="gallery_diff", hue="cluster", style="phase_type",
                     markers=marker_dict, markersize=10)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_markers = dict(zip(labels, handles))

        #ROEY COMMENTED TO AVOID ISSUES WITH GAMES THAT ARE ONLY EXPLOIT WHEN USING EFFICIENY
        plt.legend((unique_markers["explore"], unique_markers["exploit"]), ("explore", "exploit"))
        plt.xlabel("total number of steps", fontsize=14)
        plt.ylabel("steps difference", fontsize=14)
        plt.suptitle("Gallery shapes creation time, segmented by clusters", fontsize=16)
        plt.title(f"Player ID: {self.id}")
        plt.grid()

        # Show
        # plt.show()

        # Save the plot as an image file
        imagePath = f"{path}GalleryPlayer{self.id}.png"
        plt.savefig(imagePath)
        plt.close()

        return imagePath

    def plot_gallery_steps_over_dt(self, path=PATH_FROM_REP_ROOT):
        # if there is no explore / exploit --> don't plot
        if self.exploit_slices == [] or self.explore_slices == []:
            return -1

        is_gallery = self.get_gallery_mask()
        # Get the steps difference between consecutive gallery shapes
        gallery_step_idx = self.shapes_df[is_gallery].index
        gallery_steps_diffs = gallery_step_idx.diff().fillna(0).to_numpy()

        # Get the time difference between consecutive gallery shapes
        gallery_in_times = self.shapes_df[is_gallery].iloc[:, SHAPE_MOVE_TIME_IDX]
        gallery_out_times = self.shapes_df[is_gallery].iloc[:, SHAPE_MAX_MOVE_TIME_IDX]
        gallery_time_diffs = gallery_in_times - gallery_out_times.shift()
        # print(self.id) # --> for testing
        gallery_time_diffs.iloc[0] = 0
        gallery_time_diffs = gallery_time_diffs.to_numpy()

        remove_empty_steps_time = True # &&&
        if remove_empty_steps_time:
            gallery_indices =  np.where(is_gallery)[0].tolist()
            empty_steps_time = self.shapes_df.iloc[:, SHAPE_MAX_MOVE_TIME_IDX] - self.shapes_df.iloc[:, SHAPE_MOVE_TIME_IDX]
            gallery_diffs_fixed = gallery_time_diffs
            for gI, gallery_idx in enumerate(gallery_indices[:-1]):  # [:-1] makes sure we exclude the last element
                start_idx = gallery_idx + 1
                end_idx = gallery_indices[gI + 1] - 1
                gallery_diffs_fixed[gI + 1] = gallery_time_diffs[gI + 1] - sum(empty_steps_time[start_idx:end_idx])
            gallery_time_diffs = gallery_diffs_fixed

        cluster_label = np.empty(len(self.shapes_df), dtype=object)
        phase_type = np.empty(len(self.shapes_df), dtype=object)
        for phase_name, phase_slices in zip(("explore", "exploit"), (self.explore_slices, self.exploit_slices)):
            for i, (start, end) in enumerate(phase_slices):
                cluster_label[start:end] = f"{phase_name}{i}"
                phase_type[start:end] = phase_name

        df = pd.DataFrame({
            "gallery_time": gallery_in_times,
            "gallery_diff": np.nan_to_num(gallery_time_diffs/gallery_steps_diffs),
            "cluster": cluster_label[is_gallery],
            "phase_type": phase_type[is_gallery]
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        marker_dict = {"explore": "o", "exploit": "X"}
        sns.lineplot(data=df, ax=ax, x="gallery_time", y="gallery_diff", hue="cluster", style="phase_type",
                     markers=marker_dict, markersize=10)

        # Calculate the median value of gallery_diff
        # median_value = df['gallery_diff'].median()
        median_value = self.robust_median_exploit_pace
        threshold_value = self.robust_threshold_exploit_pace
        # Add a horizontal line at the median value
        ax.axhline(y=median_value, color='black', linestyle='-', label=f'Median: {median_value:.2f}')
        ax.axhline(y=threshold_value, color='black', linestyle='--', label=f'Thresh: {threshold_value:.2f}')


        #handles, labels = plt.gca().get_legend_handles_labels()
        #unique_markers = dict(zip(labels, handles))
        #ROEY COMMENTED TO AVOID ISSUES WITH GAMES THAT ARE ONLY EXPLOIT WHEN USING EFFICIENY
        #plt.legend((unique_markers["explore"], unique_markers["exploit"]), ("explore", "exploit"))

        # Getting the legend handles and labels
        ###handles, labels = plt.gca().get_legend_handles_labels()
        ###unique_markers = dict(zip(labels, handles))

        #### Collect the markers and labels that exist
        ###legend_handles = []
        ###legend_labels = []

        ###if "explore" in unique_markers:
        ###    legend_handles.append(unique_markers["explore"])
        ###    legend_labels.append("explore")
        ###if "exploit" in unique_markers:
        ###    legend_handles.append(unique_markers["exploit"])
        ###    legend_labels.append("exploit")

        # Add the median line to the legend if it's not already there
        ###median_label = f'Median: {median_value:.2f}'
        ###if median_label not in legend_labels:
        ###    legend_handles.append(ax.get_lines()[-1])
        ###    legend_labels.append(median_label)

        for ax in plt.gcf().get_axes():  # Get the current figure and all its axes
            legend = ax.get_legend()  # Get the legend from the current axis
            if legend:  # If there is a legend
                legend.remove()  # Remove the legend


        plt.xlabel("time", fontsize=14)
        plt.ylabel("time per step", fontsize=14)
        plt.suptitle("Gallery shapes creation time, segmented by clusters", fontsize=16)
        plt.title(f"Player ID: {self.id}")
        plt.grid()

        # Show
        # plt.show()

        # Save the plot as an image file
        imagePath = f"{path}GalleryPlayer{self.id}.png"
        plt.savefig(imagePath)
        plt.close()

        return imagePath
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
            exploit_clusters = player_data.get_exploit_clusters()
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
