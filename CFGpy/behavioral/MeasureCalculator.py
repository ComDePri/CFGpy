import numpy as np
import pandas as pd
from datetime import datetime
from itertools import groupby, pairwise, combinations
from _consts import *
from _utils import load_json
from collections.abc import Collection
from collections import Counter
from functools import reduce
from scipy.stats import zscore
import networkx as nx
from CFGpy.utils import get_shortest_path_len


def get_vanilla_descriptors():
    # TODO
    return 0, 0, 0, 0, 0


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


def get_steps(shapes_df: pd.DataFrame):
    shape_ids = shapes_df.iloc[:, SHAPE_ID_IDX]
    steps = list(pairwise(shape_ids))
    return steps


def get_gallery_ids(shapes_df: pd.DataFrame):
    is_gallery = _get_gallery_mask(shapes_df)
    gallery_ids = shapes_df.iloc[is_gallery, SHAPE_ID_IDX]
    return gallery_ids


def _get_n_galleries(shapes_df: pd.DataFrame):
    return sum(_get_gallery_mask(shapes_df))


def _get_self_avoidance(shapes_df: pd.DataFrame):
    shape_ids = shapes_df.iloc[:, SHAPE_ID_IDX]
    unique_shapes = np.unique(shape_ids)
    shapes_no_consecutive_duplicates = [k for k, g in groupby(shape_ids)]
    # TODO: in a future version, consecutive duplicates will be handled by Preprocessor. when this is implemented,
    #  there will be no need for shapes_no_consecutive_duplicates, it will be  equivalent to shapes.
    #  Instead of its len, it would then make sense to use n_moves (defined in self.calc)

    return len(unique_shapes) / len(shapes_no_consecutive_duplicates)


def _get_frac_unique(items, all_items_counter):
    n_unique = 0
    for item in items:
        if all_items_counter[item] == 1:
            n_unique += 1

    return n_unique / len(items)


def _get_gallery_mask(shapes_df: pd.DataFrame, phase_slices=None):
    """
    Returns a mask where true means gallery shape.
    :param shapes_df: DataFrame with shapes info.
    :param phase_slices: a list to pairs (start, end) for each phase (explore / exploit). If given, galleries will
        only be considered if part of that phase. If None (default), galleries considered over the whole game.
    :return: ndarray with shape (len(shapes_df),) and dtype bool.
    """
    if phase_slices is None:
        phase_slices = [[0, len(shapes_df)]]

    is_gallery_and_in_phase = np.zeros(len(shapes_df), dtype=bool)
    for start, end in phase_slices:
        is_gallery_in_slice = shapes_df.iloc[start:end, SHAPE_SAVE_TIME_IDX].notna()
        is_gallery_and_in_phase[start:end] = is_gallery_in_slice

    return is_gallery_and_in_phase


def _get_n_galleries_in_phase(shapes_df: pd.DataFrame, phase_slices):
    return sum(_get_gallery_mask(shapes_df, phase_slices))


def _get_last_action_time(shapes_df: pd.DataFrame):
    last_action_time = shapes_df.iloc[-1, SHAPE_SAVE_TIME_IDX]
    if np.isnan(last_action_time) or last_action_time is None:  # is None handles cases of 0-gallery games
        last_action_time = shapes_df.iloc[-1, SHAPE_MOVE_TIME_IDX]

    return last_action_time


def _get_time_in_phase(dt, phase_slices):
    dt_in_phase = []
    for start, end in phase_slices:
        dt_in_phase.extend(dt[start:end])

    return sum(dt_in_phase)


def _get_phase_efficiency(shapes_df: pd.DataFrame, phase_slices):
    # TODO: empty step with two identical galleries in a row cause shortest path of 0. how to handle this?
    actual_paths = {}
    for start, end in phase_slices:
        slice_shape_ids = shapes_df.iloc[start:end, SHAPE_ID_IDX]
        slice_is_gallery = shapes_df.iloc[start:end, SHAPE_SAVE_TIME_IDX].notna()

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
                        for (path_start, path_end), actual_path in actual_paths.items()]
    return np.median(paths_efficiency)


def _get_clusters_in_phase(shapes_df: pd.DataFrame, phase_slices):
    clusters = [tuple(shapes_df.iloc[start:end, SHAPE_ID_IDX]) for start, end in phase_slices]
    return clusters


def _get_all_players_steps(input_data):
    all_steps = []
    for player_data in input_data:
        steps = get_steps(pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY]))
        all_steps.extend(steps)

    return all_steps


def _get_all_players_galleries(input_data):
    all_galleries = []
    for player_data in input_data:
        gallery_ids = get_gallery_ids(pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY]))
        all_galleries.extend(gallery_ids)

    return all_galleries


def _get_all_players_exploit_clusters(input_data):
    all_exploit_clusters = []
    for player_data in input_data:
        shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
        exploit_slices = player_data[EXPLOIT_KEY]
        exploit_clusters = _get_clusters_in_phase(shapes_df, exploit_slices)
        all_exploit_clusters.extend(exploit_clusters)

    return all_exploit_clusters


def get_GC(input_data):
    semantic_network = nx.Graph()
    exploit_clusters = _get_all_players_exploit_clusters(input_data)
    edges = [(c1, c2) for c1, c2 in combinations(exploit_clusters, 2)
             if len(set(c1) & set(c2)) >= MIN_OVERLAP_FOR_SEMANTIC_CONNECTION]
    semantic_network.add_edges_from(edges)
    GC = max(nx.connected_components(semantic_network), key=len)

    return GC


def _is_cluster_in_GC(cluster, GC):
    for GC_cluster in GC:
        overlap = len(set(cluster) & set(GC_cluster))
        if overlap >= MIN_OVERLAP_FOR_SEMANTIC_CONNECTION:
            return True

    return False


def _get_sample_descriptors(input_data):
    step_counter = Counter(_get_all_players_steps(input_data))
    step_orig_map = get_orig_map(step_counter, group_func=lambda step: step[0])
    gallery_counter = Counter(_get_all_players_galleries(input_data))
    gallery_orig_map = get_orig_map(gallery_counter)
    GC = get_GC(input_data)

    return step_counter, step_orig_map, gallery_counter, gallery_orig_map, GC


def _calc_absolute_measures(preprocessed_data):
    absolute_measures = []
    for player_data in preprocessed_data:
        # supporting data
        shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
        explore, exploit = player_data[EXPLORE_KEY], player_data[EXPLOIT_KEY]
        delta_move_times = np.diff(shapes_df.iloc[:, SHAPE_MOVE_TIME_IDX])
        time_in_explore = _get_time_in_phase(delta_move_times, explore)
        time_in_exploit = sum(delta_move_times) - time_in_explore
        explore_lengths = [end - start for start, end in explore]
        exploit_lengths = [end - start for start, end in exploit]

        # reusable measures
        last_action_time = _get_last_action_time(shapes_df)
        n_moves = len(shapes_df)
        n_galleries = _get_n_galleries(shapes_df)
        explore_efficiency = _get_phase_efficiency(shapes_df, explore)
        exploit_efficiency = _get_phase_efficiency(shapes_df, exploit)

        absolute_measures.append({
            MEASURES_ID_KEY: player_data[PARSED_PLAYER_ID_KEY],
            MEASURES_START_TIME_KEY: datetime.fromtimestamp(player_data[PARSED_TIME_KEY]).isoformat(),
            GAME_DURATION_KEY: last_action_time,
            N_MOVES_KEY: n_moves,
            "Average Speed": n_moves / last_action_time,
            "#galleries": n_galleries,
            "self avoidance": _get_self_avoidance(shapes_df),
            "#clusters": len(exploit),
            "% galleries in exp": _get_n_galleries_in_phase(shapes_df, explore) / n_galleries,
            "% time in exp": time_in_explore / last_action_time,
            "exp efficiency": explore_efficiency,
            "scav efficiency": exploit_efficiency,
            "efficiency ratio": explore_efficiency / exploit_efficiency,
            MEDIAN_EXPLORE_LENGTH_KEY: np.median(explore_lengths),
            MEDIAN_EXPLOIT_LENGTH_KEY: np.median(exploit_lengths),
            "exp speed": sum(explore_lengths) / time_in_explore,
            "scav speed": sum(exploit_lengths) / time_in_exploit,
            LONGEST_PAUSE_KEY: max(delta_move_times[3:-4])
        })

    return pd.DataFrame(absolute_measures)


def _calc_relative_measures(preprocessed_data, step_counter, step_orig_map, gallery_counter, gallery_orig_map, GC,
                            label=None):
    label_ext = f" ({label})" if label else ""

    relative_measures = []
    for player_data in preprocessed_data:
        shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
        explore, exploit = player_data[EXPLORE_KEY], player_data[EXPLOIT_KEY]

        steps = get_steps(shapes_df)
        step_orig = [step_orig_map[step] for step in steps]
        is_gallery = _get_gallery_mask(shapes_df)
        gallery_ids = shapes_df.iloc[is_gallery, SHAPE_ID_IDX]
        gallery_orig = np.array([gallery_orig_map[shape_id] for shape_id in gallery_ids])
        is_explore_gallery = _get_gallery_mask(shapes_df, explore)[is_gallery]
        is_exploit_gallery = ~is_explore_gallery
        clusters = _get_clusters_in_phase(shapes_df, exploit)
        n_clusters_in_GC = sum([_is_cluster_in_GC(cluster, GC) for cluster in clusters])

        relative_measures.append({
            MEASURES_ID_KEY: player_data[PARSED_PLAYER_ID_KEY],
            f"Step Orig{label_ext}": np.mean(step_orig),
            f"% Steps Unique{label_ext}": _get_frac_unique(steps, step_counter),
            f"Gallery Orig{label_ext}": np.mean(gallery_orig),
            f"Gallery Orig exp{label_ext}": np.mean(gallery_orig[is_explore_gallery]),
            f"Gallery Orig scav{label_ext}": np.mean(gallery_orig[is_exploit_gallery]),
            f"% Galleries Unique{label_ext}": _get_frac_unique(gallery_ids, gallery_counter),
            f"% Galleries Unique exp{label_ext}": _get_frac_unique(gallery_ids[is_explore_gallery], gallery_counter),
            f"% Galleries Unique scav{label_ext}": _get_frac_unique(gallery_ids[is_exploit_gallery], gallery_counter),
            f"# clusters in GC{label_ext}": n_clusters_in_GC,
            f"% clusters in GC{label_ext}": n_clusters_in_GC / len(exploit),
        })

    return pd.DataFrame(relative_measures)


class MeasureCalculator:
    def __init__(self, preprocessed_data, manually_excluded_ids: Collection = MANUALLY_EXCLUDED_IDS,
                 min_n_moves: int = MIN_N_MOVES, min_game_duration: float = MIN_GAME_DURATION_SEC,
                 max_pause_duration: float = MAX_PAUSE_DURATION_SEC,
                 max_zscore_for_outlier: float = MAX_ZSCORE_FOR_OUTLIERS):
        self.input_data = preprocessed_data
        self.all_absolute_measures = None
        self.output_df = None

        self.manually_excluded_ids = manually_excluded_ids
        self.min_n_moves = min_n_moves
        self.min_game_duration = min_game_duration
        self.max_pause_duration = max_pause_duration
        self.max_zscore_for_outlier = max_zscore_for_outlier
        self.exclusions = pd.DataFrame(columns=[MEASURES_ID_KEY, "reason"])

    @classmethod
    def from_json(cls, path: str):
        return cls(load_json(path))

    def calc(self):
        self.all_absolute_measures = _calc_absolute_measures(self.input_data)
        self.output_df = self.all_absolute_measures.copy()
        self._drop_nonfirst_games()
        # TODO: uncomment once get_vanilla_descriptors is ready:
        # vanilla_relative_measures = _calc_relative_measures(self.input_data, *get_vanilla_descriptors())
        # self.output_df = self.output_df.merge(vanilla_relative_measures, on=MEASURES_ID_KEY)
        self._apply_soft_filters()
        sample_relative_measures = _calc_relative_measures(self.input_data, *_get_sample_descriptors(self.input_data),
                                                           label="sample")
        self.output_df = self.output_df.merge(sample_relative_measures, on=MEASURES_ID_KEY, how="left")

        return self.output_df

    def get_all_absolute_measures(self):
        return self.all_absolute_measures

    def _drop_nonfirst_games(self):
        """
        Keeps only the first game from each player. Allows functions downstream to assume unique IDs.
        """
        self.input_data = (pd.DataFrame(self.input_data).
                           sort_values(by=[PARSED_TIME_KEY], ascending=True).
                           drop_duplicates(subset=[PARSED_PLAYER_ID_KEY], keep="first").
                           to_dict("records"))
        self.output_df = (self.output_df.
                          sort_values(by=[MEASURES_START_TIME_KEY], ascending=True).
                          drop_duplicates(subset=[MEASURES_ID_KEY], keep="first"))

    def _apply_soft_filters(self):
        """
        Applies absolute filters first, then sample-relative filters with the remaining sample.
        """
        for filter_getter in (self._get_absolute_filters, self._get_sample_relative_filters):
            masks, reasons = filter_getter()
            self._update_exclusion_info(masks, reasons)
            is_excluded = reduce(np.logical_or, masks)

            self.input_data = pd.DataFrame(self.input_data).loc[~is_excluded].to_dict("records")
            self.output_df = self.output_df.loc[~is_excluded]

    def _get_absolute_filters(self):
        """
        Absolute filters are based on absolute measures, can be applied independently of each other. Each filter is
        represented by a textual description and a mask with **True for players to exclude**, False for players to keep.
        :return: masks, reasons.
        """
        reasons = ("Manually excluded id", "Too few moves", "Game too short", "Paused for too long")
        masks = (self.output_df[MEASURES_ID_KEY].isin(self.manually_excluded_ids),
                 self.output_df[N_MOVES_KEY] < self.min_n_moves,
                 self.output_df[GAME_DURATION_KEY] < self.min_game_duration,
                 self.output_df[LONGEST_PAUSE_KEY] > self.max_pause_duration)

        return masks, reasons

    def _get_sample_relative_filters(self):
        """
        Each filter is represented by a textual description and a mask with **True for players to exclude**, False for
        players to keep.
        :return: masks, reasons.
        """
        reasons = ("Explore duration outlier", "Exploit duration outlier")
        zscores = self.output_df[[MEDIAN_EXPLORE_LENGTH_KEY, MEDIAN_EXPLOIT_LENGTH_KEY]].apply(zscore)
        masks = (abs(zscores[MEDIAN_EXPLORE_LENGTH_KEY]) > self.max_zscore_for_outlier,
                 abs(zscores[MEDIAN_EXPLOIT_LENGTH_KEY]) > self.max_zscore_for_outlier)

        return masks, reasons

    def _update_exclusion_info(self, masks, reasons):
        """
        Updates self.to_exclude based on filters results.
        :param masks: a collection of masks, each has **True for players to exclude**, false for players to keep.
        :param reasons: a collection of strings describing exclusion reasons for the masks.
        """
        for is_excluded, reason in zip(masks, reasons):
            ids_to_exclude = self.output_df.loc[is_excluded, MEASURES_ID_KEY]
            current_exclusion = pd.DataFrame({
                MEASURES_ID_KEY: ids_to_exclude,
                "reason": [reason] * len(ids_to_exclude)
            })
            pd.concat((self.exclusions, current_exclusion))

    def dump(self, path=DEFAULT_FINAL_OUTPUT_FILENAME):
        self.output_df.to_csv(path, index=False)  # reorder columns
        # TODO: document all filtered ids and filtering criteria
        # TODO: write html with dashboards to inspect data quality and some summary stats


if __name__ == '__main__':
    import argparse
    from Preprocessor import Preprocessor

    argparser = argparse.ArgumentParser(description="Calculate measures from parsed CFG data")
    argparser.add_argument("-i", "--input", dest="input_filename",
                           help='Filename of parsed data JSON')
    argparser.add_argument("-o", "--output", default=DEFAULT_FINAL_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output CSV')
    args = argparser.parse_args()

    pp = Preprocessor.from_json(args.input_filename)
    preprocessed_data = pp.preprocess()
    mc = MeasureCalculator(preprocessed_data)
    mc.calc()
    mc.dump(args.output_filename)
