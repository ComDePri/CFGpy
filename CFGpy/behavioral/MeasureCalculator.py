import numpy as np
import pandas as pd
from datetime import datetime
from itertools import groupby, pairwise
from ._consts import *
from ._utils import load_json
from collections.abc import Collection
from collections import Counter
from functools import reduce
from scipy.stats import zscore


def get_vanilla_descriptors():
    # TODO
    return 0, 0


def get_step_orig_map(step_counter, alpha=0) -> dict[tuple[int, int], float]:
    """
    Returns a mapping from each step its originality value. The originality value of a step denoted (s1, s2) is given by
    -log10(p) where p is the transition probability P(s2 | s1), estimated over the given sample.
    :param step_counter: a Counter object with all steps in the sample.
    :param alpha: pseudocount for Laplace smoothing (see https://en.wikipedia.org/wiki/Additive_smoothing).
    :return: dict {step: orig}
    """
    total_steps_from_shape = Counter()
    for step, count in step_counter.items():
        total_steps_from_shape[step[0]] += count
    n_step_beginnings = len(total_steps_from_shape)

    # transform counts to orig values: -log10 of alpha-smoothed probability of the step
    step_orig_map = {}
    for step, count in step_counter.items():
        total = total_steps_from_shape[step[0]]
        smoothed_step_probability = (count + alpha) / (total + n_step_beginnings * alpha)
        step_orig_map[step] = -np.log10(smoothed_step_probability)

    return step_orig_map


def get_steps(shapes_df: pd.DataFrame):
    shape_ids = shapes_df.iloc[:, SHAPE_ID_IDX]
    steps = list(pairwise(shape_ids))
    return steps


def _get_n_galleries(shapes_df: pd.DataFrame):
    is_saved_to_gallery = shapes_df.iloc[:, SHAPE_SAVE_TIME_IDX].notna()
    return sum(is_saved_to_gallery)


def _get_self_avoidance(shapes_df: pd.DataFrame):
    shape_ids = shapes_df.iloc[:, SHAPE_ID_IDX]
    unique_shapes = np.unique(shape_ids)
    shapes_no_consecutive_duplicates = [k for k, g in groupby(shape_ids)]
    # TODO: in a future version, consecutive duplicates will be handled by Preprocessor. when this is implemented,
    #  there will be no need for shapes_no_consecutive_duplicates, it will be  equivalent to shapes.
    #  Instead of its len, it would then make sense to use n_moves (defined in self.calc)

    return len(unique_shapes) / len(shapes_no_consecutive_duplicates)


def _get_frac_unique_steps(steps, all_steps_counter):
    n_unique_steps = 0
    for step in steps:
        if all_steps_counter[step] == 1:
            n_unique_steps += 1

    return n_unique_steps / len(steps)


def _get_n_galleries_in_phase(shapes_df: pd.DataFrame, phase_slices):
    is_gallery_in_phase = []
    for start, end in phase_slices:
        is_gallery_in_slice = shapes_df.iloc[start:end, SHAPE_SAVE_TIME_IDX].notna()
        is_gallery_in_phase.extend(is_gallery_in_slice)

    return sum(is_gallery_in_phase)


def _get_time_in_phase(dt, phase_slices):
    dt_in_phase = []
    for start, end in phase_slices:
        dt_in_phase.extend(dt[start:end])

    return sum(dt_in_phase)


def _get_phase_efficiency(shapes_df: pd.DataFrame, phase_slices):
    actual_paths = {}
    for start, end in phase_slices:
        slice_shape_ids = shapes_df.iloc[start:end, SHAPE_ID_IDX]
        slice_is_gallery = shapes_df.iloc[start:end, SHAPE_SAVE_TIME_IDX].notna()

        gallery_start = None
        steps_between_galleries = 0
        for shape_id, is_gallery in zip(slice_shape_ids, slice_is_gallery):
            steps_between_galleries += 1
            if is_gallery:
                if gallery_start is not None:
                    actual_paths[(gallery_start, shape_id)] = steps_between_galleries

                gallery_start = shape_id
                steps_between_galleries = 0

    # TODO: compare to shortest paths
    return 1  # placeholder which is not 0 so that it can be divided by


def _get_all_players_steps(input_data):
    all_steps = []
    for player_data in input_data:
        steps = get_steps(pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY]))
        all_steps.extend(steps)

    return all_steps


def _get_sample_descriptors(input_data):
    sample_step_counter = Counter(_get_all_players_steps(input_data))
    sample_step_orig_map = get_step_orig_map(sample_step_counter)

    return sample_step_counter, sample_step_orig_map


def _calc_absolute_measures(preprocessed_data):
    absolute_measures = []
    for player_data in preprocessed_data:
        # supporting data
        shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
        explore, exploit = player_data[EXPLORE_KEY], player_data[EXPLOIT_KEY]
        delta_move_times = np.diff(shapes_df.iloc[:, SHAPE_MOVE_TIME_IDX])

        # reusable measures
        last_action_time = shapes_df.iloc[-1, SHAPE_SAVE_TIME_IDX]
        if last_action_time is None:
            last_action_time = shapes_df.iloc[-1, SHAPE_MOVE_TIME_IDX]
        n_moves = len(shapes_df)
        n_galleries = _get_n_galleries(shapes_df)
        explore_efficiency = _get_phase_efficiency(shapes_df, explore)
        exploit_efficiency = _get_phase_efficiency(shapes_df, exploit)

        absolute_measures.append({
            MEASURES_ID_KEY: player_data["id"],
            "Date/Time": datetime.fromtimestamp(player_data[PARSED_TIME_KEY]).isoformat(),
            GAME_DURATION_KEY: last_action_time,
            N_MOVES_KEY: n_moves,
            "Average Speed": n_moves / last_action_time,
            "#galleries": n_galleries,
            "self avoidance": _get_self_avoidance(shapes_df),
            "#clusters": len(exploit),
            "% galleries in exp": _get_n_galleries_in_phase(shapes_df, explore) / n_galleries,
            "% time in exp": _get_time_in_phase(delta_move_times, explore) / last_action_time,
            "exp efficiency": explore_efficiency,
            "scav efficiency": exploit_efficiency,
            "efficiency ratio": explore_efficiency / exploit_efficiency,
            MEDIAN_EXPLORE_LENGTH_KEY: 0,
            MEDIAN_EXPLOIT_LENGTH_KEY: 0,
            "exp speed": 0,
            "scav speed": 0,
            LONGEST_PAUSE_KEY: max(delta_move_times[3:-4])
        })

    return pd.DataFrame(absolute_measures)


def _calc_relative_measures(preprocessed_data, step_counter, step_orig_map, label=None):
    label_ext = f" ({label})" if label else ""

    relative_measures = []
    for player_data in preprocessed_data:
        shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
        steps = get_steps(shapes_df)
        step_orig = [step_orig_map[step] for step in steps]

        relative_measures.append({
            MEASURES_ID_KEY: player_data[PARSED_PLAYER_ID_KEY],
            f"Step Orig{label_ext}": np.mean(step_orig),
            f"% Steps Unique{label_ext}": _get_frac_unique_steps(steps, step_counter),
            f"Gallery Orig{label_ext}": 0,
            f"Gallery Orig exp{label_ext}": 0,
            f"Gallery Orig scav{label_ext}": 0,
            f"% Galleries Unique{label_ext}": 0,
            f"% Galleries Unique exp{label_ext}": 0,
            f"% Galleries Unique scav{label_ext}": 0,
            f"# clusters in GC{label_ext}": 0,
            f"% clusters in GC{label_ext}": 0,
        })

    return pd.DataFrame(relative_measures)


class MeasureCalculator:
    def __init__(self, preprocessed_data, manually_excluded_ids: Collection = MANUALLY_EXCLUDED_IDS,
                 min_n_moves: int = MIN_N_MOVES, min_game_duration: float = MIN_GAME_DURATION_SEC,
                 max_pause_duration: float = MAX_PAUSE_DURATION_SEC,
                 max_zscore_for_outlier: float = MAX_ZSCORE_FOR_OUTLIERS):
        self.input_data = preprocessed_data
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
        self.output_df = _calc_absolute_measures(self.input_data)
        # TODO: uncomment once get_vanilla_descriptors is ready:
        # vanilla_relative_measures = _calc_relative_measures(self.input_data, *get_vanilla_descriptors())
        # self.output_df = self.output_df.merge(vanilla_relative_measures, on=MEASURES_ID_KEY)
        self._apply_soft_filters()
        sample_relative_measures = _calc_relative_measures(self.input_data, *_get_sample_descriptors(self.input_data),
                                                           label="sample")
        self.output_df = self.output_df.merge(sample_relative_measures, on=MEASURES_ID_KEY)

        return self.output_df

    def _apply_soft_filters(self):
        """
        Applies absolute filters first, then sample-relative filters with the remaining sample.
        """
        for filter_getter in (self._get_absolute_filters, self._get_sample_relative_filters):
            masks, reasons = filter_getter()
            self._update_exclusion_info(masks, reasons)
            is_excluded = reduce(np.logical_or, masks)
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
