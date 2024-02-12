import pandas as pd
from datetime import datetime
from itertools import groupby
from MeasureCalculatorUtils import *
from consts import *
from utils import load_json, get_vanilla_descriptors
from collections.abc import Collection
from functools import reduce
from scipy.stats import zscore


class MeasureCalculator:
    def __init__(self, parsed_data_path: str, manually_excluded_ids: Collection = MANUALLY_EXCLUDED_IDS,
                 min_n_moves: int = MIN_N_MOVES, min_game_duration: float = MIN_GAME_DURATION_SEC,
                 max_pause_duration: float = MAX_PAUSE_DURATION_SEC,
                 max_zscore_for_outlier: float = MAX_ZSCORE_FOR_OUTLIERS):
        """
        Constructs a MeasureCalculator object.
        :param parsed_data_path: path for a .json file containing parsed data
        """
        self.input_data = load_json(parsed_data_path)
        self.output_df = None

        self.manually_excluded_ids = manually_excluded_ids
        self.min_n_moves = min_n_moves
        self.min_game_duration = min_game_duration
        self.max_pause_duration = max_pause_duration
        self.max_zscore_for_outlier = max_zscore_for_outlier
        self.exclusions = pd.DataFrame(columns=[MEASURES_ID_KEY, "reason"])

    def calc(self):
        self._get_sample_descriptors()
        self._calc_absolute_measures()
        self._calc_relative_measures(*get_vanilla_descriptors())
        self._apply_soft_filters()
        self._calc_relative_measures(*self._get_sample_descriptors(), label="sample")
        self.dump()

        return self

    def _get_sample_descriptors(self):
        sample_step_counter = Counter(get_all_players_steps(self.input_data))
        sample_step_orig_map = get_step_orig_map(self.input_data)

        return sample_step_counter, sample_step_orig_map

    def _calc_absolute_measures(self):
        absolute_measures = []
        for player_data in self.input_data:
            # supporting data
            shapes = preprocess_shapes(player_data[PARSED_ALL_SHAPES_KEY])
            explore, exploit = player_data[EXPLORE_KEY], player_data[EXPLOIT_KEY]
            dt = self._get_dt(shapes)

            # reusable measures
            total_play_time = sum(dt)
            n_moves = len(shapes)
            n_galleries = self._get_n_galleries(shapes)
            explore_efficiency = self._get_phase_efficiency(shapes, explore)
            exploit_efficiency = self._get_phase_efficiency(shapes, exploit)

            absolute_measures.append({
                MEASURES_ID_KEY: player_data["id"],
                "Date/Time": datetime.fromtimestamp(player_data["absolute start time"]).isoformat(),
                GAME_DURATION_KEY: total_play_time,
                N_MOVES_KEY: n_moves,
                "Average Speed": n_moves / total_play_time,
                "#galleries": n_galleries,
                "self avoidance": self._get_self_avoidance(shapes),
                "#clusters": len(exploit),
                "% galleries in exp": self._get_n_galleries_in_phase(shapes, explore) / n_galleries,
                "% time in exp": self._get_time_in_phase(dt, explore) / total_play_time,
                "exp efficiency": explore_efficiency,
                "scav efficiency": exploit_efficiency,
                "efficiency ratio": explore_efficiency / exploit_efficiency,
                MEDIAN_EXPLORE_LENGTH_KEY: 0,
                MEDIAN_EXPLOIT_LENGTH_KEY: 0,
                "exp speed": 0,
                "scav speed": 0,
                LONGEST_PAUSE_KEY: max(dt)
            })

        self.output_df = pd.DataFrame(absolute_measures)

    def _calc_relative_measures(self, step_counter, step_orig_map, id_list=None, label=None):
        if id_list is None:
            id_list = self.output_df[MEASURES_ID_KEY]
        label_ext = f"({label})" if label else ""

        relative_measures = []
        for player_data in self.input_data:
            player_id = player_data[PARSED_PLAYER_ID_KEY]
            if player_id in id_list:
                shapes = preprocess_shapes(player_data[PARSED_ALL_SHAPES_KEY])
                steps = get_steps(shapes)
                step_orig = [step_orig_map[step] for step in steps]

                relative_measures.append({
                    MEASURES_ID_KEY: player_id,
                    f"Step Orig {label_ext}": np.mean(step_orig),
                    f"% Steps Unique {label_ext}": self._get_frac_unique_steps(steps, step_counter),
                    f"Gallery Orig {label_ext}": 0,
                    f"Gallery Orig exp {label_ext}": 0,
                    f"Gallery Orig scav {label_ext}": 0,
                    f"% Galleries Unique {label_ext}": 0,
                    f"% Galleries Unique exp {label_ext}": 0,
                    f"% Galleries Unique scav {label_ext}": 0,
                    f"# clusters in GC {label_ext}": 0,
                    f"% clusters in GC {label_ext}": 0,
                })

        self.output_df.merge(relative_measures, on=MEASURES_ID_KEY)

    def _apply_soft_filters(self):
        """
        Applies absolute filters first, then sample-relative filters with the remaining sample.
        """
        for filter_getter in (self._get_absolute_filters, self._get_sample_relative_filters):
            masks, reasons = filter_getter()
            self._update_exclusion_info(masks, reasons)
            is_excluded = reduce(np.logical_or, masks)
            self.output_df = self.output_df.loc[~is_excluded, MEASURES_ID_KEY]

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

    def dump(self):
        # TODO: write csv with correct col order
        # TODO: document all filtered ids and filtering criteria
        # TODO: write html with dashboards to inspect data quality and some summary stats
        pass

    @staticmethod
    def _get_last_shape_time(shapes):
        last_shape = shapes[-1]
        if last_shape[SHAPE_SAVE_TIME_IDX] is None:
            return last_shape[SHAPE_MOVE_TIME_IDX]

        return last_shape[SHAPE_SAVE_TIME_IDX]

    @staticmethod
    def _get_n_galleries(shapes):
        is_saved_to_gallery = shapes[:, SHAPE_SAVE_TIME_IDX] != None
        return np.sum(is_saved_to_gallery)

    @staticmethod
    def _get_self_avoidance(shapes):
        shape_ids = shapes[:, SHAPE_ID_IDX]
        unique_shapes = np.unique(shape_ids)
        shapes_no_consecutive_duplicates = [k for k, g in groupby(shape_ids)]
        # TODO: in a future version, consecutive duplicates will be handled in post-parsing, before measure
        #  calculating. when this is implemented, there will be no need for shapes_no_consecutive_duplicates, it will be
        #  equivalent to shapes. Instead of its len, it would then make sense to use n_moves (defined in self.calc)

        return len(unique_shapes) / len(shapes_no_consecutive_duplicates)

    @staticmethod
    def _get_frac_unique_steps(steps, all_steps_counter):
        n_unique_steps = 0
        for step in steps:
            if all_steps_counter[step] == 1:
                n_unique_steps += 1

        return n_unique_steps / len(steps)

    @staticmethod
    def _get_n_galleries_in_phase(shapes, phase_slices):
        all_shapes_in_phase = []
        for start, end in phase_slices:
            all_shapes_in_phase.extend(shapes[start:end])
        all_shapes_in_phase = np.array(all_shapes_in_phase)

        return MeasureCalculator._get_n_galleries(all_shapes_in_phase)

    @staticmethod
    def _get_dt(shapes):
        shape_times = shapes[:, SHAPE_SAVE_TIME_IDX].copy()
        non_gallery_mask = shape_times == None
        shape_times[non_gallery_mask] = shapes[non_gallery_mask, SHAPE_MOVE_TIME_IDX]

        dt = np.diff(shape_times)
        return dt

    @staticmethod
    def _get_time_in_phase(dt, phase_slices):
        dt_in_phase = []
        for start, end in phase_slices:
            dt_in_phase.extend(dt[start:end])

        return sum(dt_in_phase)

    @staticmethod
    def _get_phase_efficiency(shapes: np.ndarray, phase_slices):
        actual_paths = {}
        for start, end in phase_slices:
            slice_shape_ids = shapes[start:end, SHAPE_ID_IDX]
            slice_is_gallery = shapes[start:end, SHAPE_SAVE_TIME_IDX] != None

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
        return 0


if __name__ == '__main__':
    # TODO: support run form command line, with defaults for all parameters
    path = r"test_file.json"
    mc = MeasureCalculator(path)
    mc.calc()
