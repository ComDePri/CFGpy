import numpy as np
import pandas as pd
from datetime import datetime
from CFGpy.behavioral.data_structs import PreprocessedDataset
from CFGpy.behavioral._consts import *
from CFGpy.behavioral._utils import load_json, is_semantic_connection
from collections.abc import Collection
from functools import reduce
from scipy.stats import zscore
from CFGpy.utils import get_vanilla_descriptors


def _get_frac_uniquely_covered(player_objects, objects_not_uniquely_covered):
    n_not_uniquely_covered = len(set(player_objects) & set(objects_not_uniquely_covered))
    frac_not_uniquely_covered = n_not_uniquely_covered / len(set(player_objects))
    frac_uniquely_covered = 1 - frac_not_uniquely_covered
    return frac_uniquely_covered


def is_cluster_in_GC(cluster, GC):
    for GC_cluster in GC:
        if is_semantic_connection(cluster, GC_cluster):
            return True

    return False


class MeasureCalculator:
    def __init__(self, preprocessed_data, manually_excluded_ids: Collection = MANUALLY_EXCLUDED_IDS,
                 min_n_moves: int = MIN_N_MOVES, min_game_duration: float = MIN_GAME_DURATION_SEC,
                 max_pause_duration: float = MAX_PAUSE_DURATION_SEC,
                 max_zscore_for_outlier: float = MAX_ZSCORE_FOR_OUTLIERS):
        self.input_data = PreprocessedDataset(preprocessed_data)
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
        self.all_absolute_measures = self._calc_absolute_measures()
        self.output_df = self.all_absolute_measures.copy()
        self._drop_nonfirst_games()
        vanilla_relative_measures = self._calc_relative_measures(get_vanilla_descriptors())
        self.output_df = self.output_df.merge(vanilla_relative_measures, on=MEASURES_ID_KEY)
        self._apply_soft_filters()
        sample_relative_measures = self._calc_relative_measures(self.input_data.get_descriptors(), label="sample")
        self.output_df = self.output_df.merge(sample_relative_measures, on=MEASURES_ID_KEY, how="left")

        return self.output_df

    def dump(self, path=DEFAULT_FINAL_OUTPUT_FILENAME):
        self.output_df.to_csv(path, index=False)  # reorder columns
        # TODO: document all filtered ids and filtering criteria
        # TODO: write html with dashboards to inspect data quality and some summary stats

    def get_all_absolute_measures(self):
        return self.all_absolute_measures

    def _drop_nonfirst_games(self):
        """
        Keeps only the first game from each player. Allows functions downstream to assume unique IDs.
        """
        self.input_data.drop_non_first_games()
        self.output_df = (self.output_df.
                          sort_values(by=[MEASURES_START_TIME_KEY], ascending=True).
                          drop_duplicates(subset=[MEASURES_ID_KEY], keep="first").
                          reset_index(drop=True))

    def _apply_soft_filters(self):
        """
        Applies absolute filters first, then sample-relative filters with the remaining sample.
        """
        for filter_getter in (self._get_absolute_filters, self._get_sample_relative_filters):
            masks, reasons = filter_getter()
            self._update_exclusion_info(masks, reasons)
            is_excluded = reduce(np.logical_or, masks)

            self.input_data.filter(~is_excluded)
            self.output_df = self.output_df.loc[~is_excluded].reset_index(drop=True)

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

    def _calc_absolute_measures(self):
        absolute_measures = []
        for player_data in self.input_data:
            # supporting data
            time_in_explore = player_data.total_explore_time()
            time_in_exploit = player_data.total_exploit_time()
            explore_lengths = [end - start for start, end in player_data.explore_slices]
            exploit_lengths = [end - start for start, end in player_data.exploit_slices]
            is_gallery = player_data.get_gallery_mask()
            is_explore = player_data.get_explore_mask()

            # reusable measures
            last_action_time = player_data.get_last_action_time()
            n_moves = len(player_data)
            n_galleries = sum(is_gallery)
            explore_efficiency, exploit_efficiency = player_data.get_efficiency()

            absolute_measures.append({
                MEASURES_ID_KEY: player_data.id,
                MEASURES_START_TIME_KEY: datetime.fromtimestamp(player_data.start_time).isoformat(),
                GAME_DURATION_KEY: last_action_time,
                N_MOVES_KEY: n_moves,
                "Average Speed": n_moves / last_action_time,
                "#galleries": n_galleries,
                "self avoidance": player_data.get_self_avoidance(),
                "#clusters": len(player_data.exploit_slices),
                "% galleries in exp": sum(is_gallery & is_explore) / n_galleries,
                "% time in exp": time_in_explore / last_action_time,
                "exp efficiency": explore_efficiency,
                "scav efficiency": exploit_efficiency,
                "efficiency ratio": explore_efficiency / exploit_efficiency,
                MEDIAN_EXPLORE_LENGTH_KEY: np.median(explore_lengths),
                MEDIAN_EXPLOIT_LENGTH_KEY: np.median(exploit_lengths),
                "exp speed": sum(explore_lengths) / time_in_explore,
                "scav speed": sum(exploit_lengths) / time_in_exploit,
                LONGEST_PAUSE_KEY: player_data.get_max_pause_duration()
            })

        return pd.DataFrame(absolute_measures)

    def _calc_relative_measures(self, descriptors, label=None):
        steps_not_uniquely_covered, step_orig_map, galleries_not_uniquely_covered, gallery_orig_map, GC = descriptors
        label_ext = f" ({label})" if label else ""

        relative_measures = []
        for player_data in self.input_data:
            steps = player_data.get_steps()
            step_orig = [step_orig_map[step] for step in steps]
            gallery_ids = player_data.get_gallery_ids()
            gallery_orig = np.array([gallery_orig_map[shape_id] for shape_id in gallery_ids])
            is_gallery = player_data.get_gallery_mask()
            is_explore_given_gallery = player_data.get_explore_mask()[is_gallery]
            is_exploit_given_gallery = ~is_explore_given_gallery
            exploit_clusters = player_data.get_exploit_clusters()
            n_clusters_in_GC = sum([is_cluster_in_GC(cluster, GC) for cluster in exploit_clusters])

            relative_measures.append({
                MEASURES_ID_KEY: player_data.id,
                f"Step Orig{label_ext}": np.mean(step_orig),
                f"% steps uniquely covered{label_ext}": _get_frac_uniquely_covered(steps, steps_not_uniquely_covered),
                f"Gallery Orig{label_ext}": np.mean(gallery_orig),
                f"Gallery Orig exp{label_ext}": np.mean(gallery_orig[is_explore_given_gallery]),
                f"Gallery Orig scav{label_ext}": np.mean(gallery_orig[is_exploit_given_gallery]),
                f"% galleries uniquely covered{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids, galleries_not_uniquely_covered),
                f"% galleries uniquely covered exp{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids[is_explore_given_gallery], galleries_not_uniquely_covered),
                f"% galleries uniquely covered scav{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids[is_exploit_given_gallery], galleries_not_uniquely_covered),
                f"# clusters in GC{label_ext}": n_clusters_in_GC,
                f"% clusters in GC{label_ext}": n_clusters_in_GC / len(player_data.exploit_slices),
            })

        return pd.DataFrame(relative_measures)


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
    mc = MeasureCalculator(preprocessed_data, manually_excluded_ids=["double"])
    mc.calc()
    mc.dump(args.output_filename)
