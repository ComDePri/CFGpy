import json
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from CFGpy.behavioral.data_interfaces import PostparsedDataset
from CFGpy.behavioral._consts import (FEATURES_ID_KEY, FEATURES_START_TIME_KEY, N_CLUSTERS_KEY, GAME_DURATION_KEY,
                                      N_MOVES_KEY, LONGEST_PAUSE_KEY, MEDIAN_EXPLORE_LENGTH_KEY, N_GALLERIES_KEY,
                                      SELF_AVOIDANCE_KEY, EXPLORE_EFFICIENCY_KEY, EXPLOIT_EFFICIENCY_KEY,
                                      MEDIAN_EXPLOIT_LENGTH_KEY, AVERAGE_SPEED_KEY, FRACTION_GALLERY_IN_EXPLORE_KEY,
                                      FRACTION_TIME_IN_EXPLORE_KEY, EFFICIENCY_RATIO_KEY, EXPLORE_SPEED_KEY,
                                      EXPLOIT_SPEED_KEY, DEFAULT_FINAL_OUTPUT_FILENAME, EXCLUSION_REASON_KEY,
                                      STEP_ORIG_KEY, FRACTION_STEPS_UNIQUELY_COVERED_KEY, GALLERY_ORIG_KEY,
                                      GALLERY_ORIG_EXPLORE_KEY, GALLERY_ORIG_EXPLOIT_KEY,
                                      FRACTION_GALLERIES_UNIQUELY_COVERED_KEY, FRACTION_CLUSTERS_IN_GC_KEY,
                                      FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY,
                                      FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY, N_CLUSTERS_IN_GC_KEY,
                                      ABSOLUTE_FEATURES_MESSAGE, RELATIVE_FEATURES_MESSAGE, EXPLORE_OUTLIER_REASON,
                                      EXPLOIT_OUTLIER_REASON, NO_EXPLOIT_EXCLUSION_REASON, MANUAL_EXCLUSION_REASON,
                                      GAME_LENGTH_EXCLUSION_REASON, GAME_DURATION_EXCLUSION_REASON,
                                      PAUSE_EXCLUSION_REASON, SAMPLE_RELATIVE_FEATURES_LABEL)
from CFGpy.behavioral import Configuration
from CFGpy.behavioral._utils import load_json, is_semantic_connection
from functools import reduce
from scipy.stats import zscore
from CFGpy.utils import get_vanilla_stats, step_orig_map_factory, gallery_orig_map_factory
from tqdm import tqdm


def _get_frac_uniquely_covered(player_objects, objects_not_uniquely_covered):
    set_player_objects = set(player_objects)
    n_unique_player_objects = len(set_player_objects)
    if not n_unique_player_objects:
        return None

    n_not_uniquely_covered = len(set_player_objects & set(objects_not_uniquely_covered))
    frac_not_uniquely_covered = n_not_uniquely_covered / n_unique_player_objects
    frac_uniquely_covered = 1 - frac_not_uniquely_covered
    return frac_uniquely_covered


class FeatureExtractor:
    def __init__(self, preprocessed_data, config: Configuration = None):
        self.input_data = PostparsedDataset(input_data=preprocessed_data, config=config)
        self.config = config if config is not None else Configuration.default()
        self.all_absolute_features = None
        self.output_df = None
        self.exclusions = pd.DataFrame(columns=[FEATURES_ID_KEY, EXCLUSION_REASON_KEY])
    
    @classmethod
    def from_json(cls, path: str, config=Configuration.default()):
        return cls(load_json(path), config)

    def extract(self, verbose=False):
        self.all_absolute_features = self._extract_absolute_features(verbose)
        self.output_df = self.all_absolute_features.copy()
        self._drop_nonfirst_games()
        vanilla_relative_features = self._extract_relative_features(get_vanilla_stats(), verbose=verbose)
        self.output_df = self.output_df.merge(vanilla_relative_features, on=FEATURES_ID_KEY)
        self._apply_soft_filters()
        sample_relative_features = self._extract_relative_features(self.input_data.get_stats(), verbose=verbose,
                                                                   label=SAMPLE_RELATIVE_FEATURES_LABEL)
        self.output_df = self.output_df.merge(sample_relative_features, on=FEATURES_ID_KEY, how="left")
        return self.output_df

    def dump(self, path=DEFAULT_FINAL_OUTPUT_FILENAME):
        self.output_df.to_csv(path, index=False)  # reorder columns
        self.exclusions.to_csv(f"{path}_exclusions.csv", index=False)
        self.config.to_yaml(path)

        # TODO: document all filtered ids and filtering criteria
        # TODO: write html with dashboards to inspect data quality and some summary stats

    def is_cluster_in_GC(self, cluster, GC):
        for GC_cluster in GC:
            if is_semantic_connection(cluster, GC_cluster, self.config.MIN_OVERLAP_FOR_SEMANTIC_CONNECTION):
                return True

        return False

    def get_all_absolute_features(self):
        return self.all_absolute_features

    def _drop_nonfirst_games(self):
        """
        Keeps only the first game from each player. Allows functions downstream to assume unique IDs.
        """
        self.input_data.drop_non_first_games()
        self.output_df = (self.output_df.
                          sort_values(by=[FEATURES_START_TIME_KEY], ascending=True).
                          drop_duplicates(subset=[FEATURES_ID_KEY], keep="first").
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
        Absolute filters are based on absolute features, can be applied independently of each other. Each filter is
        represented by a textual description and a mask with **True for players to exclude**, False for players to keep.
        :return: masks, reasons.
        """
        reasons = (MANUAL_EXCLUSION_REASON, NO_EXPLOIT_EXCLUSION_REASON, GAME_LENGTH_EXCLUSION_REASON,
                   GAME_DURATION_EXCLUSION_REASON, PAUSE_EXCLUSION_REASON)
        masks = (self.output_df[FEATURES_ID_KEY].isin(self.config.MANUALLY_EXCLUDED_IDS),
                 self.output_df[N_CLUSTERS_KEY] < self.config.MIN_N_CLUSTERS,
                 self.output_df[N_MOVES_KEY] < self.config.MIN_N_MOVES,
                 self.output_df[GAME_DURATION_KEY] < self.config.MIN_GAME_DURATION_SEC,
                 self.output_df[LONGEST_PAUSE_KEY] > self.config.MAX_PAUSE_DURATION_SEC)

        return masks, reasons

    def _get_sample_relative_filters(self):
        """
        Each filter is represented by a textual description and a mask with **True for players to exclude**, False for
        players to keep.
        :return: masks, reasons.
        """
        reasons = (EXPLORE_OUTLIER_REASON, EXPLOIT_OUTLIER_REASON)
        zscores = self.output_df[[MEDIAN_EXPLORE_LENGTH_KEY, MEDIAN_EXPLOIT_LENGTH_KEY]].apply(zscore)
        masks = (abs(zscores[MEDIAN_EXPLORE_LENGTH_KEY]) > self.config.MAX_ZSCORE_FOR_OUTLIERS,
                 abs(zscores[MEDIAN_EXPLOIT_LENGTH_KEY]) > self.config.MAX_ZSCORE_FOR_OUTLIERS)

        return masks, reasons

    def _update_exclusion_info(self, masks, reasons):
        """
        Updates self.to_exclude based on filters results.
        :param masks: a collection of masks, each has **True for players to exclude**, false for players to keep.
        :param reasons: a collection of strings describing exclusion reasons for the masks.
        """
        for is_excluded, reason in zip(masks, reasons):
            ids_to_exclude = self.output_df.loc[is_excluded, FEATURES_ID_KEY]
            current_exclusion = pd.DataFrame({
                FEATURES_ID_KEY: ids_to_exclude,
                EXCLUSION_REASON_KEY: [reason] * len(ids_to_exclude)
            })
            self.exclusions = pd.concat((self.exclusions, current_exclusion))

    def _extract_absolute_features(self, verbose=True):
        n_galleries_in_explore = []
        total_explore_times = []
        total_exploit_times = []
        total_explore_lengths = []
        total_exploit_lengths = []

        iterator = self.input_data
        if verbose:
            print(ABSOLUTE_FEATURES_MESSAGE)
            iterator = tqdm(iterator)

        absolute_features = []
        for player_data in iterator:
            # pre-calculations
            explore_lengths = [end - start for start, end in player_data.explore_slices]
            exploit_lengths = [end - start for start, end in player_data.exploit_slices]
            is_gallery = player_data.get_gallery_mask()
            is_explore = player_data.get_explore_mask()

            # data collection for later vectorized operations
            n_galleries_in_explore.append(sum(is_gallery & is_explore))
            total_explore_times.append(player_data.total_explore_time())
            total_exploit_times.append(player_data.total_exploit_time())
            total_explore_lengths.append(sum(explore_lengths))
            total_exploit_lengths.append(sum(exploit_lengths))

            # player-wise calculations
            explore_efficiency, exploit_efficiency = player_data.get_efficiency()
            absolute_features.append({
                FEATURES_ID_KEY: player_data.id,
                FEATURES_START_TIME_KEY: datetime.fromtimestamp(player_data.start_time).isoformat(),
                GAME_DURATION_KEY: player_data.get_last_action_time(),
                N_MOVES_KEY: len(player_data),
                N_GALLERIES_KEY: sum(is_gallery),
                SELF_AVOIDANCE_KEY: player_data.get_self_avoidance(),
                N_CLUSTERS_KEY: len(player_data.exploit_slices),
                EXPLORE_EFFICIENCY_KEY: explore_efficiency,
                EXPLOIT_EFFICIENCY_KEY: exploit_efficiency,
                MEDIAN_EXPLORE_LENGTH_KEY: np.median(explore_lengths),
                MEDIAN_EXPLOIT_LENGTH_KEY: np.median(exploit_lengths),
                LONGEST_PAUSE_KEY: player_data.get_max_pause_duration()
            })

        # vectorized operations
        features_df = pd.DataFrame(absolute_features)
        features_df[AVERAGE_SPEED_KEY] = features_df[N_MOVES_KEY] / features_df[GAME_DURATION_KEY]
        features_df[FRACTION_GALLERY_IN_EXPLORE_KEY] = pd.Series(n_galleries_in_explore) / features_df[N_GALLERIES_KEY]
        features_df[FRACTION_TIME_IN_EXPLORE_KEY] = pd.Series(total_explore_times) / features_df[GAME_DURATION_KEY]
        features_df[EFFICIENCY_RATIO_KEY] = features_df[EXPLORE_EFFICIENCY_KEY] / features_df[EXPLOIT_EFFICIENCY_KEY]
        features_df[EXPLORE_SPEED_KEY] = pd.Series(total_explore_lengths) / pd.Series(total_explore_times)
        features_df[EXPLOIT_SPEED_KEY] = pd.Series(total_exploit_lengths) / pd.Series(total_exploit_times)

        return features_df

    def _extract_relative_features(self, stats, label=None, verbose=False):
        steps_not_uniquely_covered, step_counter, galleries_not_uniquely_covered, gallery_counter, GC = stats
        label_ext = f" ({label})" if label else ""

        step_orig_map = step_orig_map_factory(step_counter, alpha=self.config.STEP_ORIG_PSEUDOCOUNT,
                                              d=self.config.STEP_ORIG_N_CATEGORIES)
        gallery_orig_map = gallery_orig_map_factory(gallery_counter, alpha=self.config.GALLERY_ORIG_PSEUDOCOUNT,
                                                    d=self.config.GALLERY_ORIG_N_CATEGORIES)

        iterator = self.input_data
        if verbose:
            print(RELATIVE_FEATURES_MESSAGE.format(label_ext))
            iterator = tqdm(iterator)

        relative_features = []
        for player_data in iterator:
            steps = player_data.get_steps()
            step_orig = [step_orig_map[step] for step in steps]
            gallery_ids = player_data.get_gallery_ids()
            gallery_orig = np.array([gallery_orig_map[shape_id] for shape_id in gallery_ids])
            is_gallery = player_data.get_gallery_mask()
            is_explore_given_gallery = player_data.get_explore_mask()[is_gallery]
            is_exploit_given_gallery = ~is_explore_given_gallery
            exploit_clusters = player_data.get_exploit_clusters()
            n_clusters_in_GC = sum([self.is_cluster_in_GC(cluster, GC) for cluster in exploit_clusters])
            frac_clusters_in_GC = (n_clusters_in_GC / len(player_data.exploit_slices)
                                   if player_data.exploit_slices else None)

            relative_features.append({
                FEATURES_ID_KEY: player_data.id,
                f"{STEP_ORIG_KEY}{label_ext}": np.mean(step_orig),
                f"{FRACTION_STEPS_UNIQUELY_COVERED_KEY}{label_ext}":
                    _get_frac_uniquely_covered(steps, steps_not_uniquely_covered),
                f"{GALLERY_ORIG_KEY}{label_ext}": np.mean(gallery_orig),
                f"{GALLERY_ORIG_EXPLORE_KEY}{label_ext}": np.mean(gallery_orig[is_explore_given_gallery]),
                f"{GALLERY_ORIG_EXPLOIT_KEY}{label_ext}": np.mean(gallery_orig[is_exploit_given_gallery]),
                f"{FRACTION_GALLERIES_UNIQUELY_COVERED_KEY}{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids, galleries_not_uniquely_covered),
                f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY}{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids[is_explore_given_gallery], galleries_not_uniquely_covered),
                f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY}{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids[is_exploit_given_gallery], galleries_not_uniquely_covered),
                f"{N_CLUSTERS_IN_GC_KEY}{label_ext}": n_clusters_in_GC,
                f"{FRACTION_CLUSTERS_IN_GC_KEY}{label_ext}": frac_clusters_in_GC,
            })

        return pd.DataFrame(relative_features)


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description="Extract features from a post-parsed CFG data")
    argparser.add_argument("-i", "--input", dest="input_filename",
                           help='Filename of post-parsed data JSON')
    argparser.add_argument("-o", "--output", default=DEFAULT_FINAL_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output CSV')
    args = argparser.parse_args()

    fe = FeatureExtractor.from_json(args.input_filename)
    fe.extract(verbose=True)
    fe.dump(args.output_filename)
