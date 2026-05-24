import os
import numpy as np
import pandas as pd
from datetime import datetime
from CFGpy.behavioral.data_interfaces import PostparsedDataset, PostParsedDatasetStats, PostparsedPlayerData, \
    get_vanilla_stats
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
                                      PAUSE_EXCLUSION_REASON, SAMPLE_RELATIVE_FEATURES_LABEL, G_KEY, ALPHA_KEY,
                                      DEFAULT_POSTPARSED_FILTERED_OUTPUT_FILENAME)
from CFGpy.behavioral import Configuration
from CFGpy.behavioral._utils import (load_json, is_semantic_connection, resolve_path, median_handle_empty as median,
                                     mean_handle_empty as mean)
from CFGpy.behavioral._logging import HasLogger
from functools import reduce
from scipy.stats import zscore
from CFGpy.utils import step_orig_map_factory, gallery_orig_map_factory

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


class FeatureExtractor(HasLogger):
    def __init__(self, *, preprocessed_data, config: Configuration = None, logger=None):
        super().__init__(logger)
        self.input_data = PostparsedDataset(input_data=preprocessed_data, config=config)
        self.config = config if config is not None else Configuration.default()
        self.all_absolute_features = None
        self.output_df = None
        self.exclusions = pd.DataFrame(columns=[FEATURES_ID_KEY, EXCLUSION_REASON_KEY])

    @classmethod
    def from_json(cls, path: str, config: Configuration = None):
        if config is None:
            config = Configuration.default()
        return cls(preprocessed_data=load_json(path), config=config)

    def _log_missing_values(self):
        # check for NaN values in the output_df and log a warning for each column that contains them, with the number of NaN values in that column, and the player IDs for which the NaN values appear
        for column in self.output_df.columns:
            n_missing = self.output_df[column].isna().sum()
            if n_missing > 0:
                missing_ids = self.output_df.loc[self.output_df[column].isna(), FEATURES_ID_KEY].tolist()
                self.log_warning(
                    f"Column '{column}' contains {n_missing} missing values for player IDs: {missing_ids}. Consider investigating the cause of these missing values and whether they should be imputed or lead to exclusion of the affected players.")

    def extract(self, verbose=False):
        self.all_absolute_features = self._extract_absolute_features(verbose)
        self.output_df = self.all_absolute_features.copy()
        # remove very short games before keeping only the first game per player
        self._drop_short_games()
        self.log_info("Keeping only the first game per player...")
        self._drop_nonfirst_games()
        vanilla_relative_features = self._extract_relative_features(get_vanilla_stats(), verbose=verbose)
        self.output_df = self.output_df.merge(vanilla_relative_features, on=FEATURES_ID_KEY)
        self.log_info(f"Applying soft filters...")
        self._apply_soft_filters()
        if len(self.output_df) > 0:
            sample_relative_features = self._extract_relative_features(self.input_data.get_stats(), verbose=verbose,
                                                                       label=SAMPLE_RELATIVE_FEATURES_LABEL)
            self.output_df = self.output_df.merge(sample_relative_features, on=FEATURES_ID_KEY, how="left")
        self._log_missing_values()
        return self.output_df

    def dump(self, name: str = None, path: str = None, with_config=True, with_exclusions=True, with_filtered_postparsed=True):
        measures_path = resolve_path(name=name, path=path, default_suffix=DEFAULT_FINAL_OUTPUT_FILENAME)
        # make sure the directory exists
        measures_dir = os.path.dirname(measures_path)
        if measures_dir and not os.path.exists(measures_dir):
            os.makedirs(measures_dir)

        self.output_df.to_csv(measures_path, index=False)
        if with_filtered_postparsed:
            postparsed_path = measures_path.replace(".csv", "") + f"_{DEFAULT_POSTPARSED_FILTERED_OUTPUT_FILENAME}"
            self.input_data.dump(postparsed_path, prettify=True)

        if with_exclusions:
            exclusions_path = measures_path.replace(".csv", "") + "_exclusions.csv"
            self.exclusions.to_csv(exclusions_path, index=False)
        if with_config:
            self.config.to_yaml(measures_path.replace(".csv", ""))
        return measures_path

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
        # check all IDs for which we find more than one game, and log a warning for each of them, since this is not expected but we want to be robust to it anyway
        id_counts = self.output_df[FEATURES_ID_KEY].value_counts()
        non_unique_ids = id_counts[id_counts > 1].index
        for user_id in non_unique_ids:
            self.log_warning(
                f"Found multiple games for player id {user_id} in the data. Only the first game will be kept for further processing.")
        self.output_df = (self.output_df.
                          sort_values(by=[FEATURES_START_TIME_KEY], ascending=True).
                          drop_duplicates(subset=[FEATURES_ID_KEY], keep="first").
                          reset_index(drop=True))

    def _apply_soft_filters(self):
        """
        Applies absolute filters first, then sample-relative filters with the remaining sample.
        """
        for filter_getter, filters_cat in zip((self._get_absolute_filters, self._get_sample_relative_filters),
                                              ("absolute filters", "sample-relative filters")):
            if len(self.output_df) == 0:
                self.log_info("No games left - skipping soft filtering")
                return
            unfiltered_df = self.output_df.copy()  # for logging purposes
            self.log_info(f"Applying {filters_cat}...")
            masks, reasons = filter_getter()
            self._update_exclusion_info(masks, reasons)
            is_excluded = reduce(np.logical_or, masks)
            self.log_info(f"Excluding {is_excluded.sum()} players based on {filters_cat}...")
            self.input_data.filter(~is_excluded)
            self.output_df = self.output_df.loc[~is_excluded].reset_index(drop=True)
            # log filtering
            for reason, mask in zip(reasons, masks):
                n_excluded = mask.sum()
                self.log_info(f"Found {n_excluded} to-be-excluded players based on filter: {reason}...")
                if n_excluded > 0:
                    excluded_ids = unfiltered_df.loc[mask, FEATURES_ID_KEY].tolist()
                    self.log_info(f"Excluded player IDs for reason '{reason}': {excluded_ids}")

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

    def _write_exclusions(self, ids_to_exclude, reason):
        current_exclusion = pd.DataFrame({
            FEATURES_ID_KEY: ids_to_exclude,
            EXCLUSION_REASON_KEY: [reason] * len(ids_to_exclude)
        })
        self.exclusions = pd.concat((self.exclusions, current_exclusion))

    def _update_exclusion_info(self, masks, reasons):
        """
        Updates self.to_exclude based on filters results.
        :param masks: a collection of masks, each has **True for players to exclude**, false for players to keep.
        :param reasons: a collection of strings describing exclusion reasons for the masks.
        """
        for is_excluded, reason in zip(masks, reasons):
            ids_to_exclude = self.output_df.loc[is_excluded, FEATURES_ID_KEY]
            self._write_exclusions(ids_to_exclude, reason)

    def _extract_absolute_features(self, verbose=True):
        n_galleries_in_explore = []
        total_explore_times = []
        total_exploit_times = []
        total_explore_lengths = []
        total_exploit_lengths = []

        iterator = self.input_data
        self.log_info(ABSOLUTE_FEATURES_MESSAGE)
        if verbose:
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
                MEDIAN_EXPLORE_LENGTH_KEY: median(explore_lengths),
                MEDIAN_EXPLOIT_LENGTH_KEY: median(exploit_lengths),
                LONGEST_PAUSE_KEY: player_data.get_max_pause_duration()
            })

        ABS_FEATURES_COLS = [FEATURES_ID_KEY, FEATURES_START_TIME_KEY, GAME_DURATION_KEY, N_MOVES_KEY, N_GALLERIES_KEY,
                             SELF_AVOIDANCE_KEY, N_CLUSTERS_KEY, EXPLORE_EFFICIENCY_KEY, EXPLOIT_EFFICIENCY_KEY,
                             MEDIAN_EXPLORE_LENGTH_KEY, MEDIAN_EXPLOIT_LENGTH_KEY, LONGEST_PAUSE_KEY]
        # vectorized operations
        features_df = pd.DataFrame(absolute_features, columns=ABS_FEATURES_COLS)
        if len(features_df) > 0:
            features_df[AVERAGE_SPEED_KEY] = features_df[N_MOVES_KEY] / features_df[GAME_DURATION_KEY]
            features_df[FRACTION_GALLERY_IN_EXPLORE_KEY] = pd.Series(n_galleries_in_explore) / features_df[
                N_GALLERIES_KEY]
            features_df[FRACTION_TIME_IN_EXPLORE_KEY] = pd.Series(total_explore_times) / features_df[GAME_DURATION_KEY]
            features_df[EFFICIENCY_RATIO_KEY] = features_df[EXPLORE_EFFICIENCY_KEY] / features_df[
                EXPLOIT_EFFICIENCY_KEY]
            features_df[EXPLORE_SPEED_KEY] = pd.Series(total_explore_lengths) / pd.Series(total_explore_times)
            features_df[EXPLOIT_SPEED_KEY] = pd.Series(total_exploit_lengths) / pd.Series(total_exploit_times)
        else:
            # add columns to empty dataframe:
            features_df[[AVERAGE_SPEED_KEY,
                         FRACTION_GALLERY_IN_EXPLORE_KEY,
                         FRACTION_TIME_IN_EXPLORE_KEY,
                         EFFICIENCY_RATIO_KEY,
                         EXPLORE_SPEED_KEY,
                         EXPLOIT_SPEED_KEY]] = None

        return features_df

    def _extract_relative_features(self, stats: PostParsedDatasetStats, label=None, verbose=False):
        steps_not_uniquely_covered = stats.steps_not_uniquely_covered
        step_counter = stats.n_times_step_taken
        galleries_not_uniquely_covered = stats.galleries_not_uniquely_covered
        gallery_counter = stats.n_times_gallery_saved
        GC = stats.giant_component

        label_ext = f" ({label})" if label else ""

        step_orig_map = step_orig_map_factory(step_counter, alpha=self.config.STEP_ORIG_PSEUDOCOUNT,
                                              d=self.config.STEP_ORIG_N_CATEGORIES)
        gallery_orig_map = gallery_orig_map_factory(gallery_counter, alpha=self.config.GALLERY_ORIG_PSEUDOCOUNT,
                                                    d=self.config.GALLERY_ORIG_N_CATEGORIES)

        iterator = self.input_data
        self.log_info(RELATIVE_FEATURES_MESSAGE.format(label_ext))
        if verbose:
            iterator = tqdm(iterator)
        player_data: PostparsedPlayerData = None  # for type hinting, can be removed without affecting functionality
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
            med_exploit_length = player_data.get_median_exploit_length()
            med_explore_length = player_data.get_median_explore_length()
            if (med_explore_length is None) or (med_exploit_length is None):
                g, alpha = None, None
            else:
                med_exploit_z = (med_exploit_length - stats.median_exploit_mean) / stats.median_exploit_std
                med_explore_z = (med_explore_length - stats.median_explore_mean) / stats.median_explore_std
                factor = 1 / (2 ** 0.5)  # to keep the result similar to the PCA computation
                g = -1 * (
                        factor * med_explore_z + factor * med_exploit_z)  # low steps -> high switching rate
                alpha = factor * med_exploit_z - factor * med_explore_z

            relative_features.append({
                FEATURES_ID_KEY: player_data.id,
                f"{STEP_ORIG_KEY}{label_ext}": mean(step_orig),
                f"{FRACTION_STEPS_UNIQUELY_COVERED_KEY}{label_ext}":
                    _get_frac_uniquely_covered(steps, steps_not_uniquely_covered),
                f"{GALLERY_ORIG_KEY}{label_ext}": mean(gallery_orig),
                f"{GALLERY_ORIG_EXPLORE_KEY}{label_ext}": mean(gallery_orig[is_explore_given_gallery]),
                f"{GALLERY_ORIG_EXPLOIT_KEY}{label_ext}": mean(gallery_orig[is_exploit_given_gallery]),
                f"{FRACTION_GALLERIES_UNIQUELY_COVERED_KEY}{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids, galleries_not_uniquely_covered),
                f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY}{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids[is_explore_given_gallery], galleries_not_uniquely_covered),
                f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY}{label_ext}":
                    _get_frac_uniquely_covered(gallery_ids[is_exploit_given_gallery], galleries_not_uniquely_covered),
                f"{N_CLUSTERS_IN_GC_KEY}{label_ext}": n_clusters_in_GC,
                f"{FRACTION_CLUSTERS_IN_GC_KEY}{label_ext}": frac_clusters_in_GC,
                f"{G_KEY}{label_ext}": g,
                f"{ALPHA_KEY}{label_ext}": alpha
            })
        columns = None
        if len(relative_features)== 0:
            columns = [FEATURES_ID_KEY, f"{STEP_ORIG_KEY}{label_ext}",
                            f"{FRACTION_STEPS_UNIQUELY_COVERED_KEY}{label_ext}", f"{GALLERY_ORIG_KEY}{label_ext}",
                            f"{GALLERY_ORIG_EXPLORE_KEY}{label_ext}",f"{GALLERY_ORIG_EXPLOIT_KEY}{label_ext}",
                            f"{FRACTION_GALLERIES_UNIQUELY_COVERED_KEY}{label_ext}",
                            f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY}{label_ext}",
                            f"{FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY}{label_ext}",
                            f"{N_CLUSTERS_IN_GC_KEY}{label_ext}",
                            f"{FRACTION_CLUSTERS_IN_GC_KEY}{label_ext}",
                            f"{G_KEY}{label_ext}",
                            f"{ALPHA_KEY}{label_ext}"
                            ]
        return pd.DataFrame(relative_features, columns=columns)

    def _drop_short_games(self):
        if self.config.MAX_IGNORED_GAME_DURATION_SEC <= 0:
            return
        self.log_info(
            f"Dropping short games below 'MAX_IGNORED_GAME_DURATION_SEC'={self.config.MAX_IGNORED_GAME_DURATION_SEC} seconds...")
        is_dropped = self.output_df[GAME_DURATION_KEY] < self.config.MAX_IGNORED_GAME_DURATION_SEC
        # if we removed all games of a player, we need to update the exclusions
        # we first check for ids that are now completely excluded with a boolean mask
        excluded_ids_mask = self.output_df.groupby(FEATURES_ID_KEY)[
                                GAME_DURATION_KEY].max() < self.config.MAX_IGNORED_GAME_DURATION_SEC
        # now we take the FEATURES_ID_KEY values for which the mask is True, which means all their games are dropped
        ids_now_excluded = excluded_ids_mask[excluded_ids_mask].index
        # update the exclusions table
        self._write_exclusions(ids_now_excluded, GAME_DURATION_EXCLUSION_REASON)
        # drop the short games for both excluded participants and the rest
        self.input_data.filter(~is_dropped)
        self.output_df = self.output_df.loc[~is_dropped].reset_index(drop=True)
