import json
from itertools import groupby

import pandas as pd

from CFGpy.behavioral._utils import load_json, CFGPipelineException, segment_explore_exploit, \
    segment_explore_exploit_mri, prettify_games_json
from CFGpy.behavioral._utils import load_json, CFGPipelineException, segment_explore_exploit, prettify_games_json, resolve_path
from CFGpy.behavioral._consts import (PARSED_ALL_SHAPES_KEY, PARSED_PLAYER_ID_KEY, EXPLORE_KEY, EXPLOIT_KEY,
                                      INVALID_SHAPE_ERROR, NOT_A_NEIGHBOR_ERROR, POSTPARSER_OUTPUT_FILENAME,
                                      ROBUST_MEDIAN_PACE_KEY, ROBUST_THRESHOLD_KEY,
                                      INVALID_SEGMENTATION_ALGORITHM_ERROR, SEG_ALG_VANILLA, SEG_ALG_MRI, VALID_SEGMENTATION_ALGORITHMS)
from CFGpy.behavioral import Configuration
from CFGpy.utils import FilesHandler
from CFGpy.behavioral._logging import HasLogger


def is_valid_transition(shape1: int, shape2: int) -> bool:
    """
    Checks whether a transition is a valid path in the CFG.
    TODO: assumes empty moves are valid. When empty moves handling is implemented, this function can be replaced with
        shape_network.has_edge(shape1, shape2)
    :param shape1: shape id, after conversion to int by PostParser.convert_shape_ids
    :param shape2: shape id, after conversion to int by PostParser.convert_shape_ids
    :return: True if the transition is valid, False if not
    """
    return shape1 == shape2 or FilesHandler().shape_network.has_edge(shape1, shape2)


class PostParser(HasLogger):
    def __init__(self, *, parsed_data, config: Configuration = None, logger=None):
        super().__init__(logger)
        self.all_players_data = parsed_data
        self.config = config or Configuration.default()

    @classmethod
    def from_json(cls, path: str, config=None):
        return cls(parsed_data=load_json(path), config=config)

    def postparse(self):
        self.convert_shape_ids()
        self.handle_empty_moves()
        self.add_explore_exploit()
        return self.all_players_data

    def convert_shape_ids(self):
        """
        Converts shape ids from their graphical representations to serial numbers.
        Raises an exception if illegal shapes are found.
        """
        from CFGpy.utils import binary_shape_to_id as bin2id

        for player_data in self.all_players_data:
            shapes = player_data[PARSED_ALL_SHAPES_KEY]
            for i, shape in enumerate(shapes):
                shape_binary_repr = shape[self.config.SHAPE_ID_IDX]
                player_id = player_data[PARSED_PLAYER_ID_KEY]
                try:
                    shape_id = bin2id(shape_binary_repr)
                    shape[self.config.SHAPE_ID_IDX] = shape_id
                except ValueError:
                    raise CFGPipelineException(INVALID_SHAPE_ERROR.format(shape_binary_repr, player_id))

                if i > 0 and not is_valid_transition(shapes[i - 1][self.config.SHAPE_ID_IDX], shape_id):
                    self.log_warning(str(CFGPipelineException(NOT_A_NEIGHBOR_ERROR.format(i - 1, i, player_id))))
                    # the exception is printed and not raised because many gaps are actually in the source data

    @staticmethod
    def group_consecutive_duplicates(elements):
        """
        Returns a list of group ids such that each group contains consecutive duplicate elements.
        :param elements: iterable
        :return: 1D list with len equal to elements
        """
        group_count = 0
        group_ids = []
        for k, g in groupby(elements):
            group_ids.extend([group_count] * len(list(g)))
            group_count += 1

        return group_ids

    def handle_empty_moves(self):
        # TODO
        for player_data in self.all_players_data:
            # cur_shape_idx = None
            # grouped_shapes = []
            # cur_group = []
            # for action in player_data:
            #     if action[self.config.SHAPE_ID_IDX] == cur_shape_idx:
            #         cur_group.append(action)
            #     else:
            #         if cur_group:
            #             grouped_shapes.append(cur_group)
            #         cur_group = [action]
            #         cur_shape_idx = action[self.config.SHAPE_ID_IDX]
            # if cur_group:
            #     grouped_shapes.append(cur_group)
            #
            # recomputed = []
            # for group in grouped_shapes:
            #     if len(group) == 1:
            #         recomputed.append(group[0] + [group[0][self.config.SHAPE_MOVE_TIME_IDX]])  # max move time is the same as move time for single moves
            #     else:
            #         # multiple consecutive identical shapes, likely due to empty moves. We keep the first and last move time, but only the save time of the first move (since the later ones are likely due to empty moves and thus not actual saves)
            #         first_move = group[0]
            #         last_move = group[-1]



            shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
            shapes_df[self.config.SHAPE_MAX_MOVE_TIME_IDX] = shapes_df[self.config.SHAPE_MOVE_TIME_IDX]
            shapes_df["group_id"] = self.group_consecutive_duplicates(shapes_df[self.config.SHAPE_ID_IDX])
            shapes_df = (shapes_df
                         .groupby("group_id", as_index=False)
                         .agg({self.config.SHAPE_ID_IDX: lambda x: int(x.iloc[0]),
                               self.config.SHAPE_MOVE_TIME_IDX: lambda x: x.iloc[0],
                               self.config.SHAPE_SAVE_TIME_IDX: lambda x: x.min(),
                               self.config.SHAPE_MAX_MOVE_TIME_IDX: lambda x: x.iloc[-1]})
                         .drop(columns="group_id"))
            shapes = (shapes_df
                      .reindex(sorted(shapes_df.columns), axis="columns")  # fixes possible column reordering by agg
                      .to_numpy(dtype=object)
                      .tolist())
            n_actions_before = len(player_data[PARSED_ALL_SHAPES_KEY])
            n_actions_after = len(shapes)
            if n_actions_before != n_actions_after:
                self.log_info(f"Player {player_data[PARSED_PLAYER_ID_KEY]}: reduced number of actions from {n_actions_before} to {n_actions_after} by handling empty moves")
            # replace nan with None for consistency with the rest of the code (and to avoid issues with json dumping later)
            for shape in shapes:
                # only need to check the gallery save time:
                if pd.isna(shape[self.config.SHAPE_SAVE_TIME_IDX]):
                    shape[self.config.SHAPE_SAVE_TIME_IDX] = None
            player_data[PARSED_ALL_SHAPES_KEY] = shapes

    def _segment_game_vanilla(self, player_data):
        explore, exploit = segment_explore_exploit(player_data[PARSED_ALL_SHAPES_KEY],
                                                   shape_move_time_idx=self.config.SHAPE_MOVE_TIME_IDX,
                                                   shape_save_time_idx=self.config.SHAPE_SAVE_TIME_IDX,
                                                   min_save_for_exploit=self.config.MIN_SAVE_FOR_EXPLOIT)
        return explore, exploit

    def _segment_game_mri(self, player_data):
        # Retrieve MRI specific params

        # Call the dedicated MRI function
        explore, exploit, robust_median, max_pace_val = segment_explore_exploit_mri(
            shapes=player_data[PARSED_ALL_SHAPES_KEY],
            min_save_for_exploit=self.config.MIN_SAVE_FOR_EXPLOIT,
            min_efficiency=self.config.MRI_SEG_MIN_EFFICIENCY_FOR_EXPLOIT,
            max_pace=self.config.MRI_SEG_MAX_PACE_FOR_MERGE,
            shape_save_time_idx=self.config.SHAPE_SAVE_TIME_IDX,
            shape_move_time_idx=self.config.SHAPE_MOVE_TIME_IDX,
            shape_max_move_time_idx=self.config.SHAPE_MAX_MOVE_TIME_IDX,
            shape_id_index=self.config.SHAPE_ID_IDX
        )

        # Save MRI stats
        player_data[ROBUST_MEDIAN_PACE_KEY] = robust_median
        player_data[ROBUST_THRESHOLD_KEY] = max_pace_val
        return explore, exploit

    def add_explore_exploit(self):
        conf_args = (self.config.SHAPE_MOVE_TIME_IDX, self.config.SHAPE_SAVE_TIME_IDX, self.config.MIN_SAVE_FOR_EXPLOIT)
        segment_func = None
        if self.config.SEGMENTATION_ALGORITHM == SEG_ALG_VANILLA:
            segment_func = self._segment_game_vanilla
        elif self.config.SEGMENTATION_ALGORITHM == SEG_ALG_MRI:
            self.log_info("using MRI segmentation algorithm instead of vanilla segmentation algorithm")
            segment_func = self._segment_game_mri
        else:
            raise ValueError(INVALID_SEGMENTATION_ALGORITHM_ERROR.format(self.config.SEGMENTATION_ALGORITHM))

        for player_data in self.all_players_data:
            explore, exploit = segment_func(player_data)
            player_data[EXPLORE_KEY] = explore
            player_data[EXPLOIT_KEY] = exploit

    def dump(self, *, name: str = None, path: str = None, pretty=False, with_config=True):
        path = resolve_path(name=name,path=path, default_suffix=POSTPARSER_OUTPUT_FILENAME)
        # dump post-parsed
        json_str = prettify_games_json(self.all_players_data) if pretty else json.dumps(self.all_players_data)
        with open(path, "w") as out_file:
            out_file.write(json_str)

        # dump config
        if with_config:
            # dump config
            self.config.to_yaml(path.replace('.json', ''))
