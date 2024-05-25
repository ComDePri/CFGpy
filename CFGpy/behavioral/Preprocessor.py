from CFGpy.behavioral._utils import *
from CFGpy.behavioral._consts import *
from CFGpy.utils import binary_matrix_to_shape_id as bin2id
from itertools import groupby
import json

DEFAULT_OUTPUT_FILENAME = "preprocessed.json"


class Preprocessor:
    def __init__(self, parsed_data):
        self.all_players_data = parsed_data

    @classmethod
    def from_json(cls, path: str):
        return cls(load_json(path))

    def preprocess(self):
        self.convert_shape_ids()
        self.handle_empty_moves()
        self.add_explore_exploit()
        return self.all_players_data

    def convert_shape_ids(self):
        """
        Converts shape ids from their graphical representations to serial numbers.
        Raises an exception if illegal shapes are found.
        """
        for player_data in self.all_players_data:
            shapes = player_data[PARSED_ALL_SHAPES_KEY]
            for shape in shapes:
                try:
                    shape[SHAPE_ID_IDX] = bin2id(shape[SHAPE_ID_IDX])
                except ValueError:
                    player_id = player_data[PARSED_PLAYER_ID_KEY]
                    shape_id = shape[SHAPE_ID_IDX]
                    msg = f"Encountered invalid shape: {shape_id}\nPlayer id: {player_id}\n" \
                          "This indicates a bug in the CFG software or in the data parsing"
                    raise CFGPipelineException(msg)

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
        for player_data in self.all_players_data:
            shapes_df = pd.DataFrame(player_data[PARSED_ALL_SHAPES_KEY])
            shapes_df[SHAPE_MAX_MOVE_TIME_IDX] = shapes_df[SHAPE_MOVE_TIME_IDX]
            shapes_df["group_id"] = self.group_consecutive_duplicates(shapes_df[SHAPE_ID_IDX])
            shapes_df = (shapes_df
                         .groupby("group_id", as_index=False)
                         .agg({SHAPE_ID_IDX: lambda x: x.iloc[0],
                               SHAPE_MOVE_TIME_IDX: lambda x: x.iloc[0],
                               SHAPE_SAVE_TIME_IDX: lambda x: x.iloc[0],
                               SHAPE_MAX_MOVE_TIME_IDX: lambda x: x.iloc[-1]})
                         .drop(columns="group_id"))
            shapes = (shapes_df
                      .reindex(sorted(shapes_df.columns), axis="columns")  # fixes possible column reordering by agg
                      .values.tolist())
            player_data[PARSED_ALL_SHAPES_KEY] = shapes

    def add_explore_exploit(self):
        for player_data in self.all_players_data:
            explore, exploit = segment_explore_exploit(player_data[PARSED_ALL_SHAPES_KEY])
            player_data[EXPLORE_KEY] = explore
            player_data[EXPLOIT_KEY] = exploit

    def dump(self, path=DEFAULT_OUTPUT_FILENAME):
        with open(path, "w") as out_file:
            json.dump(self.all_players_data, out_file)
