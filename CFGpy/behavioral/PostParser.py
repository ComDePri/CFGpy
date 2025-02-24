from CFGpy.behavioral._utils import load_json, CFGPipelineException, segment_explore_exploit, prettify_games_json
from CFGpy.behavioral._consts import (PARSED_ALL_SHAPES_KEY, PARSED_PLAYER_ID_KEY, EXPLORE_KEY, EXPLOIT_KEY,
                                      DEFAULT_FINAL_OUTPUT_FILENAME, INVALID_SHAPE_ERROR, NOT_A_NEIGHBOR_ERROR,
                                      POSTPARSER_OUTPUT_FILENAME)
from CFGpy.behavioral import Configuration
import json

# TODO: after FilesHandler feature is implemented, delete the following and use FilesHandler().shape_network instead
#   track FilesHandler's implementation in https://github.com/ComDePri/CFGpy/pull/43
import os
import networkx as nx
from CFGpy.utils import CFG_RESOURCES_PATH

shape_network = nx.read_adjlist(os.path.join(CFG_RESOURCES_PATH, "all_shapes.adjlist"), nodetype=int)


def is_valid_transition(shape1: int, shape2: int) -> bool:
    """
    Checks whether a transition is a valid path in the CFG.
    TODO: assumes empty moves are valid. When empty moves handling is implemented, this function can be replaced with
        shape_network.has_edge(shape1, shape2)
    :param shape1: shape id, after conversion to int by PostParser.convert_shape_ids
    :param shape2: shape id, after conversion to int by PostParser.convert_shape_ids
    :return: True if the transition is valid, False if not
    """
    return shape1 == shape2 or shape_network.has_edge(shape1, shape2)


class PostParser:
    def __init__(self, parsed_data, config: Configuration = None):
        self.all_players_data = parsed_data
        self.config = config if config is not None else Configuration.default()

    @classmethod
    def from_json(cls, path: str, config=None):
        return cls(load_json(path), config)

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
                    print(CFGPipelineException(NOT_A_NEIGHBOR_ERROR.format(i - 1, i, player_id)))
                    # the exception is printed and not raised because many gaps are actually in the source data

    def handle_empty_moves(self):
        # TODO
        pass

    def add_explore_exploit(self):
        conf_args = (self.config.SHAPE_MOVE_TIME_IDX, self.config.SHAPE_SAVE_TIME_IDX, self.config.MIN_SAVE_FOR_EXPLOIT)

        for player_data in self.all_players_data:
            explore, exploit = segment_explore_exploit(player_data[PARSED_ALL_SHAPES_KEY], *conf_args)
            player_data[EXPLORE_KEY] = explore
            player_data[EXPLOIT_KEY] = exploit

    def dump(self, path=POSTPARSER_OUTPUT_FILENAME, pretty=False):
        # dump post-parsed
        json_str = prettify_games_json(self.all_players_data) if pretty else json.dumps(self.all_players_data)
        with open(path, "w") as out_file:
            out_file.write(json_str)

        # dump config
        self.config.to_yaml(path)
