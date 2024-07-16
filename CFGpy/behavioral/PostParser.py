from CFGpy.behavioral._utils import load_json, CFGPipelineException, segment_explore_exploit, prettify_games_json
from CFGpy.behavioral._consts import (PARSED_ALL_SHAPES_KEY, PARSED_PLAYER_ID_KEY, EXPLORE_KEY, EXPLOIT_KEY,
                                      DEFAULT_FINAL_OUTPUT_FILENAME, INVALID_SHAPE_ERROR, POSTPARSER_OUTPUT_FILENAME)
from CFGpy.behavioral import config
from CFGpy.utils import binary_shape_to_id as bin2id
import json


class PostParser:
    def __init__(self, parsed_data):
        self.all_players_data = parsed_data

    @classmethod
    def from_json(cls, path: str):
        return cls(load_json(path))

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
        for player_data in self.all_players_data:
            shapes = player_data[PARSED_ALL_SHAPES_KEY]
            for shape in shapes:
                try:
                    shape[config.SHAPE_ID_IDX] = bin2id(shape[config.SHAPE_ID_IDX])
                except ValueError:
                    player_id = player_data[PARSED_PLAYER_ID_KEY]
                    shape_id = shape[config.SHAPE_ID_IDX]
                    raise CFGPipelineException(INVALID_SHAPE_ERROR.format(shape_id, player_id))

    def handle_empty_moves(self):
        # TODO
        pass

    def add_explore_exploit(self):
        for player_data in self.all_players_data:
            explore, exploit = segment_explore_exploit(player_data[PARSED_ALL_SHAPES_KEY])
            player_data[EXPLORE_KEY] = explore
            player_data[EXPLOIT_KEY] = exploit

    def dump(self, path=POSTPARSER_OUTPUT_FILENAME, pretty=False):
        json_str = prettify_games_json(self.all_players_data) if pretty else json.dumps(self.all_players_data)

        with open(path, "w") as out_file:
            out_file.write(json_str)


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description="Post-parse CFG data")
    argparser.add_argument("-i", "--input", dest="input_filename",
                           help='Filename of parsed data JSON')
    argparser.add_argument("-o", "--output", default=DEFAULT_FINAL_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output JSON')
    args = argparser.parse_args()

    pp = PostParser.from_json(args.input_filename)
    pp.postparse()
    pp.dump(args.output_filename)
