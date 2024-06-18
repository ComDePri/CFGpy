from CFGpy.behavioral._utils import *
from CFGpy.behavioral._consts import *
from CFGpy.utils import binary_shape_to_id as bin2id

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

    def handle_empty_moves(self):
        # TODO
        pass

    def add_explore_exploit(self):
        for player_data in self.all_players_data:
            explore, exploit = segment_explore_exploit(player_data[PARSED_ALL_SHAPES_KEY])
            player_data[EXPLORE_KEY] = explore
            player_data[EXPLOIT_KEY] = exploit

    def dump(self, path=DEFAULT_OUTPUT_FILENAME, pretty=False):
        json_str = prettify_games_json(self.all_players_data) if pretty else json.dumps(self.all_players_data)

        with open(path, "w") as out_file:
            out_file.write(json_str)


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description="Calculate measures from parsed CFG data")
    argparser.add_argument("-i", "--input", dest="input_filename",
                           help='Filename of parsed data JSON')
    argparser.add_argument("-o", "--output", default=DEFAULT_FINAL_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output JSON')
    args = argparser.parse_args()

    pp = Preprocessor.from_json(args.input_filename)
    pp.preprocess()
    pp.dump(args.output_filename)
