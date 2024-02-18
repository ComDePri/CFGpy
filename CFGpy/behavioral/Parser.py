import numpy as np
import pandas as pd
import json
import re
from datetime import datetime, timezone
from _utils import csv_coords_to_bin_coords, prettify_games_json, CFGPipelineException
from _consts import *

DEFAULT_OUTPUT_FILENAME = "parsed.json"


class Parser:
    parser_date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
    old_date_format_with_placeholder = 'DateObject<{%Y, %m, %d, %H, %M, %S.%f}, "Instant", "Gregorian", 2.>'  # The actual format has '[' instead of '<' but it makes everything easier this way
    datetime_re = '"(DateObject\[\{\d+, \d+, \d+, \d+, \d+, \d+(?:\.\d+)?}, "Instant", "Gregorian", \d+\.\])"'  # Used to remove quotes from game strings
    parse_datetime_re_day = 'DateObject\[\{(\d+), (\d+), (\d+)}, "Day", "Gregorian", \d+\.\]'
    parse_datetime_re_second = 'DateObject\[\{(\d+), (\d+), (\d+), (\d+), (\d+), (\d+)}, "Second", "Gregorian", \d+\.\]'
    parse_datetime_re_millisecond = 'DateObject\[\{(\d+), (\d+), (\d+), (\d+), (\d+), (\d+)(.\d+)?}, "Instant", "Gregorian", \d+\.\]'

    datetime_sub_expressions = [
        parse_datetime_re_day,
        parse_datetime_re_second,
        parse_datetime_re_millisecond,
    ]

    id_column_name = 'merged_id'
    parser_relevant_columns = [
        id_column_name,
        'type',
        'customData.newShape',
        'customData.shape',
        'userTime',
    ]
    default_id_columns = [
        'playerExternalId',
        'userProvidedId',
        'userId',
        'prolificId'
    ]
    default_id = 'No ID Found'

    unique_internal_id_column = 'playerId'
    json_column = 'playerCustomData'
    time_column = 'userTime'
    shape_move_column = 'customData.newShape'
    shape_save_column = 'customData.shape'
    command_type_column = 'type'
    gallery_save_time_column = 'gallery save time'
    merged_shape_column = 'merged_shapes_actions'

    game_start_command = 'start'
    tutorial_start_command = 'starttutorial'
    tutorial_end_command = 'startsearch'
    game_ended_command = 'end search'
    shape_move_command = 'movedblock'
    gallery_save_command = 'added shape to gallery'

    shape_relevant_actions = [
        shape_move_command,
        gallery_save_command,
    ]

    parsed_game_headers = [shape_move_column, time_column, gallery_save_time_column]
    first_shape = '[[0,0],[1,0],[2,0],[4,0],[5,0],[6,0],[3,0],[7,0],[8,0],[9,0]]'

    parsed_game_id_key = PARSED_PLAYER_ID_KEY
    parsed_game_time_key = PARSED_TIME_KEY
    parsed_game_actions_key = PARSED_ALL_SHAPES_KEY
    parsed_game_chosen_shapes = PARSED_CHOSEN_SHAPES_KEY

    def __init__(self, raw_data, include_in_id: list = [], id_columns: list = default_id_columns):
        self.raw_data = raw_data
        self.include_in_id = include_in_id
        self.id_columns = id_columns
        self.parsed_data = None

    @classmethod
    def from_file(cls, raw_data_filename: str, include_in_id: list = [], id_columns: list = default_id_columns):
        raw_data = pd.read_csv(raw_data_filename)
        return cls(raw_data, include_in_id, id_columns)

    def parse(self):
        preprocessed_data = self._preprocess_data()
        grouped_by_unique_id = preprocessed_data.groupby(self.unique_internal_id_column)
        hard_filtered_games = grouped_by_unique_id.filter(self._apply_hard_filters)
        self.parsed_data = self._parse_all_player_games(hard_filtered_games)
        return self.parsed_data

    def dump(self, path=DEFAULT_OUTPUT_FILENAME, pretty=False):
        json_str = prettify_games_json(self.parsed_data) if pretty else json.dumps(self.parsed_data)

        with open(path, "w") as out_file:
            out_file.write(json_str)

    def _preprocess_data(self):
        data = self.patchfix_csv_data(self.raw_data)
        data[self.json_column] = data[self.json_column].apply(json.loads)
        all_json_keys = self.get_all_json_keys_from_csv_data(data)
        for key in all_json_keys:
            # Take the json inside the csv file and turn them into columns
            data[key] = data[self.json_column].apply(lambda json_dict: json_dict.get(key))

        data[self.shape_move_column] = data[self.shape_move_column].apply(
            lambda val: sorted(json.loads(val)) if type(val) is str else np.NAN)
        data[self.shape_save_column] = data[self.shape_save_column].apply(
            lambda val: sorted(json.loads(val)) if type(val) is str else np.NAN)

        data = self.merge_id_columns(data)
        data[self.time_column] = pd.to_datetime(data[self.time_column], format=self.parser_date_format)
        data = data.sort_values(by=self.time_column).reset_index(drop=True)

        return data

    def patchfix_csv_data(self, data):
        '''Small patchy bugfix for temporary problems'''
        # Bug no.1 sometimes player external id is this instead of a random number
        data.loc[data['playerExternalId'] == '${rand://int/100000:10000000}', 'playerExternalId'] = None

        # Bug no.2 sometimes the endPosition and shape columns switch places
        switched_column_indices = np.flatnonzero(
            data['customData.endPosition'].apply(lambda x: len(json.loads(x)) == 10 if type(x) is str else False))
        data.loc[switched_column_indices, 'customData.shape'] = data.loc[
            switched_column_indices, 'customData.endPosition']
        data['customData.shape'] = data['customData.shape'].apply(
            lambda x: json.loads(x) if type(x) is str else []).apply(lambda x: str(x) if len(x) == 10 else np.NaN)

        return data

    def get_all_json_keys_from_csv_data(self, data):
        all_json_keys = np.concatenate(data[self.json_column].apply(lambda x: tuple(x.keys())).unique())

        return set(all_json_keys)

    def merge_id_columns(self, data):
        data[self.id_column_name] = None

        for id_column in self.id_columns:
            if id_column in data.columns:
                missing_indices = data[self.id_column_name].isna()
                data.loc[missing_indices, self.id_column_name] = data[id_column].loc[missing_indices].astype(str)

        missing_indices = data[self.id_column_name].isna()
        data.loc[missing_indices, self.id_column_name] = self.default_id

        return data

    def _apply_hard_filters(self, game):
        return self.is_game_started(game)

    def is_game_started(self, game):
        return game[self.command_type_column].str.contains(self.tutorial_end_command).sum() > 0

    def _parse_all_player_games(self, games):
        all_parsed_games = []
        for _, game in games.groupby(self.unique_internal_id_column):
            parsed_game = self.parse_single_game(game)
            all_parsed_games.append(parsed_game)

        return all_parsed_games

    def parse_single_game(self, game_data):
        parser_relevant_columns = self.parser_relevant_columns + self.include_in_id
        game_data = game_data[parser_relevant_columns]

        assert len(game_data[self.id_column_name].unique()) == 1
        player_id_field = game_data[self.id_column_name].iloc[0]
        game_start_time = game_data[game_data[self.command_type_column] == self.tutorial_end_command].iloc[0][
            self.time_column]

        game_data = game_data[game_data[self.time_column] >= game_start_time]
        game_data = game_data[game_data[self.command_type_column].isin(self.shape_relevant_actions)]
        first_row = [player_id_field, self.shape_move_command, self.first_shape, np.NAN, game_start_time]
        first_row_df = pd.DataFrame([first_row], columns=game_data.columns)
        game_data = pd.concat([first_row_df, game_data], ignore_index=True)

        game_data[self.time_column] = (game_data[self.time_column] - game_start_time).apply(
            lambda time_delta: time_delta.total_seconds())
        game_data[self.shape_move_column] = game_data[self.shape_move_column].apply(csv_coords_to_bin_coords)

        game_data[self.gallery_save_time_column] = None

        gallery_save_indices = game_data[self.shape_move_column].isna()[game_data[self.shape_move_column].isna()].index
        game_data.loc[gallery_save_indices - 1, self.gallery_save_time_column] = game_data.loc[
            gallery_save_indices, self.time_column].values
        game_data = game_data[game_data[self.command_type_column].isin(
            [self.shape_move_command])]  # Now that we have the save time in all move rows, we can get rid of save rows

        actions = game_data[self.parsed_game_headers]
        if self.include_in_id:
            player_id_field = [game_data[self.id_column_name].iloc[0]] + self.include_in_id
        parsed_game = {
            self.parsed_game_id_key: player_id_field,
            self.parsed_game_time_key: game_start_time.timestamp(),
            self.parsed_game_actions_key: actions.values.tolist(),
        }

        return parsed_game

    @classmethod
    def translate_parsed_results_to_mathematica(cls, json_format):
        games_in_old_format = []
        for entry in json_format:
            player_id = entry[cls.parsed_game_id_key]
            player_start_time = entry[cls.parsed_game_time_key]
            player_actions = entry[cls.parsed_game_actions_key]

            old_format_actions = [
                [list(map(str, action[0])), action[1]] if action[2] is None else [list(map(str, action[0])), action[1],
                                                                                  action[2]]
                for action in player_actions
            ]
            old_format_entry = [
                player_id,
                datetime.strftime(datetime.fromtimestamp(player_start_time, tz=timezone.utc),
                                  cls.old_date_format_with_placeholder),
                old_format_actions,
                "",
            ]
            games_in_old_format.append(old_format_entry)

        parsed_data_in_old_format = '\n'.join(
            [cls.replace_chars_to_old_format(json.dumps(game_in_old_format)) for game_in_old_format in
             games_in_old_format])

        return parsed_data_in_old_format

    @classmethod
    def replace_chars_to_old_format(cls, old_format_json_string):
        brackets_replaced = old_format_json_string.replace('[', '{').replace(']', '}')
        brackets_replaced = brackets_replaced.replace('<', '[').replace('>', ']').replace('\\', '')

        quotes_removed = re.sub(pattern=cls.datetime_re, repl='\\g<1>', string=brackets_replaced)

        return quotes_removed

    @classmethod
    def translate_mathematica_to_python(cls, mathematica_path):
        with open(mathematica_path, 'r') as f:
            data = f.read()

        games = data.split('\n')

        game_timestamps = [cls.parse_date_from_game_string(game).timestamp() for game in games]
        games = [cls.replace_datetime_with_timestamp(game_string=games[i], timestamp=game_timestamps[i]) for i in
                 range(len(games))]
        games = [re.sub(pattern='(\d\.)([\[\]\{\},])', repl='\\g<1>0\\g<2>', string=game) for game in
                 games]  # There's a bug here if we have a user with the string "0.[" in its id
        games = [game.replace('{', '[').replace('}', ']').replace('$Failed', '"$Failed"') for game in games]
        games = [json.loads(game) for game in games]

        json_format_games = []
        for game in games:
            game_id = game[0].replace('[', '{').replace(']', '}')  # { and } sometimes appear in Mathematica-parsed ids
            absolute_start_time = game[1]
            actions = [
                [list(map(int, action[0])), action[1], action[2]] if len(action) == 3 else
                [list(map(int, action[0])), action[1], None]
                for action in game[2]
            ]

            chosen_shapes = []
            if len(game) == 4 and game[3] != "":
                chosen_shapes = game[3]
                if type(chosen_shapes) is not list:
                    chosen_shapes = [chosen_shapes]

            json_format_game = {
                cls.parsed_game_id_key: game_id,
                cls.parsed_game_time_key: absolute_start_time,
                cls.parsed_game_chosen_shapes: chosen_shapes,
                cls.parsed_game_actions_key: actions,
            }

            json_format_games.append(json_format_game)

        return json_format_games

    @classmethod
    def parse_date_from_game_string(cls, game):
        game_date_string = re.findall(cls.parse_datetime_re_millisecond, game)
        if game_date_string == []:
            game_date_string = re.findall(cls.parse_datetime_re_second, game)
            if game_date_string == []:
                game_date_string = re.findall(cls.parse_datetime_re_day, game)
                if game_date_string == []:
                    raise CFGPipelineException('Was not able to parse the date in the following game:', game)
        else:
            year = int(game_date_string[0][0])
            month = int(game_date_string[0][1])
            day = int(game_date_string[0][2])
            hour = int(game_date_string[0][3])
            minute = int(game_date_string[0][4])
            second = int(game_date_string[0][5])
            microsecond = int(game_date_string[0][6][1:7])

            return datetime(year, month, day, hour, minute, second, microsecond)

        return datetime(*map(int, game_date_string[0]))

    @classmethod
    def replace_datetime_with_timestamp(cls, game_string, timestamp):
        if type(timestamp) is not str:
            timestamp = str(timestamp)

        for datetime_sub_expression in cls.datetime_sub_expressions:
            replaced_string = re.sub(pattern=datetime_sub_expression, repl=timestamp, string=game_string)
            if replaced_string != game_string:
                return replaced_string

        raise CFGPipelineException('Was not able to replace the DateObject with a timestamp in the following game:',
                                   game_string)


class ParserOldCommands(Parser):
    tutorial_end_command = 'tutorial complete'


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description="Parse raw CFG data")
    argparser.add_argument("-i", "--input", dest="input_filename",
                           help='Filename of raw data CSV')
    argparser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output CSV')
    args = argparser.parse_args()

    p = Parser.from_file(args.input_filename)
    p.parse()
    p.dump(args.output_filename)
