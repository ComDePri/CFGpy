import re
import os
import pandas as pd
import argparse
import json
import numpy as np
import subprocess
import platform
import ParserUtils
from _ctypes import PyObj_FromPtr
from datetime import datetime
from scipy import stats


class CFGParserException(Exception):
    pass


class Parser:
    MS = 1
    SECOND = 1000 * MS
    MINUTE = 60 * SECOND
    MIN_GAME_TIME = 10 * MINUTE
    MAX_SECONDS_FOR_PAUSE = 90 * SECOND
    MAX_STD_FOR_OUTLIERS = 3

    MINIMAL_ROWS_FOR_GAME_START = 2
    MIN_SHAPES_MOVED = 80

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

    json_column = 'playerCustomData'
    time_column = 'userTime'
    shape_move_column = 'customData.newShape'
    shape_save_column = 'customData.shape'
    command_type_column = 'type'
    gallery_save_time_column = 'gallery save time'
    merged_shape_column = 'merged_shapes_actions'

    disqualified_ids = 'disqualified_ids'

    game_start_command = 'start'
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

    parsed_game_id_key = 'id'
    parsed_game_time_key = 'absolute start time'
    parsed_game_actions_key = 'actions'
    parsed_game_chosen_shapes = 'chosen_shapes'

    explore_json = 'explore'
    exploit_json = 'exploit'

    def __init__(self, disqualified_ids: list = [], include_in_id: list = [], id_columns: list = default_id_columns):
        self.disqualified_ids = disqualified_ids
        self.include_in_id = include_in_id
        self.id_columns = id_columns

    @staticmethod
    def pandas_read_csv_player_custom_data_parser(data):
        if data == '' or data is None:
            return {}

        return json.loads(data)

    def parse_from_file(self, raw_data_filename: str):
        preprocessed_data = self._read_and_preprocess_data(raw_data_filename)
        sorted_players_games = self._reorganize_and_sort_data(preprocessed_data)

        filtered_players_games = []
        filtered_sorted_players_games = self._filter_disqualified_players(sorted_players_games)
        for player_games_list in filtered_sorted_players_games:
            filtered_player_games_list = self._filter_disqualified_games(player_games_list)
            if filtered_player_games_list != []:
                filtered_players_games.append(filtered_player_games_list)

        parsed_data = self._parse_all_player_games(filtered_players_games)
        # parsed_data = self.post_processing(parsed_data)

        return parsed_data

    def _read_and_preprocess_data(self, raw_data_filename):
        data = pd.read_csv(raw_data_filename, dtype=str,
                           converters={self.json_column: self.pandas_read_csv_player_custom_data_parser}, header=0,
                           encoding='utf-8')
        data = self.patchfix_csv_data(data)
        all_json_keys = self.get_all_json_keys_from_csv_data(data)
        for key in all_json_keys:
            # Take the json inside the csv file and turn them into columns
            data[key] = data[self.json_column].apply(lambda json_dict: json_dict.get(key))

        parser_relevant_columns = self.parser_relevant_columns + self.include_in_id
        data = self.merge_id_columns(data)
        data = data[parser_relevant_columns]
        data[self.time_column] = pd.to_datetime(data[self.time_column], format=self.parser_date_format)

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
                data.loc[missing_indices, self.id_column_name] = data[id_column].loc[missing_indices]

        missing_indices = data[self.id_column_name].isna()
        data.loc[missing_indices, self.id_column_name] = self.default_id

        return data

    def _reorganize_and_sort_data(self, data):
        data_split_to_players = self.split_to_players(data)
        sorted_players_data = [self.sort_and_clean_player_games(player_data) for player_data in data_split_to_players]
        data_split_to_players_games = [self.split_player_games(player_data) for player_data in sorted_players_data]

        return data_split_to_players_games

    def split_to_players(self, data):
        player_identifiers = data[self.id_column_name].unique()

        data_split_to_players = [
            data[data[self.id_column_name] == player_identifier]
            for player_identifier in player_identifiers
        ]

        return data_split_to_players

    def sort_and_clean_player_games(self, player_data):
        sorted_player_data = player_data.sort_values(by=self.time_column).reset_index(drop=True)
        sorted_player_data[self.shape_move_column] = sorted_player_data[self.shape_move_column].apply(
            lambda val: sorted(json.loads(val)) if type(val) is str else np.NAN)
        sorted_player_data[self.shape_save_column] = sorted_player_data[self.shape_save_column].apply(
            lambda val: sorted(json.loads(val)) if type(val) is str else np.NAN)
        duplicate_moves_mask = sorted_player_data[self.shape_move_column].dropna().iloc[:-1].reset_index(drop=True) == \
                               sorted_player_data[self.shape_move_column].dropna().iloc[1:].reset_index(drop=True)
        duplicate_moves_indices = sorted_player_data[self.shape_move_column].dropna().iloc[
            np.flatnonzero(duplicate_moves_mask) + 1].index

        # return sorted_player_data.drop(duplicate_moves_indices).reset_index(drop=True)
        return sorted_player_data

    def split_player_games(self, player_data):
        '''Assumes that the games are sorted by time'''
        game_start_indices = player_data.index[
            player_data[self.command_type_column] == self.game_start_command].tolist()

        if len(game_start_indices) == 1:
            return [player_data]

        player_games = []
        for i in range(len(game_start_indices) - 1):
            game_start_index = game_start_indices[i]
            game_end_index = game_start_indices[i + 1]
            game = player_data.iloc[game_start_index:game_end_index]

            player_games.append(game)

        return player_games

    def _filter_disqualified_players(self, players_list):
        return [player_games for player_games in players_list if
                player_games[0].iloc[0][self.id_column_name] not in self.disqualified_ids]

    def _filter_disqualified_games(self, player_games_list):
        player_games_list = self.filter_to_first_started_game(player_games_list)
        player_games_list = self.filter_unfinished_games(player_games_list)
        # player_games_list = self.filter_min_shapes_made(player_games_list)
        # player_games_list = self.filter_min_time_games(player_games_list)
        # player_games_list = self.filter_paused_games(player_games_list)
        # player_games_list = self.filter_games_with_illegal_shapes(player_games_list)

        return player_games_list

    def filter_to_first_started_game(self, games):
        '''Returns the first game that has actually started'''
        games = [game for game in games if
                 len(game) > self.MINIMAL_ROWS_FOR_GAME_START]  # Remove games that haven't actually started
        games = [game for game in games if game[self.command_type_column].str.contains(
            self.tutorial_end_command).sum() > 0]  # Remove games that didn't continue past the tutorial stage

        if games != []:
            return [games[0]]

        return []

    def filter_unfinished_games(self, games):
        games = [game for game in games if self.is_game_ended(game)]

        return games

    def is_game_ended(self, game):
        return game[self.command_type_column].str.contains(self.game_ended_command).sum() > 0

    def filter_min_shapes_made(self, games):
        return [game for game in games if game[self.command_type_column].str.contains(
            self.shape_move_command).sum() >= self.MIN_SHAPES_MOVED]  # Removes games where the amount of steps is below some minimum.

    def filter_min_time_games(self, games):
        filtered_games = []
        for game in games:
            start_time = game[game[self.command_type_column] == self.game_start_command].iloc[0][self.time_column]
            end_time = game[game[self.command_type_column] == self.game_ended_command].iloc[0][self.time_column]

            time_delta = end_time - start_time
            if time_delta.total_seconds() * self.SECOND >= self.MIN_GAME_TIME:
                filtered_games.append(game)

        return filtered_games

    def filter_paused_games(self, games):
        """Filters all games where there is a 90 second break at some point"""
        filtered_games = []

        for game in games:
            tutorial_end_index = np.flatnonzero(game[self.command_type_column] == self.tutorial_end_command)[0]
            game_without_tutorial = game.iloc[tutorial_end_index:]
            game_without_metamoves = game_without_tutorial[game_without_tutorial[self.command_type_column].isin(
                self.shape_relevant_actions)]  # Keep only block move/save actions
            time_deltas = game_without_metamoves[self.time_column].diff()
            time_deltas_in_seconds = time_deltas.apply(lambda time_delta: time_delta.total_seconds())
            if not any(time_deltas_in_seconds * self.SECOND > self.MAX_SECONDS_FOR_PAUSE):
                filtered_games.append(game)

        return filtered_games

    def filter_games_with_illegal_shapes(self, games):
        legal_games = []
        for game in games:
            game_shapes_coords = game[game[self.command_type_column].isin(self.shape_relevant_actions)][
                self.shape_move_column].copy()  # Keeps only rows with shape coords
            none_indices_mask = game_shapes_coords.isna()
            missing_indices = none_indices_mask.iloc[
                np.flatnonzero(none_indices_mask)].index  # There has to be a better way to do this
            gallery_save_rows = game.loc[missing_indices, self.shape_save_column]
            game_shapes_coords.loc[missing_indices] = gallery_save_rows
            game_shapes_coords = game_shapes_coords.apply(np.array)
            if all(game_shapes_coords.apply(lambda arr: arr.shape[0] == 10).values):
                legal_games.append(game)
            else:
                bad_coords = game_shapes_coords.iloc[
                    np.flatnonzero(game_shapes_coords.apply(lambda arr: arr.shape[0] != 10))[0]]
                exc = CFGParserException('Found illegal shape in game', bad_coords)
                raise exc

        return legal_games

    def _parse_all_player_games(self, player_games):
        all_parsed_games = []
        for games in player_games:
            for game in games:
                parsed_game = self.parse_single_game(game)
                all_parsed_games.append(parsed_game)

        return all_parsed_games

    def parse_single_game(self, game_data):
        game_start_time = game_data[game_data[self.command_type_column] == self.tutorial_end_command].iloc[0][
            self.time_column]
        first_row = pd.DataFrame([[ParserUtils.csv_coords_to_bin_coords(self.first_shape), 0, None]],
                                 columns=self.parsed_game_headers)
        game_data = game_data[
            game_data[self.command_type_column].isin(self.shape_relevant_actions)]  # Remove all irrelevant rows
        moves_after_tutorial = game_data[game_data[self.time_column] >= game_start_time]
        game_data = moves_after_tutorial
        game_data[self.time_column] = (game_data[self.time_column] - game_start_time).apply(
            lambda time_delta: time_delta.total_seconds())
        game_data[self.shape_move_column] = game_data[self.shape_move_column].apply(
            ParserUtils.csv_coords_to_bin_coords)

        game_data[self.gallery_save_time_column] = None

        gallery_save_indices = game_data[self.shape_move_column].isna()[game_data[self.shape_move_column].isna()].index
        game_data.loc[gallery_save_indices - 1, self.gallery_save_time_column] = game_data.loc[
            gallery_save_indices, self.time_column].values
        game_data = game_data[game_data[self.command_type_column].isin(
            [self.shape_move_command])]  # Now that we have the save time in all move rows, we can get rid of save rows

        actions = game_data[self.parsed_game_headers]
        actions = pd.concat([first_row, actions], ignore_index=True)
        player_id_field = game_data[self.id_column_name].iloc[0]
        if self.include_in_id:
            player_id_field = [game_data[self.id_column_name].iloc[0]] + self.include_in_id
        parsed_game = {
            self.parsed_game_id_key: player_id_field,
            self.parsed_game_time_key: game_start_time.timestamp(),
            self.parsed_game_actions_key: actions.values.tolist(),
        }

        return parsed_game

    def post_processing(self, parsed_data):
        parsed_data = self.add_explore_exploit(parsed_data)
        outliers_filtered = self.filter_outliers(parsed_data)

        return outliers_filtered

    def add_explore_exploit(self, parsed_data):
        for game in parsed_data:
            explore, exploit = ParserUtils.segment_explore_exploit(game[self.parsed_game_actions_key])
            game[self.explore_json] = ParserUtils.cast_list_of_tuple_to_ints(explore)
            game[self.exploit_json] = ParserUtils.cast_list_of_tuple_to_ints(exploit)

        return parsed_data

    def filter_outliers(self, parsed_data):
        all_median_explore = []
        all_median_exploit = []
        for game in parsed_data:
            explore = game[self.explore_json]
            explore_lengths = [explore_slice[1] - explore_slice[0] for explore_slice in explore]
            median_explore = np.median(explore_lengths)
            all_median_explore.append(median_explore)

            exploit = game[self.exploit_json]
            exploit_lengths = [exploit_slice[1] - exploit_slice[0] for exploit_slice in exploit]
            median_exploit = np.median(exploit_lengths)
            all_median_exploit.append(median_exploit)

        zscore_exploit = stats.zscore(all_median_exploit)
        outliers = [parsed_data[index] for index in np.where(zscore_exploit > self.MAX_STD_FOR_OUTLIERS)]
        zscore_explore = stats.zscore(all_median_explore)
        outliers += [parsed_data[index] for index in np.where(zscore_explore > self.MAX_STD_FOR_OUTLIERS)]
        outliers = list(set(outliers))

        for outlier in outliers:
            parsed_data.remove(outlier)

        return parsed_data

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
                datetime.strftime(datetime.fromtimestamp(player_start_time), cls.old_date_format_with_placeholder),
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
                 games]  # Theres a bug here if we have a user with the string "0.[" in its id
        games = [game.replace('{', '[').replace('}', ']').replace('$Failed', '"$Failed"') for game in games]
        games = [json.loads(game) for game in games]

        json_format_games = []
        for game in games:
            game_id = game[0]
            absolute_start_time = game[1]
            actions = [
                [list(map(int, action[0])), action[1], action[2]] if len(action) == 3 else [list(map(int, action[0])),
                                                                                            action[1], None] for action
                in game[2]]

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
                    raise CFGParserException('Was not able to parse the date in the following game:', game)
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

        raise CFGParserException('Was not able to replace the DateObject with a timestamp in the following game:',
                                 game_string)

    @classmethod
    def prettify_games_json(cls, parsed_games):
        prettified_games = []
        for game in parsed_games:
            game['actions'] = [NoIndent(action) for action in game['actions']]
            chosen_shapes = game.get(cls.parsed_game_chosen_shapes, None)
            if chosen_shapes is not None:
                game[cls.parsed_game_chosen_shapes] = [NoIndent(chosen_shape) for chosen_shape in chosen_shapes]

            explore = game.get(cls.explore_json, None)
            if explore:
                game[cls.explore_json] = NoIndent(explore)

            exploit = game.get(cls.exploit_json, None)
            if exploit:
                game[cls.exploit_json] = NoIndent(exploit)

            prettified_games.append(game)

        return json.dumps(prettified_games, cls=CustomIndentEncoder, sort_keys=True, indent=4)


class ParserOldCommands(Parser):
    tutorial_end_command = 'tutorial complete'

    def _filter_disqualified_games(self, player_games_list):
        player_games_list = self.filter_to_first_started_game(player_games_list)
        player_games_list = self.filter_min_shapes_made(player_games_list)
        player_games_list = self.filter_min_time_games(player_games_list)
        player_games_list = self.filter_paused_games(player_games_list)
        player_games_list = self.filter_games_with_illegal_shapes(player_games_list)

        return player_games_list

    def filter_min_time_games(self, games):
        filtered_games = []

        for game in games:
            start_time = game[game[self.command_type_column] == self.game_start_command].iloc[0][self.time_column]
            end_time = game.iloc[-1][self.time_column]

            time_delta = end_time - start_time
            if time_delta.total_seconds() * self.SECOND >= self.MIN_GAME_TIME:
                filtered_games.append(game)

        return filtered_games


# Using the answer from here https://stackoverflow.com/a/13252112 to make a prettier json file
class NoIndent(object):
    """ Value wrapper. """

    def __init__(self, value):
        self.value = value


class CustomIndentEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(CustomIndentEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(CustomIndentEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(CustomIndentEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr


if __name__ == '__main__':
    p = Parser()
    folder = r'C:\Users\roygutg\Documents\GitRepos\CFGAnalysisTools\py_math_compare'
    parsed_data = p.parse_from_file(folder + r"\event.csv")
    old_format = Parser.translate_parsed_results_to_mathematica(parsed_data)
    with open(rf"{folder}\py_parse_old_format.txt", "w") as f:
        f.write(old_format)
