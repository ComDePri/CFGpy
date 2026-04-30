import numpy as np
import pandas as pd
import json
import re
from datetime import datetime, timezone
from CFGpy.behavioral._utils import server_coords_to_binary_shape, prettify_games_json, CFGPipelineException, resolve_path, missing_str_field
from CFGpy.behavioral._consts import (PARSED_PLAYER_ID_KEY, PARSED_TIME_KEY, PARSED_ALL_SHAPES_KEY,
                                      PARSED_CHOSEN_SHAPES_KEY, MERGED_ID_KEY, DEFAULT_ID, PARSER_OUTPUT_FILENAME)
from CFGpy.behavioral import Configuration
from CFGpy.behavioral._logging import HasLogger

class Parser(HasLogger):
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

    def __init__(self, *, raw_data: pd.DataFrame, config: Configuration = None, logger=None):
        super().__init__(logger)
        self.raw_data = raw_data
        self.config = config or Configuration.default()
        self.parsed_data = None

        self.include_in_id = list(self.config.INCLUDE_IN_PARSER_ID)
        self.parser_relevant_columns = [
            MERGED_ID_KEY,
            self.config.EVENT_TYPE,
            self.config.RAW_NEW_SHAPE,
            self.config.RAW_SHAPE,
            self.config.RAW_USER_TIME,
        ]
        self.shape_relevant_event_types = [
            self.config.SHAPE_MOVE_EVENT_TYPE,
            self.config.GALLERY_SAVE_EVENT_TYPE,
        ]

    @classmethod
    def from_file(cls, raw_data_filename: str, config=None):
        raw_data = pd.read_csv(raw_data_filename)
        return cls(raw_data=raw_data, config=config)

    def parse(self):
        prepared_data = self._prepare_data()
        games_grouped_by_unique_id = prepared_data.groupby(self.config.UNIQUE_INTERNAL_ID_COLUMN)
        hard_filtered_games = games_grouped_by_unique_id.filter(self._apply_hard_filters)
        self.log_info(f"Filtered from {len(games_grouped_by_unique_id)} games to {len(hard_filtered_games.groupby(self.config.UNIQUE_INTERNAL_ID_COLUMN))} games by applying hard filters.")
        self.parsed_data = self._parse_all_player_games(hard_filtered_games)
        return self.parsed_data

    def dump(self, *, name: str = None, path: str = None, pretty=False, with_config=True):
        # dump parsed
        json_str = prettify_games_json(self.parsed_data) if pretty else json.dumps(self.parsed_data)
        path = resolve_path(name=name, path=path, default_suffix=PARSER_OUTPUT_FILENAME)
        with open(path, "w") as out_file:
            out_file.write(json_str)
        if with_config:
            # dump config
            self.config.to_yaml(path.replace('.json', ''))

    def _prepare_data(self):
        data = self.raw_data
        data = self.patchfix_csv_data(data)
        def normalize_json_dict_val(val):
            """
            create a dictionary of key values out of val that can be either a json string of a dict, a dictionary or missing value
            """
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    self.log_warning(f"Failed to parse JSON from string: {val}. Setting value to NaN.")
                    val = {}
            if isinstance(val, dict):
                return val
            elif pd.isna(val):
                return {}
            else:
                self.log_warning(f"Unexpected value type for JSON column: {type(val)}. Setting value to NaN.")
                return {}

        if self.config.PARSER_JSON_COLUMN in data.columns:
            data[self.config.PARSER_JSON_COLUMN] = data[self.config.PARSER_JSON_COLUMN].apply(normalize_json_dict_val)

            all_json_keys = self.get_all_json_keys_from_csv_data(data)
            self.log_info(f"Found {len(all_json_keys)} unique custom data json keys: {all_json_keys}.")
            for key in all_json_keys:
                # Take the json inside the csv file and turn them into columns
                data[key] = data[self.config.PARSER_JSON_COLUMN].apply(lambda json_dict: json_dict.get(key))
        else:
            self.log_info(f"No JSON column '{self.config.PARSER_JSON_COLUMN}' found in the data. Skipping JSON parsing and column extraction.")

        if self.config.SHAPE_MOVE_COLUMN in data.columns:
            data[self.config.SHAPE_MOVE_COLUMN] = data[self.config.SHAPE_MOVE_COLUMN].apply(
                lambda val: val if isinstance(val, list)
                else json.loads(val) if isinstance(val, str)
                else np.nan
                )
        else:
            self.log_warning(f"No shape move column '{self.config.SHAPE_MOVE_COLUMN}' found in the data. This column is essential for parsing shape moves, so all values will be set to NaN.")
            data[self.config.SHAPE_MOVE_COLUMN] = np.nan
        if self.config.SHAPE_SAVE_COLUMN in data.columns:
            data[self.config.SHAPE_SAVE_COLUMN] = data[self.config.SHAPE_SAVE_COLUMN].apply(
                lambda val: val if isinstance(val, list)
                else json.loads(val) if isinstance(val, str)
                else np.nan
                )
        else:
            self.log_warning(f"No shape save column '{self.config.SHAPE_SAVE_COLUMN}' found in the data. This column is essential for parsing shape saves, so all values will be set to NaN.")
            data[self.config.SHAPE_SAVE_COLUMN] = np.nan

        data = self.merge_id_columns(data)
        data = self.fix_invalid_ids(data)
        data[self.config.PARSER_TIME_COLUMN] = pd.to_datetime(data[self.config.PARSER_TIME_COLUMN],
                                                              format=self.config.SERVER_DATE_FORMAT)
        # ensure that the time parsing worked correctly by checking that there are no NaT values in the time column
        if data[self.config.PARSER_TIME_COLUMN].isna().any():
            # find the rows with NaT values in the time column and log them as a warning
            self.log_warning(f"Time parsing resulted in NaT values for the following rows:\n{data[data[self.config.PARSER_TIME_COLUMN].isna()]}")
            raise CFGPipelineException('Time parsing failed, there are NaT values in the time column after parsing.')
        data = data.sort_values(by=self.config.PARSER_TIME_COLUMN).reset_index(drop=True)

        return data

    def patchfix_csv_data(self, data):
        '''Small patchy bugfix for temporary problems'''
        # Bug no.1 sometimes player external id is this instead of a random number
        if 'playerExternalId' in data.columns: # For rm2 this column does not exist
            bad_ext_id_mask = data['playerExternalId'] == '${rand://int/100000:10000000}'
            if bad_ext_id_mask.any():
                data.loc[bad_ext_id_mask, 'playerExternalId'] = None
                self.log_info("Applied patchfix for playerExternalId column to replace '${rand://int/100000:10000000}' with None.")
        if 'customData.endPosition' in data.columns:
            # Bug no.2 sometimes the endPosition and shape columns switch places
            switched_column_indices = np.flatnonzero(
                data['customData.endPosition'].apply(lambda x: len(json.loads(x)) == 10 if type(x) is str else False))
            if len(switched_column_indices) > 0:
                self.log_info(f"Applied patchfix for switched columns for {len(switched_column_indices)} rows where 'customData.endPosition' contains shape data.")
                self.log_info(f"Switched rows indices: {switched_column_indices}")
                data.loc[switched_column_indices, 'customData.shape'] = data.loc[
                    switched_column_indices, 'customData.endPosition']
        if 'customData.shape' in data.columns:
            before_customdata_shape = data['customData.shape'].copy()

            data['customData.shape'] = data['customData.shape'].apply(
                lambda x: x if isinstance(x, list)
                else json.loads(x) if isinstance(x, str)
                else []).apply(lambda x: str(x) if len(x) == 10 else np.nan
            )
            self.log_info("Applied patchfix for 'customData.shape' column to ensure it contains valid shape data or NaN.")
            if (before_customdata_shape.notna() & data['customData.shape'].isna()).any():
                invalid_indices = data.index[before_customdata_shape.notna() & data['customData.shape'].isna()]
                shape_unique_values = before_customdata_shape[invalid_indices].unique()
                unique_events = data.loc[invalid_indices, self.config.EVENT_TYPE].unique()
                elaborate_msg = f"Changed rows had the following unique values in 'customData.shape' before the patchfix: {shape_unique_values}, and the following unique event types: {unique_events}."
                self.log_warning(f"After applying the patchfix for 'customData.shape', the following rows were found to have invalid shape data that could not be parsed and were set to NaN:\n{before_customdata_shape[invalid_indices]}.\n{elaborate_msg}")

        return data

    def get_all_json_keys_from_csv_data(self, data):
        all_json_keys = np.concatenate(data[self.config.PARSER_JSON_COLUMN].apply(lambda x: tuple(x.keys())).unique())

        return set(all_json_keys)

    def merge_id_columns(self, data):
        data[MERGED_ID_KEY] = None

        for id_column in self.config.PARSER_ID_COLUMNS:
            if id_column in data.columns:
                missing_mask = missing_str_field(data[MERGED_ID_KEY])
                new_col_missing_mask = missing_str_field(data[id_column])
                data.loc[missing_mask, MERGED_ID_KEY] = data[id_column].loc[missing_mask].astype("string")
                filled_mask = missing_mask & ~new_col_missing_mask
                self.log_info(f"Merged id column '{id_column}' into merged id column for {filled_mask.sum()} rows.\n {missing_mask.sum() - filled_mask.sum()} rows were still missing after attempting to merge this column.")
            else:
                self.log_info(f"Id column '{id_column}' is missing from the data column. Will not be used for merging ids.")


        missing_mask = missing_str_field(data[MERGED_ID_KEY])
        data.loc[missing_mask, MERGED_ID_KEY] = DEFAULT_ID
        if missing_mask.sum() > 0:
            self.log_warning(f"{missing_mask.sum()} rows were filled with the default id '{DEFAULT_ID}' after attempting to merge all id columns. This means that for these rows, all id columns specified in the config were missing or empty. These rows will be grouped together under the same id, which may affect parsing results.\n Rows indices: {data.index[missing_mask].tolist()}")

        return data

    def _apply_hard_filters(self, game):
        return self.is_game_started(game)

    def is_game_started(self, game):
        game_started = game[self.config.EVENT_TYPE].str.contains(self.config.TUTORIAL_END_EVENT_TYPE).sum() > 0
        game_id = game.name
        if not game_started:
            self.log_info(f"Game {game_id} did not pass the hard filter of containing the tutorial end event, and will be excluded from parsing.")
        return game_started

    def _parse_all_player_games(self, games):
        all_parsed_games = []
        for _, game in games.groupby(self.config.UNIQUE_INTERNAL_ID_COLUMN):
            parsed_game = self.parse_single_game(game)
            all_parsed_games.append(parsed_game)

        return all_parsed_games

    def parse_single_game(self, game_data):
        parser_relevant_columns = self.parser_relevant_columns + self.include_in_id
        game_data = game_data[parser_relevant_columns]

        assert len(game_data[MERGED_ID_KEY].unique()) == 1
        player_id_field = game_data[MERGED_ID_KEY].iloc[0]
        game_start_time = game_data[game_data[self.config.EVENT_TYPE] == self.config.TUTORIAL_END_EVENT_TYPE].iloc[0][
            self.config.PARSER_TIME_COLUMN]

        game_data = game_data[game_data[self.config.PARSER_TIME_COLUMN] >= game_start_time]
        game_data = game_data[game_data[self.config.EVENT_TYPE].isin(self.shape_relevant_event_types)]
        first_row = [player_id_field, self.config.SHAPE_MOVE_EVENT_TYPE,
                     self.config.FIRST_SHAPE_SERVER_COORDS, np.nan, game_start_time]
        first_row_df = pd.DataFrame([first_row], columns=game_data.columns)
        game_data = pd.concat([first_row_df, game_data], ignore_index=True)

        game_data[self.config.PARSER_TIME_COLUMN] = (game_data[self.config.PARSER_TIME_COLUMN] - game_start_time).apply(
            lambda time_delta: time_delta.total_seconds())
        game_data[self.config.SHAPE_MOVE_COLUMN] = game_data[self.config.SHAPE_MOVE_COLUMN].apply(
            server_coords_to_binary_shape)

        game_data[self.config.GALLERY_SAVE_TIME_COLUMN] = None

        gallery_save_indices = game_data.index[game_data[self.config.SHAPE_MOVE_COLUMN].isna()]
        game_data.loc[gallery_save_indices - 1, self.config.GALLERY_SAVE_TIME_COLUMN] = game_data.loc[
            gallery_save_indices, self.config.PARSER_TIME_COLUMN].values
        # TODO: we need to ensure that the saved shape is the same as the one before (for missing shapes cases)
        # Now that we have the save time in all move rows, we can get rid of save rows:
        game_data = game_data[game_data[self.config.EVENT_TYPE].isin([self.config.SHAPE_MOVE_EVENT_TYPE])]

        actions = game_data.loc[:, self.config.PARSED_GAME_HEADERS]
        if self.include_in_id:
            player_id_field = [game_data[MERGED_ID_KEY].iloc[0]] + self.include_in_id
        parsed_game = {
            PARSED_PLAYER_ID_KEY: player_id_field,
            PARSED_TIME_KEY: game_start_time.timestamp(),
            PARSED_ALL_SHAPES_KEY: actions.values.tolist(),
        }

        return parsed_game

    @classmethod
    def translate_parsed_results_to_mathematica(cls, json_format):
        games_in_old_format = []
        for entry in json_format:
            player_id = entry[PARSED_PLAYER_ID_KEY]
            player_start_time = entry[PARSED_TIME_KEY]
            player_actions = entry[PARSED_ALL_SHAPES_KEY]

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
                PARSED_PLAYER_ID_KEY: game_id,
                PARSED_TIME_KEY: absolute_start_time,
                PARSED_CHOSEN_SHAPES_KEY: chosen_shapes,
                PARSED_ALL_SHAPES_KEY: actions,
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

    def fix_invalid_ids(self, data):
        """
        Check if any player ID (after merging) includes invalid characters (e.g. brackets, commas, quotes) that may cause issues during parsing or later analysis, and if so, log a warning with the affected IDs and how they will be changed, and replace these characters with underscores in the merged ID column.
        """
        INVALID_CHARS = ['[', ']', '{', '}', ',', '"', "'", '\\','$','/']
        invalid_id_mask = data[MERGED_ID_KEY].apply(lambda x: any(char in str(x) for char in INVALID_CHARS))
        def replace_chars(str):
            for char in INVALID_CHARS:
                str = str.replace(char, '_')
            return str
        if invalid_id_mask.any():
            affected_ids = data.loc[invalid_id_mask, MERGED_ID_KEY].unique()
            elaborate_msg = f"The following unique merged IDs were found to contain invalid characters {INVALID_CHARS} that may cause issues during parsing or later analysis: {affected_ids}. These characters will be replaced with underscores in the merged ID column to ensure proper parsing and analysis. Affected rows indices: {data.index[invalid_id_mask].tolist()}"
            self.log_warning(elaborate_msg)
            data.loc[invalid_id_mask, MERGED_ID_KEY] = data.loc[invalid_id_mask, MERGED_ID_KEY].apply(replace_chars)
        return data
