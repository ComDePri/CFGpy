from pathlib import Path
import pytest
from CFGpy.behavioral import RedMetrics1Downloader, Parser, PostParser, FeatureExtractor, Pipeline, Configuration, \
    LocalDataRetriever, RM1DumpDataRetriever, IOCANEDataRetriever
from CFGpy.behavioral._consts import RM1, IOCANE, MERGED_ID_KEY, LOCAL
import os
import json
import numpy as np
import pandas as pd
import sys
from enum import Enum
from typing import Iterable


class Process(Enum):
    PARSER = "parser"
    POSTPARSER = "postparser"
    FEATURE_EXTRACTOR = "feature_extractor"
    DOWNLOADER = "downloader"
    PIPELINE = "pipeline"


PIPELINE_TEST_FILES_DIR = os.path.join(Path(__file__).parent, "test_files")
DUMP_TEST_FILES_DIR = os.path.join(Path(__file__).parent, "dump_test_files")
TEST_CONFIGURABLES_DIR = os.path.join(Path(__file__).parent, "test_config_logics_files")
CONFIG_FILENAME = "config.yml"
TEST_DOWNLOADED_FILENAME = "test_raw.csv"

TEST_PARSED_FILENAME = "test_parsed.json"
TEST_PARSED_OLD_FORMAT_FILENAME = "test_parsed_old_format.txt"
TEST_POSTPARSED_FILENAME = "test_postparsed.json"
TEST_FEATURES_FILENAME = "test_features.csv"

pipeline_test_dirs = [entry.path for entry in os.scandir(PIPELINE_TEST_FILES_DIR) if entry.is_dir()]
rm1_dump_test_dirs = [entry.path for entry in os.scandir(DUMP_TEST_FILES_DIR) if entry.is_dir()]
pipeline_iocane_test_dirs = [path for path in pipeline_test_dirs if "iocane" in path]



class ConfigurableParam:
    def __init__(self, name: str, possible_values, affected_process: Process):
        self.name = name
        self.possible_values = possible_values
        self.affected_process = affected_process


CONFIGURABLE_PARAMS = [
    ConfigurableParam("PARSER_ID_COLUMNS", [("userProvidedId", "userId"), ("userId", "userProvidedId")],
                      Process.PARSER),
    ConfigurableParam("MIN_SAVE_FOR_EXPLOIT", [2, 3, 4], Process.POSTPARSER),
    ConfigurableParam("MARGIN_FOR_PAUSE_DURATION", [2, 3, 4], Process.FEATURE_EXTRACTOR),
    ConfigurableParam("STEP_ORIG_PSEUDOCOUNT", [0.5, 1, 2], Process.FEATURE_EXTRACTOR),
    ConfigurableParam("GALLERY_ORIG_PSEUDOCOUNT", [0.5, 1, 2], Process.FEATURE_EXTRACTOR),
    ConfigurableParam("WRITE_G_ALPHA", [True, False], Process.FEATURE_EXTRACTOR),
    ConfigurableParam("MIN_N_MOVES", [0, 60, 90], Process.FEATURE_EXTRACTOR),
    ConfigurableParam("MIN_N_CLUSTERS", [0, 1, 4], Process.FEATURE_EXTRACTOR),
    ConfigurableParam("MIN_GAME_DURATION_SEC", [0, 600, 1200], Process.FEATURE_EXTRACTOR),
    ConfigurableParam("MAX_PAUSE_DURATION_SEC", [0, 90, 600], Process.FEATURE_EXTRACTOR),
    ConfigurableParam("MAX_ZSCORE_FOR_OUTLIERS", [0, 3, 10], Process.FEATURE_EXTRACTOR),
    # ConfigurableParam("VISUALIZATION_MAKE_PLOTS", [True, False], Process.PIPELINE), # this is not tested because it causes the tests to run very slowly, and it's not a critical parameter to test since it only affects the visualization and not the extracted features
    # ConfigurableParam("VISUALIZATION_ANIMATE", [True, False], Process.PIPELINE), # this is not tested because it causes the tests to run very slowly, and it's not a critical parameter to test since it only affects the visualization and not the extracted features
    ConfigurableParam("SAVE_INTERMEDIATE_FILES", [True, False], Process.PIPELINE)
]

@pytest.mark.parametrize("cparam", CONFIGURABLE_PARAMS, ids=[cparam.name for cparam in CONFIGURABLE_PARAMS])
def test_configurable_parameters_run(cparam: ConfigurableParam):
    """
    Checks for every defined configurable parameter in the Configuration (one that users are expected to change) that it can run with multiple possible values without errors.
    """
    test_dir = os.path.join(PIPELINE_TEST_FILES_DIR, "set6_iocane")
    cfg = Configuration.from_yaml(os.path.join(test_dir, "config.yml"))
    assert hasattr(cfg, cparam.name), f"parameter {cparam.name} not found in the Configuration object"
    for val in cparam.possible_values:
        print(f"{cparam.name} = {val}")
        setattr(cfg, cparam.name, val)
        if cparam.affected_process == Process.PARSER:
            raw_data = pd.read_csv(os.path.join(test_dir, TEST_DOWNLOADED_FILENAME))
            parser = Parser(raw_data=raw_data, config=cfg)
            parser.parse()
        elif cparam.affected_process == Process.POSTPARSER:
            postparser: PostParser = PostParser.from_json(os.path.join(test_dir, TEST_PARSED_FILENAME), config=cfg)
            postparser.postparse()
        elif cparam.affected_process == Process.FEATURE_EXTRACTOR:
            feature_extractor = FeatureExtractor.from_json(os.path.join(test_dir, TEST_POSTPARSED_FILENAME),
                                                           config=cfg)
            feature_extractor.extract(verbose=True)
        elif cparam.affected_process == Process.PIPELINE:
            cfg.DATA_SOURCE = LOCAL
            cfg.EVENT_CSV_PATH = os.path.join(test_dir, TEST_DOWNLOADED_FILENAME)
            pipeline = Pipeline(config=cfg)
            pipeline.run_pipeline()


def test_default_config_dump_compatibility():
    """
    Test that the default config can be dumped and loaded without errors, and that the loaded config has the same values as the original default config.
    """
    config = Configuration.default()
    config_dump_path = "default_config_dump.yml"
    config.to_yaml(config_dump_path)
    loaded_config = Configuration.from_yaml(config_dump_path)
    if not config == loaded_config:
        # print the diferring fields for easier debugging
        for field in config.__dataclass_fields__:
            if getattr(config, field) != getattr(loaded_config, field):
                print(
                    f"Field {field} differs: {getattr(config, field)} in original config vs {getattr(loaded_config, field)} in loaded config",
                    file=sys.stderr)
        assert False, "Loaded config is different from the original default config"


def test_user_columns_ordering():
    """
    Test that the parser takes the correct ID field based on the order defined in the config
    """
    UID_COLUMN = "userId"
    UPID_COLUMN = "userProvidedId"
    PROID_COLUMN = "prolificId"
    PEID_COLUMN = "playerExternalId"

    COLUMNS2SUFFIXES = {
        UID_COLUMN: "uid",
        UPID_COLUMN: "upid",
        PROID_COLUMN: "proid",
        PEID_COLUMN: "peid"
    }
    config = Configuration.default()
    config.PARSER_ID_COLUMNS = (UID_COLUMN, UPID_COLUMN, PROID_COLUMN, PEID_COLUMN)
    expected_col = UID_COLUMN
    expected_suffix = COLUMNS2SUFFIXES[expected_col]
    raw_data_full_user_fields = pd.read_csv(os.path.join(TEST_CONFIGURABLES_DIR, "all_user_columns_events.csv"))
    parser = Parser(config=config, raw_data=raw_data_full_user_fields)
    parsed_data = parser.parse()
    assert all(game["id"].endswith(expected_suffix) for game in
               parsed_data), f"ID field is not {expected_col} as expected based on the config"

    # same for user provided ID:
    config.PARSER_ID_COLUMNS = (UPID_COLUMN, UID_COLUMN, PROID_COLUMN, PEID_COLUMN)
    expected_col = UPID_COLUMN
    expected_suffix = COLUMNS2SUFFIXES[expected_col]
    parser = Parser(config=config, raw_data=raw_data_full_user_fields)
    parsed_data = parser.parse()
    assert all(game["id"].endswith(expected_suffix) for game in
               parsed_data), f"ID field is not {expected_col} as expected based on the config"
    # same for prolific ID:
    config.PARSER_ID_COLUMNS = (PROID_COLUMN, UID_COLUMN, UPID_COLUMN, PEID_COLUMN)
    expected_col = PROID_COLUMN
    expected_suffix = COLUMNS2SUFFIXES[expected_col]
    parser = Parser(config=config, raw_data=raw_data_full_user_fields)
    parsed_data = parser.parse()
    assert all(game["id"].endswith(expected_suffix) for game in
               parsed_data), f"ID field is not {expected_col} as expected based on the config"
    # same for user external ID:
    config.PARSER_ID_COLUMNS = (PEID_COLUMN, UID_COLUMN, UPID_COLUMN, PROID_COLUMN)
    expected_col = PEID_COLUMN
    expected_suffix = COLUMNS2SUFFIXES[expected_col]
    parser = Parser(config=config, raw_data=raw_data_full_user_fields)
    parsed_data = parser.parse()
    assert all(game["id"].endswith(expected_suffix) for game in
               parsed_data), f"ID field is not {expected_col} as expected based on the config"
    # now Check that it works when the external ID is missing
    raw_data_missing_external = pd.read_csv(os.path.join(TEST_CONFIGURABLES_DIR, "missing_userexternalid_events.csv"))
    config.PARSER_ID_COLUMNS = (PEID_COLUMN, UID_COLUMN, UPID_COLUMN, PROID_COLUMN)
    expected_col = UID_COLUMN
    expected_suffix = COLUMNS2SUFFIXES[expected_col]
    parser = Parser(config=config, raw_data=raw_data_missing_external)
    parsed_data = parser.parse()
    assert all(game["id"].endswith(expected_suffix) for game in
               parsed_data), f"ID field is not {expected_col} as expected based on the config"


def test_short_game_filtering():
    ID_WITH_SHORT_GAME = "pt_69da9d88_20180206233103346_rm_vs_radm"
    # load postparsed data
    postparsed_path = os.path.join(TEST_CONFIGURABLES_DIR, "test_postparsed_with_short_game.json")
    # start with default config
    config = Configuration.from_yaml(os.path.join(TEST_CONFIGURABLES_DIR, "test_config_with_short_game.yml"))
    feature_extractor = FeatureExtractor.from_json(postparsed_path, config=config)
    feats = feature_extractor.extract(verbose=True)
    config_ignore_shorts = Configuration.from_yaml(
        os.path.join(TEST_CONFIGURABLES_DIR, "test_config_with_short_game.yml"))
    # removed all actions after more than 4 seconds in the early game
    config_ignore_shorts.MAX_IGNORED_GAME_DURATION_SEC = (4 * 60) + 1
    feats_ignore_shorts = FeatureExtractor.from_json(postparsed_path, config=config_ignore_shorts).extract(verbose=True)
    # check that the ID with the short game is in the features with short games ignored but not in the features with default config
    assert ID_WITH_SHORT_GAME not in feats[
        "ID"].values, f"ID {ID_WITH_SHORT_GAME} is present in features with default config"
    assert ID_WITH_SHORT_GAME in feats_ignore_shorts[
        "ID"].values, f"ID {ID_WITH_SHORT_GAME} is missing from features with short games ignored"


def test_time_filtering():
    test_dir = os.path.join(PIPELINE_TEST_FILES_DIR, "set1")
    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    config.BEFORE_DATE = "2024"
    downloader_before2024 = LocalDataRetriever(events_csv_path=os.path.join(test_dir, TEST_DOWNLOADED_FILENAME),
                                               config=config)
    df_before2024 = downloader_before2024.retrieve_data(verbose=True)
    assert pd.to_datetime(df_before2024["userTime"]).max() < pd.to_datetime("2024",
                                                                            utc=True), f"Max userTime is {pd.to_datetime(df_before2024['userTime']).max()} instead of before 2024"
    # should remove two games
    N_ROWS_BEFORE2024 = 32826
    assert len(df_before2024) == N_ROWS_BEFORE2024, f"{len(df_before2024)} events instead of {N_ROWS_BEFORE2024}"
    config.BEFORE_DATE = None
    config.AFTER_DATE = "2024"
    downloader_after2024 = LocalDataRetriever(events_csv_path=os.path.join(test_dir, TEST_DOWNLOADED_FILENAME),
                                              config=config)
    df_after2024 = downloader_after2024.retrieve_data(verbose=True)
    assert pd.to_datetime(df_after2024[
                              "userTime"]).min() >= pd.to_datetime("2024",
                                                                   utc=True), f"Min userTime is {pd.to_datetime(df_after2024['userTime']).min()} instead of 2024 or later"
    N_ROWS_AFTER2024 = 6
    assert len(df_after2024) == N_ROWS_AFTER2024, f"{len(df_after2024)} events instead of {N_ROWS_AFTER2024}"
    # check both before and after
    config.BEFORE_DATE = "2024"
    config.AFTER_DATE = "2023-3"
    downloader_before2024_after2023_3 = LocalDataRetriever(
        events_csv_path=os.path.join(test_dir, TEST_DOWNLOADED_FILENAME),
        config=config)
    df_before2024_after2023_3 = downloader_before2024_after2023_3.retrieve_data(verbose=True)
    assert pd.to_datetime(df_before2024_after2023_3["userTime"]).max() < pd.to_datetime("2024",
                                                                                        utc=True), f"Max userTime is {pd.to_datetime(df_before2024_after2023_3['userTime']).max()} instead of before 2024"
    assert pd.to_datetime(df_before2024_after2023_3["userTime"]).min() >= pd.to_datetime("2023-3",
                                                                                         utc=True), f"Min userTime is {pd.to_datetime(df_before2024_after2023_3['userTime']).min()} instead of 2023-3 or later"
    N_ROWS_BEFORE2024_AFTER2023_3 = 856
    assert len(
        df_before2024_after2023_3) == N_ROWS_BEFORE2024_AFTER2023_3, f"{len(df_before2024_after2023_3)} events instead of {N_ROWS_BEFORE2024_AFTER2023_3}"


def _compare_raws(test_raw, raw, json_dict_cols=tuple(), shared_cols=False):
    assert len(test_raw) == len(raw), f"{len(raw)} events instead of {len(test_raw)}"
    if shared_cols:
        cols = set(test_raw.columns).intersection(set(raw.columns))
    else:
        cols = test_raw.columns
    for col_name in cols:
        assert col_name in raw, f"missing column {col_name}"
        if col_name in json_dict_cols:
                # compare json columns by loading them as json and comparing the resulting objects (to avoid issues with formatting differences in the json strings)
                test_json = test_raw[col_name].apply(json.loads)
                raw_json = raw[col_name].apply(json.loads)
                # compare the dictionaries for having similar content:
                for i, (test_dict, raw_dict) in enumerate(zip(test_json, raw_json)):
                    assert test_dict == raw_dict, f"Difference in column {col_name} at row {i}: {test_dict} vs {raw_dict}"
        elif test_raw[col_name].dtype == "float64":
            assert np.allclose(test_raw[col_name], raw[col_name], equal_nan=True), f"{col_name} comparison failed"
        else:
            # drop spaces after commas, added by python but not in RedMetrics' output
            test_col = test_raw[col_name].astype(str).str.replace(", ", ",")
            col = raw[col_name].astype(str).str.replace(", ", ",")
            assert test_col.equals(col), f"{col_name} comparison failed"


@pytest.mark.parametrize("test_dir", pipeline_test_dirs)
def test_downloader(test_dir):
    raw_data_filename = "raw"
    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    # get the right downloader:
    if config.DATA_SOURCE == RM1:
        downloader = RedMetrics1Downloader(output_filename=raw_data_filename, config=config)
    elif config.DATA_SOURCE == IOCANE:
        downloader = IOCANEDataRetriever(output_filename=raw_data_filename, config=config)
    else:
        raise ValueError(f"Unsupported data source for testing: {config.DATA_SOURCE}")
    downloader.retrieve_data(verbose=True)
    downloader.dump()

    test_raw = (pd.read_csv(os.path.join(test_dir, TEST_DOWNLOADED_FILENAME))
                .sort_values("id")
                .reset_index(drop=True))
    raw = pd.read_csv(downloader.output_path).sort_values("id").reset_index(drop=True)
    _compare_raws(test_raw, raw)

@pytest.mark.parametrize("test_dir", pipeline_iocane_test_dirs)
def test_cached_downloader(test_dir):
    """
    Test that if the downloader is run twice, the second time it uses the cached file and produces the same output as the first time.
    """
    raw_data_filename = "raw"
    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    downloader = IOCANEDataRetriever(output_filename=raw_data_filename, config=config)
    df = downloader.retrieve_data(verbose=True)
    downloader.dump()
    df = pd.read_csv(downloader.output_path).sort_values("id").reset_index(drop=True)

    # run the downloader again and check that it produces the same output
    downloader2 = IOCANEDataRetriever(output_filename=raw_data_filename, config=config)
    df2 = downloader2.retrieve_data(verbose=True)
    downloader2.dump()
    df2 = pd.read_csv(downloader2.output_path).sort_values("id").reset_index(drop=True)

    _compare_raws(df, df2)

@pytest.mark.parametrize("test_dir", pipeline_iocane_test_dirs)
def test_local_downloader(test_dir):
    """
    Test that the downloader can read from a local file and produce the same output as the original file downloaded using rm1.
    """
    raw_data_filename = "raw"
    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    downloader = IOCANEDataRetriever(output_filename=raw_data_filename, config=config)
    df = downloader.retrieve_data(verbose=True)
    downloader.dump()
    df = pd.read_csv(downloader.output_path).sort_values("id").reset_index(drop=True)


    # now do the same with local retriever and compare the outputs
    local_retriever = LocalDataRetriever(events_csv_path=downloader.output_path, config=config)
    local_df = local_retriever.retrieve_data(verbose=True)
    os.remove(downloader.output_path)
    local_retriever.dump()
    local_df = pd.read_csv(local_retriever.output_path).sort_values("id").reset_index(drop=True)
    os.remove(local_retriever.output_path)
    # compare the two
    _compare_raws(df, local_df)


@pytest.mark.parametrize("test_dir", rm1_dump_test_dirs)
def test_rm1_dump_server_consistency(test_dir):
    """
    Test that the downloader can read from the NAS rm1 dump and produce the same output as the original file downloaded using rm1.
    """
    dump_config = Configuration.from_yaml(os.path.join(test_dir, "rm1_dump_config.yml"))
    online_config = Configuration.from_yaml(os.path.join(test_dir, "rm1_online_config.yml"))
    dump_downloader = RM1DumpDataRetriever(output_filename="raw_dump", config=dump_config)
    dump_df = dump_downloader.retrieve_data(verbose=True)
    dump_df = dump_df.sort_values("id").reset_index(drop=True)
    online_downloader = RedMetrics1Downloader(output_filename="raw_online", config=online_config)
    online_df = online_downloader.retrieve_data(verbose=True)
    online_df = online_df.sort_values("id").reset_index(drop=True)
    _compare_raws(dump_df, online_df)


def _print_diff(df, test_comp_df, col_name, print_cols=None, allow_deviations=False):
    if not allow_deviations:
        diff_mask = df[col_name] != test_comp_df[col_name]
    else:
        diff_mask = ~np.isclose(df[col_name], test_comp_df[col_name], equal_nan=True)
    if print_cols is None:
        print_cols = [col_name]
    print(f"Difference in column {col_name}:", file=sys.stderr)
    print(df[diff_mask][print_cols], file=sys.stderr)
    print(test_comp_df[diff_mask][print_cols], file=sys.stderr)


def _assert_col_equality(df, test_comp_df, col_name, print_cols=None):
    if not df[col_name].equals(test_comp_df[col_name]):
        _print_diff(df, test_comp_df, col_name, print_cols=print_cols)
        assert False, f"{col_name} comparison failed"


def _assert_col_allclose(df, test_comp_df, col_name, print_cols=None):
    if not np.allclose(df[col_name], test_comp_df[col_name], equal_nan=True):
        _print_diff(df, test_comp_df, col_name, allow_deviations=True, print_cols=print_cols)
        assert False, f"{col_name} comparison failed"

def _compare_parsed(parsed, test_parsed):
    if len(parsed) == 0 and len(test_parsed) == 0:
        print("Both parsed and test_parsed are empty, skipping comparison")
        return
    # convert to df and compare by key, for proper float comparison in the start time column:
    test_parsed_df = pd.DataFrame(test_parsed)
    parsed_df = pd.DataFrame(parsed)
    from CFGpy.behavioral._consts import PARSED_PLAYER_ID_KEY, PARSED_TIME_KEY, PARSED_ALL_SHAPES_KEY
    _assert_col_equality(test_parsed_df, parsed_df, PARSED_PLAYER_ID_KEY,
                         print_cols=[PARSED_PLAYER_ID_KEY, PARSED_TIME_KEY])
    _assert_col_allclose(test_parsed_df, parsed_df, PARSED_TIME_KEY, print_cols=[PARSED_PLAYER_ID_KEY, PARSED_TIME_KEY])
    _assert_col_equality(test_parsed_df, parsed_df, PARSED_ALL_SHAPES_KEY,
                         print_cols=[PARSED_PLAYER_ID_KEY, PARSED_ALL_SHAPES_KEY])
    # TODO: after parser handles chosen shapes, compare those too

def _compare_parsed_to_test_dir(parsed, test_dir):
    with open(os.path.join(test_dir, TEST_PARSED_FILENAME), "r") as test_parsed_fp:
        test_parsed = json.load(test_parsed_fp)
    _compare_parsed(parsed, test_parsed)



@pytest.mark.parametrize("test_dir", pipeline_test_dirs)
def test_parser(test_dir):
    parsed_data_filename = "parsed.json"

    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    parser = Parser.from_file(os.path.join(test_dir, TEST_DOWNLOADED_FILENAME), config=config)
    parsed = parser.parse()
    parser.dump(name=parsed_data_filename)

    _compare_parsed_to_test_dir(parsed, test_dir)


@pytest.mark.parametrize("test_dir", pipeline_test_dirs)
def test_parser_conversion_to_old_format(test_dir):
    with open(os.path.join(test_dir, TEST_PARSED_FILENAME), "r") as new_format_fp:
        parsed_new_format = json.load(new_format_fp)

    parsed_old_format = Parser.translate_parsed_results_to_mathematica(parsed_new_format)
    with open("parsed_old_format.txt", "w") as parsed_old_format_fp:
        parsed_old_format_fp.write(parsed_old_format)

    with open(os.path.join(test_dir, TEST_PARSED_OLD_FORMAT_FILENAME), "r") as test_parsed_old_format_fp:
        test_parsed_old_format = test_parsed_old_format_fp.read()

    if test_parsed_old_format != parsed_old_format:
        # avoid asserting the str comparison, because if it fails python tries printing the strings and takes too long
        assert False


@pytest.mark.parametrize("test_dir", pipeline_test_dirs)
def test_parser_conversion_to_new_format(test_dir):
    mathematica_path = os.path.join(test_dir, TEST_PARSED_OLD_FORMAT_FILENAME)
    converted_to_new_format = Parser.translate_mathematica_to_python(mathematica_path)
    with open(r"parsed_converted_to_new_format.json", "w") as converted_to_new_format_fp:
        json.dump(converted_to_new_format, converted_to_new_format_fp)

    _compare_parsed_to_test_dir(converted_to_new_format, test_dir)


def _print_json_diff(json_ref, json_comp):
    # jsons are assumed to be list of games, each game has "id", "absolute start time" float and "actions" - list of actions.
    if len(json_ref) != len(json_comp):
        print(f"Different number of games: {len(json_ref)} in expected data vs {len(json_comp)} in code run",
              file=sys.stderr)
    for game_idx, (game_ref, game_comp) in enumerate(zip(json_ref, json_comp)):
        if game_ref["id"] != game_comp["id"]:
            print(
                f"Game {game_idx} has different id: {game_ref['id']} in expected data vs {game_comp['id']} in code run",
                file=sys.stderr)
        if not np.isclose(game_ref["absolute start time"], game_comp["absolute start time"], equal_nan=True):
            print(
                f"Game {game_idx} has different absolute start time: {game_ref['absolute start time']} in expected data vs {game_comp['absolute start time']} in code run",
                file=sys.stderr)
        if len(game_ref["actions"]) != len(game_comp["actions"]):
            print(
                f"Game {game_idx} has different number of actions: {len(game_ref['actions'])} in expected data vs {len(game_comp['actions'])} in code run",
                file=sys.stderr)
        for action_idx, (action_ref, action_comp) in enumerate(zip(game_ref["actions"], game_comp["actions"])):
            if action_ref != action_comp:
                print(
                    f"Game {game_idx}, action {action_idx} is different: {action_ref} in expected data vs {action_comp} in code run",
                    file=sys.stderr)


@pytest.mark.parametrize("test_dir", pipeline_test_dirs)
def test_postparser(test_dir):
    postparsed_data_filename = "test_data"

    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    postparser = PostParser.from_json(os.path.join(test_dir, TEST_PARSED_FILENAME), config=config)
    postparser.postparse()
    postparser.dump(name=postparsed_data_filename, pretty=config.DATA_SOURCE == IOCANE)

    with open(os.path.join(test_dir, TEST_POSTPARSED_FILENAME), "r") as test_postparsed_fp:
        test_postparsed = test_postparsed_fp.read()
    with open(postparsed_data_filename + "_postparsed.json", "r") as postparsed_fp:
        postparsed = postparsed_fp.read()

    if test_postparsed != postparsed:
        _print_json_diff(json.loads(test_postparsed), json.loads(postparsed))
        # avoid asserting the str comparison, because if it fails python tries printing the strings and takes too long
        assert False


def _assert_ids_match(df, test_comp_df, id_col_name):
    ids = set(df[id_col_name])
    test_ids = set(test_comp_df[id_col_name])
    if ids != test_ids:
        print(f"ID mismatch:\n", file=sys.stderr)
        print(f"IDs in code run but not in expected data: {'\n'.join(ids - test_ids)}", file=sys.stderr)
        print(f"IDs in expected data but not in code run: {'\n'.join(test_ids - ids)}", file=sys.stderr)
        assert False, "ID mismatch"


def _compare_features(test_dir, features_filename):
    test_features = pd.read_csv(os.path.join(test_dir, TEST_FEATURES_FILENAME)).sort_values("ID").reset_index(drop=True)
    features = pd.read_csv(features_filename).sort_values("ID").reset_index(drop=True)
    _assert_ids_match(features, test_features, "ID")
    assert len(test_features) == len(features), f"{len(features)} subjects instead of {len(test_features)}"
    for col in test_features:
        assert col in features.columns, f"missing feature {col}"
        if test_features[col].dtype == "float64":
            _assert_col_allclose(features, test_features, col, print_cols=["ID", col])
        elif col != "Date/Time":  # date/time causes problems with subjects that played during daylight saving
            _assert_col_equality(features, test_features, col, print_cols=["ID", col])


@pytest.mark.parametrize("test_dir", pipeline_test_dirs)
def test_feature_extractor(test_dir):
    features_filename = "test_data"

    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    feature_extractor = FeatureExtractor.from_json(os.path.join(test_dir, TEST_POSTPARSED_FILENAME), config=config)
    feature_extractor.extract(verbose=True)
    feature_extractor.dump(name=features_filename)

    _compare_features(test_dir, features_filename + "_measures.csv")


@pytest.mark.parametrize("test_dir", pipeline_test_dirs)
def test_full_pipeline(test_dir):
    features_filename = "test_data"
    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    pipeline = Pipeline(output_filename=features_filename, config=config)
    pipeline.run_pipeline()
    _compare_features(test_dir, features_filename + "_measures.csv")
