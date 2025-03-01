from pathlib import Path
import pytest
from CFGpy.behavioral import Downloader, Parser, PostParser, FeatureExtractor, Pipeline, Configuration
import os
import json
import numpy as np
import pandas as pd

TEST_FILES_DIR = os.path.join(Path(__file__).parent, "test_files")
CONFIG_FILENAME = "config.yml"
TEST_DOWNLOADED_FILENAME = "test_raw.csv"

TEST_PARSED_FILENAME = "test_parsed.json"
TEST_PARSED_OLD_FORMAT_FILENAME = "test_parsed_old_format.txt"
TEST_POSTPARSED_FILENAME = "test_postparsed.json"
TEST_FEATURES_FILENAME = "test_features.csv"

test_dirs = [entry.path for entry in os.scandir(TEST_FILES_DIR) if entry.is_dir()]


@pytest.mark.parametrize("test_dir", test_dirs)
def test_downloader(test_dir):
    raw_data_filename = "raw.csv"
    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    Downloader(output_filename=raw_data_filename, config=config).download(verbose=True)

    test_raw = (pd.read_csv(os.path.join(test_dir, TEST_DOWNLOADED_FILENAME))
                .sort_values("id")
                .reset_index(drop=True))
    raw = pd.read_csv(raw_data_filename).sort_values("id").reset_index(drop=True)

    assert len(test_raw) == len(raw), f"{len(raw)} events instead of {len(test_raw)}"
    for col_name in test_raw:
        assert col_name in raw, f"missing column {col_name}"
        if test_raw[col_name].dtype == "float64":
            assert np.allclose(test_raw[col_name], raw[col_name], equal_nan=True), f"{col_name} comparison failed"
        else:
            # drop spaces after commas, added by python but not in RedMetrics' output
            test_col = test_raw[col_name].astype(str).str.replace(", ", ",")
            col = raw[col_name].astype(str).str.replace(", ", ",")
            assert test_col.equals(col), f"{col_name} comparison failed"


def _compare_parsed(parsed, test_dir):
    with open(os.path.join(test_dir, TEST_PARSED_FILENAME), "r") as test_parsed_fp:
        test_parsed = json.load(test_parsed_fp)

    # convert to df and compare by key, for proper float comparison in the start time column:
    test_parsed_df = pd.DataFrame(test_parsed)
    parsed_df = pd.DataFrame(parsed)
    from CFGpy.behavioral._consts import PARSED_PLAYER_ID_KEY, PARSED_TIME_KEY, PARSED_ALL_SHAPES_KEY
    assert test_parsed_df[PARSED_PLAYER_ID_KEY].equals(parsed_df[PARSED_PLAYER_ID_KEY])
    assert np.allclose(test_parsed_df[PARSED_TIME_KEY], parsed_df[PARSED_TIME_KEY])
    assert test_parsed_df[PARSED_ALL_SHAPES_KEY].equals(parsed_df[PARSED_ALL_SHAPES_KEY])
    # TODO: after parser handles chosen shapes, compare those too


@pytest.mark.parametrize("test_dir", test_dirs)
def test_parser(test_dir):
    parsed_data_filename = "parsed.json"

    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    parser = Parser.from_file(os.path.join(test_dir, TEST_DOWNLOADED_FILENAME), config=config)
    parsed = parser.parse()
    parser.dump(parsed_data_filename)

    _compare_parsed(parsed, test_dir)


@pytest.mark.parametrize("test_dir", test_dirs)
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


@pytest.mark.parametrize("test_dir", test_dirs)
def test_parser_conversion_to_new_format(test_dir):
    mathematica_path = os.path.join(test_dir, TEST_PARSED_OLD_FORMAT_FILENAME)
    converted_to_new_format = Parser.translate_mathematica_to_python(mathematica_path)
    with open(r"parsed_converted_to_new_format.json", "w") as converted_to_new_format_fp:
        json.dump(converted_to_new_format, converted_to_new_format_fp)

    _compare_parsed(converted_to_new_format, test_dir)


@pytest.mark.parametrize("test_dir", test_dirs)
def test_postparser(test_dir):
    postparsed_data_filename = "postparsed.json"

    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    postparser = PostParser.from_json(os.path.join(test_dir, TEST_PARSED_FILENAME), config=config)
    postparser.postparse()
    postparser.dump(postparsed_data_filename)

    with open(os.path.join(test_dir, TEST_POSTPARSED_FILENAME), "r") as test_postparsed_fp:
        test_postparsed = test_postparsed_fp.read()
    with open(postparsed_data_filename, "r") as postparsed_fp:
        postparsed = postparsed_fp.read()

    if test_postparsed != postparsed:
        # avoid asserting the str comparison, because if it fails python tries printing the strings and takes too long
        assert False

def _compare_features(test_dir, features_filename):
    test_features = pd.read_csv(os.path.join(test_dir, TEST_FEATURES_FILENAME)).sort_values("ID").reset_index(drop=True)
    features = pd.read_csv(features_filename).sort_values("ID").reset_index(drop=True)
    assert len(test_features) == len(features), f"{len(features)} subjects instead of {len(test_features)}"
    for col in test_features:
        assert col in features, f"missing feature {col}"
        if test_features[col].dtype == "float64":
            assert np.allclose(test_features[col], features[col], equal_nan=True), f"{col} comparison failed"
        elif col != "Date/Time":  # date/time causes problems with subjects that played during daylight saving
            assert test_features[col].equals(features[col]), f"{col} comparison failed"


@pytest.mark.parametrize("test_dir", test_dirs)
def test_feature_extractor(test_dir):
    features_filename = "features.csv"

    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    feature_extractor = FeatureExtractor.from_json(os.path.join(test_dir, TEST_POSTPARSED_FILENAME), config=config)
    feature_extractor.extract(verbose=True)
    feature_extractor.dump(features_filename)

    _compare_features(test_dir, features_filename)


@pytest.mark.parametrize("test_dir", test_dirs)
def test_full_pipeline(test_dir):
    features_filename = "features.csv"
    config = Configuration.from_yaml(os.path.join(test_dir, CONFIG_FILENAME))
    pipeline = Pipeline(output_filename=features_filename, config=config)
    pipeline.run_pipeline(verbose=True)
    _compare_features(test_dir, features_filename)
