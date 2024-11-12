import pytest
from CFGpy.behavioral import Downloader, Parser, PostParser, FeatureExtractor, Pipeline, Configuration
from CFGpy import NAS_PATH
import os
import json
import numpy as np
import pandas as pd

TEST_FILES_DIR = os.path.join(NAS_PATH, "Projects", "CFG", "CFGpy test files")
RED_METRICS_URL_FILENAME = "red_metrics_url.txt"
TEST_DOWNLOADED_FILENAME = "test_event.csv"
TEST_PARSED_FILENAME = "test_parsed.json"
TEST_PARSED_OLD_FORMAT_FILENAME = "test_parsed_old_format.txt"
TEST_POSTPARSED_FILENAME = "test_postparsed.json"
TEST_FEATURES_FILENAME = "test_features.csv"

test_dirs = [os.path.join(TEST_FILES_DIR, filename)
             for filename in os.listdir(TEST_FILES_DIR)
             if os.path.isdir(os.path.join(TEST_FILES_DIR, filename))]


@pytest.mark.parametrize("test_dir", test_dirs)
def test_downloader(test_dir):
    # See https://github.com/ComDePri/CFGpy/issues/13
    with open(os.path.join(test_dir, RED_METRICS_URL_FILENAME)) as url_fp:
        url = url_fp.read()
    Downloader(url, "event.csv").download(verbose=True)

    test_event = pd.read_csv(os.path.join(test_dir, TEST_DOWNLOADED_FILENAME)).sort_values("id").reset_index(drop=True)
    event = pd.read_csv("event.csv").sort_values("id").reset_index(drop=True)
    assert test_event.equals(event)


@pytest.mark.parametrize("test_dir", test_dirs)
def test_parser(test_dir):
    parser = Parser.from_file(os.path.join(test_dir, TEST_DOWNLOADED_FILENAME))
    parser.parse()
    parser.dump("parsed.json")

    with open(os.path.join(test_dir, TEST_PARSED_FILENAME), "r") as test_parsed_fp:
        test_parsed = test_parsed_fp.read()
    with open("parsed.json", "r") as parsed_fp:
        parsed = parsed_fp.read()

    assert test_parsed == parsed


@pytest.mark.parametrize("test_dir", test_dirs)
def test_parser_conversion_to_old_format(test_dir):
    with open(os.path.join(test_dir, TEST_PARSED_FILENAME), "r") as new_format_fp:
        parsed_new_format = json.load(new_format_fp)

    parsed_old_format = Parser.translate_parsed_results_to_mathematica(parsed_new_format)
    with open(r"parsed_old_format.txt", "w") as parsed_old_format_fp:
        parsed_old_format_fp.write(parsed_old_format)

    with open(os.path.join(test_dir, TEST_PARSED_OLD_FORMAT_FILENAME), "r") as test_parsed_old_format_fp:
        test_parsed_old_format = test_parsed_old_format_fp.read()

    assert test_parsed_old_format == parsed_old_format


@pytest.mark.parametrize("test_dir", test_dirs)
def test_parser_conversion_to_new_format(test_dir):
    mathematica_path = os.path.join(test_dir, TEST_PARSED_OLD_FORMAT_FILENAME)
    converted_to_new_format = Parser.translate_mathematica_to_python(mathematica_path)
    with open(r"parsed_converted_to_new_format.json", "w") as converted_to_new_format_fp:
        json.dump(converted_to_new_format, converted_to_new_format_fp)

    with open(os.path.join(test_dir, TEST_PARSED_FILENAME), "r") as test_parsed_fp:
        test_parsed = json.load(test_parsed_fp)

    # convert to df and compare by key, for proper float comparison in the start time column:
    test_df = pd.DataFrame(test_parsed)
    converted_df = pd.DataFrame(converted_to_new_format)
    from CFGpy.behavioral._consts import PARSED_PLAYER_ID_KEY, PARSED_TIME_KEY, PARSED_ALL_SHAPES_KEY
    assert test_df[PARSED_PLAYER_ID_KEY].equals(converted_df[PARSED_PLAYER_ID_KEY])
    assert np.allclose(test_df[PARSED_TIME_KEY], converted_df[PARSED_TIME_KEY])
    assert test_df[PARSED_ALL_SHAPES_KEY].equals(converted_df[PARSED_ALL_SHAPES_KEY])
    # TODO: after parser handles chosen shapes, compare those too


@pytest.mark.parametrize("test_dir", test_dirs)
def test_postparser(test_dir):
    postparser = PostParser.from_json(os.path.join(test_dir, TEST_PARSED_FILENAME))
    postparser.postparse()
    postparser.dump("postparsed.json")

    with open(os.path.join(test_dir, TEST_POSTPARSED_FILENAME), "r") as test_postparsed_fp:
        test_postparsed = test_postparsed_fp.read()
    with open("postparsed.json", "r") as postparsed_fp:
        postparsed = postparsed_fp.read()

    assert test_postparsed == postparsed


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
    features_filename = f"features.csv"

    with open(os.path.join(test_dir, TEST_POSTPARSED_FILENAME), "r") as test_postparsed_fp:
        postparsed_data = json.load(test_postparsed_fp)
    feature_extractor = FeatureExtractor(postparsed_data)
    feature_extractor.extract(verbose=True)
    feature_extractor.dump(features_filename)

    _compare_features(test_dir, features_filename)


@pytest.mark.parametrize("test_dir", test_dirs)
def test_full_pipeline(test_dir):
    features_filename = "features.csv"

    with open(os.path.join(test_dir, RED_METRICS_URL_FILENAME)) as url_fp:
        url = url_fp.read()

    pipeline = Pipeline(url, features_filename)
    pipeline.run_pipeline(verbose=True)
    _compare_features(test_dir, features_filename)

    config = Configuration.from_yaml(f"{features_filename}_config.yml")
    print(config.RED_METRICS_CSV_URL)
    assert "&before=" in config.RED_METRICS_CSV_URL
