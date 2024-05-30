import pytest
from CFGpy.behavioral import Downloader, Parser, Preprocessor, MeasureCalculator
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
TEST_PREPROCESSED_FILENAME = "test_preprocessed.json"
TEST_MEASURES_FILENAME = "test_measures.csv"

test_dirs = [os.path.join(TEST_FILES_DIR, filename)
             for filename in os.listdir(TEST_FILES_DIR)
             if os.path.isdir(os.path.join(TEST_FILES_DIR, filename))]


@pytest.mark.parametrize("test_dir", test_dirs)
def test_downloader(test_dir):
    # Note: Downloader tests currently fail. We don't know why, as this module was not touched since its output was
    # saved for tests, and it was not developed in the lab. As far as we can tell, whatever it downloads is ground
    # truth. For better tests, maybe we should use a Red Metrics URL that restricts time (both before and after). But
    # changing the Downloader ground truth would require changing all other tests, se we're not touching it for now.
    with open(os.path.join(test_dir, RED_METRICS_URL_FILENAME)) as url_fp:
        url = url_fp.read()
    Downloader(url, "event.csv").download()

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
def test_preprocessor(test_dir):
    preprocessor = Preprocessor.from_json(os.path.join(test_dir, TEST_PARSED_FILENAME))
    preprocessor.preprocess()
    preprocessor.dump("preprocessed.json")

    with open(os.path.join(test_dir, TEST_PREPROCESSED_FILENAME), "r") as test_preprocessed_fp:
        test_preprocessed = test_preprocessed_fp.read()
    with open("preprocessed.json", "r") as preprocessed_fp:
        preprocessed = preprocessed_fp.read()

    assert test_preprocessed == preprocessed


@pytest.mark.parametrize("test_dir", test_dirs)
def test_measures_calculator(test_dir):
    with open(os.path.join(test_dir, TEST_PREPROCESSED_FILENAME), "r") as test_preprocessed_fp:
        preprocessed_data = json.load(test_preprocessed_fp)
    measures_calculator = MeasureCalculator(preprocessed_data)
    measures_calculator.calc()
    measures_calculator.dump("measures.csv")

    test_measures = pd.read_csv(os.path.join(test_dir, TEST_MEASURES_FILENAME)).sort_values("ID").reset_index(drop=True)
    measures = pd.read_csv("measures.csv").sort_values("ID").reset_index(drop=True)
    assert len(test_measures) == len(measures)
    for col in test_measures:
        assert col in measures
        if test_measures[col].dtype == "float64":
            assert np.allclose(test_measures[col], measures[col], equal_nan=True)
        elif col != "Date/Time":  # date/time causes problems with subjects that played during daylight saving
            assert test_measures[col].equals(measures[col])
