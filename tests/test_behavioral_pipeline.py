from CFGpy.behavioral import Downloader, Parser, Preprocessor, MeasureCalculator
from CFGpy import NAS_PATH
import os

TEST_FILES_DIR = os.path.join(NAS_PATH, "Projects", "CFG", "CFGpy test files")
TEST_DOWNLOADED_PATH = os.path.join(TEST_FILES_DIR, "test_event.csv")
TEST_PARSED_PATH = os.path.join(TEST_FILES_DIR, "test_parsed.json")
TEST_PARSED_OLD_FORMAT_PATH = os.path.join(TEST_FILES_DIR, "test_parsed_old_format.txt")
TEST_PREPROCESSED_PATH = os.path.join(TEST_FILES_DIR, "test_preprocessed.json")
TEST_MEASURES_PATH = os.path.join(TEST_FILES_DIR, "test_measures.csv")


def test_downloader():
    url = "https://api.creativeforagingtask.com/v1/event.csv?game=41e44d77-e341-4be0-95a8-7403f6c74647&entityType=event"
    Downloader(url, "event.csv").download()

    with open(TEST_DOWNLOADED_PATH, "r") as test_event_fp:
        test_event = test_event_fp.read()
    with open("event.csv", "r") as event_fp:
        event = event_fp.read()

    assert test_event == event


def test_parser():
    parser = Parser.from_file(TEST_DOWNLOADED_PATH)
    parser.parse()
    parser.dump("parsed.json")

    with open(TEST_PARSED_PATH, "r") as test_parsed_fp:
        test_parsed = test_parsed_fp.read()
    with open("parsed.json", "r") as parsed_fp:
        parsed = parsed_fp.read()

    assert test_parsed == parsed


def test_parser_conversion_to_old_format():
    parser = Parser.from_file(TEST_DOWNLOADED_PATH)
    parsed_new_format = parser.parse()
    parsed_old_format = Parser.translate_parsed_results_to_mathematica(parsed_new_format)

    with open(TEST_PARSED_OLD_FORMAT_PATH, "r") as test_parsed_old_format_fp:
        test_parsed_old_format = test_parsed_old_format_fp.read()

    assert test_parsed_old_format == parsed_old_format


def test_processor():
    preprocessor = Preprocessor.from_json(TEST_PARSED_PATH)
    preprocessor.preprocess()
    preprocessor.dump("preprocessed.json")

    with open(TEST_PREPROCESSED_PATH, "r") as test_preprocessed_fp:
        test_preprocessed = test_preprocessed_fp.read()
    with open("preprocessed.json", "r") as preprocessed_fp:
        preprocessed = preprocessed_fp.read()

    assert test_preprocessed == preprocessed


def test_measures_calculator():
    measures_calculator = MeasureCalculator.from_json(TEST_PARSED_PATH)
    measures_calculator.calc()
    measures_calculator.dump("measures.csv")

    with open(TEST_MEASURES_PATH, "r") as test_measures_fp:
        test_measures = test_measures_fp.read()
    with open("measures.csv", "r") as measures_fp:
        measures = measures_fp.read()

    assert test_measures == measures


if __name__ == '__main__':
    test_downloader()
    test_parser()
    test_parser_conversion_to_old_format()
    test_processor()
    test_measures_calculator()
