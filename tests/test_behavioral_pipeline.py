from CFGpy.behavioral import Downloader, Parser, Preprocessor, MeasureCalculator
from CFGpy import NAS_PATH
import os

TEST_FILES_DIR = os.path.join(NAS_PATH, "Projects", "CFG", "CFGpy test files")
RED_METRICS_URL_FILENAME = "red_metrics_url.txt"
TEST_DOWNLOADED_FILENAME = "test_event.csv"
TEST_PARSED_FILENAME = "test_parsed.json"
TEST_PARSED_OLD_FORMAT_FILENAME = "test_parsed_old_format.txt"
TEST_PREPROCESSED_FILENAME = "test_preprocessed.json"
TEST_MEASURES_FILENAME = "test_measures.csv"


def test_downloader():
    for filename in os.listdir(TEST_FILES_DIR):
        absolute_path = os.path.join(TEST_FILES_DIR, filename)
        if os.path.isdir(absolute_path):

            with open(os.path.join(absolute_path, RED_METRICS_URL_FILENAME)) as url_fp:
                url = url_fp.read()
            Downloader(url, "event.csv").download()

            with open(os.path.join(absolute_path, TEST_DOWNLOADED_FILENAME), "r") as test_event_fp:
                test_event = test_event_fp.read()
            with open("event.csv", "r") as event_fp:
                event = event_fp.read()

            assert test_event == event


def test_parser():
    for filename in os.listdir(TEST_FILES_DIR):
        absolute_path = os.path.join(TEST_FILES_DIR, filename)
        if os.path.isdir(absolute_path):
            parser = Parser.from_file(os.path.join(absolute_path, TEST_DOWNLOADED_FILENAME))
            parser.parse()
            parser.dump("parsed.json")

            with open(os.path.join(absolute_path, TEST_PARSED_FILENAME), "r") as test_parsed_fp:
                test_parsed = test_parsed_fp.read()
            with open("parsed.json", "r") as parsed_fp:
                parsed = parsed_fp.read()

            assert test_parsed == parsed


def test_parser_conversion_to_old_format():
    for filename in os.listdir(TEST_FILES_DIR):
        absolute_path = os.path.join(TEST_FILES_DIR, filename)
        if filename == "set2" and os.path.isdir(absolute_path):
            parser = Parser.from_file(os.path.join(absolute_path, TEST_DOWNLOADED_FILENAME))
            parsed_new_format = parser.parse()
            parsed_old_format = Parser.translate_parsed_results_to_mathematica(parsed_new_format)

            with open(r"parsed_old_fmt.txt", "w") as parsed_old_format_fp:
                parsed_old_format_fp.write(parsed_old_format)

            with open(os.path.join(absolute_path, TEST_PARSED_OLD_FORMAT_FILENAME), "r") as test_parsed_old_format_fp:
                test_parsed_old_format = test_parsed_old_format_fp.read()

            assert test_parsed_old_format == parsed_old_format


def test_processor():
    for filename in os.listdir(TEST_FILES_DIR):
        absolute_path = os.path.join(TEST_FILES_DIR, filename)
        if os.path.isdir(absolute_path):
            preprocessor = Preprocessor.from_json(os.path.join(absolute_path, TEST_PARSED_FILENAME))
            preprocessor.preprocess()
            preprocessor.dump("preprocessed.json")

            with open(os.path.join(absolute_path, TEST_PREPROCESSED_FILENAME), "r") as test_preprocessed_fp:
                test_preprocessed = test_preprocessed_fp.read()
            with open("preprocessed.json", "r") as preprocessed_fp:
                preprocessed = preprocessed_fp.read()

            assert test_preprocessed == preprocessed


def test_measures_calculator():
    for filename in os.listdir(TEST_FILES_DIR):
        absolute_path = os.path.join(TEST_FILES_DIR, filename)
        if os.path.isdir(absolute_path):
            measures_calculator = MeasureCalculator.from_json(os.path.join(absolute_path, TEST_PARSED_FILENAME))
            measures_calculator.calc()
            measures_calculator.dump("measures.csv")

            with open(os.path.join(absolute_path, TEST_MEASURES_FILENAME), "r") as test_measures_fp:
                test_measures = test_measures_fp.read()
            with open("measures.csv", "r") as measures_fp:
                measures = measures_fp.read()

            assert test_measures == measures


if __name__ == '__main__':
    # test_downloader()
    # test_parser()
    test_parser_conversion_to_old_format()
    # test_processor()
    # test_measures_calculator()
