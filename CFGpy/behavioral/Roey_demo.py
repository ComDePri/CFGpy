import pandas as pd
from CFGpy.behavioral import Downloader, Parser, Preprocessor
from CFGpy.behavioral.data_structs import PreprocessedPlayerData


def __from_raw_data(raw_data):
    parser = Parser(raw_data)
    print("Parsing...")
    parsed_data = parser.parse()

    preprocessor = Preprocessor(parsed_data)
    print("Segmenting...")
    return preprocessor.preprocess()


def from_url(red_metrics_csv_url):
    downloader = Downloader(red_metrics_csv_url)
    print("Downloading raw data...")
    raw_data = downloader.download()
    __from_raw_data(raw_data)


def from_file(red_metrics_csv_path):
    raw_data = pd.read_csv(red_metrics_csv_path)
    __from_raw_data(raw_data)


if __name__ == '__main__':
    preprocessor = Preprocessor.from_json(r"C:\Users\roygutg\Documents\GitRepos\CFGpy\CFGpy\behavioral\test_file1.json")
    preprocessed_data = preprocessor.preprocess()

    player_data = preprocessed_data[0]  # choose player index
    data = PreprocessedPlayerData(player_data)
    data.plot_gallery_dt()
    data.plot_shapes()
