import pandas as pd
from CFGpy.behavioral import Downloader, Parser, PostParser
from CFGpy.behavioral.data_classes import PostparsedPlayerData


def __from_raw_data(raw_data):
    parser = Parser(raw_data)
    print("Parsing...")
    parsed_data = parser.parse()
    postparser = PostParser(parsed_data)
    print("Segmenting...")
    return postparser.postparse()


def from_url(red_metrics_csv_url):
    downloader = Downloader(red_metrics_csv_url)
    print("Downloading raw data...")
    raw_data = downloader.download()
    __from_raw_data(raw_data)


def from_file(red_metrics_csv_path):
    raw_data = pd.read_csv(red_metrics_csv_path)
    __from_raw_data(raw_data)


if __name__ == '__main__':
    postparser = PostParser.from_json(r"/home/royg/home/royg/CFGpy/CFGpy/behavioral/test_file1.json")
    preprocessed_data = postparser.postparse()

    player_data = preprocessed_data[0]  # choose player index
    data = PostparsedPlayerData(player_data)
    data.plot_gallery_dt()
    data.plot_shapes()
