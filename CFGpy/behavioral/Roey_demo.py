import pandas as pd
from CFGpy.behavioral import Downloader, Parser, Preprocessor
from CFGpy.behavioral.MRIParser import MRIParser
from CFGpy.behavioral.data_structs import PreprocessedPlayerData
from CFGpy.behavioral._consts import *


def __from_raw_data(raw_data):
    parser = MRIParser(raw_data)
    print("Parsing...")
    parsed_data = parser.parse()

    preprocessor = Preprocessor(parsed_data)
    print("Segmenting...")
    return preprocessor.preprocess()


def from_url(red_metrics_csv_url):
    downloader = Downloader(red_metrics_csv_url)
    print("Downloading raw data...")
    raw_data = downloader.download()
    return __from_raw_data(raw_data)


def from_file(red_metrics_csv_path):
    raw_data = pd.read_csv(red_metrics_csv_path)
    return __from_raw_data(raw_data)


if __name__ == '__main__':
    # link Roey sent
    csv_url = "https://api.creativeforagingtask.com/v1/event.csv?game=4cb46367-7555-\
    42cb-8915-152c3f3efdfb&entityType=event&after=2021-05-23T10:51:00.\
    000Z"

    # Load data from csv
    csv_file = "/Users/avivgreenburg/Library/CloudStorage/GoogleDrive-aviv.greenburg@mail.huji.ac.il/My " \
            "Drive/שלי/לימודים/Uni_2020-2024/forth_year/lab/event.csv"
    preprocessed_data = from_file(csv_file)

    # Load data from 'test_file1'
    # JASON = "/Users/avivgreenburg/Library/CloudStorage/GoogleDrive-aviv.greenburg@mail.huji.ac.il/My Drive/שלי/לימודים/Uni_2020-2024/forth_year/lab/CFGpy/CFGpy/behavioral/test_file1.json"
    # preprocessor = Preprocessor.from_json(JASON)
    # preprocessed_data = preprocessor.preprocess()

    # Plot each subject's game
    for i in range(len(preprocessed_data)):
        player_data = preprocessed_data[i]  # choose player index
        data = PreprocessedPlayerData(player_data)
        data.plot_gallery_dt()
        # data.plot_shapes() # todo- fix matplotlib AttributeError

    # exploit_mask = data.get_exploit_mask()
    # exploit_creation_times = data.shapes_df[exploit_mask].iloc[SHAPE_MOVE_TIME_IDX]


