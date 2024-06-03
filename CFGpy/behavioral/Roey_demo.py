import os

import pandas as pd
from CFGpy.behavioral import Downloader, Parser, Preprocessor
from CFGpy.behavioral.MRIParser import MRIParser
from CFGpy.behavioral.Preprocessor import DEFAULT_OUTPUT_FILENAME
from CFGpy.behavioral.data_structs import PreprocessedPlayerData
from CFGpy.behavioral._consts import *
from pptx import Presentation
from pptx.util import Inches

CSV_URL = "https://api.creativeforagingtask.com/v1/event.csv?game=4cb46367-7555-\
42cb-8915-152c3f3efdfb&entityType=event&after=2021-05-23T10:51:00.\
000Z"  # link Roey sent
CSV_FILE_PATH = "/Users/avivgreenburg/Library/CloudStorage/GoogleDrive-aviv.greenburg@mail.huji.ac.il/My " \
                "Drive/שלי/לימודים/Uni_2020-2024/forth_year/lab/event.csv"
ROY_TEST_JASON = "/Users/avivgreenburg/Library/CloudStorage/GoogleDrive-aviv.greenburg@mail.huji.ac.il/\
My Drive\/שלי/לימודים/Uni_2020-2024/forth_year/lab/CFGpy/CFGpy/behavioral/test_file1.json"

PPT_OUTPUT_PATH = 'output/presentation.pptx'

def __from_raw_data(raw_data):
    parser = MRIParser(raw_data)
    print("Parsing...")
    parsed_data = parser.parse()

    preprocessor = Preprocessor(parsed_data)
    print("Segmenting...")

    preprocessor.dump()  # save JSON

    return preprocessor.preprocess()


def from_url(red_metrics_csv_url):
    downloader = Downloader(red_metrics_csv_url)
    print("Downloading raw data...")
    raw_data = downloader.download()
    return __from_raw_data(raw_data)


def from_file(red_metrics_csv_path):
    raw_data = pd.read_csv(red_metrics_csv_path)
    return __from_raw_data(raw_data)


def from_json(jason_path):
    preprocessor = Preprocessor.from_json(jason_path)
    return preprocessor.preprocess()


def add_plot_to_ppt(prs, image_path, title=None):
    # Add a slide with a title and content layout
    slide_layout = prs.slide_layouts[5]  # Choosing a blank slide layout
    slide = prs.slides.add_slide(slide_layout)

    # Define image placement on slide
    left = Inches(1)
    top = Inches(1)
    height = Inches(4.5)

    # Add image to slide
    slide.shapes.add_picture(image_path, left, top, height=height)

    # Add a title to the slide
    if title is not None:
        title_box = slide.shapes.title
        title_box.text = title


if __name__ == '__main__':

    # Load data from csv
    # preprocessed_data = from_file(CSV_FILE_PATH)

    # Load processed data from json
    preprocessed_data = from_json(DEFAULT_OUTPUT_FILENAME)

    # Sort by ID
    preprocessed_data = sorted(preprocessed_data, key=lambda x: x["id"])  # sort by subjects' ID

    # Create a presentation object
    prs = Presentation()

    # Plot each subject's game OR add to presentation
    for i in range(len(preprocessed_data)):
        player_data = preprocessed_data[i]  # choose player index
        data = PreprocessedPlayerData(player_data)
        imagePath = data.plot_gallery_dt()
        data.plot_shapes()

        if imagePath == -1:
            print(player_data[PARSED_PLAYER_ID_KEY] + ", missing explore or exploit phase --> can't create plot")

        else:
            # todo - fix data
            print(player_data[PARSED_PLAYER_ID_KEY] + ", plot created successfully")

            add_plot_to_ppt(prs, imagePath)
            os.remove(imagePath)  # Clean up the image file

    prs.save(PPT_OUTPUT_PATH)
    print(f"PowerPoint presentation saved as '{PPT_OUTPUT_PATH}'")

    # exploit_mask = data.get_exploit_mask()
    # exploit_creation_times = data.shapes_df[exploit_mask].iloc[SHAPE_MOVE_TIME_IDX]
