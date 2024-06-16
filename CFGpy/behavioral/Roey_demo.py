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

PPT_OUTPUT_PATH = 'output/'

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

def add_both_plots_to_ppt(prs, shapesImagePath, plotGalleryImagePath, title):
    slide_layout = prs.slide_layouts[5]  # Choosing a blank slide layout
    slide = prs.slides.add_slide(slide_layout)
    title_placeholder = slide.shapes.title
    title_placeholder.text = title

    # Add gallery plot image
    left = Inches(0.1)
    top = Inches(2)
    height = Inches(3)
    slide.shapes.add_picture(plotGalleryImagePath, left, top, height=height)

    # Add shapes image
    left = Inches(6)
    slide.shapes.add_picture(shapesImagePath, left, top, height=height)


if __name__ == '__main__':

    # Load data from url & save new JSON
    # preprocessed_data = from_url(CSV_URL)

    # Load data from csv
    # preprocessed_data = from_file(CSV_FILE_PATH)

    # Load processed data from json
    preprocessed_data = from_json(DEFAULT_OUTPUT_FILENAME)

    # Sort by ID
    preprocessed_data = sorted(preprocessed_data, key=lambda x: x["id"])  # sort by subjects' ID

    # Create a presentation object
    prs = Presentation()

    # Plot each subject's game OR add to presentation
    for player_data in preprocessed_data:
        if not player_data[PARSED_PLAYER_ID_KEY].startswith("9999"):  # skip the non-player data
            data = PreprocessedPlayerData(player_data)

            shapesImagePath = data.plot_clusters()
            plotGalleryImagePath = data.plot_gallery_dt()

            if shapesImagePath != -1 and plotGalleryImagePath != -1:
                add_both_plots_to_ppt(prs, shapesImagePath, plotGalleryImagePath,
                                f"Player {player_data[PARSED_PLAYER_ID_KEY]}")
                os.remove(shapesImagePath)  # Clean up the image file
                os.remove(plotGalleryImagePath)  # Clean up the image file
            else:
                print(f"{player_data[PARSED_PLAYER_ID_KEY]}, missing necessary phases --> can't create plot")

    # Save the presentation
    prs_name = f"{PPT_OUTPUT_PATH}player_presentation.pptx"
    prs.save(prs_name)
    print(f"PowerPoint presentation saved as '{prs_name}'")


