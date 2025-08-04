import os

import pandas as pd
from CFGpy.behavioral import Downloader, Parser, Preprocessor
from CFGpy.behavioral.MRIParser import MRIParser
from CFGpy.behavioral.Preprocessor import DEFAULT_OUTPUT_FILENAME
from CFGpy.behavioral.data_structs import PreprocessedPlayerData
from CFGpy.behavioral._consts import *
from pptx import Presentation
from pptx.util import Inches
import json # Roey added for saving the vanilla after more preprocessing
import shutil

# *** NOTE: DO NOT FORGET TO SET THE SEGMENTATION ALGORITHM IN CONSTANTS.PY ***
# (1) CSV URL for MRI games INSIDE the scanner (use MRI segmentation algorithm in constants.py)
#CSV_URL = "https://api.creativeforagingtask.com/v1/event.csv?game=4cb46367-7555-42cb-8915-152c3f3efdfb&entityType=event&after=2021-05-23T10:51:00.000Z"
CSV_URL = "https://api.creativeforagingtask.com/v1/event.csv?game=4cb46367-7555-42cb-8915-152c3f3efdfb&entityType=event&after=2025-04-28T10:51:00.000Z"
CSV_FILE_PATH = "/home/roey/Documents/CFG_data/event.csv"
# (2) CSV URL for MRI games OUTSIDE the scanner (use non-MRI segmentation algorithm in constants.py) (go to https://creativeforagingtask.com/v1/search and search for the game version "FCDBrainHebrewOnlyScan")
#CSV_URL = 'https://api.creativeforagingtask.com/v1/event.csv?game=01b164da-9cef-4dbf-aefb-17627442abe7&gameVersion=4f865647-2064-4af4-a18f-153ec24d6a5f&entityType=event'
#CSV_FILE_PATH = "/home/roey/Documents/CFG_data/event_play_outside_scanner.csv"
#ROY_TEST_JASON = "/home/roey/PycharmProjects/CFGpy/CFGpy/behavioral/test_file1.json"

PPT_OUTPUT_PATH = PATH_FROM_REP_ROOT # PATH_FROM_REP_ROOT loaded from consts above. Usually simply 'output/'
# If PPT_OUTPUT_PATH doesn't exist, create it
if not os.path.exists(PPT_OUTPUT_PATH):
    os.makedirs(PPT_OUTPUT_PATH)

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


def add_one_plot_to_ppt(prs, image_path, title=None):
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


def add_error_slide_to_ppt(prs, errors):
    """
    Adds a slide with a list of error messages to the presentation.
    :param prs: Presentation object.
    :param errors: List of error messages.
    """
    slide_layout = prs.slide_layouts[1]  # Choosing a title and content slide layout
    slide = prs.slides.add_slide(slide_layout)

    # Add title
    title_box = slide.shapes.title
    title_box.text = "Errors"

    # Add error messages
    content_box = slide.placeholders[1]
    content_box.text = "\n".join(errors)


def create_players_cluster_times_csv(preprocessed_data, output_folder=PATH_FROM_REP_ROOT+"clusters_times_csvs/"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for player_data in preprocessed_data:
        if not player_data[PARSED_PLAYER_ID_KEY].startswith("9999"):  # skip the non-player data
            player_id = player_data[PARSED_PLAYER_ID_KEY]
            #if not player_id in ['089', '096']:  # TODO: &&& REMOVE
            #    continue

            data = PreprocessedPlayerData(player_data)

            # explore_times, exploit_times = data.get_cluster_times()
            #
            csv_file_path = os.path.join(output_folder, f"Player_{player_id}_cluster_times.csv")

            # Save DataFrame to CSV
            data.clusters_times_to_csv(csv_file_path)
            #
            # def create_data_frame(phase, cluster_times, columns=("start", "end")):
            #     start, end = columns[0], columns[1]
            #     df = pd.DataFrame(cluster_times, columns=[start,end])
            #     df["phase"] = phase
            #     return df
            #
            # explore_df = create_data_frame(EXPLORE_KEY, explore_times)
            # exploit_df = create_data_frame(EXPLOIT_KEY, exploit_times)
            #
            # # # Create DataFrame for explore times
            # # explore_df = pd.DataFrame(explore_times, columns=["gallery_out_time", "gallery_in_time"])
            # # explore_df["phase"] = EXPLORE_KEY
            # #
            # # # Create DataFrame for exploit times
            # # exploit_df = pd.DataFrame(exploit_times, columns=["gallery_out_time", "gallery_in_time"])
            # # exploit_df["phase"] = EXPLOIT_KEY
            #
            # # Combine both DataFrames
            # cluster_times_df = pd.concat([explore_df, exploit_df], ignore_index=True)
            #
            # # Save DataFrame to CSV
            # cluster_times_df.to_csv(csv_file_path, index=False)

def create_empty_moves_csv(preprocessed_data, output_folder=PATH_FROM_REP_ROOT+"empty_moves_csvs/"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for player_data in preprocessed_data:
        if not player_data[PARSED_PLAYER_ID_KEY].startswith("9999"):  # skip the non-player data
            data = PreprocessedPlayerData(player_data)

            player_id = player_data[PARSED_PLAYER_ID_KEY]
            csv_file_path = os.path.join(output_folder, f"Player_{player_id}_cluster_times.csv")

            # Save DataFrame to CSV
            data.get_empty_moves_times(csv_file_path)

def create_players_game_presentation(preprocessed_data, delta_t=True):
    # Create a presentation object
    prs = Presentation()
    # Collect error messages
    error_messages = []
    # Plot each subject's game OR add to presentation
    for player_data in preprocessed_data:

        if not player_data[PARSED_PLAYER_ID_KEY].startswith("9999"):  # skip the non-player data
            data = PreprocessedPlayerData(player_data)
            shapesImagePath = data.plot_clusters()

            if delta_t:
                plotGalleryImagePath = data.plot_gallery_dt()
            else:
                plotGalleryImagePath = data.plot_gallery_steps_over_dt() #TODO: Consider adding a flag for plot_gallery_steps

            if shapesImagePath != -1 and plotGalleryImagePath != -1:
                add_both_plots_to_ppt(prs, shapesImagePath, plotGalleryImagePath,
                                      f"Player {player_data[PARSED_PLAYER_ID_KEY]}")
                os.remove(shapesImagePath)  # Clean up the image file
                os.remove(plotGalleryImagePath)  # Clean up the image file
            else:
                error_messages.append(
                    f"{player_data[PARSED_PLAYER_ID_KEY]}, missing necessary phases --> can't create plot")

    # Add a single error slide if there are any errors
    if error_messages:
        add_error_slide_to_ppt(prs, error_messages)

    return prs

def create_all_shapes_presentation(preprocessed_data):
    # Create a presentation object
    prs = Presentation()
    # Collect error messages
    error_messages = []
    # Plot each subject's game OR add to presentation
    for player_data in preprocessed_data:

        if not player_data[PARSED_PLAYER_ID_KEY].startswith("999999"):  # skip the non-player data. TODO: Roey: Make 9999 again
            data = PreprocessedPlayerData(player_data)

            shapesImagePath = data.plot_shapes()

            if shapesImagePath != -1:
                add_one_plot_to_ppt(prs, shapesImagePath, f"Player {player_data[PARSED_PLAYER_ID_KEY]}")
                os.remove(shapesImagePath)  # Clean up the image file
            else:
                error_messages.append(
                    f"{player_data[PARSED_PLAYER_ID_KEY]}, missing necessary phases --> can't create plot")

    # Add a single error slide if there are any errors
    if error_messages:
        add_error_slide_to_ppt(prs, error_messages)

    return prs

if __name__ == '__main__':
    # NOTE: Change here to one of 3 options for loading the data

    # A) Load vanilla to preprocess it with the MRI algorithm and dump it into a new json file
    #preprocessed_data = from_json("/Volumes/HartLabNAS/Projects/CFG/vanilla_data/vanilla.json")
    #path = "/Volumes/HartLabNAS/Projects/CFG/vanilla_data/vanilla_shuffled_noMRIFix.json"
    #with open(path, "w") as out_file:
    #    json.dump(preprocessed_data, out_file)

    # B) [[Default option, but see C]] Load data from url & save new JSON
    preprocessed_data = from_url(CSV_URL)

    # Dump as json file
    path = "/home/roey/PycharmProjects/CFGpy/CFGpy/behavioral/data_from_RM1.json"
    with open(path, "w") as out_file:
        json.dump(preprocessed_data, out_file)

    # Print the unique id's in preprocessed_data
    #print("Unique IDs in preprocessed_data:")

    # C) Load data from csv
    # Add a missing "startsearch" line for subject 104 (due to server communication error)
    # NOTE: This is not critical, since this is a game we exclude anyway. They saved at almost every step.
    # Define the source and destination file paths
    ###CSV_FILE_PATH_SAVED_FROM_URL = '/home/roey/PycharmProjects/CFGpy/CFGpy/behavioral/event.csv'
    ###CSV_FILE_PATH = '/home/roey/PycharmProjects/CFGpy/CFGpy/behavioral/event_manual_startsearch_added_player_104.csv'
    # Copy the contents of the original file to the new file
    ###shutil.copy(CSV_FILE_PATH_SAVED_FROM_URL, CSV_FILE_PATH)
    # Add the new line to the copied file
    ###line_to_add = 'f0eb2088-8f0eb2088-8a06-4482-a86f-7ce841f059f6,2025-03-11T10:29:26.043Z,2025-02-04T11:21:17.785Z,b241162a-93db-4213-9741-c9a825521506,db71df4c-6030-4602-be46-24ad4237f734,,,,,104,"{""expId"":""LeapsFCDBrain"",""userId"":""104"",""userAgent"":""Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"",""userProvidedId"":""104""}",startsearch,,,,,,,,,,,,'
    ###with open(CSV_FILE_PATH, 'a') as f:
    ###    f.write(line_to_add)


    # NOTE: Or use the CSV with the added startsearch line for subject 104 who is missing it on the server, probably due to network issues:
    ###CSV_FILE_PATH = '/home/roey/PycharmProjects/CFGpy/CFGpy/behavioral/event_manual_startsearch_added_player_104.csv'
    ###preprocessed_data = from_file(CSV_FILE_PATH)
    ###exit()

    # Load processed data from json
    # A - this worked
    ### preprocessed_data = from_json(DEFAULT_OUTPUT_FILENAME)
    # B - checking this option
    ###pp = Preprocessor.from_json(DEFAULT_OUTPUT_FILENAME)
    ###preprocessed_data = pp.preprocess()
    ###pp.remove_bad_games()
    # END OF B


    ## ROY G. MODIFIED CODE

#####    # For plotting specific Vanilla games (this is a little strange, because here we load a "postparsed" dataset, which means it was analyzed with a more recent version of the CFG pipeline, where we have Parser and PostParser classes, which are not present in the dev branch), and not Preprocessor and MeasurerCalculator.
    #####    import json
    #####    json_file = '/Volumes/HartLabNAS/Projects/CFG/vanilla_data/vanilla.json'
    #####    with open(json_file, 'r') as file:
    #####        preprocessed_data = json.load(file)
    #####        print('Done.')
    #####    # Just for testing, extract two games (the code will fail if it's only a single game)
    #####    preprocessed_data = preprocessed_data[0:2]
    # Sort by ID
    preprocessed_data = sorted(preprocessed_data, key=lambda x: x["id"])  # sort by subjects' ID


    def create_presentation_with_all_shapes_plot():
        prs_all_shapes =create_all_shapes_presentation(preprocessed_data)
        prs_all_shapes_name = f"{PPT_OUTPUT_PATH}all_shapes_presentation.pptx"
        prs_all_shapes.save(prs_all_shapes_name)
        print(f"PowerPoint presentation saved as '{prs_all_shapes_name}'")

    def create_presentation_with_both_plots():
        plot_by_delta_t = True # change this from False --> True: for choosing weather it'll create presentation by delta_t or steps
        prs_game = create_players_game_presentation(preprocessed_data, delta_t=plot_by_delta_t)
        if not plot_by_delta_t:
            #prs_game_name = f"{PPT_OUTPUT_PATH}games_presentation_efficiency{MIN_EFFICIENCY_FOR_EXPLOIT}_paceSpecific_5mad_rmvEmptyTimeAllSteps_efficiencyLessThan1DELETE.pptx"
            prs_game_name = f"{PPT_OUTPUT_PATH}games_presentation.pptx"
        else:
            #prs_game_name = f"{PPT_OUTPUT_PATH}games_presentation_efficiency{MIN_EFFICIENCY_FOR_EXPLOIT}_paceSpecific_5mad_rmvEmptyTimeAllSteps_efficiencyLessThan1_delta_tDELETE.pptx"
            prs_game_name = f"{PPT_OUTPUT_PATH}games_presentation.pptx"

        prs_game.save(prs_game_name)
        print(f"PowerPoint presentation saved as '{prs_game_name}'")

    def create_players_cluster_times():
        output_folder = os.path.join(PATH_FROM_REP_ROOT,"clusters_times_csvs/")
        #output_folder = PATH_FROM_REP_ROOT + "clusters_times_csvs/"

        create_players_cluster_times_csv(preprocessed_data, output_folder)
        print(f"csv's with players cluster saved in folder: '{output_folder}'")

    def create_players_empty_moves_times():
        output_folder = PATH_FROM_REP_ROOT+"empty_moves_csvs/"
        create_empty_moves_csv(preprocessed_data, output_folder)
        print(f"csv's with players cluster saved in folder: '{output_folder}'")

    # Change here to determine what the script will do
    create_presentation_with_both_plots()
    create_presentation_with_all_shapes_plot()
    cluster_times_dir = os.path.join(PPT_OUTPUT_PATH,'clusters_times_csvs/')
    if not os.path.exists(cluster_times_dir):
        os.makedirs(cluster_times_dir)
    create_players_cluster_times()
    #create_players_empty_moves_times()