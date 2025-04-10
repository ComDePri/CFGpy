import os

import pandas as pd
from CFGpy.behavioral import Downloader, Parser, Preprocessor
from CFGpy.behavioral.MRIParser import MRIParser
from CFGpy.behavioral.Preprocessor import DEFAULT_OUTPUT_FILENAME
from CFGpy.behavioral.data_structs import PreprocessedPlayerData, PATH_FROM_REP_ROOT
from CFGpy.behavioral._consts import *
from pptx import Presentation
from pptx.util import Inches
import json # Roey added for saving the vanilla after more preprocessing
from Roey_demo import  from_url, from_file, from_json

CSV_URL = "https://api.creativeforagingtask.com/v1/event.csv?game=4cb46367-7555-\
42cb-8915-152c3f3efdfb&entityType=event&after=2021-05-23T10:51:00.\
000Z"  # link Roey sent
CSV_FILE_PATH = "/home/roey/Documents/CFG_data/event.csv"
ROY_TEST_JASON = "/home/roey/PycharmProjects/CFGpy/CFGpy/behavioral/test_file1.json"

PPT_OUTPUT_PATH = PATH_FROM_REP_ROOT # Loaded from consts above, usually simply 'output/'



if __name__ == '__main__':
    # NOTE: Change here to one of 3 options for loading the data

    # Load data from url & save new JSON [Use this for the first time, or after more subjects were added. Otherwise, use the json file below
    #preprocessed_data = from_url(CSV_URL)

    # Load data from csv
    #preprocessed_data = from_file(CSV_FILE_PATH)

    # Load processed data from json
    #preprocessed_data = from_json(DEFAULT_OUTPUT_FILENAME)
    json_file = '/home/roey/PycharmProjects/CFGpy/CFGpy/behavioral/output/preprocessed_MRIalgo.json'
    preprocessed_data = from_json(json_file)

    # Sort by ID
    preprocessed_data = sorted(preprocessed_data, key=lambda x: x["id"])  # sort by subjects' ID

