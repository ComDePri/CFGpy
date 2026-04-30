from pathlib import Path
import pytest
import os
import pandas as pd
from CFGpy.behavioral.Configuration import Configuration
from CFGpy.behavioral.IOCANEDataRetriever import IOCANEDataRetriever
from CFGpy.behavioral.RedMetrics1Downloader import RedMetrics1Downloader
from CFGpy.behavioral.Parser import Parser
from tests.test_behavioral_pipeline import _compare_parsed, _compare_raws

MIGRATED_GAME_IDS_PATH = os.path.join(Path(__file__).parent, "migration_test_files/game_ids.csv")
# this is a file with list of game-IDS without column header
MIGRATED_GAME_IDS = pd.read_csv(MIGRATED_GAME_IDS_PATH, header=None)[0].tolist()

@pytest.mark.parametrize("game_id", MIGRATED_GAME_IDS)
def test_rm1_iocane_migration(game_id):
    """
    Test that the downloaded data from IOCANE matches the data downloaded from RM1 for the set of migrated game IDs
    """
    config = Configuration.default()
    config.GAME_ID = game_id
    # get data from IOCANE
    iocane_downloader = IOCANEDataRetriever(output_filename="raw_iocane", config=config)
    iocane_downloader.retrieve_data(verbose=True)
    iocane_downloader.dump()
    iocane_df = pd.read_csv(iocane_downloader.output_path)
    iocane_df = iocane_df.sort_values("id").reset_index(drop=True)
    # get data from RM1
    rm1_downloader = RedMetrics1Downloader(output_filename="raw_rm1", config=config)
    rm1_downloader.retrieve_data(verbose=True)
    rm1_downloader.dump()
    rm1_df = pd.read_csv(rm1_downloader.output_path)

    rm1_df = rm1_df.sort_values("id").reset_index(drop=True)


    _compare_raws(rm1_df, iocane_df, json_dict_cols=("playerCustomData",), shared_cols=True)
    parser_iocane = Parser(raw_data=iocane_df, config=config)
    parsed_iocane = parser_iocane.parse()
    parser_rm1 = Parser(raw_data=rm1_df, config=config)
    parsed_rm1 = parser_rm1.parse()
    # compare the parsed outputs by ID and start time (to avoid issues with order of actions or other minor differences in the raw data that don't affect the parsed output)
    _compare_parsed(parsed_rm1, parsed_iocane)