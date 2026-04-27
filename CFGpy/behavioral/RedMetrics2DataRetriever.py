import getpass
import os
from typing import Optional
import requests
from tqdm import tqdm
import pandas as pd
from CFGpy.behavioral._consts import (DATA_RETRIEVER_OUTPUT_FILENAME, CONFIG_URL_MISMATCH_ERROR)
from CFGpy.behavioral import Configuration, DataRetriever
RM2_EVENT_ID_KEY = "id"
RM2_RAW_SERVER_TIME = "serverTimestamp"
RM2_RAW_USER_TIME = "userTimestamp"
RM2_EVENT_TYPE = "type"
RM2_RAW_COORDINATES = "coordinates"

RM2_DOWNLOADER_COMMON_FIELDS = (RM2_EVENT_ID_KEY, RM2_RAW_SERVER_TIME, RM2_RAW_USER_TIME, RM2_EVENT_TYPE, RM2_RAW_COORDINATES)

class RedMetrics2DataRetriever(DataRetriever):
    def __init__(self, *, game_id: str | None = None, output_filename: str = DATA_RETRIEVER_OUTPUT_FILENAME,
                 config: Configuration = None, logger = None) -> None:
        """
        Init a RedMetrics2DataRetriever object.
        :param game_id: The game name of the game whose data you want to retrieve from RedMetrics2.
        :param game_id: The game id of the game whose data you want to retrieve from RedMetrics2.
        :param output_filename: filename for output.
        :param config: a Configuration file.
        """
        super().__init__(game_id=game_id, output_filename=output_filename,
                         config=config if config is not None else Configuration.default(), logger=logger)
        self._validate_input(input=[game_id, self._config.GAME_ID])
        self._retrieved_events_json = []
        self._session = None
    
    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._init_session()
        return self._session

        
    def _retrieve_data(self, *, verbose: bool = False) -> pd.DataFrame:
        
        self._retrieved_events_json = self._download_data_from_rm2(verbose=verbose) 
        output_json = self._create_rm2_output(verbose=verbose) 
        self._retrieved_df = self._create_df(output_json=output_json)
        
        return self._retrieved_df
    
    def _init_session(self, verbose: Optional[bool] = False) -> None:
        
        self._session = requests.Session()
        rm2_email = os.getenv("RM2_EMAIL") or input("Please enter your RedMetrics2 email: ") 
        rm2_password = os.getenv("RM2_PASSWORD") or getpass.getpass(prompt="Enter your RedMetrics2 password: ")
        self._login_to_session(email=rm2_email, password=rm2_password, verbose=verbose)
        
        return None

    
    def _login_to_session(self, email: str, password: str, verbose: Optional[bool] = False) -> None:
        self.log_info("Logging into RedMetrics2...")

            
        login_url: str = "https://api.creativeforagingtask.com/v2/login"
        login_data: dict = {
            "email": email,
            "password": password,
        }
        
        response = self.session.post(login_url, data=login_data)

        if response.status_code == 200:
            self.log_info("Successfully logged into RedMetrics2.")
        else:
            self.log_error(f"Failed to log into RedMetrics2: {response.status_code} - {response.text}")

    def _download_data_from_rm2(self, verbose: Optional[bool] = False) -> dict:
        self.log_info("Downloading data from RedMetrics2...")

            
        download_url: str = f"https://api.creativeforagingtask.com/v2/game/{self._game_id}/data.json"
        response = self.session.get(url=download_url)

        if not response.status_code == 200:
            msg = f"Error: {response.status_code} - failed to download data for game: {self._game_id}."
            self.log_error(msg)
            raise(ValueError(msg))
        self.log_info("Data downloaded successfully from RedMetrics2.")

        return response.json()

    
    def _create_rm2_output(self, verbose: Optional[bool] = False) -> pd.DataFrame:
       
        sessions_iterator = self._retrieved_events_json.get("sessions", {})
        self.log_info("Handling events...")
        if verbose:
            sessions_iterator = tqdm(sessions_iterator, desc="sessions")
            
        output_json: list[dict] = []
        
        for session in sessions_iterator:
            playerCustomData: dict = session.get("customData")
            playerID: str = session.get("id")
            events: list[dict] = session.get("events")
            for event in events:
                output_json_record: dict = self._process_event(event=event)
                output_json_record[self._config.RAW_PLAYER_CUSTOM_DATA] = playerCustomData
                output_json_record[self._config.RAW_PLAYER_ID] = playerID
                output_json.append(output_json_record)
                
        return output_json
    
    def _process_event(self, event: dict) -> dict:
        # fix naming conventions for common fields between RM2 and the other formats
        rm2_standard_field_map = {
            RM2_EVENT_ID_KEY: self._config.EVENT_ID_KEY,
            RM2_RAW_SERVER_TIME: self._config.RAW_SERVER_TIME,
            RM2_RAW_USER_TIME: self._config.RAW_USER_TIME,
            RM2_EVENT_TYPE: self._config.EVENT_TYPE,
            RM2_RAW_COORDINATES: self._config.RAW_COORDINATES
        }
        def normalize_key(key: str) -> str:
            return rm2_standard_field_map.get(key, key)
        # filter to common fields
        output_json_record = {normalize_key(k): v for (k, v) in event.items() if (k in self._config.DOWNLOADER_COMMON_FIELDS) or (k in RM2_DOWNLOADER_COMMON_FIELDS)}
        # lowercase event type:
        if self._config.EVENT_TYPE in output_json_record:
            output_json_record[self._config.EVENT_TYPE] = output_json_record[self._config.EVENT_TYPE].lower()

        # add event's custom data fields
        output_json_record = self._add_events_custom_data(event=event, output_json_record=output_json_record)
        
        return output_json_record
    
    def _add_events_custom_data(self, *, event: dict, output_json_record: dict) -> dict:
        if self._config.EVENT_CUSTOM_DATA_KEY in event:
            if isinstance(event[self._config.EVENT_CUSTOM_DATA_KEY], dict):
                # Add each key as a custom data field
                for key, value in event[self._config.EVENT_CUSTOM_DATA_KEY].items():
                    keyName = f"{self._config.EVENT_CUSTOM_DATA_KEY}.{key}"
                    output_json_record[keyName] = value
        return output_json_record

    def _create_df(self, *, output_json: dict, verbose: Optional[bool] = False) -> pd.DataFrame:
        self.log_info("Formatting DataFrame...")
        df = pd.DataFrame(output_json)
        self._extra_fields = set(df.columns) - set(self._config.DOWNLOADER_FIELD_ORDER)
        all_fields = self._config.DOWNLOADER_FIELD_ORDER + tuple(self._extra_fields)
        return df.reindex(columns=all_fields)
    