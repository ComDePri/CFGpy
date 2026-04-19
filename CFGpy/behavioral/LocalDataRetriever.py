import os
import json
from typing import Optional, Any
import pandas as pd

from CFGpy.behavioral import Configuration, DataRetriever
from CFGpy.behavioral._consts import DATA_RETRIEVER_OUTPUT_FILENAME
import warnings

class LocalDataRetriever(DataRetriever):
    def __init__(self, events_csv_path: str | None = None, output_filename: str = DATA_RETRIEVER_OUTPUT_FILENAME,
                 config: Configuration = None, logger = None) -> None:
        super().__init__(output_filename=output_filename, config=config, logger=logger)
        self._events_csv_path = events_csv_path
        self._determine_path()

    def _determine_path(self):
        config_csv_path = self._config.EVENT_CSV_PATH
        if self._events_csv_path is not None:
            if config_csv_path is None:
                return  # path already provided, no need to determine
            if self._events_csv_path != config_csv_path:
                raise ValueError(f"Conflict between provided events_csv_path ({self._events_csv_path}) and config's EVENT_CSV_PATH ({config_csv_path}). Please resolve the conflict by providing a consistent path.")
        elif config_csv_path is not None:
            self._events_csv_path = config_csv_path
        else:
            raise ValueError("No events CSV path provided. Please provide a path either through the constructor or the config.")


    def _retrieve_data(self, *args, **kwargs) -> pd.DataFrame:
        if not os.path.exists(self._events_csv_path):
            raise FileNotFoundError(f"CSV file not found at {self._events_csv_path}")

        # warn the user that we use the provided config but the original config generating the events data might be different, and that they should ensure consistency between the two configs if they want to use the original config for anything else in the pipeline
        msg = f"Loading events from {self._events_csv_path}. Make sure that the config used for this run is consistent with the config used to generate the events data. The config used to generate the events data should be saved at {self._events_csv_path.replace('.csv','')}_config.yaml if it was dumped using CFGpy."
        self.log_warning(msg)
        warnings.warn(msg, UserWarning)
        df = pd.read_csv(self._events_csv_path)
        return df