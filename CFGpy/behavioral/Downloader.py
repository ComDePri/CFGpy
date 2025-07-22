from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from CFGpy.behavioral import Configuration
from CFGpy.behavioral._consts import DOWNLOADER_OUTPUT_FILENAME, MULTIPLE_DOWNLOADER_INPUTS_ERROR, NO_DOWNLOADER_INPUT_ERROR


class Downloader(ABC):
    def __init__(self, *, game_name: str | None = None, game_id: str | None = None, output_filename: str = DOWNLOADER_OUTPUT_FILENAME, config: Configuration = None) -> None:
        self._validate_input(input=[game_id, game_name, self._config.GAME_ID, self._config.GAME_NAME])
        self._game_name = game_name or self._config.GAME_NAME
        self._game_id: str = game_id or self._config.GAME_ID
        self._output_filename = output_filename
        self._config = config
        self._downloaded_df: Optional[pd.DataFrame] = None
        self._extra_fields = set()

    @abstractmethod
    def download(self, *args, **kwargs) -> pd.DataFrame:
        pass
        
    def _validate_input(self, input: list[str]) -> None:
        none_count: int = input.count(None)
        
        # at least one URL should not be None:
        if none_count < 1:
            raise ValueError(NO_DOWNLOADER_INPUT_ERROR)

        # at most one URL should not be None:
        if  none_count > 1:
            raise ValueError(MULTIPLE_DOWNLOADER_INPUTS_ERROR)
        
        return None
    
    def dump(self, verbose: Optional[bool] = False) -> None:
        
        if self._downloaded_df is None:
            raise ValueError("No data to dump. Run download() first.")

        if verbose:
            print(f"Wrote CSV to {self._output_filename}")
            
        self._config.to_yaml(self._output_filename)
        self._downloaded_df.to_csv(self._output_filename, index=False)
