from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from CFGpy.behavioral import Configuration
from CFGpy.behavioral._consts import DATA_RETRIEVER_OUTPUT_FILENAME, MULTIPLE_DATA_RETRIEVER_INPUTS_ERROR, NO_DATA_RETRIEVER_INPUT_ERROR


class DataRetriever(ABC):
    def __init__(self, *, game_name: str | None = None, game_id: str | None = None, output_filename: str = DATA_RETRIEVER_OUTPUT_FILENAME, config: Configuration = None) -> None:
        self._game_name = game_name or config.GAME_NAME
        self._game_id: str = game_id or config.GAME_ID
        self._output_filename = output_filename
        self._config = config
        self._retrieved_df: Optional[pd.DataFrame] = None
        self._extra_fields = set()

    @abstractmethod
    def retrieve_data(self, *args, **kwargs) -> pd.DataFrame:
        pass
        
    def _validate_input(self, input: list[str]) -> None:
        count: int  = len(input) - input.count(None)
        
        # at least one URL should not be None:
        if count < 1:
            raise ValueError(NO_DATA_RETRIEVER_INPUT_ERROR)

        # at most one URL should not be None:
        if  count > 2:
            raise ValueError(MULTIPLE_DATA_RETRIEVER_INPUTS_ERROR)
        
        return None
    
    def dump(self, verbose: Optional[bool] = False) -> None:
        
        if self._retrieved_df is None:
            raise ValueError("No data to dump. Run retrieve_data() first.")

        if verbose:
            print(f"Wrote CSV to {self._output_filename}")
            
        self._config.to_yaml(self._output_filename)
        self._retrieved_df.to_csv(self._output_filename, index=False)
