import pandas as pd

from .DataRetriever import DataRetriever
from .Configuration import Configuration
from logging import Logger
from typing import Callable
import copy

class MultiGameDataRetriever(DataRetriever):
    def _retrieve_data(self, *args, **kwargs) -> pd.DataFrame:
        dfs = [retriever.retrieve_data(*args, **kwargs) for retriever in self.retrievers]
        return pd.concat(dfs).reset_index(drop=True)

    def _create_retrievers(self):
        if self._game_id:
            if isinstance(self._game_id, str):
                game_ids =[gid.strip() for gid in self._game_id.split(",")]
                # remove empty strings that may result from splitting
                game_ids = [gid for gid in game_ids if gid]
            elif isinstance(self._game_id, (list,tuple)): # check if it is a list or tuple
                game_ids = list(self._game_id)
            else:
                raise TypeError(f"Can't interpret multiple game IDs for GAME_ID field of type {type(self._game_id)}")
            retrievers = []
            for game_id in game_ids:
                cfg = copy.deepcopy(self._config)
                cfg.GAME_ID = game_id
                self.log_info(f"Creating retriever for game ID: {game_id}")
                retrievers.append(self.base_retriever_constructor(cfg, self.logger))
            return retrievers
        else:
            raise ValueError(f"Can't run MultiGameDataRetriever without GAME_ID field in the config")

    def __init__(self, base_retriever_constructor: Callable[[Configuration, Logger], DataRetriever],
                 config: Configuration = None, logger: Logger = None):
        super().__init__(config=config, logger=logger)
        self.base_retriever_constructor = base_retriever_constructor
        self.retrievers = self._create_retrievers()






