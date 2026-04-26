from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from pandas import Timestamp
from CFGpy.behavioral import Configuration
from CFGpy.behavioral._consts import DATA_RETRIEVER_OUTPUT_FILENAME, MULTIPLE_DATA_RETRIEVER_INPUTS_ERROR, NO_DATA_RETRIEVER_INPUT_ERROR
from CFGpy.behavioral._logging import HasLogger


class DataRetriever(HasLogger,ABC):
    def __init__(self, *, game_name: str | None = None, game_id: str | None = None, output_filename: str = DATA_RETRIEVER_OUTPUT_FILENAME, config: Configuration = None, logger=None) -> None:
        super().__init__(logger)
        self._game_name = game_name or config.GAME_NAME
        self._game_id: str = game_id or config.GAME_ID
        self._output_filename = output_filename
        self.output_path = f"{self._output_filename}_events.csv"
        self._config = config
        self._retrieved_df: Optional[pd.DataFrame] = None
        self._extra_fields = set()

    @abstractmethod
    def _retrieve_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        The internal implementation of the data retrieval logic. This method should be implemented by subclasses to
        specify how to retrieve data from the desired source.
        """
        pass


    def _filter_df_by_date(self, df: pd.DataFrame, *, after: Timestamp | None, before: Timestamp | None, verbose: bool = False) -> pd.DataFrame:
        """Filter a DF by inclusive datetime range [after, before] on a best-effort basis.

        - If timestamp column is missing or can't be parsed, returns df unchanged.
        """
        if df is None or df.empty:
            return df
        if after is None and before is None:
            return df

        time_col = self._config.RAW_USER_TIME


        if not time_col:
            self.log_warning("No time column specified in config; skipping date filtering.")
            return df
        if time_col not in df.columns:
            self.log_warning(f"Time column '{time_col}' not found in data; skipping date filtering.")
            return df

        self.log_info(f"Attempting to filter data by date using column '{time_col}' with after={after} and before={before}.")


        # Coerce to datetime; keep original column untouched.
        series = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        mask = series.notna()

        if after is not None:
            mask &= series >= after
            self.log_info(f"Filtering from {after} onwards...")
        if before is not None:
            mask &= series <= before
            self.log_info(f"Filtering until {before}...")
        self.log_info(f"Filtered from {len(df)} rows to {mask.sum()} rows by date.")
        return df.loc[mask].reset_index(drop=True).copy()

    def _to_ts(self, s: str) -> pd.Timestamp:
        """Helper to parse a datetime string to a pandas Timestamp, with error handling."""
        try:
            return pd.to_datetime(s, errors="raise", utc=True)
        except Exception as e:
            raise ValueError(f"Invalid datetime string for filtering: '{s}'.\nError: {e}")


    def retrieve_data(self, *args, **kwargs) -> pd.DataFrame:
        """Retrieve data and return it as a pandas DataFrame.

        Generic optional filtering (applies to all data sources):
            before: datetime string (inclusive upper bound)
            after: datetime string (inclusive lower bound)
        """
        # validate date filtering early before doing any work:
        before = self._config.BEFORE_DATE
        after = self._config.AFTER_DATE
        before_ts = self._to_ts(before) if before else None
        after_ts = self._to_ts(after) if after else None

        self._retrieved_df = self._retrieve_data(*args, **kwargs)

        verbose = kwargs.pop("verbose", False)
        self._retrieved_df = self._filter_df_by_date(self._retrieved_df, after=after_ts, before=before_ts, verbose=verbose)
        return self._retrieved_df

    def _validate_input(self, input: list[str]) -> None:
        def normalize_input(inp: str | None | list[str]) -> str | None | tuple[str]:
            if isinstance(inp, list):
                return tuple(inp)
            return inp

        unique_inputs = set([normalize_input(inp) for inp in input if inp is not None])
        count: int = len(unique_inputs)

        # at least one URL should not be None:
        if count < 1:
            raise ValueError(NO_DATA_RETRIEVER_INPUT_ERROR)

        # at most one URL should not be None:
        if count > 1:
            raise ValueError(MULTIPLE_DATA_RETRIEVER_INPUTS_ERROR)

        return None

    def dump(self) -> None:

        if self._retrieved_df is None:
            raise ValueError("No data to dump. Run retrieve_data() first.")

        self._retrieved_df.to_csv(self.output_path, index=False)
        self.log_info(f"Wrote data to: {self.output_path}")

        self._config.to_yaml(self._output_filename)
