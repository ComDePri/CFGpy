from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from pandas import Timestamp
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
    def _retrieve_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        The internal implementation of the data retrieval logic. This method should be implemented by subclasses to
        specify how to retrieve data from the desired source.
        """
        pass

    # def _resolve_time_column_for_filtering(self, df: pd.DataFrame) -> str | None:
    #     """Choose which timestamp column to use for before/after filtering.
    #
    #     Preference order:
    #     1) configured RAW_SERVER_TIME
    #     2) configured RAW_USER_TIME
    #
    #     Returns column name if present in df, else None.
    #     """
    #     if self._config is None:
    #         return None
    #
    #     for attr in ("RAW_SERVER_TIME", "RAW_USER_TIME"):
    #         col = getattr(self._config, attr, None)
    #         if col and col in df.columns:
    #             return col
    #     return None

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
            return df

        if verbose:
            print(f"Filtering data by date using column '{time_col}'...")

        # Coerce to datetime; keep original column untouched.
        series = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        mask = series.notna()

        if after is not None:
            mask &= series >= after
            if verbose:
                print(f"Filtering from {after} onwards...")
        if before is not None:
            mask &= series <= before
            if verbose:
                print(f"Filtering until {before}...")
        if verbose:
            print(f"Filtered from {len(df)} rows to {mask.sum()} rows by date.")
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

        These are popped from kwargs before delegating to the subclass' `_retrieve_data`, so subclasses that don't
        accept them won't break.
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
        count: int = len(input) - input.count(None)

        # at least one URL should not be None:
        if count < 1:
            raise ValueError(NO_DATA_RETRIEVER_INPUT_ERROR)

        # at most one URL should not be None:
        if count > 1:
            raise ValueError(MULTIPLE_DATA_RETRIEVER_INPUTS_ERROR)

        return None

    def dump(self, verbose: Optional[bool] = False) -> None:

        if self._retrieved_df is None:
            raise ValueError("No data to dump. Run retrieve_data() first.")

        self._retrieved_df.to_csv(f"{self._output_filename}_events.csv", index=False)
        if verbose:
            print(f"Wrote CSV to {self._output_filename}")

        self._config.to_yaml(self._output_filename)
