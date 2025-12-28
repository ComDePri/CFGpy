import json
import os
from typing import Optional
import pandas as pd
from pathlib import Path
from CFGpy.behavioral._consts import (DATA_RETRIEVER_OUTPUT_FILENAME, CONFIG_URL_MISMATCH_ERROR)
from CFGpy.behavioral import Configuration, DataRetriever
from CFGpy.utils._nas_path import get_nas_path

class RedMetrics1DataRetriever(DataRetriever):
    def __init__(self, *, game_name: str | None = None, game_id: str | None = None, game_version_ids: list[str] | None = None, 
                 output_filename: str = DATA_RETRIEVER_OUTPUT_FILENAME, config: Configuration = None,
                 csv_directory: str = None) -> None:
        """
        :param game_name: The game name of the game whose data you want to retrieve from local CSV files.
        :param game_id: The game id of the game whose data you want to retrieve from local CSV files.
        :param game_version_ids: List of game version IDs to filter by.
        :param output_filename: filename for output.
        :param config: a Configuration file.
        :param csv_directory: Directory containing the CSV files (events.csv, players.csv, games.csv, game_versions.csv)
        """
        super().__init__(game_name=game_name, game_id=game_id, output_filename=output_filename, 
                         config=config if config is not None else Configuration.default(is_rm1=False))

        self._validate_input(input=[game_id, game_name, game_version_ids, self._config.GAME_ID, self._config.GAME_NAME, self._config.GAME_VERSION_IDS])
        self._validate_config()
        self._game_version_ids = self._config.GAME_VERSION_IDS or game_version_ids
        self.nas_path = get_nas_path()
        self.csv_path = os.path.join(self.nas_path, "Projects", "CFG", "all_data_from_aws", "redmetrics")
        self._csv_directory = Path(csv_directory if csv_directory else self.csv_path)
        self._load_csv_files()

    def _load_csv_files(self):
        """Load all CSV files from the local directory"""
        try:
            self._events_df = pd.read_csv(self._csv_directory / "events.csv")
            self._players_df = pd.read_csv(self._csv_directory / "players.csv")
            self._games_df = pd.read_csv(self._csv_directory / "games.csv")
            self._game_versions_df = pd.read_csv(self._csv_directory / "game_versions.csv")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing CSV file in {self._csv_directory}: {e}")
        except Exception as e:
            raise Exception(f"Error loading CSV files from {self._csv_directory}: {e}")

    def retrieve_data(self, *, verbose: bool = False, after: str = None, before: str = None, event_type: str = None, 
                       section: str = None) -> pd.DataFrame:
        self._retrieved_df = self.fetch_all_data(verbose=verbose, after=after, before=before, event_type=event_type, section=section)
        return self._format_df(verbose=verbose)
        
    def _validate_config(self) -> None:
        if not self._config.is_rm1:
            raise ValueError(CONFIG_URL_MISMATCH_ERROR)
        return None
    
    def get_game_version_ids(self):
        if self._game_version_ids:
            return self._game_version_ids
        
        elif self._game_id:
            matching_versions = self._game_versions_df[self._game_versions_df['game_id'] == self._game_id]
            if not matching_versions.empty:
                return matching_versions['id'].tolist()
            else:
                raise ValueError(f"No game_versions found for game_id: {self._game_id}")

        else:
            merged_df = self._game_versions_df.merge(
                self._games_df, 
                left_on='game_id', 
                right_on='id', 
                suffixes=('', '_game')
            )
            matching_versions = merged_df[merged_df['name_game'] == self._game_name]
            if not matching_versions.empty:
                return matching_versions['id'].tolist()
            else:
                raise ValueError(f"No game_versions found for game name: {self._game_name}")

    def _create_df(self, *, game_version_id: str, after: str = None, before: str = None, event_type: str = None, 
                          section: str = None) -> pd.DataFrame:
        """
        Returns a dataframe of events for a given game version id.
        """
        
        filtered_df = self._events_df[self._events_df['gameVersion_id'] == game_version_id].copy()
        
        if after:
            filtered_df = filtered_df[filtered_df['serverTime'] >= after]
        if before:
            filtered_df = filtered_df[filtered_df['serverTime'] <= before]
        if event_type:
            filtered_df = filtered_df[filtered_df['type'] == event_type]
        if section:
            filtered_df = filtered_df[filtered_df['section'].str.contains(section, na=False)]
        
        result_df = filtered_df.merge(
            self._players_df,
            left_on='player_id',
            right_on='id',
            suffixes=('', '_player')
        )
        
        if 'id_player' in result_df.columns:
            result_df = result_df.drop(columns=['id_player'])
        if 'customData' in result_df.columns:
            result_df = result_df.rename(columns={'customData': 'eventCustomData'})
        if 'customData_player' in result_df.columns:
            result_df = result_df.rename(columns={'customData_player': 'playerCustomData'})

        return result_df

    def fetch_all_data(self, *, verbose: bool = False, after: str = None, before: str = None,
                    event_type: str = None, section: str = None) -> pd.DataFrame:
        """Fetch all data from local CSV files"""
        all_dfs = []
        game_version_ids = self.get_game_version_ids()

        for game_version_id in game_version_ids:
            if verbose:
                print(f"Fetching game_version_id={game_version_id}...")
            
            df = self._create_df(
                game_version_id=game_version_id,
                after=after,
                before=before,
                event_type=event_type,
                section=section
            )
            
            if not df.empty:
                all_dfs.append(df)

        result_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        if verbose:
            print(f"Fetched {len(result_df)} rows total from {len(game_version_ids)} game version(s).")

        return result_df

    def parse_json_column(self, *, df: pd.DataFrame, column_name: str, prefix: str):
        """Parse JSON columns in the DataFrame"""
        def try_parse(val):
            if pd.isna(val):
                return {}
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return {}

        parsed_df = df[column_name].apply(try_parse).apply(pd.Series)
        parsed_df.columns = [f"{prefix}.{col}" for col in parsed_df.columns]
        
        return pd.concat([df.drop(columns=[column_name]), parsed_df], axis=1)

    def convert_to_iso8601_millis(self, *, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Convert specified datetime columns in a DataFrame to ISO 8601 format with millisecond precision.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (list): List of column names to convert.

        Returns:
            pd.DataFrame: Modified DataFrame with formatted datetime columns.
        """
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce") \
                            .dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ') \
                            .str.slice(stop=-4) + 'Z'
        return df

    def _format_df(self, *, verbose: Optional[bool] = False) -> pd.DataFrame:
        """Format the retrieved DataFrame to match expected output format"""
        if not self._retrieved_df.empty:
            
            if verbose:
                print("Formatting dataframe...")
                
            self._retrieved_df.rename(columns={
                "gameVersion_id": "gameVersion",
                "player_id": "playerId",
                "birthDate": "playerBirthdate",
                "region": "playerRegion",
                "country": "playerCountry",
                "gender": "playerGender",
                "externalId": "playerExternalId",
            }, inplace=True)

            if "eventCustomData" in self._retrieved_df.columns:
                self._retrieved_df = self.parse_json_column(df=self._retrieved_df, column_name="eventCustomData", prefix="customData")
            
            self._retrieved_df = self.convert_to_iso8601_millis(df=self._retrieved_df, columns=["serverTime", "userTime"])
    
        self._extra_fields = set(self._retrieved_df.columns) - set(self._config.DOWNLOADER_FIELD_ORDER)
        all_fields = self._config.DOWNLOADER_FIELD_ORDER + tuple(self._extra_fields)
        self._retrieved_df = self._retrieved_df.reindex(columns=all_fields)
        
        return self._retrieved_df
