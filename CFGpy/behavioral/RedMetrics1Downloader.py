import json
import os
from typing import Any, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from CFGpy.behavioral._consts import (DOWNLOADER_OUTPUT_FILENAME, CONFIG_URL_MISMATCH_ERROR, PER_PAGE, MAX_PAGES)
from CFGpy.behavioral import Configuration, Downloader

class RedMetrics1Downloader(Downloader):
    def __init__(self, *, game_name: str | None = None, game_id: str | None = None, game_version_ids: list[str] | None = None, output_filename: str = DOWNLOADER_OUTPUT_FILENAME, 
                 config: Configuration = None) -> None:
        """
        Init a RedMetrics1Downloader object.
        :param game_id: The game name of the game whose data you want to download from RedMetrics.
        :param game_id: The game id of the game whose data you want to download from RedMetrics.
        :param output_filename: filename for output.
        :param config: a Configuration file.
        """
        super().__init__(game_name=game_name, game_id=game_id, output_filename=output_filename, 
                         config=config if config is not None else Configuration.default())
        self._validate_input(input=[game_id, game_name, self._config.GAME_ID, self._config.GAME_NAME, game_version_ids])
        self._validate_config()
        self._game_version_ids = game_version_ids
        self._engine = create_engine(
            f'postgresql+psycopg2://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}'
        )

    def download(self, *, verbose: bool = False, after: str = None, before: str = None, event_type: str = None, 
                       section: str = None) -> pd.DataFrame:
        self._downloaded_df = self.fetch_all_data(verbose=verbose, after=after, before=before, event_type=event_type, section=section)
        return self._format_df(verbose=verbose)
        
    def _validate_config(self) -> None:
        if self._config.is_rm2:
            raise ValueError(CONFIG_URL_MISMATCH_ERROR)
        return None
    
    def get_game_version_ids(self, *, connection):
        
        if self._game_version_ids:
            return self._game_version_ids
        
        elif self._game_id:
            query = text("SELECT id FROM game_versions WHERE game_id = :game_id")
            result = connection.execute(query, {"game_id": self._game_id}).fetchall()
            return [row[0] for row in result]

        else:
            query = text("""
                SELECT gv.id
                FROM game_versions gv
                JOIN games g ON gv.game_id = g.id
                WHERE g.name = :name
            """)
            result = connection.execute(query, {"name": self._game_name}).fetchall()
            if result:
                return [row[0] for row in result]
            else:
                raise ValueError(f"No game_versions found for game name: {self._game_name}")

    def build_event_query(self, *, game_version_id: str, page: int, after: str = None, before: str = None, event_type: str = None, 
                          section: str = None) -> Any: # TODO: check types
        
        filters = ['e."gameVersion_id" = :game_version_id']
        params = {"game_version_id": game_version_id}

        if after:
            filters.append('e."serverTime" >= :after')
            params["after"] = after
        if before:
            filters.append('e."serverTime" <= :before')
            params["before"] = before
        if event_type:
            filters.append('e."type" = :event_type')
            params["event_type"] = event_type
        if section:
            filters.append('e."section" ~ :section')
            params["section"] = section

        where_clause = " AND ".join(filters)
        offset = (page - 1) * PER_PAGE

        query = text(f"""
            SELECT 
                e."id", e."serverTime", e."userTime", e."gameVersion_id", e."player_id", 
                e."type", e."coordinates", e."section", 
                p."birthDate", p."region", p."country", p."gender", 
                p."externalId", 
                p."customData" AS "playerCustomData",
                e."customData" AS "eventCustomData"
            FROM events e
            JOIN players p ON e."player_id" = p."id"
            WHERE {where_clause}
            ORDER BY e."serverTime" ASC
            LIMIT :limit OFFSET :offset
        """)

        params["limit"] = PER_PAGE
        params["offset"] = offset
        
        return query, params

    def fetch_all_data(self, *, verbose: bool = False, after: str = None, before: str = None,
                    event_type: str = None, section: str = None) -> pd.DataFrame:
        
        all_dfs = []
        with self._engine.begin() as connection:
            game_version_ids = self.get_game_version_ids(connection=connection)

            for game_version_id in game_version_ids:
                for page in range(1, MAX_PAGES + 1):
                    if verbose:
                        print(f"Fetching game_version_id={game_version_id}, page {page}...")
                    query, params = self.build_event_query(
                        game_version_id=game_version_id,
                        page=page,
                        after=after,
                        before=before,
                        event_type=event_type,
                        section=section
                    )
                    df = pd.read_sql_query(query, connection, params=params)
                    if df.empty:
                        break
                    all_dfs.append(df)

        df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        if verbose:
            print(f"Fetched {len(df)} rows total from {len(game_version_ids)} game version(s).")

        return df


    def parse_json_column(self, *, df: pd.DataFrame, column_name: str, prefix: str):
        
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

        if not self._downloaded_df.empty:
            
            if verbose:
                print("Formatting dataframe...")
                
            self._downloaded_df.rename(columns={
                "gameVersion_id": "gameVersion",
                "player_id": "playerId",
                "birthDate": "playerBirthdate",
                "region": "playerRegion",
                "country": "playerCountry",
                "gender": "playerGender",
                "externalId": "playerExternalId",
            }, inplace=True)

            self._downloaded_df = self.parse_json_column(df=self._downloaded_df, column_name="eventCustomData", prefix="customData")
            self._downloaded_df = self.convert_to_iso8601_millis(df=self._downloaded_df, columns=["serverTime", "userTime"])
    
        self._extra_fields = set(self._downloaded_df.columns) - set(self._config.DOWNLOADER_FIELD_ORDER)
        all_fields = self._config.DOWNLOADER_FIELD_ORDER + tuple(self._extra_fields)
        
        return self._downloaded_df.reindex(columns=all_fields)
