import os
import getpass
import requests
import json
from typing import Optional, Any
import pandas as pd

from CFGpy.behavioral import Configuration, DataRetriever
from CFGpy.behavioral._consts import DATA_RETRIEVER_OUTPUT_FILENAME
from CFGpy.behavioral._utils import parse_json_column
import warnings


class CFGAppSyncDataRetriever(DataRetriever):
    """
    Downloader for the new CFG platform backed by Amplify/AppSync.

    Expected config fields:
        GAME_NAME
        GAME_ID
        CFG_GRAPHQL_URL
        CFG_AUTH_MODE            # "apiKey" or "userPool"
        CFG_API_KEY              # if authMode == "apiKey"
        CFG_COGNITO_LOGIN_URL    # if authMode == "userPool"
        CFG_USERNAME             # if authMode == "userPool"
        CFG_PASSWORD             # if authMode == "userPool"

    Optional config fields:
        DOWNLOADER_FIELD_ORDER
    """



    def __init__(
        self,
        *,
        game_name: str | None = None,
        game_id: str | None = None,
        output_filename: str = DATA_RETRIEVER_OUTPUT_FILENAME,
        config: Configuration = None,
    ) -> None:
        # warn that this is an experimental backend for now:
        warnings.warn("The CFGAppSyncDataRetriever is an experimental data retriever for the new CFG platform. "
                      "Please report any issues or inaccuracies you encounter when using it.", UserWarning, stacklevel=2)
        super().__init__(
            game_name=game_name,
            game_id=game_id,
            output_filename=output_filename,
            config=config,
        )
        self._validate_input([game_id, game_name, self._config.GAME_ID, self._config.GAME_NAME])
        self._session: Optional[requests.Session] = None
        self._games_cache: Optional[list[dict[str, Any]]] = None
        self._versions_cache: dict[str, list[dict[str, Any]]] = {}

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._init_session()
        return self._session

    def _init_session(self, verbose: bool = False) -> None:
        """
        Initialize an authenticated requests.Session for the new CFG platform.

        Auth credentials are intentionally NOT read from the config object, to avoid
        accidental sharing through config files.

        Current implementation supports only Cognito userPool-style login:
        - username/email from env var CFG_USERNAME or interactive input
        - password from env var CFG_PASSWORD or hidden interactive prompt

        Required config fields:
            CFG_COGNITO_LOGIN_URL
        """
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._login_userpool(verbose=verbose)

    def _import_auth_dependencies(self):
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "CFGAppSyncDataRetriever requires the optional dependency 'boto3'. "
                "Install it with: pip install boto3"
            ) from e

        try:
            from pycognito.aws_srp import AWSSRP
        except ImportError as e:
            raise ImportError(
                "CFGAppSyncDataRetriever requires the optional dependency 'pycognito'. "
                "Install it with: pip install pycognito"
            ) from e

        return boto3, AWSSRP

    def _login_userpool(self, verbose: bool = False) -> None:
        """
        Log in via a userPool-backed endpoint and store the returned JWT token
        in the session Authorization header.

        Credentials are taken from environment variables first:
            CFG_USERNAME
            CFG_PASSWORD

        If missing, they are requested interactively.
        """
        boto3, AWSSRP = self._import_auth_dependencies()
        username = os.getenv("CFG_USERNAME") or input("Please enter your CFG username/email: ")
        password = os.getenv("CFG_PASSWORD") or getpass.getpass("Please enter your CFG password: ")

        region = self._config.CFG_COGNITO_REGION
        client_id = self._config.CFG_COGNITO_CLIENT_ID
        user_pool_id = self._config.CFG_COGNITO_USER_POOL_ID

        if not region:
            raise ValueError("CFG_COGNITO_REGION is required")
        if not client_id:
            raise ValueError("CFG_COGNITO_CLIENT_ID is required")
        if not user_pool_id:
            raise ValueError("CFG_COGNITO_USER_POOL_ID is required")

        if verbose:
            print("Logging into CFG user pool with SRP...")

        cognito = boto3.client("cognito-idp", region_name=region)

        aws = AWSSRP(
            username=username,
            password=password,
            pool_id=user_pool_id,
            client_id=client_id,
            client=cognito,
        )

        tokens = aws.authenticate_user()
        auth_result = tokens.get("AuthenticationResult", {})
        token = auth_result.get("IdToken")

        if not token:
            raise ValueError("Login succeeded but no IdToken was returned")

        self._session.headers.update({"Authorization": token})

        if verbose:
            print("Successfully logged into CFG.")

    def _retrieve_data(
        self,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        include_game_data: bool = True,
        include_player_data: bool = True,
        include_session_data: bool = True,
        version_id: Optional[str] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        if self._game_name and not self._game_id:
            self._game_id = self._get_game_id_by_name(self._game_name, verbose=verbose)

        if not self._game_id:
            raise ValueError("Could not determine game_id")

        if verbose:
            print(f"Using game_id={self._game_id}")

        sessions = self._fetch_sessions(
            game_id=self._game_id,
            version_id=version_id,
            date_from=date_from,
            date_to=date_to,
            verbose=verbose,
        )

        events = self._fetch_events_for_sessions(sessions, verbose=verbose)

        player_map = {}
        if include_player_data:
            player_map = self._fetch_players_for_sessions(sessions, verbose=verbose)

        game_map = {g["id"]: g for g in self._list_games(verbose=verbose)}
        version_map = self._build_version_map_for_game(self._game_id, verbose=verbose)

        rows = self._build_rows(
            game_map=game_map,
            sessions=sessions,
            events=events,
            player_map=player_map,
            version_map=version_map,
        )

        # groups = {
        #     "game": include_game_data,
        #     "player": include_player_data,
        #     "session": include_session_data,
        # }
        # filtered_rows = self._filter_columns(rows, groups)

        self._retrieved_df = self._create_df(rows)
        self._retrieved_df = parse_json_column(df=self._retrieved_df, column_name=self._config.EVENT_CUSTOM_DATA_KEY, prefix=self._config.EVENT_CUSTOM_DATA_KEY)
        self._retrieved_df = self._order_df(self._retrieved_df)
        return self._retrieved_df

    def _graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        response = self.session.post(
            self._config.CFG_GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()

        if "errors" in payload:
            raise ValueError(f"GraphQL error: {payload['errors']}")

        return payload["data"]

    def _list_games(self, *, verbose: bool = False) -> list[dict]:
        if self._games_cache is not None:
            return self._games_cache

        if verbose:
            print("Fetching games...")

        query = """
        query ListGames($nextToken: String) {
          listGames(nextToken: $nextToken) {
            items {
              id
              name
              description
            }
            nextToken
          }
        }
        """

        items = self._list_all_graphql("listGames", query, variables={})
        self._games_cache = items
        return items

    def _list_versions_for_game(self, game_id: str, *, verbose: bool = False) -> list[dict]:
        if game_id in self._versions_cache:
            return self._versions_cache[game_id]

        if verbose:
            print(f"Fetching versions for game_id={game_id}...")

        query = """
        query ListGameVersions($filter: ModelGameVersionFilterInput, $nextToken: String) {
          listGameVersions(filter: $filter, nextToken: $nextToken) {
            items {
              id
              gameId
              version
              description
              releasedAt
            }
            nextToken
          }
        }
        """

        items = self._list_all_graphql(
            "listGameVersions",
            query,
            variables={"filter": {"gameId": {"eq": game_id}}},
        )
        self._versions_cache[game_id] = items
        return items

    def _build_version_map_for_game(self, game_id: str, *, verbose: bool = False) -> dict[str, str]:
        versions = self._list_versions_for_game(game_id, verbose=verbose)
        return {v["id"]: v.get("version", "") for v in versions}

    def _get_game_id_by_name(self, game_name: str, *, verbose: bool = False) -> str:
        games = self._list_games(verbose=verbose)
        matches = [g for g in games if g.get("name") == game_name]

        if not matches:
            raise ValueError(f"Game not found: {game_name}")
        if len(matches) > 1:
            raise ValueError(f"Multiple games matched name={game_name}; use game_id instead")

        return matches[0]["id"]

    def _fetch_sessions(
        self,
        *,
        game_id: str,
        version_id: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        verbose: bool = False,
    ) -> list[dict]:
        if verbose:
            print("Fetching sessions...")

        query = """
        query ListSessions($filter: ModelSessionFilterInput, $nextToken: String) {
          listSessions(filter: $filter, nextToken: $nextToken) {
            items {
              id
              playerId
              gameId
              gameVersionId
              startedAt
              endedAt
              metadata
            }
            nextToken
          }
        }
        """

        conditions: list[dict[str, Any]] = [{"gameId": {"eq": game_id}}]

        if version_id:
            conditions.append({"gameVersionId": {"eq": version_id}})
        if date_from:
            conditions.append({"startedAt": {"ge": self._date_to_iso_start(date_from)}})
        if date_to:
            conditions.append({"startedAt": {"le": self._date_to_iso_end(date_to)}})

        if len(conditions) == 1:
            filter_obj = conditions[0]
        else:
            filter_obj = {"and": conditions}

        return self._list_all_graphql(
            "listSessions",
            query,
            variables={"filter": filter_obj},
        )

    def _fetch_events_for_sessions(self, sessions: list[dict], *, verbose: bool = False) -> list[dict]:
        if verbose:
            print(f"Fetching events for {len(sessions)} sessions...")

        query = """
        query ListEvents($filter: ModelEventFilterInput, $nextToken: String) {
          listEvents(filter: $filter, nextToken: $nextToken) {
            items {
              id
              sessionId
              gameId
              type
              occurredAt
              data
            }
            nextToken
          }
        }
        """

        all_events: list[dict] = []
        for i, sess in enumerate(sessions, start=1):
            if verbose:
                print(f"  session {i}/{len(sessions)}")

            events = self._list_all_graphql(
                "listEvents",
                query,
                variables={"filter": {"sessionId": {"eq": sess["id"]}}},
            )
            all_events.extend(events)

        all_events.sort(key=lambda x: x.get("occurredAt", ""))
        return all_events

    def _fetch_players_for_sessions(self, sessions: list[dict], *, verbose: bool = False) -> dict[str, dict]:
        if verbose:
            print("Fetching player records...")

        query = """
        query GetPlayer($id: ID!) {
          getPlayer(id: $id) {
            id
            anonymousId
            firstSeenAt
            metadata
          }
        }
        """

        player_ids = sorted({s["playerId"] for s in sessions if s.get("playerId")})
        player_map: dict[str, dict] = {}

        for i, pid in enumerate(player_ids, start=1):
            if verbose and (i == 1 or i % 100 == 0):
                print(f"  player {i}/{len(player_ids)}")

            data = self._graphql(query, {"id": pid})
            player = data.get("getPlayer")
            if player:
                player_map[player["id"]] = player

        return player_map

    def _list_all_graphql(
        self,
        root_field: str,
        query: str,
        variables: Optional[dict] = None,
    ) -> list[dict]:
        results: list[dict] = []
        next_token = None
        variables = dict(variables or {})

        while True:
            variables["nextToken"] = next_token
            data = self._graphql(query, variables)
            page = data[root_field]
            results.extend(page["items"])
            next_token = page.get("nextToken")
            if not next_token:
                break

        return results

    @staticmethod
    def _date_to_iso_start(date_str: str) -> str:
        return f"{date_str}T00:00:00.000Z"

    @staticmethod
    def _date_to_iso_end(date_str: str) -> str:
        return f"{date_str}T23:59:59.000Z"

    @staticmethod
    def _parse_metadata(raw: Any) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, dict):
                return raw
            return {}
        except Exception:
            return {}

    def _build_rows(
        self,
        *,
        game_map: dict[str, dict],
        sessions: list[dict],
        events: list[dict],
        player_map: dict[str, dict],
        version_map: dict[str, str],
    ) -> list[dict[str, str]]:
        session_map = {s["id"]: s for s in sessions}
        rows: list[dict[str, str]] = []

        for ev in events:
            sess = session_map.get(ev["sessionId"], {})

            player = player_map.get(sess.get("playerId", ""), {})
            sess_meta = self._parse_metadata(sess.get("metadata"))

            duration_seconds = ""
            started_at = sess.get("startedAt")
            ended_at = sess.get("endedAt")
            if started_at and ended_at:
                try:
                    start_ts = pd.Timestamp(started_at)
                    end_ts = pd.Timestamp(ended_at)
                    duration_seconds = str(round((end_ts - start_ts).total_seconds()))
                except Exception:
                    duration_seconds = ""

            row = {
                # "gameId": ev.get("gameId", ""),
                # "gameName": game_map.get(ev.get("gameId", ""), {}).get("name", ""),
                self._config.RAW_GAME_VERSION: sess.get("gameVersionId", "") or "",
                # "gameVersionLabel": version_map.get(sess.get("gameVersionId", ""), "") or "",
                self._config.RAW_PLAYER_ID: sess.get("playerId", "") or "",
                # "playerAnonymousId": player.get("anonymousId", "") or "",
                # "playerFirstSeenAt": player.get("firstSeenAt", "") or "",
                "playerMetadata": player.get("metadata") if player.get("metadata") is not None else {},
                # "sessionId": ev.get("sessionId", ""),
                # "sessionStartedAt": started_at or "",
                # "sessionEndedAt": ended_at or "",
                # "sessionDurationSeconds": duration_seconds,
                "sessionMetadata": json.loads(sess.get("metadata")) if sess.get("metadata") is not None else {},
                # "externalId": sess_meta.get("externalId", "") or "",
                # "expId": sess_meta.get("expId", "") or "",
                # "userId": sess_meta.get("userId", "") or "",
                # "userProvidedId": sess_meta.get("userProvidedId", "") or "",
                self._config.EVENT_ID_KEY: ev.get("id", ""),
                self._config.EVENT_TYPE: ev.get("type", ""),
                self._config.RAW_USER_TIME: ev.get("occurredAt", ""),
                self._config.EVENT_CUSTOM_DATA_KEY: ev.get("data") if ev.get("data") is not None else "",
            }
            if row["sessionMetadata"]:
                row["playerCustomData"] = row["sessionMetadata"].get("customData") if (row["sessionMetadata"].get("customData") is not None) else {}
            player_metadata = row.get("playerMetadata", {})
            row[self._config.RAW_PLAYER_BIRTHDATE] = player.get("birthDate", None)
            row[self._config.RAW_PLAYER_REGION] = player.get("region", None)
            row[self._config.RAW_PLAYER_COUNTRY] = player.get("country", None)
            row[self._config.RAW_PLAYER_GENDER] = player.get("gender", None)
            row[self._config.RAW_PLAYER_EXTERNAL_ID] = player.get("externalId", None)
            # remove the general metadata fields
            row.pop("playerMetadata", None)
            row.pop("sessionMetadata", None)
            rows.append(row)

        return rows


    def _order_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._config.DOWNLOADER_FIELD_ORDER:
            ordered_cols = [col for col in self._config.DOWNLOADER_FIELD_ORDER if col in df.columns]
            extra_cols = [col for col in df.columns if col not in ordered_cols]
            return df[ordered_cols + extra_cols]
        else:
            return df
    def _create_df(self, rows: list[dict[str, str]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)

        return df