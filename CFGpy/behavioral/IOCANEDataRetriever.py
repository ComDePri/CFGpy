import os
import getpass
import requests
import json
from typing import Optional, Any
import pandas as pd
import tqdm
from platformdirs import user_cache_dir

from CFGpy.behavioral import Configuration, DataRetriever
from CFGpy.behavioral._consts import DATA_RETRIEVER_OUTPUT_FILENAME, IOCANE_BOOTSTRAP_URL
from CFGpy.behavioral._utils import parse_json_column
from CFGpy._version import __version__ as CFGPY_VERSION
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed


class IOCANEDataRetriever(DataRetriever):
    """
    Downloader for the new CFG platform backed by Amplify/AppSync.

    Expected config fields:
        GAME_NAME
        GAME_ID

    Optional config fields:
        DOWNLOADER_FIELD_ORDER
    """

    def __init__(
            self,
            *,
            game_id: str | None = None,
            output_filename: str = DATA_RETRIEVER_OUTPUT_FILENAME,
            config: Configuration = None,
            logger=None
    ) -> None:
        super().__init__(
            game_id=game_id,
            output_filename=output_filename,
            config=config,
            logger=logger
        )
        # warn that this is an experimental backend for now:
        warnings.warn("The IOCANEDataRetriever is an experimental data retriever for the new CFG platform. "
                      "Please report any issues or inaccuracies you encounter when using it.", UserWarning,
                      stacklevel=2)
        self.log_warning("The IOCANEDataRetriever is an experimental data retriever for the new CFG platform. ")
        self._validate_input([game_id, self._config.GAME_ID])
        self._session: Optional[requests.Session] = None
        self._games_cache: Optional[list[dict[str, Any]]] = None
        self._versions_cache: dict[str, list[dict[str, Any]]] = {}
        self._graphql_url = None

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
                "IOCANEDataRetriever requires the optional dependency 'boto3'. "
                "Install it with: pip install boto3"
            ) from e

        try:
            from pycognito.aws_srp import AWSSRP
        except ImportError as e:
            raise ImportError(
                "IOCANEDataRetriever requires the optional dependency 'pycognito'. "
                "Install it with: pip install pycognito"
            ) from e

        return boto3, AWSSRP

    def _login_userpool(self, verbose: bool = False) -> None:
        boto3, AWSSRP = self._import_auth_dependencies()

        username = os.getenv("CFG_USERNAME") or input(
            "Please enter your CFG username/email: "
        )
        password = os.getenv("CFG_PASSWORD") or getpass.getpass(
            "Please enter your CFG password: "
        )

        # ─── Step 1: Fetch runtime config from Lambda ─────────────────────────────
        bootstrap_url = IOCANE_BOOTSTRAP_URL

        self.log_info("Fetching backend configuration from bootstrap endpoint...")

        resp = requests.post(
            bootstrap_url,
            json={"username": username, "password": password},
            timeout=30,
        )

        if resp.status_code != 200:
            raise ValueError(f"Bootstrap login failed: {resp.text}")

        cfg = resp.json()

        # Inject dynamically (DO NOT persist)
        region = cfg["CFG_COGNITO_REGION"]
        client_id = cfg["CFG_COGNITO_CLIENT_ID"]
        user_pool_id = cfg["CFG_COGNITO_USER_POOL_ID"]
        graphql_url = cfg["CFG_GRAPHQL_URL"]


        self._graphql_url = graphql_url

        if verbose:
            self.log_info("Received backend configuration successfully")

        # ─── Step 2: SRP login (existing logic) ───────────────────────────────────
        self.log_info("Logging into Cognito via SRP...")

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
            raise ValueError("Login succeeded but no IdToken returned")

        self._session.headers.update({"Authorization": token})

        self.log_info(
            f"Logged in as {username} (region={region}, pool={user_pool_id})"
        )


    def _retrieve_data(
            self,
            *,
            date_from: Optional[str] = None,
            date_to: Optional[str] = None,
            version_id: Optional[str] = None,
            verbose: bool = False,
    ) -> pd.DataFrame:

        if not self._game_id:
            raise ValueError("Could not determine game_id")

        self.log_info(f"Using game_id={self._game_id}")

        sessions = self._fetch_sessions(
            game_id=self._game_id,
            version_id=version_id,
            date_from=date_from,
            date_to=date_to,
            verbose=verbose,
        )

        events = self._fetch_events_for_sessions(sessions, verbose=verbose)

        player_map = {}
        player_map = self._fetch_players_for_sessions(sessions, verbose=verbose)

        game_map = {g["id"]: g for g in self._list_games(verbose=verbose)}


        rows = self._build_rows(
            game_map=game_map,
            sessions=sessions,
            events=events,
            player_map=player_map,
        )

        self._retrieved_df = self._create_df(rows)

        self._retrieved_df = parse_json_column(df=self._retrieved_df, column_name=self._config.EVENT_CUSTOM_DATA_KEY,
                                               prefix=self._config.EVENT_CUSTOM_DATA_KEY)
        self._retrieved_df = self._order_df(self._retrieved_df)
        return self._retrieved_df

    def _graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        response = self.session.post(
            self._graphql_url,
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

        self.log_info("Listing games...")

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
        self.log_info(
            f"Fetching sessions via game.sessions for game_id={game_id}, "
            f"version_id={version_id}, date_from={date_from}, date_to={date_to}..."
        )

        query = """
        query GetGameWithSessions($id: ID!, $filter: ModelSessionFilterInput, $nextToken: String) {
          getGame(id: $id) {
            sessions(filter: $filter, nextToken: $nextToken) {
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
        }
        """

        conditions: list[dict[str, Any]] = []

        if version_id:
            conditions.append({"gameVersionId": {"eq": version_id}})
        if date_from:
            conditions.append({"startedAt": {"ge": self._date_to_iso_start(date_from)}})
        if date_to:
            conditions.append({"startedAt": {"le": self._date_to_iso_end(date_to)}})

        if len(conditions) == 0:
            filter_obj = None
        elif len(conditions) == 1:
            filter_obj = conditions[0]
        else:
            filter_obj = {"and": conditions}

        sessions: list[dict] = []
        next_token = None

        while True:
            data = self._graphql(
                query,
                {
                    "id": game_id,
                    "filter": filter_obj,
                    "nextToken": next_token,
                },
            )

            game = data.get("getGame")
            if not game:
                raise ValueError(f"Game not found: {game_id}")

            page = game["sessions"]
            sessions.extend(page["items"])

            next_token = page.get("nextToken")
            if not next_token:
                break

        return sessions


    def _fetch_events_for_one_session(self, session_id: str) -> list[dict]:
        query = """
        query GetSessionWithEvents($id: ID!, $nextToken: String) {
          getSession(id: $id) {
            events(nextToken: $nextToken) {
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
        }
        """
        session_events = []
        next_token = None

        while True:
            data = self._graphql(query, {"id": session_id, "nextToken": next_token})
            events_block = data["getSession"]["events"]
            session_events.extend(events_block["items"])
            next_token = events_block.get("nextToken")
            if not next_token:
                break

        return session_events

    def _use_event_cache(self) -> bool:
        return bool(getattr(self._config, "IOCANE_USE_EVENT_CACHE", False))



    def _get_event_cache_dir(self) -> str:
        # user override
        cfg_dir = getattr(self._config, "IOCANE_EVENT_CACHE_DIR", None)
        if cfg_dir:
            base_dir = cfg_dir
        else:
            base_dir = user_cache_dir("CFGpy", "ComDePriLab")  # e.g. ~/.cache/CFGpy/
        self.log_info(f"Using cache dir: {base_dir}")
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def _get_event_cache_path(self) -> str:
        game_id = self._game_id or "unknown_game"
        version = CFGPY_VERSION.replace(".", "_")
        base_dir = self._get_event_cache_dir()
        return os.path.join(base_dir, f"iocane_event_cache_{game_id}_{version}.json")

    def _load_event_cache(self) -> dict[str, list[dict]]:
        if not self._use_event_cache():
            return {}

        path = self._get_event_cache_path()
        if not os.path.exists(path):
            return {}

        self.log_info(f"Loading IOCANE event cache from {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                sessions = json.load(f)

            if not isinstance(sessions, dict):
                return {}


            return sessions

        except Exception as e:
            self.log_warning(f"Failed to load event cache: {e}")
            return {}

    def _save_event_cache(self, cache: dict[str, list[dict]]) -> None:
        if not self._use_event_cache():
            return

        path = self._get_event_cache_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

        os.replace(tmp_path, path)
        self.log_info(f"Saved IOCANE event cache to {path}")

    def _fetch_events_for_sessions(self, sessions: list[dict], *, verbose: bool = False) -> list[dict]:
        if not sessions:
            return []

        session_ids = [str(s["id"]) for s in sessions if s.get("id")]

        cache = self._load_event_cache()

        cached_events: list[dict] = []
        missing_session_ids: list[str] = []

        for sid in session_ids:
            if self._use_event_cache() and sid in cache:
                cached_events.extend(cache[sid])
            else:
                missing_session_ids.append(sid)

        self.log_info(
            f"Event cache hit for {len(session_ids) - len(missing_session_ids)} / "
            f"{len(session_ids)} sessions."
        )

        fetched_by_session: dict[str, list[dict]] = {}

        if missing_session_ids:
            self.log_info(f"Fetching events for {len(missing_session_ids)} uncached sessions.")

            max_workers = min(16, max(1, len(missing_session_ids)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._fetch_events_for_one_session, sid): sid
                    for sid in missing_session_ids
                }

                iterator = as_completed(futures)
                if verbose:
                    iterator = tqdm.tqdm(
                        iterator,
                        total=len(futures),
                        desc="sessions",
                        unit="session",
                    )

                for fut in iterator:
                    sid = futures[fut]
                    events = fut.result()
                    fetched_by_session[sid] = events

            if self._use_event_cache():
                cache.update(fetched_by_session)
                self._save_event_cache(cache)

        all_events = cached_events
        for events in fetched_by_session.values():
            all_events.extend(events)

        all_events.sort(key=lambda x: x.get("occurredAt", ""))
        return all_events


    def _fetch_one_player(self, pid: str) -> tuple[str, dict | None]:
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
        data = self._graphql(query, {"id": pid})
        player = data.get("getPlayer")
        return pid, player

    def _fetch_players_for_sessions(self, sessions: list[dict], *, verbose: bool = False) -> dict[str, dict]:
        player_ids = sorted({s["playerId"] for s in sessions if s.get("playerId")})
        if not player_ids:
            return {}

        player_map = {}
        max_workers = min(16, max(1, len(player_ids)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._fetch_one_player, pid): pid for pid in player_ids}
            iterator = as_completed(futures)
            if verbose:
                iterator = tqdm.tqdm(iterator, total=len(futures), desc="players", unit="player")

            for fut in iterator:
                pid, player = fut.result()
                if player:
                    player_map[pid] = player

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
                self._config.RAW_GAME_VERSION: sess.get("gameVersionId", "") or "",
                self._config.RAW_PLAYER_ID: sess.get("playerId", "") or "",
                # self._config.RAW_PLAYER_EXTERNAL_ID: json.loads(player.get("metadata","{}")).get("externalId", "") or "",
                "playerMetadata": player.get("metadata") if player.get("metadata") is not None else {},
                "sessionMetadata": json.loads(sess.get("metadata")) if sess.get("metadata") is not None else {},
                self._config.EVENT_ID_KEY: ev.get("id", ""),
                self._config.EVENT_TYPE: ev.get("type", "").lower(),
                self._config.RAW_USER_TIME: ev.get("occurredAt", ""),
                self._config.EVENT_CUSTOM_DATA_KEY: ev.get("data") if ev.get("data") is not None else "",
            }
            # ensure user time includes decimal points for seconds, for consistency with other backends:
            if row[self._config.RAW_USER_TIME] and ('.' not in row[self._config.RAW_USER_TIME]):
                row[self._config.RAW_USER_TIME] = row[self._config.RAW_USER_TIME].replace("Z",".000Z")
            # check if there are missing zeros between the '.' and 'Z' and add them if needed (e.g. .1Z -> .100Z)
            if row[self._config.RAW_USER_TIME] and ('.' in row[self._config.RAW_USER_TIME]):
                time_part = row[self._config.RAW_USER_TIME].split("T")[1]
                if time_part.endswith("Z") and len(time_part.split(".")[1].rstrip("Z")) < 3:
                    missing_zeros = 3 - len(time_part.split(".")[1].rstrip("Z"))
                    row[self._config.RAW_USER_TIME] = row[self._config.RAW_USER_TIME].replace("Z", "0" * missing_zeros + "Z")
            if row["sessionMetadata"]:
                row[self._config.RAW_PLAYER_CUSTOM_DATA] = json.dumps(row["sessionMetadata"].get("customData")) if (
                        row["sessionMetadata"].get("customData") is not None) else "{}"

            row[self._config.RAW_PLAYER_BIRTHDATE] = player.get("birthDate", None)
            row[self._config.RAW_PLAYER_REGION] = player.get("region", None)
            row[self._config.RAW_PLAYER_COUNTRY] = player.get("country", None)
            row[self._config.RAW_PLAYER_GENDER] = player.get("gender", None)
            row[self._config.RAW_PLAYER_EXTERNAL_ID] = self._get_player_external_id(player)
            # remove the general metadata fields
            row.pop("playerMetadata", None)
            row.pop("sessionMetadata", None)
            rows.append(row)
        return rows

    def _get_player_external_id(self, player: dict) -> Optional[str]:
        metadata = player.get("metadata", None)
        if not metadata:
            return None
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                return None
        if isinstance(metadata, dict):
            return metadata.get("externalId", None)
        return None

    def _order_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._config.DOWNLOADER_FIELD_ORDER:
            ordered_cols = [col for col in self._config.DOWNLOADER_FIELD_ORDER if col in df.columns]
            extra_cols = [col for col in df.columns if col not in ordered_cols]
            return df[ordered_cols + extra_cols]
        else:
            return df

    def _create_df(self, rows: list[dict[str, str]]) -> pd.DataFrame:
        raw_cols = [
            self._config.RAW_GAME_VERSION,
            self._config.RAW_PLAYER_ID,
            self._config.EVENT_ID_KEY,
            self._config.EVENT_TYPE,
            self._config.RAW_USER_TIME,
            self._config.EVENT_CUSTOM_DATA_KEY,
            self._config.RAW_PLAYER_CUSTOM_DATA,
            self._config.RAW_PLAYER_BIRTHDATE,
            self._config.RAW_PLAYER_REGION,
            self._config.RAW_PLAYER_COUNTRY,
            self._config.RAW_PLAYER_GENDER,
            self._config.RAW_PLAYER_EXTERNAL_ID]
        if len(rows) == 0:
            self.log_warning("No events found for the specified game and filters. Returning an empty DataFrame.")
        df = pd.DataFrame(rows, columns=raw_cols)
        raw_cols = set(raw_cols)  # all columns in the df are considered "raw" at this stage, since we haven't done any parsing yet
        # check if rows include keys that were removed and warn
        ignored_keys = set()
        for row in rows:
            for key in row.keys():
                if key not in raw_cols:
                    ignored_keys.add(key)
        if ignored_keys:
            self.log_warning(f"Some keys in the retrieved data were not included in the output DataFrame: {ignored_keys}. "
                             f"These keys were ignored and will not be included in the output. "
                             f"Consider adding them to the config.DOWNLOADER_FIELD_ORDER if you want them included in the output.")

        return df
