import csv
import requests
from tqdm import tqdm
import pandas as pd
from CFGpy.behavioral._consts import (DOWNLOADER_OUTPUT_FILENAME, NO_DOWNLOADER_URL_ERROR, DOWNLOADER_URL_NO_CSV_ERROR,
                                      TWO_DOWNLOADER_URL_ERROR, EVENTS_PER_PAGE, PAGE_REPETITION_LIMIT_REACHED)
from CFGpy.behavioral._utils import CFGPipelineException
from CFGpy.behavioral import Configuration


class Downloader:
    def __init__(self, csv_url: str | None = None, output_filename: str = DOWNLOADER_OUTPUT_FILENAME,
                 config: Configuration = None) -> None:
        """
        Init a Downloader object.
        :param csv_url: Web address of the "Download all pages as CSV" in RedMetrics. Optional. If None, URL is expected
                         as part of config.
        :param output_filename: filename for output.
        :param config: a Configuration file. If this defines a RedMetrics URL, `csv_url` shouldn't.
        """
        self.config = config if config is not None else Configuration.default()
        self.csv_url = csv_url
        self._validate_url()
        self.json_url = self.csv_url.replace("/event.csv", "/event.json")
        self.downloaded_events_json = []
        self.output_filename = output_filename

        self.players = dict()
        self.custom_data_fields = set()
        self.required_net_request = []
        self.extra_fields = set()

    def download(self, verbose: bool = False) -> pd.DataFrame:
        self.download_events_json(verbose)
        output_json = self.create_output(verbose)
        self.write_csv(output_json, verbose)
        self.dump_config()

        return pd.read_csv(self.output_filename)  # why not return output_json? see to-do in create_output

    def _validate_url(self) -> None:
        # at least one URL should not be None:
        if self.csv_url is None and self.config.RED_METRICS_CSV_URL is None:
            raise ValueError(NO_DOWNLOADER_URL_ERROR)

        # at most one URL should not be None:
        if self.csv_url is not None and self.config.RED_METRICS_CSV_URL is not None:
            raise ValueError(TWO_DOWNLOADER_URL_ERROR)

        self.csv_url = self.config.RED_METRICS_CSV_URL if self.csv_url is None else self.csv_url

        # URL should point to an event.csv file:
        if not "/event.csv" in self.csv_url:
            raise ValueError(DOWNLOADER_URL_NO_CSV_ERROR.format(self.csv_url))

    def _get_page(self, page_i: int) -> requests.Response:
        """
        Gets a page of events in JSON format, using the RedMetrics1 API (https://github.com/CyberCRI/RedMetrics/blob/master/API.md)
        :param page_i: page index
        :return: a Response object, see RedMetrics1 API for info.
        """
        params = {"page": page_i, "perPage": EVENTS_PER_PAGE, 'orderBy': 'userTime:asc'}
        response = requests.get(self.json_url, params)
        response.raise_for_status()

        return response

    def download_events_json(self, verbose: bool = False) -> None:
        """
        Populates self.downloaded_events_json from RedMetrics1.
        :param verbose: whether to output progress info during download.
        """
        if verbose:
            print("Download events...")

        first_page = self._get_page(1)
        page_count = int(first_page.headers["x-page-count"])
        total_results = int(first_page.headers['x-Total-Count'])

        page_iterator = range(1, page_count + 1)
        if verbose:
            page_iterator = tqdm(page_iterator, desc="page")
        for page_i in page_iterator:
            goal_n_events = page_i * EVENTS_PER_PAGE if page_i < page_count else total_results
            self.downloaded_events_json += self._get_page_events(page_i, goal_n_events)

            assert self.downloaded_events_json == [event for i, event in enumerate(self.downloaded_events_json) if
                                                   event not in self.downloaded_events_json[:i]]
            # TODO: assertion can be dropped if sure that no duplicates find their way in

    def _get_page_events(self, page_i: int, goal_n_events: int, repetition_limit: int = 5) -> list:
        """
        Retrieves the page's events from RedMetrics1.
        Deals with inconsistent paging by repeatedly downloading this and the previous page, until all events are
        found or the repetition limit is reached (in which case, raises an error).
        :param page_i: page index.
        :param goal_n_events: number of events that should be reached with this page.
        :param repetition_limit: maximum number of retries per page # TODO: move this parameter to config
        :return: a list of events.
        """
        if page_i < 1:
            raise ValueError(f"RedMetrics1 page indexing is 1-based. Got invalid page index {page_i}")

        existing_event_ids = [event['id'] for event in self.downloaded_events_json]
        n_missing_events = goal_n_events - len(self.downloaded_events_json)
        discovered_events = []
        for repetition in range(repetition_limit + 1):  # add 1 because the first iteration is not a repetition
            extra_events = []
            if page_i > 1 and repetition:  # no need to check previous page on the first try, only in repetitions
                extra_events = self._get_page(page_i - 1).json()
            new_events = self._get_page(page_i).json() + extra_events

            # add events discovered in this iteration and remove duplicates:
            discovered_events += [event for event in new_events if event['id'] not in existing_event_ids]
            discovered_events = [event for i, event in enumerate(discovered_events) if
                                 event not in discovered_events[:i]]

            if len(discovered_events) == n_missing_events:
                return discovered_events

        raise CFGPipelineException(PAGE_REPETITION_LIMIT_REACHED.format(page_i, repetition_limit))

    def _get_player(self, player_id: str) -> dict:
        """
        Get the player in cache, or make a network request
        """
        if player_id in self.players:
            player = self.players[player_id]
        else:
            r = requests.get(self.config.DOWNLOAD_PLAYER_REQUEST.format(player_id))
            player = r.json()
            self.players[player_id] = player
            self.required_net_request.append(player_id)

        return player

    def create_output(self, verbose=False) -> list:
        # TODO: this is not really creating the final output, since write_csv() does more work to format it. The logic
        #  from write_csv may be implemented here to create an equivalent pandas DataFrame, and then write_csv can be
        #  replaced by calling to_csv on that DataFrame. This will also make the return statement in self.download
        #  cleaner: instead of writing a CSV and then reading it just to be able to return it in the correct format,
        #  self will have a correctly formatted DataFrame to return from download
        #  after doing this, Downloader should only write to disk using a dump method, which will also dump config

        output_json = []

        event_iterator = self.downloaded_events_json
        if verbose:
            print("\nHandling events...")
            event_iterator = tqdm(event_iterator, desc="events")

        for event in event_iterator:
            # filter to common fields
            output_json_record = {k: v for (k, v) in event.items() if k in self.config.DOWNLOADER_COMMON_FIELDS}

            # add event's custom data fields
            if self.config.EVENT_CUSTOM_DATA_KEY in event:
                if isinstance(event[self.config.EVENT_CUSTOM_DATA_KEY], dict):
                    # Add each key as a custom data field
                    for key, value in event[self.config.EVENT_CUSTOM_DATA_KEY].items():
                        keyName = f"{self.config.EVENT_CUSTOM_DATA_KEY}.{key}"
                        self.custom_data_fields.add(keyName)
                        output_json_record[keyName] = value
                else:
                    self.custom_data_fields.add(self.config.EVENT_CUSTOM_DATA_KEY)

            # add player's fields
            player_id = event[self.config.EVENT_PLAYER_ID_KEY]
            player = self._get_player(player_id)

            output_json_record[self.config.RAW_PLAYER_ID] = player_id
            output_json_record[self.config.RAW_PLAYER_BIRTHDATE] = player.get("birthDate")
            output_json_record[self.config.RAW_PLAYER_REGION] = player.get("region")
            output_json_record[self.config.RAW_PLAYER_COUNTRY] = player.get("country")
            output_json_record[self.config.RAW_PLAYER_GENDER] = player.get("gender")
            output_json_record[self.config.RAW_PLAYER_EXTERNAL_ID] = player.get("externalId")
            output_json_record[self.config.RAW_PLAYER_CUSTOM_DATA] = player.get("customData")

            output_json.append(output_json_record)

        return output_json

    def dump_config(self) -> None:
        self.config.to_yaml(self.output_filename)

    def write_csv(self, output_json, verbose=False) -> None:
        """
        Writes the CSV while ensuring existence and oder of all fields defined in self.config.DOWNLOADER_FIELD_ORDER
        """
        self.extra_fields = set(self.custom_data_fields) - set(self.config.DOWNLOADER_FIELD_ORDER)

        all_fields = self.config.DOWNLOADER_FIELD_ORDER + tuple(self.extra_fields)
        with open(self.output_filename, "w", newline="", encoding="utf-8") as output_file:
            output_file_writer = csv.DictWriter(output_file, fieldnames=all_fields, quoting=csv.QUOTE_ALL)

            output_file_writer.writeheader()
            for output_json_record in output_json:
                output_file_writer.writerow(output_json_record)

        if verbose:
            print(f"Wrote CSV to {self.output_filename}")

    def get_net_requested_players(self) -> list:
        """
        Returns the list of players that required a network request.
        """
        return self.required_net_request

    def get_extra_fields(self) -> set:
        """
        Returns the set of extra/unexpected fields found in the data.
        """
        return self.extra_fields
