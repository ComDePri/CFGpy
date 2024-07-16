import csv
import requests
from tqdm import tqdm
import pandas as pd
from CFGpy.behavioral._consts import DOWNLOADER_OUTPUT_FILENAME, DOWNLOADER_URL_ERROR, EVENTS_PER_PAGE
from CFGpy.behavioral import config


class Downloader:
    def __init__(self, csv_url, output_filename=DOWNLOADER_OUTPUT_FILENAME):
        self.json_url = self.convert_url(csv_url)  # runs on init so that ValueError is raised immediately if needed
        self.output_filename = output_filename
        self.common_fields = config.DOWNLOADER_COMMON_FIELDS
        self.field_order = config.DOWNLOADER_FIELD_ORDER

        self.players = dict()
        self.custom_data_fields = set()
        self.required_net_request = []
        self.extra_fields = set()

    def download(self, verbose=False):
        input_json = self.get_events(verbose)
        output_json = self.create_output(input_json, verbose)
        self.write_csv(output_json, verbose)

        return pd.read_csv(self.output_filename)  # why not return output_json? see to-do in create_output

    @staticmethod
    def convert_url(csv_url):
        # Expecting address like:
        # https://api.creativeforagingtask.com/v1/event.csv?game=c9d8979c-94ad-498f-8d2b-a37cff3c5b41&gameVersion=40f2894d-1891-456b-af26-a386c6111287&entityType=event

        if csv_url.find("/event.csv") == -1:
            raise ValueError(DOWNLOADER_URL_ERROR.format(csv_url))

        return csv_url.replace("/event.csv", "/event.json")

    def get_events(self, verbose=False):
        if verbose:
            print("Download events...")
        input_json = []
        page_count = int(requests.get(self.json_url, {"page": 1, "perPage": EVENTS_PER_PAGE}).headers['x-page-count'])
        # TODO: see if a similar request can give the total events number and avoid downloading if the output file
        #  exists with the correct length

        page_iterator = range(1, page_count + 1)
        if verbose:
            page_iterator = tqdm(page_iterator, desc="pages")
        for page in page_iterator:
            r = requests.get(self.json_url, {"page": page, "perPage": EVENTS_PER_PAGE})
            page_json = r.json()
            input_json = input_json + page_json

        return input_json

    def _get_player(self, player_id):
        """
        Get the player in cache, or make a network request
        """
        if player_id in self.players:
            player = self.players[player_id]
        else:
            r = requests.get(config.DOWNLOAD_PLAYER_REQUEST.format(player_id))
            player = r.json()
            self.players[player_id] = player
            self.required_net_request.append(player_id)

        return player

    def create_output(self, input_json, verbose=False):
        # TODO: this is not really creating the final output, since write_csv() does more work to format it. The logic
        #  from write_csv may be implemented here to create an equivalent pandas DataFrame, and then write_csv can be
        #  replaced by calling to_csv on that DataFrame. This will also make the return statement in self.download
        #  cleaner: instead of writing a CSV and then reading it just to be able to return it in the correct format,
        #  self will have a correctly formatted DataFrame to return from download

        output_json = []

        event_iterator = input_json
        if verbose:
            print("\nHandling events...")
            event_iterator = tqdm(event_iterator, desc="events")

        for event in event_iterator:
            # Filter out common fields
            output_json_record = {k: v for (k, v) in event.items() if k in self.common_fields}

            # Handle custom data
            if config.EVENT_CUSTOM_DATA_KEY in event:
                if isinstance(event[config.EVENT_CUSTOM_DATA_KEY], dict):
                    # Add each key as a custom data field
                    for key, value in event[config.EVENT_CUSTOM_DATA_KEY].items():
                        keyName = f"{config.EVENT_CUSTOM_DATA_KEY}.{key}"
                        self.custom_data_fields.add(keyName)
                        output_json_record[keyName] = value
                else:
                    self.custom_data_fields.add(config.EVENT_CUSTOM_DATA_KEY)

            # Handle player
            player_id = event[config.EVENT_PLAYER_ID_KEY]
            player = self._get_player(player_id)

            output_json_record[config.RAW_PLAYER_ID] = player_id
            output_json_record[config.RAW_PLAYER_BIRTHDATE] = player.get("birthDate")
            output_json_record[config.RAW_PLAYER_REGION] = player.get("region")
            output_json_record[config.RAW_PLAYER_COUNTRY] = player.get("country")
            output_json_record[config.RAW_PLAYER_GENDER] = player.get("gender")
            output_json_record[config.RAW_PLAYER_EXTERNAL_ID] = player.get("externalId")
            output_json_record[config.RAW_PLAYER_CUSTOM_DATA] = player.get("customData")

            output_json.append(output_json_record)

        return output_json

    def write_csv(self, output_json, verbose=False):
        """
        Writes the CSV while ensuring existence and oder of all fields defined in self.field_order
        """
        self.extra_fields = set(self.custom_data_fields) - set(self.field_order)

        all_fields = self.field_order + tuple(self.extra_fields)
        with open(self.output_filename, "w", newline="") as output_file:
            output_file_writer = csv.DictWriter(output_file, fieldnames=all_fields, quoting=csv.QUOTE_ALL)

            output_file_writer.writeheader()
            for output_json_record in output_json:
                output_file_writer.writerow(output_json_record)

        if verbose:
            print(f"Wrote CSV to {self.output_filename}")

    def get_net_requested_players(self):
        """
        Returns the list of players that required a network request.
        """
        return self.required_net_request

    def get_extra_fields(self):
        """
        Returns the set of extra/unexpected fields found in the data.
        """
        return self.extra_fields


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description="Download CSV from RedMetrics")
    argparser.add_argument("url", help='Web address of the "Download all pages as CSV"')
    argparser.add_argument("-o", "--output", default=DOWNLOADER_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output CSV')
    args = argparser.parse_args()

    d = Downloader(args.url, args.output_filename)
    d.download()
