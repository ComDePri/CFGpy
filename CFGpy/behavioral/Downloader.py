import csv
import requests
from tqdm import tqdm
import pandas as pd

PER_PAGE = 500

COMMON_FIELDS = ("id", "serverTime", "userTime", "gameVersion", "type", "coordinates", "section")
COMMON_PLAYER_FIELDS = ("playerId", "playerBirthdate", "playerRegion", "playerCountry", "playerGender",
                        "playerExternalId", "playerCustomData")
FIELD_ORDER = (
    "id",
    "serverTime",
    "userTime",
    "gameVersion",
    "playerId",
    "playerBirthdate",
    "playerRegion",
    "playerCountry",
    "playerGender",
    "playerExternalId",
    "playerCustomData",
    "type",
    "coordinates",
    "section",
    "customData.startPosition",
    "customData.shapeIndices",
    "customData.shapeIndex",
    "customData.isSelected",
    "customData.newShape",
    "customData.timeSinceLastMouseUp",
    "customData.time",
    "customData.shapes",
    "customData.shape",
    "customData.endPosition",
)

DEFAULT_OUTPUT_FILENAME = "event.csv"


class Downloader:
    def __init__(self, csv_url, output_filename=DEFAULT_OUTPUT_FILENAME, common_fields=COMMON_FIELDS,
                 common_player_fields=COMMON_PLAYER_FIELDS, field_order=FIELD_ORDER):
        self.json_url = self.convert_url(csv_url)  # runs on init so that ValueError is raised immediately if needed
        self.output_filename = output_filename
        self.common_fields = common_fields
        self.common_player_fields = common_player_fields
        self.field_order = field_order

        self.players = dict()
        self.custom_data_fields = set()

    def download(self):
        input_json = self.get_events()
        output_json = self.create_output(input_json)
        self.write_csv(output_json)

        return pd.read_csv(self.output_filename)  # why not return output_json? see to-do in create_output

    @staticmethod
    def convert_url(csv_url):
        # Expecting address like:
        # https://api.creativeforagingtask.com/v1/event.csv?game=c9d8979c-94ad-498f-8d2b-a37cff3c5b41&gameVersion=40f2894d-1891-456b-af26-a386c6111287&entityType=event

        if csv_url.find("/event.csv") == -1:
            raise ValueError(f"URL is incorrect: '{csv_url}'")

        return csv_url.replace("/event.csv", "/event.json")

    def get_events(self):
        print("Download events...")
        input_json = []
        page_count = int(requests.get(self.json_url, {"page": 1, "perPage": PER_PAGE}).headers['x-page-count'])
        # TODO: see if a similar request can give the total events number and avoid downloading if the output file
        #  exists with the correct length

        for page in tqdm(range(1, page_count + 1), desc="pages"):
            r = requests.get(self.json_url, {"page": page, "perPage": PER_PAGE})
            page_json = r.json()
            input_json = input_json + page_json

        print(f"\nDone. Found {len(input_json)} events")
        return input_json

    def _get_player(self, player_id):
        """
        Get the player in cache, or make a network request
        """

        used_net_request = False
        if player_id in self.players:
            player = self.players[player_id]
        else:
            used_net_request = True
            r = requests.get(f"https://api.creativeforagingtask.com/v1/player/{player_id}")
            player = r.json()
            self.players[player_id] = player

        return player, used_net_request

    def create_output(self, input_json):
        # TODO: this is not really creating the final output, since write_csv() does more work to format it. The logic
        #  from write_csv may be implemented here to create an equivalent pandas DataFrame, and then write_csv can be
        #  replaced by calling to_csv on that DataFrame. This will also make the return statement in self.download
        #  cleaner: instead of writing a CSV and then reading it just to be able to return it in the correct format,
        #  self will have a correctly formatted DataFrame to return from download

        output_json = []
        requested_players = []

        print("\nHandling events...")
        for event in tqdm(input_json, desc="events"):
            # Filter out common fields
            output_json_record = {k: v for (k, v) in event.items() if k in self.common_fields}

            # Handle custom data
            if "customData" in event:
                if isinstance(event["customData"], dict):
                    # Add each key as a custom data field
                    for key, value in event["customData"].items():
                        keyName = f"customData.{key}"
                        self.custom_data_fields.add(keyName)
                        output_json_record[keyName] = value
                else:
                    self.custom_data_fields.add("customData")

            # Handle player
            player_id = event["player"]
            player, used_net_req = self._get_player(player_id)
            if used_net_req:
                requested_players.append(player_id)

            output_json_record["playerId"] = player_id
            output_json_record["playerBirthdate"] = player.get("birthDate")
            output_json_record["playerRegion"] = player.get("region")
            output_json_record["playerCountry"] = player.get("country")
            output_json_record["playerGender"] = player.get("gender")
            output_json_record["playerExternalId"] = player.get("externalId")
            output_json_record["playerCustomData"] = player.get("customData")

            output_json.append(output_json_record)

        print(f"\nPlayers that required a network request: {requested_players}")

        return output_json

    def write_csv(self, output_json):
        """
        Writes the CSV while ensuring existence and oder of all fields defined in self.field_order
        """
        extra_fields = set(self.custom_data_fields) - set(self.field_order)
        extra_fields_msg = f"Found extra fields: {extra_fields}" if extra_fields else "No extra fields found"
        print(extra_fields_msg)

        all_fields = self.field_order + tuple(extra_fields)
        with open(self.output_filename, "w", newline="") as output_file:
            output_file_writer = csv.DictWriter(output_file, fieldnames=all_fields, quoting=csv.QUOTE_ALL)

            output_file_writer.writeheader()
            for output_json_record in output_json:
                output_file_writer.writerow(output_json_record)

        print(f"Wrote CSV to {self.output_filename}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Download CSV from RedMetrics")
    parser.add_argument("url", help='Web address of the "Download all pages as CSV"')
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILENAME, dest="output_filename",
                        help='Filename of output CSV')
    args = parser.parse_args()

    d = Downloader(args.url, args.output_filename)
    d.download()
