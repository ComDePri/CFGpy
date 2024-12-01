import json
from CFGpy.behavioral import Downloader, Parser, PostParser, FeatureExtractor, config


class RM2_JSON2CSV(Downloader):
    def __init__(self, rm2_json_filename, *downloader_args, **downloader_kwargs):
        super().__init__(*downloader_args, **downloader_kwargs)
        self.rm2_json_filename = rm2_json_filename
        self.rm1_format = None

    def convert(self):
        with open(self.rm2_json_filename) as f:
            rm2 = json.load(f)

        game_version = rm2["publisherId"]
        rm1 = []
        for session in rm2["sessions"]:
            for rm2_event in session["events"]:
                rm1.append({
                    "gameVersion": game_version,
                    "player": session["id"],
                    "serverTime": rm2_event['serverTimestamp'],
                    "userTime": rm2_event['userTimestamp'],
                    "type": rm2_event["type"].lower(),
                    "id": rm2_event["id"],
                    "customData": rm2_event["customData"],
                    "playerExternalId": session["externalId"],
                    "playerCustomData": "{}"  # should be session["customData"], but then Parser wouldn't read it a json
                })

        self.rm1_format = rm1

    def create_output(self, verbose):
        output_json = []

        for event in self.rm1_format:
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

            player_id = event[config.EVENT_PLAYER_ID_KEY]
            output_json_record[config.RAW_PLAYER_ID] = player_id
            output_json_record[config.RAW_PLAYER_EXTERNAL_ID] = event["playerExternalId"]
            output_json_record[config.RAW_PLAYER_CUSTOM_DATA] = event["playerCustomData"]
            output_json.append(output_json_record)

        return output_json

    def process_json(self, verbose=True):
        self.convert()
        output_json = self.create_output(verbose)
        self.write_csv(output_json, verbose)


rm2_json_filename = r"/home/royg/home/royg/CFGpy/CFGpy/behavioral/rm2.json"
raw_data_filename = "event.csv"
rm2_json2csv = RM2_JSON2CSV(rm2_json_filename=rm2_json_filename, output_filename=raw_data_filename)
rm2_json2csv.process_json()

parsed_data = Parser.from_file(raw_data_filename)
postparsed_data = PostParser(parsed_data).postparse()
feature_extractor = FeatureExtractor(postparsed_data)
feature_extractor.extract(verbose=True)

features_filename = "features.csv"
feature_extractor.dump(features_filename)
