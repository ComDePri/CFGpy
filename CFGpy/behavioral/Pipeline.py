from datetime import datetime, timezone
from CFGpy.behavioral import DataRetriever, RM1DumpDataRetriever, RedMetrics2DataRetriever, Parser, PostParser, \
    FeatureExtractor, Configuration, RedMetrics1Downloader, CFGAppSyncDataRetriever, LocalDataRetriever
from CFGpy.behavioral._consts import DEFAULT_FINAL_OUTPUT_FILENAME, RM1, RM1_NAS_DUMP, RM2, \
    UNSUPPORTED_DATA_SOURCE_ERROR, APPSync, VALID_DATA_SOURCES, LOCAL
from CFGpy.behavioral._utils import CFGPipelineException


class Pipeline:
    def __init__(self, game_name: str | None = None, game_id: str | None = None,
                 game_version_ids: list[str] | None = None,
                 output_filename=DEFAULT_FINAL_OUTPUT_FILENAME, config: Configuration = None,
                 input_events_csv_path: str | None = None) -> None:

        self._game_name = game_name
        self._game_id: str = game_id
        self._game_version_ids = game_version_ids
        self._input_events_csv_path = input_events_csv_path

        self.output_filename = output_filename
        self.config = config or Configuration.default()

        self.data_retriever = None
        self.raw_data = None
        self.parser: Parser = None
        self.parsed_data = None
        self.postparser: PostParser = None
        self.postparsed_data = None
        self.feature_extractor: FeatureExtractor = None
        self.features_df = None

    def _get_now_str(self) -> str:
        """
        Returns a string representation of the current time, formatted like server's time (given in self.config).

        Python's datetime only allows specifying sub-second precision in microseconds (6 decimal places), but RedMetrics
        URL only accept milliseconds (3 decimal places). Therefore, if the server's time format contains microseconds,
        we manually replace that with milliseconds, to accommodate RedMetrics.
        """
        now = datetime.now(timezone.utc)
        now_str = (
            now.strftime(
                self.config.SERVER_DATE_FORMAT
                .replace("%f", "{}"))  # plants a placeholder instead of microseconds
            .format(f"{now.microsecond // 1000:0>3}")  # fills in millisecond info, 0-padded to three digits
        )
        return now_str

    def _add_input_params_to_config(self):
        self.config.GAME_NAME = self.data_retriever._game_name
        self.config.GAME_ID = self.data_retriever._game_id
        if self.config.DATA_SOURCE == RM1_NAS_DUMP:
            self.config.GAME_VERSION_IDS = self.data_retriever._game_version_ids

    def _get_data_retriever(self) -> DataRetriever:
        if self.config.DATA_SOURCE == RM1_NAS_DUMP:
            return RM1DumpDataRetriever(game_name=self._game_name, game_id=self._game_id,
                                        game_version_ids=self._game_version_ids, config=self.config,
                                        output_filename=self.output_filename)
        elif self.config.DATA_SOURCE == RM2:
            return RedMetrics2DataRetriever(game_name=self._game_name, game_id=self._game_id, config=self.config,
                                            output_filename=self.output_filename)
        elif self.config.DATA_SOURCE == RM1:
            return RedMetrics1Downloader(csv_url=self.config.RED_METRICS_CSV_URL, config=self.config,
                                         output_filename=self.output_filename)
        elif self.config.DATA_SOURCE == APPSync:
            return CFGAppSyncDataRetriever(game_name=self._game_name, game_id=self._game_id, config=self.config,
                                           output_filename=self.output_filename)
        elif self.config.DATA_SOURCE == LOCAL:
            return LocalDataRetriever(config=self.config, output_filename=self.output_filename,
                                      events_csv_path=self._input_events_csv_path)
        else:
            raise ValueError(UNSUPPORTED_DATA_SOURCE_ERROR.format(self.config.DATA_SOURCE))

    def _retrieve_data(self, verbose):
        """
        This method contains the data retrieval process exclusively. This can be overridden by deriving classes.
        :param verbose: whether to print info during the data retrieval process
        :return: raw data
        """
        return self.data_retriever.retrieve_data(verbose=verbose)

    def retrieve_data(self, verbose=True):
        """
        Wraps raw data retrieval with extra necessary functionality.
        If you wish to override the data retrieval method, override _retrieve_data, not this.
        :param verbose: whether to print info during the data retrieval process
        """
        if self.raw_data is not None:
            raise CFGPipelineException("Raw data has already been retrieved")

        self.data_retriever = self._get_data_retriever()
        self._add_input_params_to_config()

        if verbose:
            print("Retrieving raw data...")

        self.raw_data = self._retrieve_data(verbose=verbose)
        self.data_retriever.dump(verbose=verbose)

    def _parse(self):
        """
        This method contains the parsing process exclusively. This can be overridden by deriving classes.
        :return: parsed data
        """
        self.parser = Parser(raw_data=self.raw_data, config=self.config)
        return self.parser.parse()

    def parse(self, verbose):
        """
        Wraps data parsing with extra necessary functionality.
        If you wish to override the parsing method, override _parse, not this.
        :param verbose: whether to print info during the parsing process
        """
        if self.raw_data is None:
            raise CFGPipelineException("Raw data has to be retrieved before parsing")
        if self.parsed_data is not None:
            raise CFGPipelineException("Data already parsed")

        if verbose:
            print("Parsing...")
        self.parsed_data = self._parse()
        self.parser.dump(name=self.output_filename)

    def _postparse(self):
        """
        This method contains the post-parsing process exclusively. This can be overridden by deriving classes.
        :return: post-parsed data
        """
        self.postparser = PostParser(parsed_data=self.parsed_data, config=self.config)
        return self.postparser.postparse()

    def postparse(self, verbose):
        """
        Wraps data post-parsing with extra necessary functionality.
        If you wish to override the post-parsing method, override _postparse, not this.
        :param verbose: whether to print info during the post-parsing process
        """
        if self.parsed_data is None:
            raise CFGPipelineException("Data has to be parsed before post-parsing (duh!)")
        if self.postparsed_data is not None:
            raise CFGPipelineException("Data already post-parsed")

        if verbose:
            print("Post-parsing...")
        self.postparsed_data = self._postparse()

    def _extract_features(self, verbose):
        self.feature_extractor = FeatureExtractor(preprocessed_data=self.postparsed_data, config=self.config)
        return self.feature_extractor.extract(verbose)

    def extract_features(self, verbose):
        features_output_path = f"{self.output_filename}_features.csv"
        if self.postparsed_data is None:
            raise CFGPipelineException("Data has to be post-parsed before feature extraction")
        if self.features_df is not None:
            raise CFGPipelineException("Features already extracted")

        if verbose:
            print("Calculating measures...")

        self.features_df = self._extract_features(verbose)
        self.feature_extractor.dump(name=self.output_filename, with_exclusions=True)

        if verbose:
            print(f"Results written successfully to: {features_output_path}")

    def run_pipeline(self, verbose=True):
        self.retrieve_data(verbose=verbose)
        self.parse(verbose=verbose)
        self.postparse(verbose=verbose)
        self.extract_features(verbose=verbose)
        return self.features_df


def main():
    import argparse

    argparser = argparse.ArgumentParser(description="Run CFG behavioral data pipeline")
    # can give any of the valid data sources as argument to override the config data source
    argparser.add_argument("--data-source", choices=VALID_DATA_SOURCES,
                           help="The data source to retrieve data from. Should be provided only if it doesn't appear in the config file.")
    argparser.add_argument("--game-name", help='The name of the name.')
    argparser.add_argument("--game-id", help='The id of the game.')
    argparser.add_argument("--game-version-ids", nargs="+",
                           help='A list of the game version ids that you want to retrieve.')
    argparser.add_argument("--config-path", help='The path to the yml file that contains the configuration')
    argparser.add_argument("--events-csv-path",
                           help='The path to the events CSV file. Only needed if the data source is Local or if you want to provide a custom path to the events CSV file instead of providing it in the config.')
    argparser.add_argument("-o", "--output", default=DEFAULT_FINAL_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output CSV')
    args = argparser.parse_args()

    config: Configuration | None = Configuration.from_yaml(
        yaml_path=args.config_path) if args.config_path else Configuration.default()
    if args.data_source:
        config.DATA_SOURCE = args.data_source

    pl = Pipeline(game_name=args.game_name, game_id=args.game_id, game_version_ids=args.game_version_ids,
                  output_filename=args.output_filename, config=config, input_events_csv_path=args.events_csv_path)

    pl.run_pipeline()


if __name__ == '__main__':
    main()
