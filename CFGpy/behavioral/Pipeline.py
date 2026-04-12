import os

import tqdm

from datetime import datetime, timezone
from CFGpy.behavioral import DataRetriever, RM1DumpDataRetriever, RedMetrics2DataRetriever, Parser, PostParser, \
    FeatureExtractor, Configuration, RedMetrics1Downloader, CFGAppSyncDataRetriever, LocalDataRetriever
from CFGpy.behavioral._consts import DEFAULT_FINAL_OUTPUT_FILENAME, RM1, RM1_NAS_DUMP, RM2, \
    UNSUPPORTED_DATA_SOURCE_ERROR, APPSync, VALID_DATA_SOURCES, LOCAL, ARG_TO_CONF_MAP, DATA_SOURCE_ARG, GAME_NAME_ARG, \
    GAME_ID_ARG, GAME_VERSION_IDS_ARG, BEFORE_DATE_ARG, AFTER_DATE_ARG, EVENTS_CSV_PATH_ARG, PARSED_PLAYER_ID_KEY
from CFGpy.behavioral._utils import CFGPipelineException
from CFGpy.behavioral._logging import build_pipeline_logger, HasLogger
from CFGpy.utils import visualization

class Pipeline(HasLogger):
    def __init__(self, game_name: str | None = None, game_id: str | None = None,
                 game_version_ids: list[str] | None = None,
                 output_filename=DEFAULT_FINAL_OUTPUT_FILENAME, config: Configuration = None,
                 input_events_csv_path: str | None = None, verbose=True) -> None:
        self.output_filename = output_filename
        logger = build_pipeline_logger(self.output_filename, verbose=verbose)
        super().__init__(logger)
        self._game_name = game_name
        self._game_id: str = game_id
        self._game_version_ids = game_version_ids
        self._input_events_csv_path = input_events_csv_path


        self.config = config or Configuration.default()

        self.data_retriever = None
        self.raw_data = None
        self.parser: Parser = None
        self.parsed_data = None
        self.postparser: PostParser = None
        self.postparsed_data = None
        self.feature_extractor: FeatureExtractor = None
        self.features_df = None
        self.verbose = verbose


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
                                        output_filename=self.output_filename, logger=self.logger)
        elif self.config.DATA_SOURCE == RM2:
            return RedMetrics2DataRetriever(game_name=self._game_name, game_id=self._game_id, config=self.config,
                                            output_filename=self.output_filename, logger=self.logger)
        elif self.config.DATA_SOURCE == RM1:
            return RedMetrics1Downloader(csv_url=self.config.RED_METRICS_CSV_URL, game_id=self._game_id, config=self.config,
                                         output_filename=self.output_filename, logger=self.logger)
        elif self.config.DATA_SOURCE == APPSync:
            return CFGAppSyncDataRetriever(game_name=self._game_name, game_id=self._game_id, config=self.config,
                                           output_filename=self.output_filename, logger=self.logger)
        elif self.config.DATA_SOURCE == LOCAL:
            return LocalDataRetriever(config=self.config, output_filename=self.output_filename,
                                      events_csv_path=self._input_events_csv_path, logger=self.logger)
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
        self.logger.info("Retrieving data...")

        self.raw_data = self._retrieve_data(verbose=verbose)
        self.data_retriever.dump()

    def _parse(self):
        """
        This method contains the parsing process exclusively. This can be overridden by deriving classes.
        :return: parsed data
        """
        self.parser = Parser(raw_data=self.raw_data, config=self.config, logger=self.logger)
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
        self.log_info("Parsing data...")

        self.parsed_data = self._parse()
        self.parser.dump(name=self.output_filename, with_config=False, pretty=self.config.PRETTIFY_PARSER_OUPUT)

    def _postparse(self):
        """
        This method contains the post-parsing process exclusively. This can be overridden by deriving classes.
        :return: post-parsed data
        """
        self.postparser = PostParser(parsed_data=self.parsed_data, config=self.config, logger=self.logger)
        postparsed = self.postparser.postparse()
        self.postparser.dump(name=self.output_filename, with_config=False, pretty=self.config.PRETTIFY_PARSER_OUPUT)
        return postparsed

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
        self.log_info("Post-parsing data...")
        self.postparsed_data = self._postparse()

    def _extract_features(self, verbose):
        self.feature_extractor = FeatureExtractor(preprocessed_data=self.postparsed_data, config=self.config, logger=self.logger)
        return self.feature_extractor.extract(verbose)

    def extract_features(self, verbose):
        if self.postparsed_data is None:
            raise CFGPipelineException("Data has to be post-parsed before feature extraction")
        if self.features_df is not None:
            raise CFGPipelineException("Features already extracted")
        self.log_info("Calculating measures...")

        self.features_df = self._extract_features(verbose)
        features_path = self.feature_extractor.dump(name=self.output_filename, with_exclusions=True, with_config=False)
        self.log_info(f"Results written to: {features_path}")

    def visualize(self, verbose):
        postparsed_data = self.postparsed_data
        viz_dir = self.output_filename + "_visualizations"
        os.makedirs(viz_dir, exist_ok=True)

        if self.config.VISUALIZATION_ANIMATE:
            self.log_info("Visualizing games with animation...")
        else:
            self.log_info("Visualizing games without animation...")
        if verbose:
            postparsed_data = tqdm.tqdm(self.postparsed_data, desc="Visualizing games", unit="game")
        for game in postparsed_data:
            if self.config.VISUALIZATION_ANIMATE:
                visualization.animate_game(game=game, speed=self.config.VISUALIZATION_ANIMATION_SPEED, output_dir_path=os.path.join(viz_dir, "animations"))

            #visualization.plot_game(game=game, output_dir_path=os.path.join(viz_dir, "plots"))
            visualization.plot_game(game=game, output_dir_path=os.path.join(viz_dir, "plots"))

    def run_pipeline(self):
        self.retrieve_data(verbose=self.verbose)
        self.parse(verbose=self.verbose)
        self.postparse(verbose=self.verbose)
        self.visualize(verbose=self.verbose)
        self.extract_features(verbose=self.verbose)
        return self.features_df


def safe_update(config: Configuration, key: str, new_value):
    if not hasattr(config, key):
        raise AttributeError(f"Configuration object has no attribute '{key}'")
    current_value = getattr(config, key, None)
    if current_value is not None and new_value is not None and current_value != new_value:
        raise ValueError(
            f"Conflict for config key '{key}': current value '{current_value}' vs new value '{new_value}'. Please resolve the conflict by providing a consistent value.")
    if new_value is not None:
        setattr(config, key, new_value)


def update_config_with_args(config: Configuration, args) -> Configuration:
    for arg_attr, config_attr in ARG_TO_CONF_MAP.items():
        arg_attr = arg_attr.replace("-", "_")  # argparse converts dashes to underscores for attribute names
        arg_value = getattr(args, arg_attr, None)
        safe_update(config, config_attr, arg_value)
    return config


def main():
    import argparse

    argparser = argparse.ArgumentParser(description="Run CFG behavioral data pipeline")
    # can give any of the valid data sources as argument to override the config data source
    argparser.add_argument(f"--{DATA_SOURCE_ARG}", choices=VALID_DATA_SOURCES,
                           help="The data source to retrieve data from. Should be provided only if it doesn't appear in the config file.")
    argparser.add_argument(f"--{GAME_NAME_ARG}", help='The name of the name.')
    argparser.add_argument(f"--{GAME_ID_ARG}", help='The id of the game.')
    argparser.add_argument(f"--{GAME_VERSION_IDS_ARG}", nargs="+",
                           help='A list of the game version ids that you want to retrieve.')
    argparser.add_argument(f"--{BEFORE_DATE_ARG}", type=str, default=None,
                           help='The end of the date range of the games you want to retrieve. Should be in a pandas-parseable datetime format. Only needed if you want to provide it as an argument instead of providing it in the config.')
    argparser.add_argument(f"--{AFTER_DATE_ARG}", type=str, default=None,
                           help='The start date of the games you want to retrieve. Should be in a pandas-parseable datetime format. Only needed if you want to provide it as an argument instead of providing it in the config.')
    argparser.add_argument("--config-path", help='The path to the yml file that contains the configuration')
    argparser.add_argument(f"--{EVENTS_CSV_PATH_ARG}",
                           help='The path to the events CSV file. Only needed if the data source is Local or if you want to provide a custom path to the events CSV file instead of providing it in the config.')
    argparser.add_argument("-o", "--output", default="cfg", dest="output_filename",
                           help='Filename of output files. This filename will be used as a prefix for all output files generated by the pipeline. The final features dataframe will be saved as <output_filename>_features.csv. Default is "cfg".')
    argparser.add_argument("-v", "--verbose", action="store_true", help="Whether to print info during the pipeline run. Default is False.")
    args = argparser.parse_args()

    config: Configuration | None = Configuration.from_yaml(
        yaml_path=args.config_path) if args.config_path else Configuration.default()
    arg_data_source = getattr(args, DATA_SOURCE_ARG.replace("-", "_"), None)
    if arg_data_source:
        config.DATA_SOURCE = arg_data_source  # set data source early to allow validation of other args
    config = update_config_with_args(config, args)  # keep config as single source of truth for downstream usage

    pl = Pipeline(output_filename=args.output_filename, config=config, verbose=args.verbose)

    pl.run_pipeline()


if __name__ == '__main__':
    main()
