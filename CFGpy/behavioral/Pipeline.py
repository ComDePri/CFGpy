from datetime import datetime, timezone
from CFGpy.behavioral import DataRetriever, RedMetrics1DataRetriever, RedMetrics2DataRetriever, Parser, PostParser, FeatureExtractor, Configuration
from CFGpy.behavioral._consts import DEFAULT_FINAL_OUTPUT_FILENAME
from CFGpy.behavioral._utils import CFGPipelineException


class Pipeline:
    def __init__(self, game_name: str | None = None, game_id: str | None = None, game_version_ids: list[str] | None = None, is_rm1: bool = False, 
                 output_filename=DEFAULT_FINAL_OUTPUT_FILENAME, config: Configuration = None):
       
        self._game_name = game_name
        self._game_id: str = game_id
        self._game_version_ids = game_version_ids
        self._is_rm1 = is_rm1
        
        self.output_filename = output_filename
        self.config = config or Configuration.default(is_rm1=is_rm1) 
        
        self.data_retriever = None
        self.raw_data = None
        self.parser = None
        self.parsed_data = None
        self.postparser = None
        self.postparsed_data = None
        self.feature_extractor = None
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
        if self._is_rm1:
            self.config.GAME_VERSION_IDS = self.data_retriever._game_version_ids
            
    def _get_downloader(self) -> DataRetriever:
        return (RedMetrics2DataRetriever(game_name=self._game_name, game_id=self._game_id, config=self.config) if not self._is_rm1 
                else RedMetrics1DataRetriever(game_name=self._game_name, game_id=self._game_id, game_version_ids=self._game_version_ids, config=self.config))
    
    def _retrieve_data(self, verbose):
        """
        This method contains the downloading process exclusively. This can be overridden by deriving classes.
        :param verbose: whether to print info during the downloading process
        :return: raw data
        """
        return self.data_retriever.retrieve_data(verbose=verbose)
    
    def download(self, verbose=True):
        """
        Wraps raw data downloading with extra necessary functionality.
        If you wish to override the downloading method, override _download, not this.
        :param verbose: whether to print info during the downloading process
        """
        if self.raw_data is not None:
            raise CFGPipelineException("Raw data already downloaded")

        self.data_retriever = self._get_downloader()
        self._add_input_params_to_config()

        if verbose:
            print("Downloading raw data...")
            
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
            raise CFGPipelineException("Raw data has to be downloaded before parsing")
        if self.parsed_data is not None:
            raise CFGPipelineException("Data already parsed")

        if verbose:
            print("Parsing...")
        self.parsed_data = self._parse()
        self.parser.dump()

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
        if self.postparsed_data is None:
            raise CFGPipelineException("Data has to be post-parsed before feature extraction")
        if self.features_df is not None:
            raise CFGPipelineException("Features already extracted")

        if verbose:
            print("Calculating measures...")

        self.features_df = self._extract_features(verbose)
        self.feature_extractor.dump(self.output_filename)

        if verbose:
            print(f"Results written successfully to: {self.output_filename}")

    def run_pipeline(self, verbose=True):
        self.download(verbose=verbose)
        self.parse(verbose=verbose)
        self.postparse(verbose=verbose)
        self.extract_features(verbose=verbose)
        return self.features_df


def main():
    import argparse

    argparser = argparse.ArgumentParser(description="Run CFG behavioral data pipeline")
    argparser.add_argument("--game-name", help='The name of the name.')
    argparser.add_argument("--game-id", help='The id of the game.')
    argparser.add_argument("--game-version-ids", nargs="+", help='A list of the game version ids that you want to download.')
    argparser.add_argument("--config-path", help='The path to the yml file that contains the configuration')
    argparser.add_argument("-o", "--output", default=DEFAULT_FINAL_OUTPUT_FILENAME, dest="output_filename",
                        help='Filename of output CSV')
    argparser.add_argument("--rm1", action="store_true", help="Use RM1 data")
    args = argparser.parse_args()
    
    config: Configuration | None = Configuration.from_yaml(yaml_path=args.config_path) if args.config_path else None
    
    pl = Pipeline(game_name=args.game_name, game_id=args.game_id, game_version_ids=args.game_version_ids, is_rm1=(not args.rm1), 
                  output_filename=args.output_filename, config=config)
    
    pl.run_pipeline()


if __name__ == '__main__':
    main()
