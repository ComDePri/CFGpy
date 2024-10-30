from datetime import datetime, timezone
from CFGpy.behavioral import Downloader, Parser, PostParser, FeatureExtractor, Configuration
from CFGpy.behavioral._consts import DEFAULT_FINAL_OUTPUT_FILENAME
from CFGpy.behavioral._utils import CFGPipelineException


class Pipeline:
    def __init__(self, red_metrics_csv_url: str | None = None, output_filename=DEFAULT_FINAL_OUTPUT_FILENAME,
                 config=Configuration.default()):
        self.output_filename = output_filename
        self.config = config
        self.red_metrics_csv_url = red_metrics_csv_url

        self.downloader = None
        self.raw_data = None
        self.parser = None
        self.parsed_data = None
        self.postparser = None
        self.postparsed_data = None
        self.feature_extractor = None
        self.features_df = None

    def _add_url_to_config(self):
        csv_url = self.downloader.csv_url
        if "&before=" not in csv_url:
            now_str = datetime.now(timezone.utc).strftime(self.config.SERVER_DATE_FORMAT)
            csv_url += f"&before={now_str}"

        self.config.RED_METRICS_CSV_URL = csv_url

    def _download(self, verbose):
        """
        This method contains the downloading process exclusively. This can be overridden by deriving classes.
        :param verbose: whether to print info during the downloading process
        :return: raw data
        """
        return self.downloader.download(verbose)

    def download(self, verbose=True):
        """
        Wraps raw data downloading with extra necessary functionality.
        If you wish to override the downloading method, override _download, not this.
        :param verbose: whether to print info during the downloading process
        """
        if self.raw_data is not None:
            raise CFGPipelineException("Raw data already downloaded")

        self.downloader = Downloader(self.red_metrics_csv_url, config=self.config)
        self._add_url_to_config()

        if verbose:
            print("Downloading raw data...")
        self.raw_data = self._download(verbose)

    def _parse(self):
        """
        This method contains the parsing process exclusively. This can be overridden by deriving classes.
        :return: parsed data
        """
        self.parser = Parser(self.raw_data, self.config)
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
        self.postparser = PostParser(self.parsed_data, self.config)
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
        self.feature_extractor = FeatureExtractor(self.postparsed_data, self.config)
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
        self.download(verbose)
        self.parse(verbose)
        self.postparse(verbose)
        self.extract_features(verbose)
        return self.features_df


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description="Run CFG behavioral data pipeline")
    argparser.add_argument("url", help='Web address of the "Download all pages as CSV"')
    argparser.add_argument("-o", "--output", default=DEFAULT_FINAL_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output CSV')
    args = argparser.parse_args()

    pl = Pipeline(args.url, args.output_filename)
    pl.run_pipeline()
