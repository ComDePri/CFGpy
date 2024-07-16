from CFGpy.behavioral import Downloader, Parser, PostParser, FeatureExtractor
from CFGpy.behavioral._consts import DEFAULT_FINAL_OUTPUT_FILENAME


class Pipeline:
    def __init__(self, red_metrics_csv_url, output_filename=DEFAULT_FINAL_OUTPUT_FILENAME):
        self.output_filename = output_filename
        self.downloader = Downloader(red_metrics_csv_url)
        self.parser = None
        self.postparser = None
        self.feature_extractor = None

    def run_pipeline(self, verbose=True):
        print("Downloading raw data...")
        raw_data = self.downloader.download(verbose)

        print("Parsing...")
        self.parser = Parser(raw_data)
        parsed_data = self.parser.parse()
        self.parser.dump()

        print("Post-parsing...")
        self.postparser = PostParser(parsed_data)
        postparsed_data = self.postparser.postparse()

        print("Calculating measures...")
        self.feature_extractor = FeatureExtractor(postparsed_data)
        features_df = self.feature_extractor.extract(verbose)
        self.feature_extractor.dump(self.output_filename)
        print(f"Results written successfully to {self.output_filename}")

        return features_df


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description="Run CFG behavioral data pipeline")
    argparser.add_argument("url", help='Web address of the "Download all pages as CSV"')
    argparser.add_argument("-o", "--output", default=DEFAULT_FINAL_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output CSV')
    args = argparser.parse_args()

    pl = Pipeline(args.url, args.output_filename)
    pl.run_pipeline()
