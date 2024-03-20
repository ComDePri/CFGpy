from CFGpy.behavioral import Downloader, Parser, Preprocessor, MeasureCalculator
from CFGpy.behavioral._consts import DEFAULT_FINAL_OUTPUT_FILENAME


class Pipeline:
    def __init__(self, red_metrics_csv_url, output_filename=DEFAULT_FINAL_OUTPUT_FILENAME):
        self.output_filename = output_filename
        self.downloader = Downloader(red_metrics_csv_url)
        self.parser = None
        self.preprocessor = None
        self.measure_calculator = None

    def run_pipeline(self):
        print("Downloading raw data...")
        raw_data = self.downloader.download()

        print("Parsing...")
        self.parser = Parser(raw_data)
        parsed_data = self.parser.parse()
        self.parser.dump()

        print("Calculating measures...")
        self.preprocessor = Preprocessor(parsed_data)
        preprocessed_data = self.preprocessor.preprocess()
        self.measure_calculator = MeasureCalculator(preprocessed_data)
        measures_df = self.measure_calculator.calc()
        self.measure_calculator.dump(self.output_filename)
        print(f"Results written successfully to {self.output_filename}")

        return measures_df


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(description="Run CFG behavioral data pipeline")
    argparser.add_argument("url", help='Web address of the "Download all pages as CSV"')
    argparser.add_argument("-o", "--output", default=DEFAULT_FINAL_OUTPUT_FILENAME, dest="output_filename",
                           help='Filename of output CSV')
    args = argparser.parse_args()

    pl = Pipeline(args.url, args.output_filename)
    pl.run_pipeline()
