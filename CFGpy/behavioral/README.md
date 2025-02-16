# CFG Behavioral Data Pipline

After running a behavioral CFG experiment, the data is processed using this package.
The pipeline ends in a CSV file where each row represents a participant that met our filtering criteria and each column is one of the standard [Creative Foraging Game measures](https://comdepri.slab.com/posts/nmhuytz6).

## Basic Usage
As of v0.0.1, this package can only retrieve raw data from RedMetrics1, and see issue https://github.com/ComDePri/CFGpy/issues/26.

### Command line
Given a URL for downloading raw data as CSV from RM1, you can run the Pipeline from a terminal like so:
```
run_pipeline --url <raw_data_url> --config-path <config_file_path> -o <output_filename>
```
Where the output filename argument is optional, and you need to have either the raw_data_url or the config_file_path argument but not both.

### Python Script
```python
from CFG.behavioral import Pipeline

Pipeline(raw_data_url, output_filename).run_pipeline()
```

Alternatively, pass the raw data URL as part of a configuration object:
```python
from CFG.behavioral import Configuration, Pipeline

config = Configuration.default()
config.RED_METRICS_CSV_URL = csv_url
Pipeline(config=config, output_filename=output_filename).run_pipeline()
```

In all cases, `output_filename` is optional.

> 📝 When a pipeline finishes, it outputs its configuration to enable full reproducibility. The outputted configuration will always include the raw data URL (even if the inputted configuration did not), and can be loaded into python using `config = Configuration.from_yaml(config_filename)`

## Modules and Logic
Below is an overview of the different modules composing the pipeline and their functions.

> 📝 To learn more about the different types of data mentioned here, see [CFG data types and structure](https://comdepri.slab.com/posts/b90dhs98)

### Downloader

Downloads raw data from the RedMetrics1 server.

### Parser

This module is responsible for turning raw data into parsed data. It's an objective translator — converts the data's structure for easier processing downstream, without changing its contents.

The exception to the rule is that Parser removes games that did not start (e.g., no game data, only tutorial data). We call this a "hard" filter, as it excludes data that cannot be parsed.

As a result, if a player accessed the game but did not start it, or for some reason played the entire thing in the tutorial stage, they will not be present in the parsed dataset.

After the hard filter is applied, the remaining data is parsed.

### Post-Parser

This module prepares parsed data for feature extraction. Namely:

1. Converting shape IDs from graphic representations to integers.
2. Throws an exception if the data contains "illegal" shapes, meaning shapes that are not supposed to be playable in the game (e.g., not all squares are connected).

    > ⚠️ This error indicates a bug in the game software or in the analysis pipeline, so processing cannot continue before it is investigated and resolved

    > 📝 Processing can't continue with illegal shapes because it relies on pre-calculated values for all possible shapes and transitions between them

3. _In future versions (see https://github.com/ComDePri/CFGpy/issues/2, https://github.com/ComDePri/CFGpy/pull/17):_ Deleting "empty" moves: moves that have the same shape as the previous move.

    > 📝 An empty move occurs when a player chooses a block and releases it in the same position where it started. This can happen for example if they regretted the move while making it, or as a result of the GUI supporting the release of a block far from the shape, by placing it at the nearest legal position (some players have reported using this to get "random" block placements).
    >
    > It follows that empty moves may convey meaningful information about the way the game was played (that's why they're not eliminated at the Parser level). Maybe in the future handling empty moves will change to support novel features of the creative process (see [here](https://comdepri.slab.com/posts/cfg-open-issues-v8w6zxv9#hc7c6-transitions-between-game-states-are-not-unique)).

4. Segmenting each game to exploration and exploitation phases.

### Feature Extractor

> 📝 For details on the distinction between absolute and relative features, see the [Terminology](https://comdepri.slab.com/posts/creative-foraging-game-measures-nmhuytz6#hut2h-terminology) section in [Creative Foraging Game measures](https://comdepri.slab.com/posts/nmhuytz6)

This module produces the end result CSV:

1. Calculates all absolute measures.
2. Drops non-first games for each player.

    > ⚠️ Players with prior CFG experience play the game differently. This stage is here to ensure we only keep the first game for each player ID in the sample. If some players are experienced but have unique IDs, those should be flagged for manual exclusion in the `Configuration` object passed to `FeatureExtractor`

3. Calculates all [vanilla](https://comdepri.slab.com/posts/e6tzqtk1)-relative features.
4. Applies "soft" filters, i.e., filters that can be chosen and configured by the person analyzing the data, to match their preferences and the studied population:
    1. Applies absolute filters, i.e., those that exclude participants based on their absolute measures, such as game duration.
    1. Applies sample-relative filters, e.g., outlier rejection.
5. Calculates all sample-relative measures.

## Pipeline Customization

### Custom modules

To use different classes for any of the modules, create a new class extending pipeline, and override one its private methods (whose names begin with an underscore).

The public methods contain core logic that should not be overriden, e.g., exception management.

For example:
```python
class MyPipline(Pipeline):
    def _parse():
        # use customized parser here
```

### Custom logic

Say you have a use case that requires customizing the pipeline in a way that cannot be encapsulated in a customized module. For example: you have a single sample saved in several different RedMetrics games. To get accurate relative features, you need the sample united before it's passed onto feature extraction.

Generally, the intention of this package is to be operated with the `Pipeline` API, with personal modification possible through configuration files and class inheritance, as outlined above.

But for some use cases, the flexibility of using each module directly might be required.

For the example above, a solution may look something like this:

```python
parsed_united = []
for url in urls:
    raw_data = Downloader(url).download()
    parsed = Parser(raw_data).parse()
    parsed_united += parsed

postparsed = PostParser(parsed_united).postparse()
feature_extractor = FeactureExtractor(postparsed)
features = feature_extractor.extract()
```

> ⚠️ When using the modules individually, without the assitance of the `Pipeline` class, it is the user's responsibility to ensure all modules use appropriate configurations.

> ⚠️ When using the modules individually, without the assitance of the `Pipeline` class, the outputted configuration file will not have the raw data URL injected, which may compromise reproducibility. It is the user's responsibility to enable reproducing their results.
