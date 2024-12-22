"""
This file contains all non-configurable and/or arbitrarily chosen values in the behavioral data pipeline.
Some of these are only used for internal reference and their actual value doesn't matter. Others appear in outputs and
external scripts may rely on them.
Please do not change this file unless you're absolutely sure you know what you're doing.
"""

# configuration
CONFIG_PACKAGE = "CFGpy.behavioral"
CONFIG_FILENAME = "default_config.yml"
CFGPY_VERSION_ERROR = "Configuration file requires CFGpy version {}. Installed version is {}"
CONFIG_DUMP_EXTENSION = ".yml"

# downloader
DOWNLOADER_OUTPUT_FILENAME = "event.csv"
NO_DOWNLOADER_URL_ERROR = "RedMetrics URL undefined. Specify URL either as a parameter or in config"
TWO_DOWNLOADER_URL_ERROR = "RedMetrics URL was defined both as a parameter and in config. Define URL exactly once"
DOWNLOADER_URL_NO_CSV_ERROR = "URL is incorrect: '{}'\nCopy the address from 'Download all pages as CSV' in RedMetrics"
EVENTS_PER_PAGE = 500

# parser
PARSER_OUTPUT_FILENAME = "parsed.json"
MERGED_ID_KEY = "merged_id"
DEFAULT_ID = 'No ID Found'
NOT_A_NEIGHBOR_ERROR = "Found adjacent non neighboring shapes {} and {}, Player id:{}\n" \
                       "Check the source data to see if the game recorded an illegal move, otherwise it's the parser."

# parsed data format
PARSED_PLAYER_ID_KEY = 'id'
PARSED_TIME_KEY = 'absolute start time'
PARSED_ALL_SHAPES_KEY = 'actions'
PARSED_CHOSEN_SHAPES_KEY = 'chosen_shapes'

# post-parser
POSTPARSER_OUTPUT_FILENAME = "postparsed.json"
EXPLORE_KEY = "explore"
EXPLOIT_KEY = "exploit"
INVALID_SHAPE_ERROR = "Encountered invalid shape: {}\nPlayer id: {}\n" \
                      "This indicates a bug in the CFG software or in the data parsing"

# feature extractor
FEATURES_ID_KEY = "ID"
FEATURES_START_TIME_KEY = "Date/Time"
N_MOVES_KEY = "Total # moves"
N_GALLERIES_KEY = "#galleries"
SELF_AVOIDANCE_KEY = "self avoidance"
N_CLUSTERS_KEY = "#clusters"
EXPLORE_EFFICIENCY_KEY = "exp efficiency"
EXPLOIT_EFFICIENCY_KEY = "scav efficiency"
GAME_DURATION_KEY = "Total Play Time"
MEDIAN_EXPLORE_LENGTH_KEY = "median exp steps"
MEDIAN_EXPLOIT_LENGTH_KEY = "median scav steps"
LONGEST_PAUSE_KEY = "max dt"
AVERAGE_SPEED_KEY = "Average Speed"
FRACTION_GALLERY_IN_EXPLORE_KEY = "% galleries in exp"
FRACTION_TIME_IN_EXPLORE_KEY = "% time in exp"
EFFICIENCY_RATIO_KEY = "efficiency ratio"
EXPLORE_SPEED_KEY = "exp speed"
EXPLOIT_SPEED_KEY = "scav speed"
STEP_ORIG_KEY = "Step Orig"
FRACTION_STEPS_UNIQUELY_COVERED_KEY = "% steps uniquely covered"
GALLERY_ORIG_KEY = "Gallery Orig"
GALLERY_ORIG_EXPLORE_KEY = "Gallery Orig exp"
GALLERY_ORIG_EXPLOIT_KEY = "Gallery Orig scav"
FRACTION_GALLERIES_UNIQUELY_COVERED_KEY = "% galleries uniquely covered"
FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLORE_KEY = "% galleries uniquely covered exp"
FRACTION_GALLERIES_UNIQUELY_COVERED_EXPLOIT_KEY = "% galleries uniquely covered scav"
N_CLUSTERS_IN_GC_KEY = "# clusters in GC"
FRACTION_CLUSTERS_IN_GC_KEY = "% clusters in GC"

EXCLUSION_REASON_KEY = "reason"
SAMPLE_RELATIVE_FEATURES_LABEL = "sample"
EXPLORE_OUTLIER_REASON = "Explore length outlier"
EXPLOIT_OUTLIER_REASON = "Exploit length outlier"
MANUAL_EXCLUSION_REASON = "Manually excluded id"
NO_EXPLOIT_EXCLUSION_REASON = "Did not exploit"
GAME_LENGTH_EXCLUSION_REASON = "Game length too short"
GAME_DURATION_EXCLUSION_REASON = "Game duration too short"
PAUSE_EXCLUSION_REASON = "Paused for too long"

ABSOLUTE_FEATURES_MESSAGE = "Extracting absolute features..."
RELATIVE_FEATURES_MESSAGE = "Extracting relative{} features..."
DEFAULT_FINAL_OUTPUT_FILENAME = "CFG_measures.csv"

# utils
SERVER_COORDS_TYPE_ERROR = "Received incorrect type as csv_coords, should be str or list, received {}"
PRETTIFY_WARNING = "Creating a pretty JSON may take a while! Avoid if the file is very big."
