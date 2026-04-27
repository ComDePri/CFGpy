"""
This file contains all non-configurable and/or arbitrarily chosen values in the behavioral data pipeline.
Some of these are only used for internal reference and their actual value doesn't matter. Others appear in outputs and
external scripts may rely on them.
Please do not change this file unless you're absolutely sure you know what you're doing.
"""

# script
DATA_SOURCE_ARG = "data-source"
GAME_ID_ARG = "game-id"
GAME_VERSION_IDS_ARG = "game-version-ids"
BEFORE_DATE_ARG = "before"
AFTER_DATE_ARG = "after"
EVENTS_CSV_PATH_ARG = "events-csv-path"


ARG_TO_CONF_MAP = {
        DATA_SOURCE_ARG: "DATA_SOURCE",
        GAME_ID_ARG: "GAME_ID",
        GAME_VERSION_IDS_ARG: "GAME_VERSION_IDS",
        BEFORE_DATE_ARG: "BEFORE_DATE",
        AFTER_DATE_ARG: "AFTER_DATE",
        EVENTS_CSV_PATH_ARG: "EVENT_CSV_PATH",
    }
# configuration
CONFIG_PACKAGE = "CFGpy.behavioral"
CONFIG_FILENAME = "default_config.yml"
RM2_CONFIG_FILENAME = "default_rm2_config.yml"
CFGPY_VERSION_ERROR = "Configuration file requires CFGpy version {}. Installed version is {}"
CONFIG_DUMP_EXTENSION = ".yml"

# valid data sources
RM1 = "RedMetrics1"
RM2 = "RedMetrics2"
RM1_NAS_DUMP = "RedMetrics1Dump"
IOCANE = "IOCANE"
LOCAL = "Local"
VALID_DATA_SOURCES =  (RM1, RM2, RM1_NAS_DUMP, IOCANE, LOCAL)
DATA_SOURCES_ALIASES_LOW = {
    RM1: [RM1.lower(), "rm1", "red metrics 1", "redmetrics1"],
    RM2: [RM2.lower(), "rm2", "red metrics 2", "redmetrics2"],
    RM1_NAS_DUMP: [RM1_NAS_DUMP.lower(), "rm1 dump", "redmetrics1 dump", "redmetrics1dump"],
    IOCANE: [IOCANE.lower(), "iocane"],
    LOCAL: [LOCAL.lower(), "local"]
}
UNSUPPORTED_DATA_SOURCE_ERROR = "Unsupported data source: {}. Valid options are: {}".format("{}", VALID_DATA_SOURCES)


# data retrievers
DATA_RETRIEVER_OUTPUT_FILENAME = "event"
NO_DATA_RETRIEVER_INPUT_ERROR = "No input defined for data retriever. Define one of RedMetrics1 URL / game ID / game name exactly once - either as a parameter or in the config."
MULTIPLE_DATA_RETRIEVER_INPUTS_ERROR = "Input was defined in multiple ways. Define one of RedMetrics1 URL / game ID / game name exactly once - either as a parameter or in the config."
DOWNLOADER_URL_NO_CSV_ERROR = "URL is incorrect: '{}'\nCopy the address from 'Download all pages as CSV' in RedMetrics"
CONFIG_URL_MISMATCH_ERROR = "The config and the url or game id must both be either RedMetrics1 or RedMetrics2"
PAGE_REPETITION_LIMIT_REACHED = "Was not able to get all events from page {} after {} retries."
PER_PAGE = 10000
RM1_EVENTS_PER_PAGE = 500
MAX_PAGES = 1000

# parser
PARSER_OUTPUT_FILENAME = "parsed.json"
MERGED_ID_KEY = "merged_id"
DEFAULT_ID = 'No ID Found'
NOT_A_NEIGHBOR_ERROR = "Found adjacent non-neighboring shapes, indices {}-{}, Player id: {}\n" \
                       "Check the source data for illegal moves, otherwise this indicates a bug in the pipeline."

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
G_KEY = "explore-exploit switching rate"
ALPHA_KEY = "tendency to exploit"

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
DEFAULT_FINAL_OUTPUT_FILENAME = "measures.csv"
DEFAULT_POSTPARSED_FILTERED_OUTPUT_FILENAME = "postparsed_clean.json"

# utils
SERVER_COORDS_TYPE_ERROR = "Received incorrect type as csv_coords, should be str or list, received {}"
PRETTIFY_WARNING = "Creating a pretty JSON may take a while! Avoid if the file is very big."

# Visualization
VIS_SHAPE_COLOR = "#32CD32"  # CSS "limegreen", as used in the game
VIS_EXPLOIT_SHAPE_COLOR = '#1DA7EF'
VIS_SHAPE_BG_COLOR = "k"
VIS_GALLERY_BG_COLOR = "r"