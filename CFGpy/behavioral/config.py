"""
This is a pipeline configuration file. The values defined here affect the results of the pipeline, not only it's
structure. These values may change on a case to case basis (for example, setting stricter or more lenient participant
exclusion criteria, to suit some atypical population, for exploratory purposes, or for any other reason).

The version of this file found in the `main` branch of the CFGpy package should always reflect the lab's defaults.
Do NOT change unless you're absolutely sure you know what you're doing, AND you have Yuval's approval for the new
default config.
"""

from CFGpy.behavioral._utils import server_coords_to_binary_shape
from CFGpy.utils import binary_shape_to_id
from CFGpy.behavioral import _consts

# downloading
DOWNLOAD_PLAYER_REQUEST = "https://api.creativeforagingtask.com/v1/player/{}"

# raw data
"""
    Key names like these usually appear in _consts.py. These, though, are here because they're defined in the CFG 
    javascript code and RedMetrics DB configuration, both beyond the scope of this package. The package receives them 
    as part of the pipeline configuration, to signify that it's the user's responsibility to supply the correct 
    values based on their version of CFG javascript and RedMetrics database.
"""
EVENT_ID_KEY = "id"
EVENT_CUSTOM_DATA_KEY = "customData"
EVENT_PLAYER_ID_KEY = "player"
RAW_PLAYER_ID = "playerId"
RAW_SERVER_TIME = "serverTime"
RAW_USER_TIME = "userTime"
RAW_GAME_VERSION = "gameVersion"
RAW_PLAYER_BIRTHDATE = "playerBirthdate"
RAW_PLAYER_REGION = "playerRegion"
RAW_PLAYER_COUNTRY = "playerCountry"
RAW_PLAYER_GENDER = "playerGender"
RAW_PLAYER_EXTERNAL_ID = "playerExternalId"
RAW_PLAYER_CUSTOM_DATA = "playerCustomData"
EVENT_TYPE = "type"
RAW_COORDINATES = "coordinates"
RAW_SECTION = "section"
RAW_NEW_SHAPE = "customData.newShape"
RAW_SHAPE = "customData.shape"
DOWNLOADER_COMMON_FIELDS = (
    EVENT_ID_KEY, RAW_SERVER_TIME, RAW_USER_TIME, RAW_GAME_VERSION, EVENT_TYPE, RAW_COORDINATES, RAW_SECTION)
DOWNLOADER_FIELD_ORDER = (
    EVENT_ID_KEY,
    RAW_SERVER_TIME,
    RAW_USER_TIME,
    RAW_GAME_VERSION,
    RAW_PLAYER_ID,
    RAW_PLAYER_BIRTHDATE,
    RAW_PLAYER_REGION,
    RAW_PLAYER_COUNTRY,
    RAW_PLAYER_GENDER,
    RAW_PLAYER_EXTERNAL_ID,
    RAW_PLAYER_CUSTOM_DATA,
    EVENT_TYPE,
    RAW_COORDINATES,
    RAW_SECTION,
    "customData.startPosition",
    "customData.shapeIndices",
    "customData.shapeIndex",
    "customData.isSelected",
    RAW_NEW_SHAPE,
    "customData.timeSinceLastMouseUp",
    "customData.time",
    "customData.shapes",
    RAW_SHAPE,
    "customData.endPosition",
)
TUTORIAL_END_EVENT_TYPE = 'startsearch'  # in very old data versions, this is 'tutorial complete'
SHAPE_MOVE_EVENT_TYPE = 'movedblock'
GALLERY_SAVE_EVENT_TYPE = 'added shape to gallery'

# parsing
FIRST_SHAPE_SERVER_COORDS = '[[0,0],[1,0],[2,0],[4,0],[5,0],[6,0],[3,0],[7,0],[8,0],[9,0]]'
UNIQUE_INTERNAL_ID_COLUMN = 'playerId'
PARSER_ID_COLUMNS = (
    RAW_PLAYER_EXTERNAL_ID,
    'userProvidedId',
    'userId',
    'prolificId'
)
# PARSER_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'  # for RedMetrics1
PARSER_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S%z'  # for RedMetrics2
PARSER_TIME_COLUMN = RAW_USER_TIME
PARSER_JSON_COLUMN = RAW_PLAYER_CUSTOM_DATA
SHAPE_MOVE_COLUMN = RAW_NEW_SHAPE
SHAPE_SAVE_COLUMN = RAW_SHAPE
PARSED_GAME_HEADERS = [SHAPE_MOVE_COLUMN, PARSER_TIME_COLUMN, _consts.GALLERY_SAVE_TIME_COLUMN]
SHAPE_ID_IDX = PARSED_GAME_HEADERS.index(SHAPE_MOVE_COLUMN)
SHAPE_MOVE_TIME_IDX = PARSED_GAME_HEADERS.index(PARSER_TIME_COLUMN)
SHAPE_SAVE_TIME_IDX = PARSED_GAME_HEADERS.index(_consts.GALLERY_SAVE_TIME_COLUMN)

# post-parsing
MIN_SAVE_FOR_EXPLOIT = 3

# feature extraction
FIRST_SHAPE_ID = binary_shape_to_id(server_coords_to_binary_shape(FIRST_SHAPE_SERVER_COORDS))
MARGIN_FOR_PAUSE_DURATION = 3
MIN_OVERLAP_FOR_SEMANTIC_CONNECTION = 2

# default soft filters
MANUALLY_EXCLUDED_IDS = ()
MIN_N_MOVES = 80
MIN_N_CLUSTERS = 1
MIN_GAME_DURATION_SEC = 600
MAX_PAUSE_DURATION_SEC = 90
MAX_ZSCORE_FOR_OUTLIERS = 3
