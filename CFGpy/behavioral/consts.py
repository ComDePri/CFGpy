# parsed data format
PARSED_PLAYER_ID_KEY = 'id'
PARSED_TIME_KEY = 'absolute start time'
PARSED_ALL_SHAPES_KEY = 'actions'
SHAPE_ID_IDX = 0
SHAPE_MOVE_TIME_IDX = 1
SHAPE_SAVE_TIME_IDX = 2
PARSED_CHOSEN_SHAPES_KEY = 'chosen_shapes'

# preprocessing
EXPLORE_KEY = "explore"
EXPLOIT_KEY = "exploit"

# measure calculation
MEASURES_ID_KEY = "ID"
N_MOVES_KEY = "Total # moves"
GAME_DURATION_KEY = "Total Play Time"
MEDIAN_EXPLORE_LENGTH_KEY = "median exp steps"
MEDIAN_EXPLOIT_LENGTH_KEY = "median scav steps"
LONGEST_PAUSE_KEY = "max dt"
# default soft filters
MANUALLY_EXCLUDED_IDS = ()
MIN_N_MOVES = 80
MIN_GAME_DURATION_SEC = 600
MAX_PAUSE_DURATION_SEC = 90
MAX_ZSCORE_FOR_OUTLIERS = 3
