# parsed data format
PARSED_PLAYER_ID_KEY = 'id'
PARSED_TIME_KEY = 'absolute start time'
PARSED_ALL_SHAPES_KEY = 'actions'
SHAPE_ID_IDX = 0
SHAPE_MOVE_TIME_IDX = 1
SHAPE_SAVE_TIME_IDX = 2
SHAPE_MAX_MOVE_TIME_IDX = 3
PARSED_CHOSEN_SHAPES_KEY = 'chosen_shapes'

segmentation_algorithm = "MRI"  # MRI or non-MRI
if segmentation_algorithm=="non-MRI":
    # (1) ***  Similar to non-MRI algorithm: ***
    PATH_FROM_REP_ROOT = "output/non_mri_segmentation_algorithm/"
    MIN_EFFICIENCY_FOR_EXPLOIT = 2 # This should be 2 for regular games (actually, just greater than 1), and 0.8 for MRI games
    MAX_PACE_FOR_MERGE = 10000 # If a merge based on efficiency includes two shapes for which the average time per step is more than 10 seconds, the merge will be cancled (set very high for non-MRI behvaior)
    REMOVE_EMPTY_TIME_STEPS = False #False for regular games, True for MRI games
    USE_PACE_CRITERION = False
elif segmentation_algorithm=="MRI":
    PATH_FROM_REP_ROOT = "output/mri_segmentation_algorithm/"
    # (2) *** Values for the MRI algorithms: ***
    MIN_EFFICIENCY_FOR_EXPLOIT = 0.8 # This should be 2 for regular games (actually, just greater than 1), and 0.8 for MRI games
    MAX_PACE_FOR_MERGE = 10 # If a merge based on efficiency includes two shapes for which the average time per step is more than 10 seconds, the merge will be cancled (set very high for non-MRI behvaior)
    REMOVE_EMPTY_TIME_STEPS = True #False for regular games, True for MRI games
    USE_PACE_CRITERION = True

# preprocessing
MIN_SAVE_FOR_EXPLOIT = 3 # Minimal number of consecutive gallery shapes to be considered an exploit

EXPLORE_KEY = "explore"
EXPLOIT_KEY = "exploit"

# measure calculation
MEASURES_ID_KEY = "ID"
MEASURES_START_TIME_KEY = "Date/Time"
N_MOVES_KEY = "Total # moves"
GAME_DURATION_KEY = "Total Play Time"
MEDIAN_EXPLORE_LENGTH_KEY = "median exp steps"
MEDIAN_EXPLOIT_LENGTH_KEY = "median scav steps"
LONGEST_PAUSE_KEY = "max dt"
DEFAULT_FINAL_OUTPUT_FILENAME = "CFG measures.csv"
MIN_OVERLAP_FOR_SEMANTIC_CONNECTION = 2

# default soft filters
MANUALLY_EXCLUDED_IDS = ('999999')
MIN_N_MOVES = 80
MIN_GAME_DURATION_SEC = 600
MAX_PAUSE_DURATION_SEC = 90
MAX_ZSCORE_FOR_OUTLIERS = 3
