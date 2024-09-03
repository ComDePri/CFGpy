from dataclasses import dataclass
from CFGpy.behavioral._consts import CONFIG_PACKAGE, CONFIG_FILENAME
from CFGpy.behavioral._utils import server_coords_to_binary_shape
from CFGpy.utils import binary_shape_to_id
import yaml
import dacite
import importlib.resources as ir
import sys
from pathlib import Path


@dataclass
class Configuration:
    @classmethod
    def default(cls):
        if sys.version_info[1] >= 9:
            config_path = ir.files(CONFIG_PACKAGE).joinpath(CONFIG_FILENAME)
        else:
            config_path = ir.path(CONFIG_PACKAGE, CONFIG_FILENAME)

        return cls.from_yaml(config_path)

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        with open(yaml_path) as yaml_fp:
            config_dict = yaml.safe_load(yaml_fp)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = dacite.from_dict(data_class=cls, data=config_dict, config=dacite.Config(cast=[tuple]))
        config._post_init()
        return config

    def _post_init(self):
        self._validate()
        self._add_parsed_game_header_indices()
        self._add_first_shape_id()

    def _validate(self):
        pass

    def _add_parsed_game_header_indices(self) -> None:
        self.SHAPE_ID_IDX: int = self.PARSED_GAME_HEADERS.index(self.SHAPE_MOVE_COLUMN)
        self.SHAPE_MOVE_TIME_IDX: int = self.PARSED_GAME_HEADERS.index(self.PARSER_TIME_COLUMN)
        self.SHAPE_SAVE_TIME_IDX: int = self.PARSED_GAME_HEADERS.index(self.GALLERY_SAVE_TIME_COLUMN)

    def _add_first_shape_id(self) -> None:

        self.FIRST_SHAPE_ID = binary_shape_to_id(server_coords_to_binary_shape(self.FIRST_SHAPE_SERVER_COORDS))

    # Raw data column names
    """
    Key names like these usually appear in _consts.py. These, though, are here because they're defined in the CFG
    javascript code and RedMetrics DB configuration, both beyond the scope of this package. The package receives them as
    part of the pipeline configuration, to signify that it's the user's responsibility to supply the correct values
    based on their version of CFG javascript and RedMetrics database.
    """
    EVENT_ID_KEY: str
    EVENT_CUSTOM_DATA_KEY: str
    EVENT_PLAYER_ID_KEY: str
    RAW_PLAYER_ID: str
    RAW_SERVER_TIME: str
    RAW_USER_TIME: str
    RAW_GAME_VERSION: str
    RAW_PLAYER_BIRTHDATE: str
    RAW_PLAYER_REGION: str
    RAW_PLAYER_COUNTRY: str
    RAW_PLAYER_GENDER: str
    RAW_PLAYER_EXTERNAL_ID: str
    RAW_PLAYER_CUSTOM_DATA: str
    EVENT_TYPE: str
    RAW_COORDINATES: str
    RAW_SECTION: str
    RAW_NEW_SHAPE: str
    RAW_SHAPE: str

    # Downloading
    DOWNLOAD_PLAYER_REQUEST: str
    DOWNLOADER_COMMON_FIELDS: tuple
    DOWNLOADER_FIELD_ORDER: tuple
    TUTORIAL_END_EVENT_TYPE: str
    SHAPE_MOVE_EVENT_TYPE: str
    GALLERY_SAVE_EVENT_TYPE: str

    # Parsing
    FIRST_SHAPE_SERVER_COORDS: str
    UNIQUE_INTERNAL_ID_COLUMN: str
    PARSER_ID_COLUMNS: tuple
    PARSER_DATE_FORMAT: str
    PARSER_TIME_COLUMN: str
    PARSER_JSON_COLUMN: str
    SHAPE_MOVE_COLUMN: str
    SHAPE_SAVE_COLUMN: str
    GALLERY_SAVE_TIME_COLUMN: str
    PARSED_GAME_HEADERS: tuple

    # Post-Parsing
    MIN_SAVE_FOR_EXPLOIT: int

    # Feature Extraction
    MARGIN_FOR_PAUSE_DURATION: int
    MIN_OVERLAP_FOR_SEMANTIC_CONNECTION: int

    # Soft Filtering
    MANUALLY_EXCLUDED_IDS: tuple
    MIN_N_MOVES: int
    MIN_N_CLUSTERS: int
    MIN_GAME_DURATION_SEC: float
    MAX_PAUSE_DURATION_SEC: float
    MAX_ZSCORE_FOR_OUTLIERS: float
