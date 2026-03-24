from dataclasses import dataclass, asdict

from numpy._core.numeric import False_
from CFGpy._version import __version__ as CFGpy_version
from CFGpy.behavioral._consts import CONFIG_PACKAGE, CONFIG_FILENAME, CFGPY_VERSION_ERROR, CONFIG_DUMP_EXTENSION, VALID_DATA_SOURCES, UNSUPPORTED_DATA_SOURCE_ERROR
from CFGpy.behavioral._utils import server_coords_to_binary_shape, get_default_data_source
from CFGpy.utils import binary_shape_to_id
import yaml
import dacite
import importlib.resources as ir
import sys


@dataclass
class Configuration:

    @classmethod
    def default(cls):
        config_filename = CONFIG_FILENAME
        if sys.version_info[1] >= 9:
            config_path = ir.files(CONFIG_PACKAGE).joinpath(config_filename)
        else:
            config_path = ir.path(CONFIG_PACKAGE, config_filename)
        config = cls.from_yaml(config_path)
        return config

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path) as yaml_fp:
            config_dict = yaml.safe_load(yaml_fp)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = dacite.from_dict(data_class=cls, data=config_dict, config=dacite.Config(cast=[tuple]))
        config._post_init()
        return config

    def _post_init(self):
        self._add_CFGpy_version()
        self._add_data_source()
        self._add_parsed_game_header_indices()
        self._add_first_shape_id()
        self._validate()


    def to_yaml(self, yaml_path: str):
        if not yaml_path.endswith(CONFIG_DUMP_EXTENSION):
            yaml_path += "_config" + CONFIG_DUMP_EXTENSION

        config_dict = asdict(self)
        with open(yaml_path, 'w') as yaml_fp:
            yaml.safe_dump(config_dict, yaml_fp)

    def _validate(self):
        required_version = self.CFGPY_VERSION
        if required_version is not None and required_version != CFGpy_version:
            raise ValueError(CFGPY_VERSION_ERROR.format(required_version, CFGpy_version))
        if self.DATA_SOURCE not in VALID_DATA_SOURCES:
            raise ValueError(UNSUPPORTED_DATA_SOURCE_ERROR.format(self.DATA_SOURCE))


    def _add_CFGpy_version(self):
        if self.CFGPY_VERSION is None:
            self.CFGPY_VERSION = CFGpy_version

    def _add_data_source(self):
        if self.DATA_SOURCE is None:
            self.DATA_SOURCE = get_default_data_source(self.CFGPY_VERSION)


    def _add_parsed_game_header_indices(self) -> None:
        self.SHAPE_ID_IDX: int = self.PARSED_GAME_HEADERS.index(self.SHAPE_MOVE_COLUMN)
        self.SHAPE_MOVE_TIME_IDX: int = self.PARSED_GAME_HEADERS.index(self.PARSER_TIME_COLUMN)
        self.SHAPE_SAVE_TIME_IDX: int = self.PARSED_GAME_HEADERS.index(self.GALLERY_SAVE_TIME_COLUMN)

    def _add_first_shape_id(self) -> None:
        self.FIRST_SHAPE_ID = binary_shape_to_id(server_coords_to_binary_shape(self.FIRST_SHAPE_SERVER_COORDS))

    # Class data (unchanged)

    CFGPY_VERSION: str | None
    EVENT_ID_KEY: str
    EVENT_CUSTOM_DATA_KEY: str
    EVENT_PLAYER_ID_KEY: str
    RAW_PLAYER_ID: str
    RAW_SERVER_TIME: str
    RAW_USER_TIME: str

    RAW_PLAYER_CUSTOM_DATA: str
    EVENT_TYPE: str
    RAW_COORDINATES: str
    RAW_NEW_SHAPE: str
    RAW_SHAPE: str

    DOWNLOADER_COMMON_FIELDS: tuple
    DOWNLOADER_FIELD_ORDER: tuple
    TUTORIAL_END_EVENT_TYPE: str
    SHAPE_MOVE_EVENT_TYPE: str
    GALLERY_SAVE_EVENT_TYPE: str

    FIRST_SHAPE_SERVER_COORDS: str
    UNIQUE_INTERNAL_ID_COLUMN: str
    PARSER_ID_COLUMNS: tuple
    INCLUDE_IN_PARSER_ID: tuple
    SERVER_DATE_FORMAT: str
    PARSER_TIME_COLUMN: str
    PARSER_JSON_COLUMN: str
    SHAPE_MOVE_COLUMN: str
    SHAPE_SAVE_COLUMN: str
    GALLERY_SAVE_TIME_COLUMN: str
    PARSED_GAME_HEADERS: tuple

    MIN_SAVE_FOR_EXPLOIT: int

    MARGIN_FOR_PAUSE_DURATION: int
    MIN_OVERLAP_FOR_SEMANTIC_CONNECTION: int
    STEP_ORIG_PSEUDOCOUNT: int
    GALLERY_ORIG_PSEUDOCOUNT: int
    STEP_ORIG_N_CATEGORIES: int
    GALLERY_ORIG_N_CATEGORIES: int

    MANUALLY_EXCLUDED_IDS: tuple
    MIN_N_MOVES: int
    MIN_N_CLUSTERS: int
    MIN_GAME_DURATION_SEC: float
    MAX_PAUSE_DURATION_SEC: float
    MAX_ZSCORE_FOR_OUTLIERS: float

    GAME_NAME: str | None = None
    GAME_ID: str | None = None
    GAME_VERSION_IDS: list[str] | None = None
    BEFORE_DATE: str | None = None
    AFTER_DATE: str | None = None

    RED_METRICS_CSV_URL: str | None = None
    DOWNLOAD_PLAYER_REQUEST: str = None
    EVENT_CSV_PATH: str = None
    RAW_GAME_VERSION: str = None
    RAW_PLAYER_BIRTHDATE: str = None
    RAW_PLAYER_REGION: str = None
    RAW_PLAYER_COUNTRY: str = None
    RAW_PLAYER_GENDER: str = None
    RAW_PLAYER_EXTERNAL_ID: str = None
    RAW_SECTION: str = None

    DATA_SOURCE: str | None = None

    CFG_GRAPHQL_URL: str | None = None
    CFG_COGNITO_REGION: str | None = None
    CFG_COGNITO_CLIENT_ID: str | None = None
    CFG_COGNITO_USER_POOL_ID: str | None = None

    PRETTIFY_PARSER_OUPUT: bool = True
    WRITE_G_ALPHA: bool = False # to support old data

    MAX_IGNORED_GAME_DURATION_SEC: float = 0