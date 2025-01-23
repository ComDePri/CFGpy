import json
import os
import time
import boto3
import hashlib
import networkx as nx
import numpy as np
import pandas as pd
from appdirs import user_cache_dir


class FileNames:

    CACHE_DIR: str = user_cache_dir("cfgpy")
    CFGPY_BUCKET: str = "cfgpy-data"

    VANILLA_DATA: str = "vanilla_data/vanilla.json" # TODO: upload file
    VANILLA_FEATURES: str = "vanilla_data/vanilla_features.csv" # TODO: upload file
    VANILLA_GALLERY_COUNTER: str = "vanilla_data/gallery_counter.json"
    VANILLA_GIANT_COMPONENT: str = "vanilla_data/giant_component.json"
    VANILLA_STEP_COUNTER: str = "vanilla_data/step_counter.json"
    SHAPES_NETWORK: str = "all_shapes.adjlist"
    ID2COORD: str = "grid_coords.npy"
    SHORTEST_PATHS: str = "shortest_path_len.json"

    ALL_FILES: list[str] = [VANILLA_DATA, VANILLA_FEATURES, 
                            VANILLA_GALLERY_COUNTER, VANILLA_GIANT_COMPONENT, 
                            VANILLA_STEP_COUNTER, SHAPES_NETWORK, ID2COORD, SHORTEST_PATHS]


class BotoUtils:

    def __init__(self):
        os.makedirs(FileNames.CACHE_DIR, exist_ok=True)
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id="AKIAUWHEKK3UPKUQVBPY",
            aws_secret_access_key="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"
            )

    def get_file_from_s3(self, bucket_name: str, key: str, local_filename: str = None) -> str:

        if local_filename is None:
            local_filename = os.path.join(FileNames.CACHE_DIR, hashlib.md5(key.encode()).hexdigest())
        
        if os.path.exists(local_filename):
            print(f"File is already cached: {local_filename}")
            return local_filename
        
        print(f"Downloading {key} from bucket {bucket_name}...")
        self.s3_client.download_file(bucket_name, key, local_filename)
        print(f"File downloaded and cached at: {local_filename}")
        
        return local_filename
    
    def get_files_from_s3(self, bucket_name: str, keys: list[str]) -> dict[str, str]:
        return {key: self.get_file_from_s3(bucket_name, key) for key in keys}
    
    def clean_cache(expiration_time: int = 86400):

        current_time = time.time()
        for file in os.listdir(FileNames.CACHE_DIR):
            file_path = os.path.join(FileNames.CACHE_DIR, file)
            if os.path.isfile(file_path) and (current_time - os.path.getmtime(file_path) > expiration_time):
                os.remove(file_path)
                print(f"Removed expired cache file: {file_path}")

    
class FilesHandler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'): 
            self.boto_utils = BotoUtils()
            self.file_names_dict: dict[str, str] = self.boto_utils.get_files_from_s3(bucket_name=FileNames.CFGPY_BUCKET, keys=FileNames.ALL_FILES)
            self._vanilla_data: dict = {}
            self._vanilla_features: pd.DataFrame = {}
            self._vanilla_gallery_counter: dict = {}
            self._vanilla_giant_component: dict = {}
            self._vanilla_step_counter: dict = {}
            self._shape_network = {}
            self._id2coord = {}
            self._shortest_paths_dict: dict = {}
            self._initialized = True
    
    @property
    def vanilla_data(self) -> dict:
        if not self._vanilla_data:
            with open(os.path.join(FileNames.CACHE_DIR, self.file_names_dict[FileNames.VANILLA_DATA])) as f:
                self._vanilla_data = json.load(f)
        return self._vanilla_data
    
    @property
    def vanilla_features(self) -> dict:
        if not self._vanilla_features:
            self._vanilla_features = pd.read_csv(os.path.join(FileNames.CACHE_DIR, self.file_names_dict[FileNames.VANILLA_FEATURES]))
        return self._vanilla_features
    
    @property
    def vanilla_gallery_counter(self) -> dict:
        if not self._vanilla_gallery_counter:
            with open(os.path.join(FileNames.CACHE_DIR, self.file_names_dict[FileNames.VANILLA_GALLERY_COUNTER])) as f:
                self._vanilla_gallery_counter = json.load(f)
        return self._vanilla_gallery_counter
    
    @property
    def vanilla_giant_component(self) -> dict:
        if not self._vanilla_giant_component:
            with open(os.path.join(FileNames.CACHE_DIR, self.file_names_dict[FileNames.VANILLA_GIANT_COMPONENT])) as f:
                self._vanilla_giant_component = json.load(f)
        return self._vanilla_giant_component
    
    @property
    def vanilla_step_counter(self) -> dict:
        if not self._vanilla_step_counter:
            with open(os.path.join(FileNames.CACHE_DIR, self.file_names_dict[FileNames.VANILLA_STEP_COUNTER])) as f:
                self._vanilla_step_counter = json.load(f)
        return self._vanilla_step_counter
    
    @property
    def shape_network(self) -> dict:
        if not self._shape_network:
            self._shape_network = nx.read_adjlist(os.path.join(FileNames.CACHE_DIR, self.file_names_dict[FileNames.SHAPES_NETWORK]), nodetype=int)
        return self._shape_network
    
    @property
    def id2coord(self) -> dict:
        if not self._id2coord:
           self._id2coord = np.load(os.path.join(FileNames.CACHE_DIR, self.file_names_dict[FileNames.ID2COORD]))
        return self._id2coord
    
    @property
    def shortest_paths_dict(self) -> dict:
        if not self._shortest_paths_dict:
            with open(os.path.join(FileNames.CACHE_DIR, self.file_names_dict[FileNames.SHORTEST_PATHS])) as f:
                self._shortest_paths_dict = json.load(f)
        return self._shortest_paths_dict
