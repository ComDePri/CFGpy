import json
import os
import networkx as nx
import numpy as np
import pandas as pd
from appdirs import user_cache_dir
import requests
from requests.auth import HTTPBasicAuth


class FileNames:

    CACHE_DIR: str = user_cache_dir("cfgpy")
    VANILLA_DIR: str = os.path.join(CACHE_DIR, "vanilla_data")
    CFGPY_BUCKET: str = "cfgpy-data"

    VANILLA_DATA: str = "vanilla_data/vanilla.json"
    VANILLA_FEATURES: str = "vanilla_data/vanilla_features.csv"
    VANILLA_GALLERY_COUNTER: str = "vanilla_data/gallery_counter.json"
    VANILLA_GIANT_COMPONENT: str = "vanilla_data/giant_component.json"
    VANILLA_STEP_COUNTER: str = "vanilla_data/step_counter.json"
    SHAPE_NETWORK: str = "all_shapes.adjlist"
    ID2COORD: str = "grid_coords.npy"
    SHORTEST_PATHS: str = "shortest_path_len.json"

    ALL_FILES: list[str] = [VANILLA_DATA, VANILLA_FEATURES, 
                            VANILLA_GALLERY_COUNTER, VANILLA_GIANT_COMPONENT, 
                            VANILLA_STEP_COUNTER, SHAPE_NETWORK, ID2COORD, SHORTEST_PATHS]

    
class FilesHandler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.token: str = os.getenv("GITHUB_TOKEN")  # TODO: remove token when the repo is public
            self._vanilla_data: dict = {}
            self._vanilla_features: pd.DataFrame = {}
            self._vanilla_gallery_counter: dict = {}
            self._vanilla_giant_component: dict = {}
            self._vanilla_step_counter: dict = {}
            self._shape_network = {}
            self._id2coord = {}
            self._shortest_paths_dict: dict = {}
            self._initialized = True

            if not os.path.isdir(FileNames.CACHE_DIR):
                os.mkdir(FileNames.CACHE_DIR)

            if not os.path.isdir(FileNames.VANILLA_DIR):
                os.mkdir(FileNames.VANILLA_DIR)
    
    def get_raw_file_from_github(self, file_name: str) -> None:  # TODO: remove token when the repo is public
        
        if file_name not in FileNames.ALL_FILES:
            raise ValueError(f"{file_name} is an invalid file. Only the following files: {FileNames.ALL_FILES} can be retrieved.")
        
        url = f"https://raw.githubusercontent.com/ComDePri/CFGpy/move_files_to_s3/CFGpy/files/{file_name}" # TODO: verify url and change branch when merging to dev
        response = requests.get(url, auth=HTTPBasicAuth('username', self.token))

        if response.status_code == 200:
            print("File content retrieved successfully:")
            print(f"Writing file {file_name} to the cache directory {FileNames.CACHE_DIR}.")
            if file_name.endswith(".npy"):
                with open(os.path.join(FileNames.CACHE_DIR, file_name), "wb") as f:
                        f.write(response.content)
            else:
                with open(os.path.join(FileNames.CACHE_DIR, file_name), "w", encoding="utf-8") as f:
                        f.write(response.text)
                
        else:
            print(f"Error: {response.status_code} - {response.text}")

        return None
    
    def get_file(self, file_name: str) -> None:
        if not os.path.exists(os.path.join(FileNames.CACHE_DIR, file_name)):
            self.get_raw_file_from_github(file_name=file_name)
        return None
    
    def load_json_data(self, file_name: str) -> dict:
        with open(os.path.join(FileNames.CACHE_DIR, file_name)) as f:
            json_data = json.load(f)
        return json_data

    @property
    def vanilla_data(self) -> dict:
        self.get_file(file_name=FileNames.VANILLA_DATA)
        if not self._vanilla_data:
           self._vanilla_data = self.load_json_data(file_name=FileNames.VANILLA_DATA)
        return self._vanilla_data
    
    @property
    def vanilla_features(self) -> dict:
        self.get_file(file_name=FileNames.VANILLA_FEATURES)
        if not self._vanilla_features:
            self._vanilla_features = pd.read_csv(os.path.join(FileNames.CACHE_DIR, FileNames.VANILLA_FEATURES))
        return self._vanilla_features
    
    @property
    def vanilla_gallery_counter(self) -> dict:
        self.get_file(file_name=FileNames.VANILLA_GALLERY_COUNTER)
        if not self._vanilla_gallery_counter:
            self._vanilla_gallery_counter = self.load_json_data(file_name=FileNames.VANILLA_GALLERY_COUNTER)
        return self._vanilla_gallery_counter
    
    @property
    def vanilla_giant_component(self) -> dict:
        self.get_file(file_name=FileNames.VANILLA_GIANT_COMPONENT)
        if not self._vanilla_giant_component:
            self._vanilla_giant_component = self.load_json_data(file_name=FileNames.VANILLA_GIANT_COMPONENT)
        return self._vanilla_giant_component
    
    @property
    def vanilla_step_counter(self) -> dict:
        self.get_file(file_name=FileNames.VANILLA_STEP_COUNTER)
        if not self._vanilla_step_counter:
            self._vanilla_step_counter = self.load_json_data(file_name=FileNames.VANILLA_STEP_COUNTER)
        return self._vanilla_step_counter
    
    @property
    def shape_network(self) -> dict:
        self.get_file(file_name=FileNames.SHAPE_NETWORK)
        if not self._shape_network:
            self._shape_network = nx.read_adjlist(os.path.join(FileNames.CACHE_DIR, FileNames.SHAPE_NETWORK), nodetype=int)
        return self._shape_network
    
    @property
    def id2coord(self) -> dict:
        self.get_file(file_name=FileNames.ID2COORD)
        if not self._id2coord:
           self._id2coord = np.load(os.path.join(FileNames.CACHE_DIR, FileNames.ID2COORD))
        return self._id2coord
    
    @property
    def shortest_paths_dict(self) -> dict:
        self.get_file(file_name=FileNames.SHORTEST_PATHS)
        if not self._shortest_paths_dict:
            self._shortest_paths_dict = self.load_json_data(file_name=FileNames.SHORTEST_PATHS)
        return self._shortest_paths_dict
