import json
import os
from pathlib import Path
import shutil
from typing import Optional
import networkx as nx
import numpy as np
import pandas as pd
from appdirs import user_cache_dir
import requests


class FileNames:

    CACHE_DIR: str = user_cache_dir("ComDePri-CFGpy")
    VANILLA_DIR: str = os.path.join(CACHE_DIR, "vanilla_data")

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
            self._vanilla_data: dict = {}
            self._vanilla_features: pd.DataFrame = {}
            self._vanilla_gallery_counter: dict = {}
            self._vanilla_giant_component: dict = {}
            self._vanilla_step_counter: dict = {}
            self._shape_network: nx.Graph = None
            self._id2coord: np.ndarray = None
            self._shortest_paths_dict: dict = {}
            self._initialized = True
    
    @staticmethod
    def clear_cache() -> None:

        cache_dir = Path(FileNames.CACHE_DIR)
        for item in cache_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item) 

        return None
    
    @staticmethod
    def get_raw_file_from_github(*, file_name: str, branch: Optional[str] = "main") -> None: 
        
        if file_name not in FileNames.ALL_FILES:
            raise ValueError(f"{file_name} is an invalid file. Only the following files: {FileNames.ALL_FILES} can be retrieved.")
        
        url = f"https://raw.githubusercontent.com/ComDePri/CFGpy/{branch}/CFGpy/files/{file_name}"
        response = requests.get(url)

        if response.status_code == 200:

            dir_path: str = os.path.join(FileNames.CACHE_DIR, os.path.dirname(file_name))

            if not os.path.exists(dir_path):
                os.makedirs(dir_path) # Create directories if they don't exist

            file_path: str = os.path.join(FileNames.CACHE_DIR, file_name)
            print(f"Retrived file {file_name} successfully. Saving file to {file_path}.")

            if file_name.endswith(".npy"):
                with open(file_path, "wb") as f:
                        f.write(response.content)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                
        else:
            raise FileNotFoundError(f"Error: {response.status_code} - {response.text}")

        return None
    
    def get_file(self, *, file_name: str) -> None:
        if not os.path.exists(os.path.join(FileNames.CACHE_DIR, file_name)):
            self.get_raw_file_from_github(file_name=file_name, branch=os.getenv("GIT_BRANCH", "main"))
        return None
    
    def load_json_data(self, *, file_name: str, dir_path: Optional[str] = FileNames.CACHE_DIR) -> dict:
        with open(os.path.join(dir_path, file_name)) as f:
            json_data = json.load(f)
        return json_data

    @property
    def vanilla_data(self) -> dict:
        if not self._vanilla_data:
           self.get_file(file_name=FileNames.VANILLA_DATA)
           self._vanilla_data = self.load_json_data(file_name=FileNames.VANILLA_DATA)
        return self._vanilla_data
    
    @property
    def vanilla_features(self) -> pd.DataFrame:
        if not self._vanilla_features:
            self.get_file(file_name=FileNames.VANILLA_FEATURES)
            self._vanilla_features = pd.read_csv(os.path.join(FileNames.CACHE_DIR, FileNames.VANILLA_FEATURES))
        return self._vanilla_features
    
    @property
    def vanilla_gallery_counter(self) -> dict:
        if not self._vanilla_gallery_counter:
            self.get_file(file_name=FileNames.VANILLA_GALLERY_COUNTER)
            self._vanilla_gallery_counter = self.load_json_data(file_name=FileNames.VANILLA_GALLERY_COUNTER)
        return self._vanilla_gallery_counter
    
    @property
    def vanilla_giant_component(self) -> dict:
        if not self._vanilla_giant_component:
            self.get_file(file_name=FileNames.VANILLA_GIANT_COMPONENT)
            self._vanilla_giant_component = self.load_json_data(file_name=FileNames.VANILLA_GIANT_COMPONENT)
        return self._vanilla_giant_component
    
    @property
    def vanilla_step_counter(self) -> dict:
        if not self._vanilla_step_counter:
            self.get_file(file_name=FileNames.VANILLA_STEP_COUNTER)
            self._vanilla_step_counter = self.load_json_data(file_name=FileNames.VANILLA_STEP_COUNTER)
        return self._vanilla_step_counter
    
    @property
    def shape_network(self) -> nx.Graph:
        if self._shape_network is None:
            self.get_file(file_name=FileNames.SHAPE_NETWORK)
            self._shape_network = nx.read_adjlist(os.path.join(FileNames.CACHE_DIR, FileNames.SHAPE_NETWORK), nodetype=int)
        return self._shape_network
    
    @property
    def id2coord(self) -> np.ndarray:
        if self._id2coord is None:
           self.get_file(file_name=FileNames.ID2COORD)
           self._id2coord = np.load(os.path.join(FileNames.CACHE_DIR, FileNames.ID2COORD))
        return self._id2coord
    
    @property
    def shortest_paths_dict(self) -> dict:
        if not self._shortest_paths_dict:
            self.get_file(file_name=FileNames.SHORTEST_PATHS)
            self._shortest_paths_dict = self.load_json_data(file_name=FileNames.SHORTEST_PATHS)
        return self._shortest_paths_dict
