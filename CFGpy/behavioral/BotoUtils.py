import os
import time
import boto3
import hashlib


CACHE_DIR: str = os.getenv("CACHE_DIR", "./cache")
CFGPY_BUCKET: str = "cfgpy-data"


class BotoUtils:

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id="AKIAUWHEKK3UPKUQVBPY",
            aws_secret_access_key="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"
            )

    def get_file_from_s3(self, bucket_name: str, key: str, local_filename: str = None) -> str:

        if local_filename is None:
            local_filename = os.path.join(CACHE_DIR, hashlib.md5(key.encode()).hexdigest())
        
        if os.path.exists(local_filename):
            print(f"File is already cached: {local_filename}")
            return local_filename
        
        print(f"Downloading {key} from bucket {bucket_name}...")
        self.s3_client.download_file(bucket_name, key, local_filename)
        print(f"File downloaded and cached at: {local_filename}")
        
        return local_filename
    
    def get_files_from_s3(self, bucket_name: str, keys: list[str]) -> list[str]:
        return [self.get_file_from_s3(bucket_name, key) for key in keys]
    
    def clean_cache(expiration_time: int = 86400):

        current_time = time.time()
        for file in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, file)
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
            self.vanilla_gallery_counter: dict = self.boto_utils.get_file_from_s3(bucket_name=CFGPY_BUCKET, key="vanilla_data/gallery_counter.json")
            self.vanilla_giant_component: dict = self.boto_utils.get_file_from_s3(bucket_name=CFGPY_BUCKET, key="vanilla_data/giant_component.json")
            self.vanilla_step_counter: dict = self.boto_utils.get_file_from_s3(bucket_name=CFGPY_BUCKET, key="vanilla_data/step_counter.json")
            self.shape_network = self.boto_utils.get_file_from_s3(bucket_name=CFGPY_BUCKET, key="all_shapes.adjlist")
            self.id2coord = self.boto_utils.get_file_from_s3(bucket_name=CFGPY_BUCKET, key="shortest_path_len.json")
            self.shortest_paths_dict: dict = self.boto_utils.get_file_from_s3(bucket_name=CFGPY_BUCKET, key="all_shapes.adjlist")
            self._initialized = True
