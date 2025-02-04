import os
from pathlib import Path
import unittest
from CFGpy.utils import FilesHandler, FileNames


class FilesHandlerTests(unittest.TestCase):
    
    def setUp(self):
        self.files_handler: FilesHandler = FilesHandler()
    
    def test_clear_cache(self):
        self.files_handler.clear_cache()
        for f in FileNames.ALL_FILES:
            self.assertFalse(os.path.exists(os.path.join(FileNames.CACHE_DIR, f)))
            
    def test_get_files_from_github(self):
        self.files_handler.clear_cache()
        for f in FileNames.ALL_FILES:
            self.assertFalse(os.path.exists(os.path.join(FileNames.CACHE_DIR, f)))
            self.files_handler.get_raw_file_from_github(file_name=f, branch=os.getenv("GIT_BRANCH", "main"))
            self.assertTrue(os.path.exists(os.path.join(FileNames.CACHE_DIR, f)))
    
    def test_load_json_data(self):
        json_data: dict = self.files_handler.load_json_data(file_name="test_json_file.json", dir_path=os.path.join(Path(__file__).parent, "test_files"))
        self.assertEqual(type(json_data), dict)
        self.assertEqual(len(json_data), 50)
        self.assertEqual(set(json_data.values()), {0, 1, 2, 3, 4})
        self.assertEqual(len(json_data.keys()), 50)
    
    def test_files_handler_properties(self):
        self.files_handler.clear_cache()
        self.assertGreater(len(self.files_handler.vanilla_data), 1)
        self.assertGreater(len(self.files_handler.vanilla_features), 1)
        self.assertGreater(len(self.files_handler.vanilla_gallery_counter), 1)
        self.assertGreater(len(self.files_handler.vanilla_giant_component), 1)
        self.assertGreater(len(self.files_handler.vanilla_step_counter), 1)
        self.assertGreater(len(self.files_handler.shape_network), 1)
        self.assertGreater(len(self.files_handler.id2coord), 1)
        self.assertGreater(len(self.files_handler.shortest_paths_dict), 1)

if __name__ == "__main__":
    unittest.main()