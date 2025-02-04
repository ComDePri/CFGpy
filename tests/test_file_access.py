import pytest
import os
from pathlib import Path
from CFGpy.utils import FilesHandler, FileNames


@pytest.fixture
def files_handler():
    return FilesHandler()


def test_clear_cache(files_handler):
    files_handler.clear_cache()
    for f in FileNames.ALL_FILES:
        assert not os.path.exists(os.path.join(FileNames.CACHE_DIR, f))


def test_get_files_from_github(files_handler):
    files_handler.clear_cache()
    for f in FileNames.ALL_FILES:
        assert not os.path.exists(os.path.join(FileNames.CACHE_DIR, f))
        files_handler.get_raw_file_from_github(file_name=f, branch=os.getenv("GIT_BRANCH", "main"))
        assert os.path.exists(os.path.join(FileNames.CACHE_DIR, f))


def test_load_json_data(files_handler):
    json_data = files_handler.load_json_data(file_name="test_json_file.json", dir_path=os.path.join(Path(__file__).parent, "test_files"))
    assert isinstance(json_data, dict)
    assert len(json_data) == 50
    assert set(json_data.values()) == {0, 1, 2, 3, 4}
    assert len(json_data.keys()) == 50


def test_files_handler_properties(files_handler):
    files_handler.clear_cache()
    assert len(files_handler.vanilla_data) > 1
    assert len(files_handler.vanilla_features) > 1
    assert len(files_handler.vanilla_gallery_counter) > 1
    assert len(files_handler.vanilla_giant_component) > 1
    assert len(files_handler.vanilla_step_counter) > 1
    assert len(files_handler.shape_network) > 1
    assert len(files_handler.id2coord) > 1
    assert len(files_handler.shortest_paths_dict) > 1
