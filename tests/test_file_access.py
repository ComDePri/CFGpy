import os
from pathlib import Path
import pytest
import CFGpy.utils 
print("hello " + str(dir(CFGpy.utils))) 
from CFGpy.utils.FilesHandler import FilesHandler, FileNames


@pytest.fixture
def files_handler():
    """Fixture to provide a fresh FilesHandler for each test."""
    files_handler = FilesHandler()
    files_handler.clear_cache()  # Ensure cache is cleared before each test
    return files_handler


def test_clear_cache(files_handler):
    """Test that cache is cleared."""
    for f in FileNames.ALL_FILES:
        assert not os.path.exists(os.path.join(FileNames.CACHE_DIR, f))


def test_get_files_from_github(files_handler):
    """Test downloading files from GitHub."""
    for f in FileNames.ALL_FILES:
        assert not os.path.exists(os.path.join(FileNames.CACHE_DIR, f))
        files_handler.get_raw_file_from_github(file_name=f, branch=os.getenv("GIT_BRANCH", "main"))
        assert os.path.exists(os.path.join(FileNames.CACHE_DIR, f))
    
    with pytest.raises(FileNotFoundError):
        files_handler.get_raw_file_from_github(file_name=FileNames.VANILLA_DATA, branch="not_a_branch")

    with pytest.raises(ValueError):
        files_handler.get_raw_file_from_github(file_name="not_a_file", branch=os.getenv("GIT_BRANCH", "main"))


def test_load_json_data(files_handler):
    """Test loading JSON data from file."""
    json_data: dict = files_handler.load_json_data(file_name="test_json_file.json", dir_path=os.path.join(Path(__file__).parent, "test_files"))
    assert isinstance(json_data, dict)
    assert len(json_data) == 50
    assert set(json_data.values()) == {0, 1, 2, 3, 4}
    assert len(json_data.keys()) == 50


def test_files_handler_properties(files_handler):
    """Test various properties of the FilesHandler."""
    assert len(files_handler.vanilla_data) > 1
    assert len(files_handler.vanilla_features) > 1
    assert len(files_handler.vanilla_gallery_counter) > 1
    assert len(files_handler.vanilla_giant_component) > 1
    assert len(files_handler.vanilla_step_counter) > 1
    assert len(files_handler.shape_network) > 1
    assert len(files_handler.id2coord) > 1
    assert len(files_handler.shortest_paths_dict) > 1
