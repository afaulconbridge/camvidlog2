import os.path
from pathlib import Path

from pytest import fixture


@fixture(name="data_directory")
def fixture_data_directory() -> Path:
    data_path = Path(os.path.dirname(os.path.realpath(__file__))) / "data"
    assert data_path.exists()
    assert data_path.is_dir()
    return data_path


@fixture(name="video_path")
def fixture_video_path(data_directory: Path) -> Path:
    video_path = data_directory / "test.mp4"
    assert video_path.exists()
    assert video_path.is_file()
    return video_path
