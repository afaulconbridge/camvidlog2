import os.path
from pathlib import Path

from pytest import fixture


@fixture(name="data_directory")
def fixture_data_directory() -> Path:
    return Path(os.path.dirname(os.path.realpath(__file__))) / "data"


@fixture(name="video_path")
def fixture_video_path(data_directory: Path) -> Path:
    return data_directory / "test.mp4"
