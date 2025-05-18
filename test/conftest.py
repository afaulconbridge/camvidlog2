import os.path
from pathlib import Path

from pytest import fixture

from camvidlog2.ai import get_video_embeddings
from camvidlog2.data import create
from camvidlog2.vid import generate_frames_cv2


@fixture(name="data_directory", scope="session")
def fixture_data_directory() -> Path:
    data_path = Path(os.path.dirname(os.path.realpath(__file__))) / "data"
    assert data_path.exists()
    assert data_path.is_dir()
    return data_path


@fixture(name="video_path", scope="session")
def fixture_video_path(data_directory: Path) -> Path:
    video_path = data_directory / "test.mp4"
    assert video_path.exists()
    assert video_path.is_file()
    return video_path


@fixture(name="video_embeddings_path", scope="session")
def fixture_video_embeddings_path(video_path: Path) -> Path:
    # create a database
    db_path = video_path.parent / "test.feather"
    # support re-use in other tests and runs
    # may cause a race condition in rare circumstances
    if not db_path.exists():
        # add embeddings to database
        df = create(video_path, get_video_embeddings(generate_frames_cv2(video_path)))
        # save database
        df.to_feather(db_path)

    return db_path
