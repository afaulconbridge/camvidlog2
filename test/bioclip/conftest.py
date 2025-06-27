from pathlib import Path

import pytest

from camvidlog2.bioclip.ai import get_video_embeddings
from camvidlog2.bioclip.data import create
from camvidlog2.vid import generate_frames_cv2


@pytest.fixture(name="video_embeddings_path", scope="session")
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
