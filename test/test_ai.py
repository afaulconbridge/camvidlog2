import math
from pathlib import Path

import numpy as np

from camvidlog2.ai import get_string_embedding, get_video_embeddings
from camvidlog2.vid import get_frame_by_no


def test_get_text_embeddings():
    embedding = get_string_embedding("hello world")

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 512
    assert embedding.dtype == np.float32

    # magnitude of the embedding should be close to one
    assert math.isclose(np.linalg.norm(embedding), 1.0, rel_tol=0.0001, abs_tol=0.0001)


def test_get_video_embeddings(video_path: Path):
    frame = get_frame_by_no(video_path, 1)

    _, embedding = next(get_video_embeddings([(0, frame)]))

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 512
    assert embedding.dtype == np.float32

    # magnitude of the embedding should be close to one
    assert math.isclose(np.linalg.norm(embedding), 1.0, rel_tol=0.0001, abs_tol=0.0001)
