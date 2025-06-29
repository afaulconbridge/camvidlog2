import math
from pathlib import Path

import numpy as np

from camvidlog2.bioclip.ai import (
    get_string_embedding,
    get_string_embeddings,
    get_video_embeddings,
)
from camvidlog2.vid import get_frame_by_no


def test_get_string_embedding():
    embedding = get_string_embedding("hello world")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32

    # magnitude of the embedding should be close to one
    assert math.isclose(np.linalg.norm(embedding), 1.0, rel_tol=0.0001, abs_tol=0.0001)


def test_get_string_embeddings():
    embeddings = get_string_embeddings(["hello", "world"])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 512)
    assert embeddings.dtype == np.float32

    # magnitude of the embedding should be close to one
    assert math.isclose(
        np.linalg.norm(embeddings[0]), 1.0, rel_tol=0.0001, abs_tol=0.0001
    )
    assert math.isclose(
        np.linalg.norm(embeddings[1]), 1.0, rel_tol=0.0001, abs_tol=0.0001
    )


def test_get_video_embeddings(video_path: Path):
    frame = get_frame_by_no(video_path, 1)

    _, embedding = next(get_video_embeddings([(0, frame)]))

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 512
    assert embedding.dtype == np.float32

    # magnitude of the embedding should be close to one
    assert math.isclose(np.linalg.norm(embedding), 1.0, rel_tol=0.0001, abs_tol=0.0001)
