import math
from pathlib import Path

import numpy as np
import pandas as pd

from camvidlog2.bioclip.queries import load_embedding_dataframe
from camvidlog2.common.config import (
    EmbeddingCollection,
    EmbeddingGroup,
    FrameEmbedding,
    StringEmbedding,
)


def test_strings_load_embedding_via_json_filename() -> None:
    embedding_collection = EmbeddingCollection(
        groups=[EmbeddingGroup(items=[StringEmbedding(query="hello world")])]
    )

    df = load_embedding_dataframe(embedding_collection, pd.DataFrame())

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 512)
    assert math.isclose(np.linalg.norm(df), 1.0, rel_tol=0.0001, abs_tol=0.0001)


def test_frame_load_embedding_via_json_filename(
    video_path: Path,
    video_embeddings_path: Path,
) -> None:
    embedding_collection = EmbeddingCollection(
        groups=[
            EmbeddingGroup(
                items=[FrameEmbedding(filepath=video_path, frame_no=1)],
            )
        ],
    )
    video_embeddings = pd.read_feather(video_embeddings_path)

    df = load_embedding_dataframe(embedding_collection, video_embeddings)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 512)
    assert math.isclose(np.linalg.norm(df), 1.0, rel_tol=0.0001, abs_tol=0.0001)


def test_frame_load_embedding_via_json_filename_multi_group(
    video_path: Path,
    video_embeddings_path: Path,
) -> None:
    embedding_collection = EmbeddingCollection(
        groups=[
            EmbeddingGroup(
                items=[FrameEmbedding(filepath=video_path, frame_no=1)],
            ),
            EmbeddingGroup(
                items=[FrameEmbedding(filepath=video_path, frame_no=2)],
            ),
        ],
    )
    video_embeddings = pd.read_feather(video_embeddings_path)

    df = load_embedding_dataframe(embedding_collection, video_embeddings)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 512)
    assert math.isclose(np.linalg.norm(df.T[0]), 1.0, rel_tol=0.0001, abs_tol=0.0001)
    assert math.isclose(np.linalg.norm(df.T[1]), 1.0, rel_tol=0.0001, abs_tol=0.0001)


def test_frame_load_embedding_via_json_filename_multi_embeddings(
    video_path: Path,
    video_embeddings_path: Path,
) -> None:
    embedding_collection = EmbeddingCollection(
        groups=[
            EmbeddingGroup(
                items=[
                    FrameEmbedding(filepath=video_path, frame_no=1),
                    FrameEmbedding(filepath=video_path, frame_no=2),
                ],
            ),
        ]
    )
    video_embeddings = pd.read_feather(video_embeddings_path)

    df = load_embedding_dataframe(embedding_collection, video_embeddings)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 512)
    assert math.isclose(np.linalg.norm(df.T[0]), 1.0, rel_tol=0.0001, abs_tol=0.0001)
    assert math.isclose(np.linalg.norm(df.T[1]), 1.0, rel_tol=0.0001, abs_tol=0.0001)
