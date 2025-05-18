from pathlib import Path

import pandas as pd

from camvidlog2.data import EmbeddingGroup, FrameEmbedding, StringEmbedding
from camvidlog2.queries import load_embedding_group_dataframe


def test_strings_load_embedding_via_json_filename(tmp_path: Path) -> None:
    embedding_group = EmbeddingGroup(items=[StringEmbedding(query="hello world")])

    df = load_embedding_group_dataframe(embedding_group, pd.DataFrame())

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 512)


def test_frame_load_embedding_via_json_filename(
    video_path: Path, video_embeddings_path: Path
) -> None:
    embedding_group = EmbeddingGroup(
        items=[FrameEmbedding(filepath=video_path, frame_no=1)]
    )
    video_embeddings = pd.read_feather(video_embeddings_path)

    df = load_embedding_group_dataframe(embedding_group, video_embeddings)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 512)
