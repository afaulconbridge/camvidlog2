import numpy as np
import pandas as pd

from camvidlog2.ai import get_string_embeddings
from camvidlog2.data import (
    EmbeddingGroup,
    FrameEmbedding,
    StringEmbedding,
)


def load_embedding_group_dataframe(
    embedding_group: EmbeddingGroup,
    frame_embedding_data: pd.DataFrame,
) -> pd.DataFrame:
    embedding_strings = [
        i.query for i in embedding_group.items if isinstance(i, StringEmbedding)
    ]
    nda_strings = get_string_embeddings(embedding_strings)

    embedding_frames = [
        i for i in embedding_group.items if isinstance(i, FrameEmbedding)
    ]
    nda_frames = (
        np.stack(
            [
                frame_embedding_data.loc[(str(f.filepath), f.frame_no)]
                for f in embedding_frames
            ]
        )
        if embedding_frames
        else None
    )

    # combine the embeddings from strings and frames, in the original order!

    interleafing = [
        (
            nda_strings[i, :] if isinstance(e, StringEmbedding) else None,
            nda_frames[i] if isinstance(e, FrameEmbedding) else None,
        )
        for i, e in enumerate(embedding_group.items)
    ]

    nda_combined = pd.DataFrame(
        np.stack([a if a is not None else b for a, b in interleafing])
    )

    # make sure column names are strings
    nda_combined.columns = [str(x) for x in nda_combined.columns]  # type: ignore

    return nda_combined


def calculate_distances(
    video_embeddings: pd.DataFrame,
    search_embeddings: pd.DataFrame,
    roll: int = 0,
) -> pd.DataFrame:
    # dot product of unit vectors is the alignment between them (1 = equal, 0 = perpendicular)
    # matrices must be sizes XxY and YxZ
    # indexes must match (i.e. nunindexed)
    video_embeddings_for_dot = video_embeddings.reset_index(drop=True)
    distances = video_embeddings_for_dot.dot(search_embeddings.T)

    # reuse the existing index for the new distances
    distances.index = video_embeddings.index

    # apply a rolling mean calculation if appropriate
    if roll > 0:
        # note: this can use a lot of memory!
        distances = (
            distances.groupby(level="filename").rolling(window=roll, center=True).mean()
        )
        # avoid index duplication bug: https://stackoverflow.com/questions/42119793/doing-a-groupby-and-rolling-window-on-a-pandas-dataframe-with-a-multilevel-index
        distances = distances.droplevel(0)

    return distances
