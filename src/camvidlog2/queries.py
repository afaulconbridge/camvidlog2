import numpy as np
import pandas as pd

from camvidlog2.ai import get_string_embeddings
from camvidlog2.data import (
    EmbeddingCollection,
    FrameEmbedding,
    StringEmbedding,
)


def load_embedding_dataframe(
    embedding_collection: EmbeddingCollection,
    frame_embedding_data: pd.DataFrame,
) -> pd.DataFrame:
    # pre-calculate all the string embeddings we'll need in one go for efficiency
    embedding_strings = [
        item.query
        for embedding_group in embedding_collection.groups
        for item in embedding_group.items
        if isinstance(item, StringEmbedding)
    ]
    nda_strings = get_string_embeddings(embedding_strings)

    # combine the embeddings from strings and frames, in the original order!
    nda_combined_list: list[np.ndarray | pd.DataFrame] = []
    string_count = 0
    for embedding_group in embedding_collection.groups:
        for item in embedding_group.items:
            if isinstance(item, StringEmbedding):
                nda_combined_list.append(nda_strings[string_count, :])
                string_count += 1
            elif isinstance(item, FrameEmbedding):
                nda_combined_list.append(
                    frame_embedding_data.loc[(str(item.filepath), item.frame_no), :],
                )
            else:
                raise TypeError("Unexpected item in embedding_group")
    nda_combined = pd.DataFrame(np.stack(nda_combined_list))

    # make sure column names are strings
    nda_combined.columns = [str(x) for x in nda_combined.columns]  # type: ignore

    # return a combined dataframe of all embeddings
    # NOTE: this does _not_ include which embeddings are part of which groups!
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
        # if there aren't enough to roll, pandas will create na results; discard these
        distances.dropna()

    return distances


def calculate_results(distances: pd.Series) -> pd.DataFrame:
    # get the index of the most aligned frame in each file
    index_max = distances.groupby("filename").idxmax()
    # create a new dataframe of only the rows that are the max in each file
    results = pd.DataFrame({"distance": distances.loc[index_max.tolist()]})
    # from the frame_no from the index as there is now only one frame per file
    # still keep the column around so it can be referred to later
    results.reset_index(names=["filename", "frame_no"], level=[1], inplace=True)
    # sort the files by best first
    results.sort_values(
        by="distance",
        ascending=False,
        inplace=True,
    )
    return results
