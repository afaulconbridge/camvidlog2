from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def create(
    video: Path, frame_embeddings: Iterable[tuple[int, np.ndarray]]
) -> pd.DataFrame:
    frame_nos = []
    embeddings = []
    for frame_no, embedding in frame_embeddings:
        frame_nos.append(frame_no)
        embeddings.append(embedding)
    array = pd.DataFrame.from_records(embeddings)
    array.insert(0, "frame_no", frame_nos)
    array.insert(0, "filename", str(video))

    # make sure all column names are strings so they roundtrip correctly
    array.columns = [str(x) for x in array.columns]  # type: ignore

    # apply the appropriate index
    array.set_index(["filename", "frame_no"], inplace=True)
    return array
