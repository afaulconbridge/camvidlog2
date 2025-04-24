from pathlib import Path
import pandas as pd
import numpy as np
from typing import Iterable


def load(path: Path) -> pd.DataFrame | None:
    # load existing array, if any
    existing_array = None
    if path.exists():
        existing_array = pd.read_feather("tmp.feather")

    return existing_array


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

    return array
