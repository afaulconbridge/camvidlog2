from pathlib import Path
from collections.abc import Iterable
from typing import Annotated, Literal, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class StringEmbedding(BaseModel):
    source: Literal["string"] = "string"
    query: str
    human_label: str = ""


class FrameEmbedding(BaseModel):
    source: Literal["frame"] = "frame"
    filepath: Path
    frame_no: int


# see https://stackoverflow.com/a/70917353/932342
AnyEmbedding = Annotated[
    Union[StringEmbedding, FrameEmbedding], Field(discriminator="source")
]


class EmbeddingGroup(BaseModel):
    items: list[AnyEmbedding]


def load_embedding_group_json(filename: str | Path) -> EmbeddingGroup:
    with open(Path(filename), "r") as json_file:
        json_string = json_file.read()
    return EmbeddingGroup.model_validate_json(json_string)


def load(path: Path) -> pd.DataFrame | None:
    # load existing array, if any
    existing_array = None
    if path.exists():
        existing_array = pd.read_feather(path)
        existing_array.set_index(["filename", "frame_no"], inplace=True)

        # make sure all column names are strings so they roundtrip correctly
        existing_array.columns = [str(x) for x in existing_array.columns]  # type: ignore

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

    # apply the appropriate index
    array.set_index(["filename", "frame_no"], inplace=True)
    return array
