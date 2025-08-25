from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class EmbeddingSource(StrEnum):
    STRING = "string"
    FRAME = "frame"
    IMAGE = "image"


class StringEmbedding(BaseModel):
    source: Literal["string"] = "string"
    query: str
    human_label: str = ""


class FrameEmbedding(BaseModel):
    source: Literal["frame"] = "frame"
    filepath: Path
    frame_no: int


class ImageEmbedding(BaseModel):
    source: Literal["image"] = "image"
    filepath: Path


# see https://stackoverflow.com/a/70917353/932342
AnyEmbedding = Annotated[
    StringEmbedding | FrameEmbedding | ImageEmbedding, Field(discriminator="source")
]


class EmbeddingGroup(BaseModel):
    name: str = ""
    items: list[AnyEmbedding]


class EmbeddingCollection(BaseModel):
    groups: list[EmbeddingGroup]


def load_embedding_json(filename: str | Path) -> EmbeddingCollection:
    with open(Path(filename), "r") as json_file:
        json_string = json_file.read()
    return EmbeddingCollection.model_validate_json(json_string)
