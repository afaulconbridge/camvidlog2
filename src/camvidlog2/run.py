import typer
from pathlib import Path
import numpy as np

import pandas as pd  # consider fireducks

import open_clip
import cv2
from typing import Generator
from PIL import Image
import torch.nn.functional

pd.options.mode.copy_on_write = True


def generate_frames_cv2(
    filename: str | Path,
) -> Generator[tuple[int, np.ndarray], None, None]:
    video_capture = cv2.VideoCapture(str(filename), cv2.CAP_ANY)
    success = True
    frame_no = 1
    while success:
        success, array = video_capture.read()
        if success:
            yield frame_no, array
        frame_no += 1
    video_capture.release()


def get_video_embeddings(
    filename: str,
) -> Generator[tuple[int, np.ndarray], None, None]:
    model, _, processor = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    model.eval()
    for frame_no, frame in generate_frames_cv2(filename):
        frame_pil = Image.fromarray(frame.squeeze())
        image_features = torch.nn.functional.normalize(
            model.encode_image(processor(frame_pil).unsqueeze(0)),
            dim=-1,
        )

        yield (
            frame_no,
            image_features[0].detach().numpy().astype(np.float32),
        )


def get_string_embedding(query: str) -> np.ndarray:
    model, _, processor = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    features = torch.nn.functional.normalize(
        model.encode_text(tokenizer(query)), dim=-1
    )
    return features[0].detach().numpy().astype(np.float32)


app = typer.Typer()


@app.command()
def load(videos: list[str]):
    # load existing array, if any
    existing_array = None
    if Path("tmp.feather").exists():
        existing_array = pd.read_feather("tmp.feather")

    for video in videos:
        if (
            existing_array is not None
            and existing_array["filename"].str.contains(video).any()
        ):
            print("File already loaded")
            print(video)
            continue

        frame_nos = []
        embeddings = []
        for frame_no, embedding in get_video_embeddings(video):
            frame_nos.append(frame_no)
            embeddings.append(embedding)
        array = pd.DataFrame.from_records(embeddings)
        array.insert(0, "frame_no", frame_nos)
        array.insert(0, "filename", video)

        # make sure all column names are strings so they roundtrip correctly
        array.columns = [str(x) for x in array.columns]  # type: ignore

        # add new data to existing array
        if existing_array is None:
            existing_array = array
        else:
            existing_array = pd.concat([existing_array, array])

        # save array (new or existing) to disk
        existing_array.to_feather("tmp.feather")
        print("saved file")


@app.command()
def query(query: str):
    df = pd.read_feather("tmp.feather")
    embedding = get_string_embedding(query)
    df_embeddings = df.drop(columns=["filename", "frame_no"])

    # dot product of unit vectors is the alignment between them (1 = equal, 0 = perpendicular)
    df["distances"] = df_embeddings.dot(embedding)

    # drop embedding columns and cleanup
    df = df[["filename", "frame_no", "distances"]]
    df.reset_index(drop=True, inplace=True)

    # get the rows that are the closest match
    grouped = df.loc[df.groupby("filename")["distances"].idxmax()]

    grouped.sort_values(by="distances", ascending=False, inplace=True)

    for i, row in enumerate(grouped.itertuples()):
        print(row)
        if i >= 10:
            break


if __name__ == "__main__":
    app()
