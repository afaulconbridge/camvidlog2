import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from camvidlog2.ai import get_string_embedding, get_video_embeddings
from camvidlog2.data import create
from camvidlog2.data import load as data_load
from camvidlog2.vid import generate_frames_cv2, get_frame_by_no, save

app = typer.Typer()


@app.command()
def load(videos: list[str]):
    # load existing array, if any
    existing_array = data_load(Path("tmp.feather"))
    for video in videos:
        if (
            existing_array is not None
            and existing_array["filename"].str.contains(video).any()
        ):
            print("File already loaded")
            print(video)
            continue

        video_path = Path(video)
        if not video_path.exists():
            print("file does not exist")
            print(video)
            continue

        array = create(
            video_path, get_video_embeddings(generate_frames_cv2(video_path))
        )

        # add new data to existing array
        if existing_array is None:
            existing_array = array
        else:
            existing_array = pd.concat([existing_array, array])

        # save array (new or existing) to disk
        existing_array.to_feather("tmp.feather")
        print("saved file")


@app.command()
def query(
    query: Annotated[str, typer.Argument()],
    outdir: Annotated[Path | None, typer.Option()] = None,
):
    df = data_load(Path("tmp.feather"))
    if df is None:
        raise ValueError("Unable to load database")

    if outdir:
        # ensure the target location exists
        outdir = outdir.absolute().resolve()
        os.makedirs(outdir, exist_ok=True)

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
        _, filename, frame_no, distance = row
        print(f"{i:3d} {frame_no:4d} {distance:.3f} {filename}")
        if outdir:
            img_array = get_frame_by_no(filename, frame_no)
            save(outdir / f"{i:03d}.jpg", img_array)
        if i >= 10:  # TODO make this a command line option
            break


if __name__ == "__main__":
    app()
