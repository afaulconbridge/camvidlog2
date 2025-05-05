import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from camvidlog2.ai import get_string_embeddings, get_video_embeddings
from camvidlog2.data import create
from camvidlog2.data import load as data_load
from camvidlog2.vid import FrameError, generate_frames_cv2, get_frame_by_no, save

app = typer.Typer()


@app.command()
def load(
    videos: list[str],
    db: Annotated[Path, typer.Option()] = Path("tmp.feather"),
):
    # load existing array, if any
    existing_array = data_load(db)
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
        existing_array.to_feather(db)
        print("saved file")


@app.command()
def query(
    queries: Annotated[list[str], typer.Argument()],
    outdir: Annotated[Path | None, typer.Option()] = None,
    db: Annotated[Path, typer.Option()] = Path("tmp.feather"),
    num: Annotated[int, typer.Option("--num", "-n")] = 10,
    roll: Annotated[int, typer.Option("--rolling", "-r")] = 0,
):
    df = data_load(db)
    if df is None:
        raise ValueError("Unable to load database")

    if outdir:
        # ensure the target location exists
        outdir = outdir.absolute().resolve()
        os.makedirs(outdir, exist_ok=True)

    string_embeddings = get_string_embeddings(queries)
    frame_embeddings = df.drop(columns=["filename", "frame_no"])

    # dot product of unit vectors is the alignment between them (1 = equal, 0 = perpendicular)
    distances = frame_embeddings.dot(string_embeddings.transpose())
    # drop frame embedding columns and cleanup
    df = df[["filename", "frame_no"]]

    # bolt distances onto existing dataframe
    df = pd.concat([df, distances], axis=1)

    # apply a rolling mean calculation if appropriate
    if roll > 0:
        # reindex on the combination we need
        df.set_index(["filename", "frame_no"], inplace=True)
        df = df.groupby(level="filename").rolling(window=roll, center=True).mean()
        # avoid index duplication bug: https://stackoverflow.com/questions/42119793/doing-a-groupby-and-rolling-window-on-a-pandas-dataframe-with-a-multilevel-index
        df = df.droplevel(0)

    # handle each query separately from this point onward
    df.reset_index(inplace=True)
    for j, _ in enumerate(queries, 1):
        # get the frame in each file that is the closest match
        grouped = df.loc[df.groupby("filename")[j - 1].idxmax()]

        grouped.sort_values(
            by=(j - 1),
            ascending=False,
            inplace=True,
        )  # type: ignore
        # this is a typing bug - it is legit to have non-string column names

        for i, row in enumerate(grouped.itertuples(), 1):
            _, filename, frame_no, *distances = row
            distance = distances[j - 1]
            print(f"{j:3d} {i:3d} {frame_no:4d} {distance:.3f} {filename}")
            if outdir:
                try:
                    img_array = get_frame_by_no(filename, frame_no)
                except FrameError:
                    continue
                outpath = outdir / f"{j:03d}" / f"{i:03d}.jpg"
                os.makedirs(outpath.parent, exist_ok=True)
                save(outpath, img_array)
                del img_array
            if i >= num:
                break


if __name__ == "__main__":
    app()
