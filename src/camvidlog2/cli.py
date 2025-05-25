import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from camvidlog2.ai import get_video_embeddings
from camvidlog2.data import (
    EmbeddingGroup,
    StringEmbedding,
    create,
    load_embedding_group_json,
)
from camvidlog2.data import load as data_load
from camvidlog2.queries import (
    calculate_distances,
    calculate_results,
    load_embedding_group_dataframe,
)
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
    queries: Annotated[list[str] | None, typer.Argument()] = None,
    outdir: Annotated[Path | None, typer.Option()] = None,
    db: Annotated[Path, typer.Option()] = Path("tmp.feather"),
    num: Annotated[int, typer.Option("--num", "-n", min=0)] = 10,
    roll: Annotated[int, typer.Option("--rolling", "-r", min=0)] = 0,
    json: Annotated[Path | None, typer.Option()] = None,
):
    video_embeddings = data_load(db)
    if video_embeddings is None:
        raise ValueError("Unable to load database")

    if json:
        if queries:
            raise ValueError("Cannot use --json and [QUERIES]")
        embedding_group = load_embedding_group_json(json)
    else:
        if not queries:
            raise ValueError("Must provide either --json or [QUERIES]")
        embedding_group = EmbeddingGroup(
            items=[StringEmbedding(query=q) for q in queries],
        )

    if outdir:
        # ensure the target location exists
        outdir = outdir.absolute().resolve()
        os.makedirs(outdir, exist_ok=True)
        # record what search was run
        with open(outdir / "query.json", "w") as json_out:
            json_out.write(embedding_group.model_dump_json(indent=2))

    search_embeddings = load_embedding_group_dataframe(
        embedding_group,
        video_embeddings,
    )

    distances = calculate_distances(video_embeddings, search_embeddings, roll=roll)

    # output for an average of all distances
    distances["avg"] = distances.mean(axis=1)
    for result in calculate_results(distances["avg"], num, roll):
        print(
            f"avg {result.rank:3d} {result.frame_no:4d} {result.score:.3f} {result.filename}"
        )
        if outdir:
            # ensure output subdir exists
            os.makedirs(outdir / "avg", exist_ok=True)
            try:
                img_array = get_frame_by_no(result.filename, result.frame_no)
            except FrameError:
                continue
            outpath = outdir / "avg" / f"{result.rank:03d}.jpg"
            save(outpath, img_array)
            del img_array
    # TODO output csv

    # output for each query separately
    for j, _ in enumerate(embedding_group.items, 0):
        for result in calculate_results(distances[j], num, roll):
            print(
                f"{j + 1:3d} {result.rank:3d} {result.frame_no:4d} {result.score:.3f} {result.filename}"
            )
            if outdir:
                # ensure output subdir exists
                os.makedirs(outdir / f"{j + 1:3d}", exist_ok=True)
                try:
                    img_array = get_frame_by_no(result.filename, result.frame_no)
                except FrameError:
                    continue
                outpath = outdir / f"{j + 1:3d}" / f"{result.rank:03d}.jpg"
                save(outpath, img_array)
                del img_array
        # TODO output csv


if __name__ == "__main__":
    app()
