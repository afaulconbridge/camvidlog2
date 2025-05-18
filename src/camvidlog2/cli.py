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
from camvidlog2.queries import calculate_distances, load_embedding_group_dataframe
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

    # handle each query separately from this point onward
    # get the frame in each file that is the closest match
    index_loc_max = distances.groupby("filename").idxmax()
    for j, _ in enumerate(embedding_group.items, 0):
        query_max = pd.DataFrame(distances.loc[index_loc_max[j].tolist()][j])
        query_max.reset_index(names=["filename", "frame_no"], level=[1], inplace=True)
        query_max.sort_values(
            by=j,
            ascending=False,
            inplace=True,
        )  # type: ignore
        # this is a typing bug - it is legit to have non-string column names

        if outdir:
            # ensure query output subdir exists
            os.makedirs(outdir / f"{j + 1:03d}", exist_ok=True)
            # record results to a file
            query_max.to_csv(outdir / f"{j + 1:03d}" / "result.csv")

        for i, (filename, frame_no, distance) in enumerate(query_max.itertuples(), 1):
            print(f"{j + 1:3d} {i:3d} {frame_no:4d} {distance:.3f} {filename}")
            if outdir:
                try:
                    img_array = get_frame_by_no(filename, frame_no)
                except FrameError:
                    continue
                outpath = outdir / f"{j + 1:03d}" / f"{i:03d}.jpg"
                save(outpath, img_array)
                del img_array
            if i >= num:
                break


if __name__ == "__main__":
    app()
