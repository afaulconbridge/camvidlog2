import os
from itertools import islice
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from camvidlog2.bioclip.ai import get_video_embeddings
from camvidlog2.bioclip.data import (
    create,
)
from camvidlog2.bioclip.plots import plot_distances
from camvidlog2.bioclip.queries import (
    calculate_distances,
    calculate_results,
    load_embedding_dataframe,
)
from camvidlog2.common.config import (
    EmbeddingCollection,
    EmbeddingGroup,
    StringEmbedding,
    load_embedding_json,
)
from camvidlog2.common.data import load as data_load
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
        video_path = Path(video).resolve()
        if (
            existing_array is not None
            and (
                existing_array.index.get_level_values("filename") == str(video_path)
            ).any()
        ):
            print("File already loaded")
            print(video)
            continue

        if not video_path.exists():
            print("File does not exist")
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
        embedding_collection = load_embedding_json(json)
    else:
        if not queries:
            raise ValueError("Must provide either --json or [QUERIES]")
        embedding_collection = EmbeddingCollection(
            groups=[
                EmbeddingGroup(
                    items=[StringEmbedding(query=q) for q in queries],
                ),
            ]
        )

    if outdir:
        # ensure the target location exists
        outdir = outdir.resolve()
        os.makedirs(outdir, exist_ok=True)
        # record what search was run
        with open(outdir / "query.json", "w") as json_out:
            json_out.write(embedding_collection.model_dump_json(indent=2))

    search_embeddings = load_embedding_dataframe(
        embedding_collection,
        video_embeddings,
    )

    distances = calculate_distances(video_embeddings, search_embeddings, roll=roll)

    distances_index = 0
    for j, embedding_group in enumerate(embedding_collection.groups):
        # calculate the average distance over the embeddings in this group
        group_distances = distances.iloc[
            :, distances_index : distances_index + len(embedding_group.items)
        ]
        mean_distances = group_distances.mean(axis=1)
        # update the index pointer
        distances_index += len(embedding_group.items)
        # calculate the results
        results = calculate_results(mean_distances)

        group_name = embedding_group.name if embedding_group.name else f"{j + 1:03d}"

        if outdir:
            # ensure output subdir exists
            os.makedirs(outdir / group_name, exist_ok=True)
            # record all results to a file
            results.to_csv(outdir / group_name / "result.csv")
            # draw plot(s) for output
            plot_distances(results, outdir / group_name / "result.png")
        # for top results, print and save image
        for rank, (filename, frame_no, score) in enumerate(
            islice(results.itertuples(), num),
            1,
        ):
            print(
                f"{group_name} {rank:3d} {frame_no:4d} {score:.3f} {filename}",
            )
            if outdir:
                # save the best frames
                try:
                    img_array = get_frame_by_no(filename, frame_no)
                except FrameError:
                    print(
                        f"Warning: Failed to extract frame {frame_no} from {filename}"
                    )
                    continue
                outpath = outdir / group_name / f"{rank:03d}.jpg"
                save(outpath, img_array)
                del img_array


if __name__ == "__main__":
    app()
