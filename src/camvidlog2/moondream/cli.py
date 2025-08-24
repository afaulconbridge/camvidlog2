from pathlib import Path
from typing import Annotated

import click
import pandas as pd
import typer

from camvidlog2.common.data import load as data_load
from camvidlog2.moondream.ai import generate_bboxes
from camvidlog2.moondream.data import create
from camvidlog2.moondream.plot import overlay_detections
from camvidlog2.vid import generate_frames_cv2, get_video_stats, save_video

app = typer.Typer()


@app.command()
def load(
    videos: list[Path],
    classes: Annotated[list[str], typer.Option("--class", "-c")],
    db: Annotated[Path, typer.Option()] = Path("tmp.feather"),
) -> None:
    # load existing array, if any
    existing_array = data_load(db)

    # moondream currently supports a single prompt; keep UI honest
    if len(classes) != 1:
        raise typer.BadParameter("exactly one --class is supported for now")

    for video in videos:
        video = video.resolve()
        if (
            existing_array is not None
            and (existing_array.index.get_level_values("filename") == str(video)).any()
        ):
            typer.echo(f"Already loaded: {video}")
            continue

        if not video.exists():
            typer.echo(f"Missing file: {video}")
            continue

        # inference
        array = create(
            video,
            generate_bboxes(
                (v for _, v in generate_frames_cv2(video)),
                class_names=classes,
            ),
            classes,
        )

        # add new data to existing array
        if existing_array is None:
            existing_array = array
        else:
            existing_array = pd.concat([existing_array, array])

        # save array (new or existing) to disk
        db.parent.mkdir(parents=True, exist_ok=True)
        existing_array.to_feather(db)
        typer.echo(f"Saved database: {db}")


@app.command()
def show(
    video: Path,
    output: Path,
    db: Annotated[Path, typer.Option()] = Path("tmp.feather"),
) -> None:
    # load existing array, if any
    array = data_load(db)
    if array is None:
        raise click.FileError(str(db), hint="database file must exist")

    video = video.resolve()
    array = array.loc[array.index.get_level_values("filename") == str(video)]
    if array.shape[0] == 0:
        raise typer.BadParameter("video not in database")

    stats = get_video_stats(video)
    with save_video(output, stats=stats) as video_writer:
        for frame in overlay_detections(
            (f for _, f in generate_frames_cv2(video)),
            array,
        ):
            video_writer.write(frame)


if __name__ == "__main__":
    app()
