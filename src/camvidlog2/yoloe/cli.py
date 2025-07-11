from pathlib import Path
from typing import Annotated

import click
import pandas as pd
import typer
from ultralytics import YOLOE

from camvidlog2.common.data import load as data_load
from camvidlog2.vid import generate_frames_cv2, get_video_stats, save_video
from camvidlog2.yoloe.ai import generate_tracked_bboxes
from camvidlog2.yoloe.data import create
from camvidlog2.yoloe.plot import overlay_detections

app = typer.Typer(help="YOLOE video processing and model export CLI.")


@app.command(help="Run inference on videos and store results in a database.")
def load(
    videos: list[Path],
    classes: Annotated[
        list[str],
        typer.Option("--class", "-c", help="List of class names for detection"),
    ],
    onnx: Annotated[
        Path, typer.Option("--onnx", "-o", help="Path to YOLOE ONNX model file")
    ] = Path("yoloe-11l-seg.onnx"),
    db: Annotated[Path, typer.Option(help="Path to the output database file")] = Path(
        "tmp.feather"
    ),
):
    # load existing array, if any
    existing_array = data_load(db)

    for video in videos:
        video = video.resolve()
        if (
            existing_array is not None
            and (existing_array.index.get_level_values("filename") == str(video)).any()
        ):
            print("File already loaded")
            print(video)
            continue

        if not video.exists():
            print("File does not exist")
            print(video)
            continue

        # inference
        array = create(
            video,
            generate_tracked_bboxes(
                generate_frames_cv2(video),
                onnx_path=onnx,
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
        existing_array.to_feather(db)
        print("saved file")


@app.command(help="Display detections on a video using results from the database.")
def show(
    video: Annotated[Path, typer.Argument(help="Path to the input video file")],
    output: Annotated[Path, typer.Argument(help="Path to the output video file")],
    db: Annotated[Path, typer.Option("--db", help="Path to the database file")] = Path(
        "tmp.feather"
    ),
    confidence_threshold: Annotated[
        float,
        typer.Option(
            "--confidence", "-c", help="Minimum confidence required to display a track"
        ),
    ] = 0.5,
):
    # load existing array, if any
    array = data_load(db)
    if array is None:
        raise click.FileError("database file must exist")

    video = video.resolve()
    array = array.loc[array.index.get_level_values("filename") == str(video)]
    if array.shape[0] == 0:
        raise ValueError("video must be in database")

    stats = get_video_stats(video)
    with save_video(output, stats=stats) as video_writer:
        for frame in overlay_detections(
            (f for _, f in generate_frames_cv2(video)),
            array,
            confidence_threshold=confidence_threshold,
        ):
            video_writer.write(frame)


@app.command(help="Export a YOLOE model to ONNX format with custom classes.")
def prepare(
    classes: Annotated[
        list[str],
        typer.Option(
            "--class",
            "-c",
            help="List of class names for the model",
        ),
    ],
    model_path: Annotated[
        str, typer.Option(help="Path to YOLOE model file")
    ] = "yoloe-11l-seg.pt",
    onnx_path: Annotated[
        str, typer.Option(help="Path to ONNX file output")
    ] = "yoloe-11l-seg.onnx",
    force: Annotated[
        bool,
        typer.Option(
            False,
            "--force",
            "-f",
            help="Overwrite output ONNX file if it exists",
        ),
    ] = False,
    batch: Annotated[
        int,
        typer.Option(
            10,
            "--batch",
            "-b",
            help="Batch size for ONNX export",
        ),
    ] = 10,
):
    onnx_path_obj = Path(onnx_path)
    # Check if output file exists and handle according to --force flag
    if onnx_path_obj.exists():
        if force:
            onnx_path_obj.unlink()
        else:
            raise FileExistsError(f"Destination ONNX file already exists: {onnx_path}")
    # Load via ultralytics, will download if appropriate
    model = YOLOE(model_path)
    # Set class names for the model - this is then baked into the ONNX export
    model.set_classes(classes, model.get_text_pe(classes))
    """
The technical challenge involves handling the visual embeddings during export since they require reference image processing.

Based on our documentation, you might be able to work around this by:

    Using model.get_visual_pe() to extract embeddings from your reference image
    Passing these embeddings to model.set_classes() which accepts both text and visual embeddings

This approach would "bake in" the visual prompts before export.

1076,699,2545,1761
"""
    # Export the modified model to ONNX format
    out_str = model.export(
        format="onnx",
        name=str(onnx_path),
        simplify=True,  # uses onnxslim to optimize the model
        batch=batch,
    )
    out_path = Path(out_str)
    # Verify that the export location matches the requested output path
    if out_path != onnx_path_obj:
        raise ValueError("Unexpected export location")


if __name__ == "__main__":
    app()
