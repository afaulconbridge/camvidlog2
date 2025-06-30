from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from camvidlog2.common.data import load as data_load
from camvidlog2.vid import generate_frames_cv2
from camvidlog2.yoloe.ai import generate_tracked_bboxes
from camvidlog2.yoloe.data import create

app = typer.Typer()


@app.command()
def load(
    videos: list[Path],
    classes: Annotated[list[str], typer.Option("--class", "-c")],
    onnx: Annotated[Path, typer.Option("--onnx", "-o")] = Path("yoloe-11l-seg.onnx"),
    db: Annotated[Path, typer.Option()] = Path("tmp.feather"),
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
                (v for _, v in generate_frames_cv2(video)),
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


if __name__ == "__main__":
    app()
