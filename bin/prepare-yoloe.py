#!/usr/bin/env python3
from pathlib import Path

import typer
from typing_extensions import Annotated
from ultralytics import YOLOE


def main(
    classes: Annotated[
        list[str],
        typer.Option(
            "--class",
            "-c",
            help="List of class names for the model",
        ),
    ],
    model_path: str = typer.Option(
        "yoloe-11l-seg.pt",
        help="Path to YOLOE model file.",
    ),
    onnx_path: str = typer.Option(
        "yoloe-11l-seg.onnx",
        help="Output ONNX file path",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output ONNX file if it exists",
    ),
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
    # Export the modified model to ONNX format
    out_str = model.export(
        format="onnx",
        name=str(onnx_path),
        simplify=True,  # uses onnxslim to optimize the model
    )
    out_path = Path(out_str)
    # Verify that the export location matches the requested output path
    if out_path != onnx_path_obj:
        raise ValueError("Unexpected export location")


if __name__ == "__main__":
    typer.run(main)
