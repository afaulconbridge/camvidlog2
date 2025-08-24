import time
from collections.abc import Iterable
from typing import Generator

import numpy as np
import pandas as pd
from PIL import Image
from supervision import ByteTrack, Detections, DetectionsSmoother
from transformers import AutoModelForCausalLM


def generate_bboxes(
    frames: Iterable[np.ndarray],
    class_names: list[str],
    *,
    cloud: bool = False,
) -> Generator[pd.DataFrame]:
    """

    Yields:
        pd.DataFrame with columns: [frame_no, x1, y1, x2, y2, class]
        - 'class' is a pandas Categorical column with class names.
    """
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
        # device_map={"": "cuda"},
    )

    tracker = ByteTrack()
    smoother = DetectionsSmoother()

    size_xy = None

    for i, frame in enumerate(frames):
        # Object Detection
        start = time.time()
        print(f"Handling frame {i}")

        if size_xy is None:
            size_xy = (frame.shape[1], frame.shape[0])

        image = Image.fromarray(frame)
        image = image.resize((3840 // 8, 2160 // 8))

        # only supports one prompt at a time currently?
        bboxes = model.detect(image, class_names[0])["objects"]
        if not bboxes:
            yield pd.DataFrame(columns=["frame_no", "x1", "y1", "x2", "y2", "class"])
            continue

        # [{'x_min': 0.1953125, 'y_min': 0.46826171875, 'x_max': 0.6171875, 'y_max': 0.79541015625}]
        df = pd.DataFrame(bboxes).rename(
            columns={"x_min": "x1", "x_max": "x2", "y_min": "y1", "y_max": "y2"}
        )
        df["x1"] *= size_xy[0]
        df["x2"] *= size_xy[0]
        df["y1"] *= size_xy[1]
        df["y2"] *= size_xy[1]
        detections_coords = df[["x1", "y1", "x2", "y2"]].to_numpy()

        # smoother neads tracking ids
        # tracking ids need confidences
        # invent confidence!
        detections = Detections(
            xyxy=detections_coords[:, :4],
            confidence=np.full(len(bboxes), 0.9),
        )
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        data = {
            "frame_no": pd.Series(
                np.full(detections.xyxy.shape[0], i, dtype=np.uint32),
                dtype=pd.UInt32Dtype(),
            ),
            "x1": pd.Series(np.floor(detections.xyxy[:, 0]), dtype=np.uint32),
            "y1": pd.Series(np.floor(detections.xyxy[:, 1]), dtype=np.uint32),
            "x2": pd.Series(np.ceil(detections.xyxy[:, 2]), dtype=np.uint32),
            "y2": pd.Series(np.ceil(detections.xyxy[:, 3]), dtype=np.uint32),
            "class": class_names[0],
        }
        df = pd.DataFrame(data)

        end = time.time()

        print(f"Handled frame {i} in {end - start:.02f}s")

        yield df
