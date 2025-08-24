from collections.abc import Generator, Iterator

import numpy as np
import pandas as pd
import supervision as sv


def overlay_detections(
    frames: Iterator[np.ndarray],
    detections: pd.DataFrame,
) -> Generator[np.ndarray, None, None]:
    """Overlay bounding boxes on frames."""

    box_annotator = sv.BoxAnnotator(
        color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX
    )

    for i, frame in enumerate(frames):
        # Select detections for the current frame via multilevel index
        if i not in detections.index.get_level_values("frame_no"):
            # no detections for this frame, yield the original frame
            yield frame
            continue
        detections_frame = detections.xs(i, level="frame_no")
        detections_coords = detections_frame[["x1", "y1", "x2", "y2"]].to_numpy()

        print(detections_coords)

        detections_sv = sv.Detections(
            xyxy=detections_coords,
        )

        scene = frame.copy()

        scene = box_annotator.annotate(scene=scene, detections=detections_sv)

        yield scene
