from collections.abc import Generator, Iterator

import numpy as np
import pandas as pd
import supervision as sv


def overlay_detections(
    frames: Iterator[np.ndarray],
    detections: pd.DataFrame,
) -> Generator[np.ndarray]:
    """Overlay bounding boxes on frames."""

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.RichLabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    for i, frame in enumerate(frames):
        # Select detections for the current frame via multilevel index
        detections_frame = detections[detections["frame_no"] == i]
        if detections_frame.empty:
            # no detections for this frame, yield the original frame
            yield frame
            continue
        detections_frame = detections.xs(i, level="frame_no")

        detections_sv = sv.Detections(
            xyxy=detections_frame[["x1", "y1", "x2", "y2"]].to_numpy(),
            confidence=detections_frame["conf"].to_numpy(),
            class_id=detections_frame["class"].cat.codes.astype(int).to_numpy(),
            tracker_id=detections_frame["tracker"].astype(int).to_numpy(),
        )

        scene = frame.copy()

        scene = box_annotator.annotate(scene=scene, detections=detections_sv)

        labels = [
            f"{detections_frame['class'].cat.categories[cls]} {conf:.2f}"
            for cls, conf in zip(
                detections_sv.class_id, detections_sv.confidence, strict=True
            )
        ]
        scene = label_annotator.annotate(
            scene=scene, detections=detections_sv, labels=labels
        )

        scene = trace_annotator.annotate(scene=scene, detections=detections_sv)

        yield scene
