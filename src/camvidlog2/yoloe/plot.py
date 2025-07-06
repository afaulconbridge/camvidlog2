from collections.abc import Generator, Iterator

import numpy as np
import pandas as pd
import supervision as sv


def overlay_detections(
    frames: Iterator[np.ndarray], detections: pd.DataFrame
) -> Generator[np.ndarray, None, None]:
    """Overlay bounding boxes on frames."""

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.RichLabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    for i, frame in enumerate(frames):
        # Select detections for the current frame via multilevel index
        if i not in detections.index.get_level_values("frame_no"):
            # no detections for this frame, yield the original frame
            yield frame
            continue
        detections_frame = detections.xs(i, level="frame_no")

        detections_sv = sv.Detections(
            xyxy=detections_frame.iloc[:, :4].to_numpy(),
            confidence=detections_frame.iloc[:, 4].to_numpy(),
            class_id=detections_frame.iloc[:, 5].cat.codes.astype(int).to_numpy(),
            tracker_id=detections_frame.iloc[:, 6].astype(int).to_numpy(),
        )

        # Prepare labels for annotation
        labels = [
            f"{detections_frame.iloc[:, 5].cat.categories[cls]} {conf:.2f}"
            for cls, conf in zip(
                detections_sv.class_id, detections_sv.confidence, strict=True
            )
        ]
        # create a copy of the frame to annotate
        scene = frame.copy()
        # Annotate boxes
        scene = box_annotator.annotate(scene=scene, detections=detections_sv)
        # Annotate labels if available
        if labels is not None:
            scene = label_annotator.annotate(
                scene=scene, detections=detections_sv, labels=labels
            )
        # Annotate traces if tracker IDs are available
        scene = trace_annotator.annotate(scene=scene, detections=detections_sv)
        yield scene
