from collections.abc import Generator, Iterator

import numpy as np
import pandas as pd
import supervision as sv


def overlay_detections(
    frames: Iterator[np.ndarray],
    detections: pd.DataFrame,
    confidence_threshold: float = 0.5,
) -> Generator[np.ndarray, None, None]:
    """
    Overlay bounding boxes, labels, and traces on video frames.

    Args:
        frames: An iterator of video frames as numpy arrays (H, W, C).
        detections: A pandas DataFrame with a MultiIndex including 'frame_no',
            and columns: ['x1', 'y1', 'x2', 'y2', 'tracker', [class confidence]].
            'x1', 'y1', 'x2', 'y2' are bounding box coordinates,
            'tracker' is a unique identifier to group bounding boxes across frames,
            'class confidence' is a variable number of columns, each representing a named class's confidence score.
        confidence_threshold: Minimum confidence required to display a detection (default: 0.5)

    Yields:
        np.ndarray: The frame with overlaid bounding boxes, labels, and traces for each frame.
        If no detections exist for a frame, yields the original frame.
    """

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.RichLabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK)

    classes: list[str] = [
        col
        for col in detections.columns
        if col not in {"x1", "y1", "x2", "y2", "tracker"}
    ]

    # Group by tracker, drop all rows for a tracker if none of its confidences exceed threshold
    detections_confidence_max = pd.DataFrame(
        {
            "confidence_max": detections[classes].max(axis=1),
            "tracker": detections["tracker"],
        }
    )
    tracker_max_conf = detections_confidence_max.groupby("tracker").max().reset_index()
    tracker_to_keep = frozenset(
        tracker_max_conf[tracker_max_conf["confidence_max"] >= confidence_threshold][
            "tracker"
        ]
    )
    print(
        f"Keeping {len(tracker_to_keep)} trackers with confidence >= {confidence_threshold:.2f}"
    )
    detections = detections[detections["tracker"].isin(tracker_to_keep)]

    for i, frame in enumerate(frames):
        # Select detections for the current frame via multilevel index
        if i not in detections.index.get_level_values("frame_no"):
            # no detections for this frame, yield the original frame
            print(f"No detections for frame {i}.")
            yield frame
            continue
        detections_frame = detections.xs(i, level="frame_no")

        confidences = detections_frame[classes]

        confidence_max = confidences.max(axis=1)
        label_max = confidences.idxmax(axis=1)

        detections_sv = sv.Detections(
            xyxy=detections_frame[["x1", "y1", "x2", "y2"]].to_numpy(),
            tracker_id=detections_frame["tracker"].astype(int).to_numpy(),
            confidence=confidence_max.to_numpy(),
        )

        scene = frame.copy()

        scene = box_annotator.annotate(scene=scene, detections=detections_sv)

        labels = [
            f"{cls} {conf:.2f} [{tracker}]"
            for cls, conf, tracker in zip(
                label_max, confidence_max, detections_frame["tracker"], strict=True
            )
        ]
        scene = label_annotator.annotate(
            scene=scene, detections=detections_sv, labels=labels
        )

        scene = trace_annotator.annotate(scene=scene, detections=detections_sv)

        yield scene
