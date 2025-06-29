from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from supervision import ByteTrack, Detections

from camvidlog2.common.nms.np import nms


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a frame for YOLOE ONNX inference.

    Args:
        frame: np.ndarray of shape (H, W, 3), BGR format, dtype uint8.

    Returns:
        np.ndarray: Preprocessed image of shape (1, 3, 640, 640), dtype float32, values in [0, 1].
            - Channel order: [R, G, B]
            - Pixel values normalized to [0, 1]
            - Suitable for ONNX model input
    """
    frm = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    frm = frm.transpose(2, 0, 1)  # (3, 640, 640)
    frm = frm.reshape(1, 3, 640, 640)
    frm = (frm / 255.0).astype(np.float32)
    return frm


def convert_xyhw_into_x1y1x2y2(array: np.ndarray) -> None:
    """
    Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2) format in-place.
    Beware: this modifies the input numpy array directly.

    Args:
        array: numpy array of shape (N, 4) or (N, >=4) with (x, y, w, h) in the first 4 columns.
            - array[:, 0]: x center
            - array[:, 1]: y center
            - array[:, 2]: width
            - array[:, 3]: height
    """
    array[:, 0] = array[:, 0] - (array[:, 2] / 2)
    array[:, 1] = array[:, 1] - (array[:, 3] / 2)
    array[:, 2] = array[:, 0] + array[:, 2]
    array[:, 3] = array[:, 1] + array[:, 3]


def apply_scaling(array: np.ndarray, width: int, height: int) -> None:
    """
    Scale bounding box coordinates in-place from 640x640 to the given width and height.
    Beware: this modifies the input numpy array directly.

    Args:
        array: numpy array of shape (N, 4) or (N, >=4) with (x1, y1, x2, y2) in the first 4 columns.
            - array[:, 0]: x1 (top-left x)
            - array[:, 1]: y1 (top-left y)
            - array[:, 2]: x2 (bottom-right x)
            - array[:, 3]: y2 (bottom-right y)
        width: target image width
        height: target image height
    """
    array[:, 0] *= width / 640
    array[:, 1] *= height / 640
    array[:, 2] *= width / 640
    array[:, 3] *= height / 640


def filter_bboxes(
    conf: float,
    nms_thresh: float,
    num_classes: int,
    img_h: int,
    img_w: int,
    results_all: np.ndarray,
) -> np.ndarray:
    """
    Filter and process YOLOE model outputs to obtain bounding boxes after confidence thresholding and NMS.

    Args:
        conf: Confidence threshold for filtering detections.
        nms_thresh: IoU threshold for Non-Maximum Suppression (NMS).
        num_classes: Number of classes in the model.
        img_h: Height of the original image.
        img_w: Width of the original image.
        results_all: np.ndarray of shape (N, 4 + num_classes), dtype float32.
            - columns 0:4: [x_center, y_center, width, height]
            - columns 4:4+num_classes: class confidence scores

    Returns:
        np.ndarray: Array of filtered detections after NMS, shape (M, 6), dtype float32.
            - columns 0:4: [x1, y1, x2, y2] (top-left and bottom-right corners, scaled to original image size)
            - column 4: confidence score
            - column 5: class_id (float, but should be int-castable)
    """
    # Extract bounding box coordinates (x_center, y_center, width, height)
    boxes = results_all[:, :4]
    # Extract class confidence scores
    class_scores = results_all[:, 4 : 4 + num_classes]
    # Get class with highest confidence for each detection
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    # Stack boxes, confidences, and class_ids: shape (N, 6)
    results_top = np.hstack([boxes, confidences[:, None], class_ids[:, None]])
    # Filter by confidence threshold
    results_filtered_conf = results_top[results_top[:, 4] > conf]
    # Convert (x_center, y_center, width, height) to (x1, y1, x2, y2) in-place
    convert_xyhw_into_x1y1x2y2(results_filtered_conf)
    # Scale coordinates from 640x640 to original image size in-place
    apply_scaling(results_filtered_conf, img_w, img_h)
    # Apply Non-Maximum Suppression (NMS)
    results_filtered_conf_nms = nms(results_filtered_conf, nms_thresh)
    return results_filtered_conf_nms


def generate_tracked_bboxes(
    frames: Iterable[np.ndarray],
    onnx_path: str | Path,
    class_names: List[str],
    conf: float = 0.1,
    nms_thresh: float = 0.7,
    providers: Optional[Sequence[str]] = [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ],
) -> Iterator[pd.DataFrame]:
    """
    Generator that yields tracked bounding boxes for each frame as a pandas DataFrame.

    Each row: [frame_no, x1, y1, x2, y2, confidence, classes, track_id]
    - The 'classes' column is a pandas Categorical type with class names as categories.
    - Missing track_id will be represented as pd.NA (nullable integer column).

    Args:
        frames: iterable of np.ndarray (frames in BGR format)
        onnx_path: path to ONNX model
        class_names: list of class names
        conf: confidence threshold
        nms_thresh: NMS IoU threshold
        providers: ONNX runtime providers

    Yields:
        pd.DataFrame with columns: [frame_no, x1, y1, x2, y2, conf, class, track_id]
        - 'class' is a pandas Categorical column with class names.
        - 'tracker' is a nullable integer column for track IDs.
    """
    onnx_model = ort.InferenceSession(str(onnx_path), providers=providers)

    num_classes = len(class_names)

    tracker = ByteTrack()
    columns = ["frame_no", "x1", "y1", "x2", "y2", "conf", "class", "tracker"]
    for frame_no, frame in enumerate(frames):
        img_h, img_w = frame.shape[:2]

        # Preprocess the frame
        frame_prepared = preprocess(frame)

        # Run inference
        outputs = onnx_model.run(None, {"images": frame_prepared})

        # post-process the results
        results_raw = outputs[0]
        results_all = results_raw.transpose().squeeze()
        if len(results_all) == 0:
            yield pd.DataFrame(columns=columns)
            continue
        results_filtered = filter_bboxes(
            conf, nms_thresh, num_classes, img_h, img_w, results_all
        )
        if len(results_filtered) == 0:
            yield pd.DataFrame(columns=columns)
            continue

        # Do tracking
        detections = Detections(
            xyxy=results_filtered[:, :4],
            confidence=results_filtered[:, 4],
            class_id=results_filtered[:, 5].astype(int),
        )
        tracked_detections = tracker.update_with_detections(detections)
        n = len(tracked_detections)
        if n == 0:
            yield pd.DataFrame(columns=columns)
            continue

        # Prepare data output
        data = {
            "frame_no": [frame_no] * n,
            "x1": np.floor(tracked_detections.xyxy[:, 0]).astype(np.uint32),
            "y1": np.floor(tracked_detections.xyxy[:, 1]).astype(np.uint32),
            "x2": np.ceil(tracked_detections.xyxy[:, 2]).astype(np.uint32),
            "y2": np.ceil(tracked_detections.xyxy[:, 3]).astype(np.uint32),
            "conf": tracked_detections.confidence,
            "class": pd.Series(
                pd.Categorical(
                    tracked_detections.class_id,
                    categories=np.arange(len(class_names)),
                    ordered=False,
                ).rename_categories(class_names),
                dtype=pd.CategoricalDtype(class_names, ordered=False),
            ),
            "tracker": pd.Series(
                [
                    tid if tid is not None else pd.NA
                    for tid in tracked_detections.tracker_id
                ]
                if tracked_detections.tracker_id is not None
                else [pd.NA] * n,
                dtype=np.uint32,
            ),
        }
        df = pd.DataFrame(data)
        yield df
