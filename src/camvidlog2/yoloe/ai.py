import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from supervision import ByteTrack, Detections, DetectionsSmoother

from camvidlog2.common.nms.np import nms
from camvidlog2.vid import slice_frame


# this is a compile time object, so downstream packages can modify it
# e.g. to support OpenVINO or CUDA and associated dependencies
class ProvidersList:
    providers = [
        # "CUDAExecutionProvider",
        # "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a frame for YOLOE ONNX inference.

    Args:
        frame: np.ndarray of shape (H, W, 3), BGR format, dtype uint8.

    Returns:
        np.ndarray: Preprocessed image of shape (3, 640, 640), dtype float32, values in [0, 1].
            - Channel order: [R, G, B]
            - Pixel values normalized to [0, 1]
            - Suitable for ONNX model input
    """
    frm = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    frm = frm.transpose(2, 0, 1)  # (3, 640, 640)
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
    num_classes: int,
    img_h: int,
    img_w: int,
    results_all: np.ndarray,
) -> np.ndarray:
    """
    Filter and process YOLOE model outputs to obtain bounding boxes after confidence thresholding.
    (NMS is not applied here.)

    Args:
        conf: Confidence threshold for filtering detections.
        num_classes: Number of classes in the model.
        img_h: Height of the original image.
        img_w: Width of the original image.
        results_all: np.ndarray of shape (N, 4 + num_classes), dtype float32.
            - columns 0:4: [x_center, y_center, width, height]
            - columns 4:4+num_classes: class confidence scores

    Returns:
        np.ndarray: Array of filtered detections, shape (M, 6), dtype float32.
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
    if img_h != 640 or img_w != 640:
        apply_scaling(results_filtered_conf, img_w, img_h)
    # Ensure x1 and y1 are no less than zero
    results_filtered_conf[:, 0] = np.maximum(results_filtered_conf[:, 0], 0)
    results_filtered_conf[:, 1] = np.maximum(results_filtered_conf[:, 1], 0)
    # Ensure x2 and y2 are no more than img_w and img_h
    results_filtered_conf[:, 2] = np.minimum(results_filtered_conf[:, 2], img_w)
    results_filtered_conf[:, 3] = np.minimum(results_filtered_conf[:, 3], img_h)
    return results_filtered_conf


def generate_tracked_bboxes(
    frames: Iterable[np.ndarray],
    onnx_path: str | Path,
    class_names: list[str],
    *,
    conf: float = 0.1,
    nms_thresh: float = 0.7,
    slice_overlap: float = 0.25,
    max_batch_size: int = 10,
) -> Iterator[pd.DataFrame]:
    """
    Generator that yields tracked bounding boxes for each frame as a pandas DataFrame.

    Each row: [frame_no, x1, y1, x2, y2, confidence, classes, tracker]
    - The 'classes' column is a pandas Categorical type with class names as categories.
    - Missing tracker will be represented as pd.NA (nullable integer column).

    Args:
        frames: iterable of np.ndarray (frames in BGR format)
        onnx_path: path to ONNX model
        class_names: list of class names
        conf: confidence threshold
        nms_thresh: NMS IoU threshold
        max_batch_size: maximum number of chunks to process in a single ONNX inference batch (default 30)
        slice_overlap: overlap ratio for frame slicing (default 0.25)
        providers: ONNX runtime providers

    Yields:
        pd.DataFrame with columns: [frame_no, x1, y1, x2, y2, conf, class, tracker]
        - 'class' is a pandas Categorical column with class names.
        - 'tracker' is a nullable integer column for track IDs.
    """
    onnx_model = ort.InferenceSession(
        str(onnx_path),
        providers=ProvidersList.providers,
        #    sess_options=session_options,
    )
    onnx_model.disable_fallback()

    num_classes = len(class_names)
    tracker = ByteTrack()
    smoother = DetectionsSmoother()
    for frame_no, frame in enumerate(frames):
        all_detections = []

        # beware, batch size is baked into the model too
        for batch in itertools.batched(
            slice_frame(
                frame,
                slice_width=640,
                slice_height=640,
                slice_overlap=slice_overlap,
                strict=True,
            ),
            max_batch_size,
        ):
            regions, slices = zip(*batch, strict=True)
            slices = [preprocess(slice) for slice in slices]
            # model _must_ have a specific batch size, no smaller, no larger
            if len(slices) < max_batch_size:
                # Only create padding tensor if we need it, and reuse it
                padding_tensor = np.zeros((3, 640, 640), dtype=np.float32)
                while len(slices) < max_batch_size:
                    slices.append(padding_tensor)
            # Run inference on the batch
            outputs = onnx_model.run(
                None,
                input_feed={"images": np.stack(slices, axis=0)},
            )

            # For each chunk in the batch, post-process detections
            for i, region in enumerate(regions):
                results_all = outputs[0][i].transpose().squeeze()
                # Filter and scale detections for this chunk (no NMS yet)
                chunk_detections = filter_bboxes(
                    conf, num_classes, 640, 640, results_all
                )
                # Offset chunk detections to original frame coordinates
                chunk_detections[:, 0] += region.x1  # x1
                chunk_detections[:, 2] += region.x1  # x2
                chunk_detections[:, 1] += region.y1  # y1
                chunk_detections[:, 3] += region.y1  # y2
                all_detections.append(chunk_detections)
        # Combine all detections from all chunks
        all_detections = np.vstack(all_detections)

        # Run NMS on the combined detections (in full-frame coordinates)
        results_filtered = nms(all_detections, nms_thresh)

        # Do tracking on the filtered detections
        detections = Detections(
            xyxy=results_filtered[:, :4],
            confidence=results_filtered[:, 4],
            class_id=results_filtered[:, 5].astype(int),
        )
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)
        # Prepare output DataFrame for this frame
        n = len(detections)
        data = {
            "frame_no": [frame_no] * n,
            "x1": pd.Series(np.floor(detections.xyxy[:, 0]), dtype=np.uint32),
            "y1": pd.Series(np.floor(detections.xyxy[:, 1]), dtype=np.uint32),
            "x2": pd.Series(np.ceil(detections.xyxy[:, 2]), dtype=np.uint32),
            "y2": pd.Series(np.ceil(detections.xyxy[:, 3]), dtype=np.uint32),
            "conf": pd.Series(detections.confidence, dtype=np.float32),
            "class": pd.Series(
                pd.Categorical(
                    detections.class_id,
                    categories=np.arange(len(class_names)),
                    ordered=False,
                ).rename_categories(class_names),
                dtype=pd.CategoricalDtype(class_names, ordered=False),
            ),
            "tracker": pd.Series(
                [tid if tid is not None else 0 for tid in detections.tracker_id]
                if detections.tracker_id is not None
                else [0] * n,
                dtype=np.uint32,
            ),
        }
        df = pd.DataFrame(data)
        yield df
