from enum import Enum

import numpy as np


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single box and an array of boxes.
    box: shape (6,)  [x1, y1, x2, y2, confidence, class_label]
    boxes: shape (N, 6)
    Returns: IoU values, shape (N,)
    """
    # Intersection
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = area_box + area_boxes - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


class NMSMode(Enum):
    AREA = "area"
    CONF_MAX = "conf_max"
    CONF_SUM = "conf_sum"


def nms(
    boxes: np.ndarray,
    iou_threshold: float = 0.5,
    mode: NMSMode = NMSMode.CONF_MAX,
) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on an array of boxes.
    boxes: shape (N, 4+C) [x1, y1, x2, y2, [class confidence]]
    iou_threshold: float, threshold for IoU to suppress boxes
    mode: NMSMode - determines sorting method
    Returns: filtered boxes, shape (M, 4+C) [x1, y1, x2, y2, [class confidence]]
    """
    if len(boxes) == 0:
        return np.empty((0, boxes.shape[1]))

    if mode == NMSMode.AREA:
        sort_key = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    elif mode == NMSMode.CONF_SUM:
        sort_key = boxes[:, 4:].sum(axis=1)
    elif mode == NMSMode.CONF_MAX:
        sort_key = boxes[:, 4:].max(axis=1)
    else:
        raise ValueError(f"Unknown NMS mode: {mode}")
    boxes = boxes[sort_key.argsort()[::-1]]

    keep = []
    while len(boxes) > 0:
        curr = boxes[0]
        keep.append(curr)
        if len(boxes) == 1:
            break
        ious = iou(curr, boxes[1:])
        inds = np.where(ious <= iou_threshold)[0]
        boxes = boxes[1:][inds]
    return np.stack(keep) if keep else np.empty((0, boxes.shape[1]))


def weighted_nms(
    boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    Perform Weighted Non-Maximum Suppression (box voting) on an array of boxes.
    boxes: shape (N, 4+C) [x1, y1, x2, y2, [class confidence]]
    iou_threshold: float, threshold for IoU to merge boxes
    Returns: filtered boxes, shape (M, 4+C) [x1, y1, x2, y2, [class confidence]]
    """
    if len(boxes) == 0:
        return np.empty((0, boxes.shape[1]))

    sort_key = boxes[:, 4:].max(axis=1)
    boxes = boxes[sort_key.argsort()[::-1]]

    keep = []
    used = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if used[i]:
            continue
        curr = boxes[i]
        ious = iou(curr, boxes)
        overlapping = np.where((ious > iou_threshold) & (~used))[0]
        overlapping = np.append(overlapping, i)  # include self
        group = boxes[overlapping]

        # Weighted average (across all classes)
        weights = group[:, 4:].max(axis=1)
        xyxy = np.average(group[:, :4], axis=0, weights=weights)
        confs = np.average(group[:, 4:], axis=0, weights=weights)

        merged = np.concatenate([xyxy, confs])
        keep.append(merged)
        used[overlapping] = True
    return np.stack(keep) if keep else np.empty((0, boxes.shape[1]))
