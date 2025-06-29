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


def nms(boxes: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on an array of boxes.
    boxes: shape (N, 6) [x1, y1, x2, y2, confidence, class_label]
    Returns: filtered boxes, shape (M, 6)
    """
    if len(boxes) == 0:
        return np.empty((0, 6))

    # Sort boxes by confidence descending
    order = boxes[:, 4].argsort()[::-1]
    boxes = boxes[order]

    keep = []
    while len(boxes) > 0:
        current = boxes[0]
        keep.append(current)

        # Filter out boxes with the same class and IoU > threshold
        rest = boxes[1:]
        if len(rest) == 0:
            break
        # Only compare with same class label
        same_class_mask = rest[:, 5] == current[5]
        same_class_boxes = rest[same_class_mask]
        other_boxes = rest[~same_class_mask]

        if len(same_class_boxes) > 0:
            ious = iou(current, same_class_boxes)
            keep_mask = ious <= iou_threshold
            same_class_boxes = same_class_boxes[keep_mask]

        # Concatenate boxes of other classes and those not suppressed
        boxes = np.vstack([same_class_boxes, other_boxes])

    return np.vstack(keep)
