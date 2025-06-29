import torch


def iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between a single box and an array of boxes (PyTorch tensors).
    box: shape (6,)  [x1, y1, x2, y2, conf, cls]
    boxes: shape (N, 6)
    Returns: IoU values, shape (N,)
    """
    # Intersection
    inter_x1 = torch.maximum(box[0], boxes[:, 0])
    inter_y1 = torch.maximum(box[1], boxes[:, 1])
    inter_x2 = torch.minimum(box[2], boxes[:, 2])
    inter_y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = area_box + area_boxes - inter_area
    iou = torch.where(
        union_area > 0, inter_area / union_area, torch.zeros_like(union_area)
    )
    return iou


def nms(boxes: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Perform Non-Maximum Suppression (NMS) on a tensor of boxes.
    boxes: shape (N, 6) [x1, y1, x2, y2, conf, cls]
    Returns: filtered boxes, shape (M, 6)
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 6))

    # Sort boxes by confidence descending
    _, order = boxes[:, 4].sort(descending=True)
    boxes = boxes[order]

    keep = []
    while boxes.size(0) > 0:
        current = boxes[0]
        keep.append(current.unsqueeze(0))

        if boxes.size(0) == 1:
            break

        rest = boxes[1:]
        same_class_mask = rest[:, 5] == current[5]
        same_class_boxes = rest[same_class_mask]
        other_boxes = rest[~same_class_mask]

        if same_class_boxes.size(0) > 0:
            ious = iou(current, same_class_boxes)
            keep_mask = ious <= iou_threshold
            same_class_boxes = same_class_boxes[keep_mask]

        boxes = torch.cat([same_class_boxes, other_boxes], dim=0)

    return torch.cat(keep, dim=0)
