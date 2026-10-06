"""Box/mask primitives shared by detectors; no model imports."""

import numpy as np


def bbox_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): First bounding box in format [x1, y1, x2, y2]
        box2 (list): Second bounding box in format [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0 and 1
    """
    # Intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def apply_nms(boxes, scores=None, iou_threshold=0.6):
    """
    Apply non-maximum suppression to eliminate redundant bounding boxes.

    Args:
        boxes (list or numpy.ndarray): List of bounding boxes in format [[x1, y1, x2, y2], ...]
        scores (list or numpy.ndarray, optional): Confidence scores for each box. If None, boxes are processed in given order.
        iou_threshold (float): IoU threshold for considering boxes as overlapping. Increase to be more strict.

    Returns:
        list: Indices of boxes to keep after NMS
    """
    if len(boxes) == 0:
        return []

    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)

    keep_indices = []

    # Visit boxes by score (highest first), or in the given order without scores
    if scores is not None:
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        order = scores.argsort()[::-1]
    else:
        order = np.arange(len(boxes))

    while len(order) > 0:
        # Pick the box with highest score or first in order
        current_index = order[0]
        keep_indices.append(current_index)

        if len(order) == 1:
            break

        ious = [bbox_iou(boxes[current_index], boxes[j]) for j in order[1:]]

        # Keep only boxes below the IoU threshold
        filtered_indices = np.where(np.array(ious) < iou_threshold)[0]
        order = order[filtered_indices + 1]  # +1 because we skipped the first element

    return keep_indices
