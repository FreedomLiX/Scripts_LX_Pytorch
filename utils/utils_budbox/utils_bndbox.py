"""
bounding box
"""

import torch
import numpy as np


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = float((x[0] + x[2]) / 2)  # x center
    y[1] = float((x[1] + x[3]) / 2)  # y center
    y[2] = float(x[2] - x[0])        # width
    y[3] = float(x[3] - x[1])        # height
    return y


def xyxy2xywh_opencv(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = float(x[0])         # x left
    y[1] = float(x[1])         # y left
    y[2] = float(x[2] - x[0])  # width
    y[3] = float(x[3] - x[1])  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = float(x[0] - x[2] / 2)  # top left x
    y[1] = float(x[1] - x[3] / 2)  # top left y
    y[2] = float(x[0] + x[2] / 2)  # bottom right x
    y[3] = float(x[1] + x[3] / 2)  # bottom right y
    return y


def box_iou(box1, box2):
    """
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (min(box1[:, None, 2:], box2[:, 2:]) - max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)