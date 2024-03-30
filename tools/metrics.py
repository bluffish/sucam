import torch
from torchmetrics.functional.regression import relative_squared_error


def get_iou(preds, targets):
    with torch.no_grad():
        pred = (preds > .5)
        tgt = targets.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()

    return intersect, union, intersect / union if (union > 0) else 1.0


def get_rse(preds, targets):
    mask = targets > 0
    return relative_squared_error(preds[mask], targets[mask])
