import cv2
import numpy as np
import torch
from sklearn.metrics import *
from sklearn.calibration import *
import torchmetrics


def bin_predictions(y_score, y_true, n_bins=10):
    max_prob, max_ind = y_score.max(dim=1)

    acc_binned = torch.zeros((n_bins,))
    conf_binned = torch.zeros((n_bins,))
    bin_cardinalities = torch.zeros((n_bins,))

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    lower_bin_boundary = bin_boundaries[:-1]
    upper_bin_boundary = bin_boundaries[1:]

    corrects = max_ind == y_true.argmax(dim=1)

    for b in range(n_bins):
        in_bin = (max_prob < upper_bin_boundary[b]) & (max_prob >= lower_bin_boundary[b])
        bin_cardinality = in_bin.sum()
        bin_cardinalities[b] = bin_cardinality

        if bin_cardinality > 0:
            acc_binned[b] = corrects[in_bin].float().mean()
            conf_binned[b] = max_prob[in_bin].mean()

    return acc_binned, conf_binned, bin_cardinalities


def expected_calibration_error(pred, label, exclude=None, n_bins=10):
    y_score = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    y_true = label.permute(0, 2, 3, 1).reshape(-1, label.shape[1])

    if exclude is not None:
        include = ~exclude.permute(0, 2, 3, 1).flatten()
        y_score = y_score[include]
        y_true = y_true[include]

    acc, conf, bin_cardinalities = bin_predictions(y_score, y_true, n_bins=n_bins)
    ece = torch.abs(acc - conf) * bin_cardinalities
    ece = ece.sum() / (y_true.shape[0])

    return conf, acc, ece


def get_iou(preds, labels, exclude=None):
    classes = preds.shape[1]
    iou = [0] * classes

    pmax = preds.argmax(dim=1).unsqueeze(1)
    lmax = labels.argmax(dim=1).unsqueeze(1)

    with torch.no_grad():
        for i in range(classes):
            p = (pmax == i).bool()
            l = (lmax == i).bool()

            if exclude is not None:
                p &= ~exclude
                l &= ~exclude

            intersect = (p & l).sum().float().item()
            union = (p | l).sum().float().item()
            iou[i] = intersect / union if union > 0 else 0
    return iou


def unc_iou(y_score, y_true, thresh=.5):
    pred = (y_score > thresh).bool()
    target = y_true.bool()

    intersect = (pred & target).sum()
    union = (pred | target).sum()

    return intersect / union


def patch_metrics(uncertainty_scores, uncertainty_labels, quantile=False):
    thresholds = np.linspace(0, 1, 11)

    pavpus = []
    agcs = []
    ugis = []

    for thresh in thresholds:
        if quantile:
            perc = torch.quantile(uncertainty_scores, thresh).item()
            pavpu, agc, ugi = calculate_pavpu(uncertainty_scores, uncertainty_labels,
                                              uncertainty_threshold=perc)
        else:
            pavpu, agc, ugi = calculate_pavpu(uncertainty_scores, uncertainty_labels, uncertainty_threshold=thresh)

        pavpus.append(pavpu)
        agcs.append(agc)
        ugis.append(ugi)

    return pavpus, agcs, ugis, thresholds, auc(thresholds, pavpus), auc(thresholds, agcs), auc(thresholds, ugis)


def calculate_pavpu(uncertainty_scores, uncertainty_labels, accuracy_threshold=0.5, uncertainty_threshold=0.2,
                    window_size=1):
    if window_size == 1:
        uncertainty_labels = uncertainty_labels.cuda()
        uncertainty_scores = uncertainty_scores.cuda()

        accurate = ~uncertainty_labels.long()
        uncertain = uncertainty_scores >= uncertainty_threshold

        au = torch.sum(accurate & uncertain).cpu()
        ac = torch.sum(accurate & ~uncertain).cpu()
        iu = torch.sum(~accurate & uncertain).cpu()
        ic = torch.sum(~accurate & ~uncertain).cpu()
    else:
        ac, ic, au, iu = 0., 0., 0., 0.

        anchor = (0, 0)
        last_anchor = (uncertainty_labels.shape[1] - window_size, uncertainty_labels.shape[2] - window_size)

        while anchor != last_anchor:
            label_window = uncertainty_labels[:,
                           anchor[0]:anchor[0] + window_size,
                           anchor[1]:anchor[1] + window_size
                           ]

            uncertainty_window = uncertainty_scores[:,
                                 anchor[0]:anchor[0] + window_size,
                                 anchor[1]:anchor[1] + window_size
                                 ]

            accuracy = torch.sum(label_window, dim=(1, 2)) / (window_size ** 2)
            avg_uncertainty = torch.mean(uncertainty_window, dim=(1, 2))

            accurate = accuracy < accuracy_threshold
            uncertain = avg_uncertainty >= uncertainty_threshold

            au += torch.sum(accurate & uncertain)
            ac += torch.sum(accurate & ~uncertain)
            iu += torch.sum(~accurate & uncertain)
            ic += torch.sum(~accurate & ~uncertain)

            if anchor[1] < uncertainty_labels.shape[1] - window_size:
                anchor = (anchor[0], anchor[1] + 1)
            else:
                anchor = (anchor[0] + 1, 0)

    a_given_c = ac / (ac + ic + 1e-10)
    u_given_i = iu / (ic + iu + 1e-10)

    pavpu = (ac + iu) / (ac + au + ic + iu + 1e-10)

    return pavpu, a_given_c, u_given_i


def roc_pr(uncertainty_scores, uncertainty_labels, exclude=None):
    y_true = uncertainty_labels.flatten().numpy()
    y_score = uncertainty_scores.flatten().numpy()

    if exclude is not None:
        include = ~exclude.flatten().numpy()
        y_true = y_true[include]
        y_score = y_score[include]

    pr, rec, tr = precision_recall_curve(y_true, y_score, drop_intermediate=True)
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=True)

    auroc = auc(fpr, tpr)
    aupr = average_precision_score(y_true, y_score)

    no_skill = np.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, aupr, no_skill


def ece_score(y_pred, y_true, n_bins=10, exclude=None):
    y_true = y_true.argmax(dim=1)

    if exclude is not None:
        y_true[exclude[:, 0] == 1] = -1

    return torchmetrics.functional.calibration_error(
        y_pred, y_true, "multiclass",
        num_classes=y_pred.shape[1],
        n_bins=n_bins,
        ignore_index=-1
    )


def brier_score(y_pred, y_true, exclude=None):
    brier = torch.nn.functional.mse_loss(y_pred, y_true, reduction='none')

    if exclude is not None:
        brier = brier[~exclude.repeat(1, y_pred.shape[1], 1, 1)]

    return brier.mean()
