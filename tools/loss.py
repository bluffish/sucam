import torch
import torch.distributions as D
import torch.nn.functional as F
from tools.uncertainty import *
import torch.nn as nn


def ce_loss(logits, target, weights=None):
    return F.cross_entropy(logits, target, weight=weights, reduction='none')


def a_loss(logits, target, weights=None):
    ce = ce_loss(logits, target, weights=weights)
    al = entropy(logits)[:, 0, :, :].detach()

    return ce * al


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def depth_loss(output, target):
    depth_ratio_map = torch.log(output + 1e-8) - \
                      torch.log(target + 1e-8)

    n = output.shape[-1] * output.shape[-2]

    loss_1 = torch.sum(depth_ratio_map * depth_ratio_map, dim=(1, 2, 3)) / n
    sum_2 = torch.sum(depth_ratio_map, dim=(1, 2, 3))
    loss_2 = sum_2 * sum_2 / n**2
    return torch.mean(loss_1 + loss_2)


def bce_loss(logits, target, weights=None):
    return F.binary_cross_entropy_with_logits(logits, target, reduction='none')


def focal_loss(logits, target, weights=None, n=2):
    c = logits.shape[1]
    x = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, c)
    target = target.argmax(dim=1).long()
    target = target.view(-1)

    log_p = F.log_softmax(x, dim=-1)
    ce = F.nll_loss(log_p, target, weight=weights, reduction='none')
    all_rows = torch.arange(len(x))
    log_pt = log_p[all_rows, target]

    pt = log_pt.exp()
    focal_term = (1 - pt + 1e-12) ** n

    loss = focal_term * ce
    return loss.view(-1, 200, 200)


def focal_loss_o(logits, target, weights=None, n=2):
    target = target.argmax(dim=1)
    log_p = F.log_softmax(logits, dim=1)

    ce = F.nll_loss(log_p, target, weight=weights, reduction='none')

    log_pt = log_p.gather(1, target[None])
    pt = log_pt.exp()
    loss = ce * (1 - pt + 1e-8) ** n

    return loss


def uce_loss(alpha, y, weights=None):
    S = torch.sum(alpha, dim=1, keepdim=True)
    B = y * (torch.digamma(S) - torch.digamma(alpha) + 1e-10)

    if weights is not None:
        B *= weights.view(1, -1, 1, 1)

    A = torch.sum(B, dim=1, keepdim=True)

    return A


def u_focal_loss(alpha, y, weights=None, n=2):
    S = torch.sum(alpha, dim=1, keepdim=True)

    a0 = S
    aj = torch.gather(alpha, 1, torch.argmax(y, dim=1, keepdim=True))

    B = y * torch.exp((torch.lgamma(a0 - aj + n) + torch.lgamma(a0)) -
                      (torch.lgamma(a0 + n) + torch.lgamma(a0 - aj))) * (torch.digamma(a0 + n) - torch.digamma(aj))

    if weights is not None:
        B *= weights.view(1, -1, 1, 1)

    A = torch.sum(B, dim=1, keepdim=True)

    return A


def entropy_reg(alpha, beta_reg=.001):
    alpha = alpha.permute(0, 2, 3, 1)

    reg = D.Dirichlet(alpha).entropy().unsqueeze(1)

    return -beta_reg * reg


def ood_reg(alpha, ood):
    if ood.long().sum() == 0:
        return 0

    alpha = alpha.permute(0, 2, 3, 1)

    alpha_d = D.Dirichlet(alpha)
    target_d = D.Dirichlet(torch.ones_like(alpha))

    reg = D.kl.kl_divergence(alpha_d, target_d).unsqueeze(1)

    return reg[ood.bool()].mean()


def gamma(x):
    return torch.exp(torch.lgamma(x))