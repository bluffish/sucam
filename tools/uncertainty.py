import numpy as np
import torch


def vacuity(alpha):
    class_num = alpha.shape[1]
    S = torch.sum(alpha, dim=1, keepdim=True)
    v = class_num / S

    return v


def dissonance(alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)

    evidence = alpha - 1
    belief = evidence / S
    uncertainty = torch.zeros_like(S)

    for k in range(belief.shape[0]):
        for i in range(belief.shape[1]):
            bi = belief[k][i]
            term_bal = 0.0
            term_bj = 0.0
            for j in range(belief.shape[1]):
                if j != i:
                    bj = belief[k][j]
                    term_bal += bj * bal(bi, bj)
                    term_bj += bj

            uncertainty[k] += bi * term_bal / (term_bj + 1e-7)

    return uncertainty


def bal(b_i, b_j):
    return 1 - torch.abs(b_i - b_j) / (b_i + b_j + 1e-7)


def entropy(pred, dim=1):
    class_num = 4
    prob = torch.softmax(pred, dim=dim) + 1e-10

    e = -prob * (torch.log(prob) / np.log(class_num))
    u = torch.sum(e, dim=dim, keepdim=True)

    return u


