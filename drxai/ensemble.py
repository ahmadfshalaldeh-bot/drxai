"""Fusion rules (Section 3.6, Algorithm 1) and the uniform-weight bound.

All functions operate on probability arrays of shape (M, N, K): M members,
N images, K classes.
"""
from __future__ import annotations

import numpy as np

from .metrics import macro_auc


def auc_weights(val_aucs) -> np.ndarray:
    """w_i = AUC_i^val / sum_j AUC_j^val   (Eq. (8))."""
    a = np.asarray(val_aucs, dtype=float)
    return a / a.sum()


def fuse(probs: np.ndarray, rule: str, weights=None) -> np.ndarray:
    """Return fused class scores (N, K).

    auc_weighted : sum_i w_i p_i
    mean         : (1/M) sum_i p_i
    majority     : fraction of members whose argmax equals c  (vote shares,
                   so that a ranking-based AUC is defined for hard voting;
                   ties in the final argmax are broken by the mean probability)
    """
    probs = np.asarray(probs, dtype=float)
    m, n, k = probs.shape
    if rule == "auc_weighted":
        w = np.asarray(weights, float).reshape(m, 1, 1)
        return (w * probs).sum(0)
    if rule == "mean":
        return probs.mean(0)
    if rule == "majority":
        votes = np.zeros((n, k))
        for i in range(m):
            votes[np.arange(n), probs[i].argmax(1)] += 1.0 / m
        return votes + 1e-6 * probs.mean(0)       # tie-break, keeps rows ~ summing to 1
    raise ValueError(rule)


def weight_deviation(weights) -> float:
    """delta = sum_i |w_i - 1/M| (Proposition 1)."""
    w = np.asarray(weights, float)
    return float(np.abs(w - 1.0 / len(w)).sum())


def uniform_bound(val_aucs) -> dict:
    """Proposition 1: |P^w_c - P^u_c| <= delta/2 <= (b - a) / (4 * mean(AUC))."""
    a = np.asarray(val_aucs, float)
    w = auc_weights(a)
    d = weight_deviation(w)
    return {"weights": w, "delta": d, "bound_delta": d / 2,
            "bound_range": (a.max() - a.min()) / (4 * a.mean())}


def near_tie_fraction(scores_u: np.ndarray, y: np.ndarray, eps: float) -> float:
    """Fraction of (positive, negative) pairs, averaged over one-vs-rest
    classes, whose uniform-fusion scores differ by less than ``eps``. Only these
    pairs can change order when weights replace uniform averaging, so this is
    an upper bound on |AUC_w - AUC_u| per class (Corollary 1)."""
    k = scores_u.shape[1]
    fr = []
    for c in range(k):
        pos, neg = scores_u[y == c, c], scores_u[y != c, c]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = np.abs(pos[:, None] - neg[None, :])
        fr.append(float((diff < eps).mean()))
    return float(np.mean(fr))


def compare_rules(probs, y, val_aucs) -> dict:
    out = {}
    w = auc_weights(val_aucs)
    for rule in ("auc_weighted", "mean", "majority"):
        out[rule] = macro_auc(y, fuse(probs, rule, w))
    b = uniform_bound(val_aucs)
    out["bound"] = b
    out["near_tie_fraction"] = near_tie_fraction(fuse(probs, "mean"), y, 2 * b["bound_delta"])
    return out
