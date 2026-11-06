"""Evaluation metrics exactly as defined in Section 3.9 (Eqs. (11)-(19)).

y : int array (N,) of reference grades 0..K-1
P : float array (N, K) of class probabilities (rows sum to 1)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

K_DEFAULT = 5


def confusion(y, yhat, k: int = K_DEFAULT) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (np.asarray(y), np.asarray(yhat)), 1)
    return cm


def auc_binary(pos_scores, neg_scores) -> float:
    """Mann-Whitney form of the AUC with ties counted 1/2 (Eq. (11))."""
    s = np.concatenate([pos_scores, neg_scores])
    from scipy.stats import rankdata

    r = rankdata(s)
    m, n = len(pos_scores), len(neg_scores)
    return float((r[:m].sum() - m * (m + 1) / 2) / (m * n))


def per_class_auc(y, P) -> np.ndarray:
    y = np.asarray(y)
    return np.array([auc_binary(P[y == c, c], P[y != c, c]) if (y == c).any() and (y != c).any()
                     else np.nan for c in range(P.shape[1])])


def macro_auc(y, P) -> float:
    """Macro-averaged one-vs-rest AUC (Eq. (12))."""
    return float(np.nanmean(per_class_auc(y, P)))


def per_class_sens_spec(y, yhat, k: int = K_DEFAULT):
    cm = confusion(y, yhat, k)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        prec = tp / (tp + fp)
        f1 = 2 * prec * sens / (prec + sens)
    return sens, spec, np.nan_to_num(f1)


def quadratic_weighted_kappa(y, yhat, k: int = K_DEFAULT) -> float:
    """kappa_w = 1 - sum W O / sum W E,  W_ij = (i-j)^2/(K-1)^2  (Eq. (15))."""
    O = confusion(y, yhat, k).astype(float)
    i, j = np.indices((k, k))
    W = (i - j) ** 2 / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1.0 - (W * O).sum() / (W * E).sum())


def expected_calibration_error(y, P, bins: int = 10) -> float:
    """Top-label ECE with equal-width confidence bins (Eq. (16))."""
    conf = P.max(1)
    pred = P.argmax(1)
    correct = (pred == np.asarray(y)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(y, P, bins: int = 10):
    conf, pred = P.max(1), P.argmax(1)
    correct = (pred == np.asarray(y)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    acc = np.array([correct[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    cf = np.array([conf[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    cnt = np.bincount(idx, minlength=bins)
    return cf, acc, cnt


def brier_multiclass(y, P) -> float:
    """(1/N) sum_n sum_c (P_nc - 1[y_n = c])^2, range [0, 2] (Eq. (17))."""
    Y = np.eye(P.shape[1])[np.asarray(y)]
    return float(((P - Y) ** 2).sum(1).mean())


def referable(y, P, yhat=None, threshold: int = 2):
    """Referable DR derived from the five-class output (Eq. (18)).

    Score  : P(grade >= threshold) = sum_{c >= threshold} P_c
    Decision: argmax grade >= threshold
    """
    yb = (np.asarray(y) >= threshold).astype(int)
    score = P[:, threshold:].sum(1)
    yhat = P.argmax(1) if yhat is None else yhat
    dec = (np.asarray(yhat) >= threshold).astype(int)
    tp = int(((dec == 1) & (yb == 1)).sum()); fn = int(((dec == 0) & (yb == 1)).sum())
    tn = int(((dec == 0) & (yb == 0)).sum()); fp = int(((dec == 1) & (yb == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"auc": float(roc_auc_score(yb, score)) if 0 < yb.sum() < len(yb) else np.nan,
            "sensitivity": sens, "specificity": spec, "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def predictive_values(sens: float, spec: float, prevalence: float):
    """PPV and NPV by Bayes' theorem at an assumed prevalence (Eq. (19))."""
    ppv = sens * prevalence / (sens * prevalence + (1 - spec) * (1 - prevalence))
    npv = spec * (1 - prevalence) / (spec * (1 - prevalence) + (1 - sens) * prevalence)
    return ppv, npv


def all_metrics(y, P, k: int = K_DEFAULT, ece_bins: int = 10) -> dict:
    y = np.asarray(y)
    yhat = P.argmax(1)
    sens, spec, f1 = per_class_sens_spec(y, yhat, k)
    return {
        "accuracy": float((yhat == y).mean()),
        "macro_sensitivity": float(np.nanmean(sens)),
        "macro_specificity": float(np.nanmean(spec)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.nanmean(sens)),
        "macro_auc": macro_auc(y, P),
        "qwk": quadratic_weighted_kappa(y, yhat, k),
        "ece": expected_calibration_error(y, P, ece_bins),
        "brier": brier_multiclass(y, P),
        "per_class_recall": sens.tolist(),
        "per_class_auc": per_class_auc(y, P).tolist(),
    }
