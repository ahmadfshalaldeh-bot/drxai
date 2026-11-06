"""Statistical inference (Section 3.10).

* ``delong_macro``: paired test for the difference in macro one-vs-rest AUC of
  two models scored on the same images. It is the stratified influence-function
  (structural-component) estimator of Proposition 2; for a single binary AUC it
  coincides exactly with DeLong et al. (1988) -- see ``delong_binary`` and the
  unit test ``tests/test_stats.py``.
* ``mcnemar``: exact (binomial) or continuity-corrected chi-square McNemar test
  on paired correct/incorrect decisions.
* ``holm``: Holm step-down adjustment.
* ``cluster_bootstrap``: percentile CIs resampling patients (clusters) within
  grade strata, so that paired eyes are resampled together and intervals are
  confined to the metric's range.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import per_class_auc


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------
def _placements(pos: np.ndarray, neg: np.ndarray):
    """V10(i) = mean_j psi(pos_i, neg_j);  V01(j) = mean_i psi(pos_i, neg_j)."""
    d = pos[:, None] - neg[None, :]
    psi = (d > 0) + 0.5 * (d == 0)
    return psi.mean(1), psi.mean(0)


def influence_macro_auc(y: np.ndarray, P: np.ndarray, classes=None) -> tuple[float, np.ndarray]:
    """Return (macro AUC, per-image influence xi_n) for the one-vs-rest macro AUC.

    xi_n = (1/K) [ (V10^{y_n}(n) - AUC_{y_n}) / m_{y_n}
                   + sum_{k != y_n} (V01^k(n) - AUC_k) / n_k ]
    """
    y = np.asarray(y)
    classes = list(range(P.shape[1])) if classes is None else list(classes)
    K = len(classes)
    xi = np.zeros(len(y))
    aucs = []
    for k in classes:
        pos_idx, neg_idx = np.where(y == k)[0], np.where(y != k)[0]
        v10, v01 = _placements(P[pos_idx, k], P[neg_idx, k])
        auc = v10.mean()
        aucs.append(auc)
        xi[pos_idx] += (v10 - auc) / len(pos_idx) / K
        xi[neg_idx] += (v01 - auc) / len(neg_idx) / K
    return float(np.mean(aucs)), xi


def _stratified_var(xi: np.ndarray, strata: np.ndarray) -> float:
    v = 0.0
    for g in np.unique(strata):
        x = xi[strata == g]
        if len(x) > 1:
            v += len(x) / (len(x) - 1) * ((x - x.mean()) ** 2).sum()
    return float(v)


def delong_macro(y, P_a, P_b, classes=None, groups=None) -> dict:
    """Paired two-sided test of H0: macroAUC_a = macroAUC_b (Proposition 2).

    groups=None : images are the sampling units, strata = reference grade
                  (reduces exactly to DeLong et al. for a single AUC).
    groups=ids  : patient-clustered version; influence values are summed within
                  each patient and the variance is taken across patients,
                  stratified by the patient's maximum grade.
    """
    y = np.asarray(y)
    a, xi_a = influence_macro_auc(y, P_a, classes)
    b, xi_b = influence_macro_auc(y, P_b, classes)
    d = xi_a - xi_b
    if groups is None:
        var = _stratified_var(d, y)
    else:
        g = pd.Series(d).groupby(np.asarray(groups)).sum()
        s = pd.Series(y).groupby(np.asarray(groups)).max().loc[g.index]
        var = _stratified_var(g.to_numpy(), s.to_numpy())
    z = (a - b) / np.sqrt(var) if var > 0 else np.inf * np.sign(a - b)
    p = float(2 * stats.norm.sf(abs(z)))
    se = np.sqrt(var)
    return {"auc_a": a, "auc_b": b, "diff": a - b, "se": float(se), "z": float(z), "p": p,
            "ci95": (a - b - 1.959964 * se, a - b + 1.959964 * se)}


def delong_binary(yb, s_a, s_b) -> dict:
    """Classical DeLong et al. (1988) paired test for two binary AUCs."""
    yb = np.asarray(yb).astype(bool)
    out = []
    for s in (np.asarray(s_a), np.asarray(s_b)):
        out.append(_placements(s[yb], s[~yb]))
    m, n = yb.sum(), (~yb).sum()
    V10 = np.vstack([out[0][0], out[1][0]])
    V01 = np.vstack([out[0][1], out[1][1]])
    auc = V10.mean(1)
    S10, S01 = np.cov(V10), np.cov(V01)
    L = np.array([1.0, -1.0])
    var = L @ S10 @ L / m + L @ S01 @ L / n
    z = (auc[0] - auc[1]) / np.sqrt(var)
    return {"auc_a": float(auc[0]), "auc_b": float(auc[1]), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z))), "var": float(var)}


# ---------------------------------------------------------------------------
# McNemar and multiplicity
# ---------------------------------------------------------------------------
def mcnemar(correct_a, correct_b) -> dict:
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())       # a right, b wrong
    n10 = int((~a & b).sum())       # a wrong, b right
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "stat": 0.0, "p": 1.0, "method": "none"}
    if n < 25:
        p = float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
        return {"n01": n01, "n10": n10, "stat": float(min(n01, n10)), "p": p, "method": "exact"}
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    return {"n01": n01, "n10": n10, "stat": float(chi2), "p": float(stats.chi2.sf(chi2, 1)),
            "method": "chi2_cc"}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap(metric: Callable[[np.ndarray], float], groups, strata, B: int = 1000,
                      seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile CI of ``metric(idx)`` where idx are resampled image indices.

    Clusters (patients) are resampled with replacement within strata defined by
    the cluster's maximum grade; all images of a drawn cluster enter together.
    Percentile intervals inherit the range of the metric (e.g. [0, 1]).
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    cstrata = np.array([np.max(np.asarray(strata)[mbr]) for mbr in members])
    by_stratum = {s: np.where(cstrata == s)[0] for s in np.unique(cstrata)}
    est = metric(np.arange(len(groups)))
    reps = np.empty(B)
    for b in range(B):
        draw = np.concatenate([rng.choice(ids, size=len(ids), replace=True)
                               for ids in by_stratum.values()])
        idx = np.concatenate([members[c] for c in draw])
        reps[b] = metric(idx)
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(est), "lo": float(lo), "hi": float(hi), "se": float(np.nanstd(reps))}


def fold_summary(values) -> tuple[float, float]:
    """Mean and sample SD across the five test folds (Tables 4-6)."""
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1))


def paired_t(a, b) -> dict:
    t, p = stats.ttest_rel(a, b)
    return {"t": float(t), "p": float(p), "mean_diff": float(np.mean(np.asarray(a) - np.asarray(b)))}


def wilcoxon(a, b) -> dict:
    w, p = stats.wilcoxon(a, b)
    return {"W": float(w), "p": float(p)}


__all__ = ["delong_macro", "delong_binary", "influence_macro_auc", "mcnemar", "holm",
           "cluster_bootstrap", "fold_summary", "paired_t", "wilcoxon", "per_class_auc"]
