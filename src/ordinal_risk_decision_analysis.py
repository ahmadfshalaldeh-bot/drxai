#!/usr/bin/env python3
"""Compare argmax and ordinal squared-distance decisions on fixed fold-assigned probabilities.

This is a prediction-level sensitivity analysis. It does not retrain EfficientNet-B4,
change the primary argmax result, or constitute external validation. For grades
0,...,4 and C(j,k)=(j-k)^2, the posterior-risk minimizer is the integer grade
nearest to the posterior mean sum_j j p_j, with deterministic lower-grade tie breaking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

PROB_COLS = [f"prob_class_{i}" for i in range(5)]
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "ordinal_mae", "within_one_grade", "severe_miss_rate_ge2"]


def metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.abs(y - pred)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "log_loss": float(log_loss(y, p, labels=list(range(5)))),
        "macro_auc_ovr": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "ordinal_mae": float(err.mean()),
        "within_one_grade": float(np.mean(err <= 1)),
        "severe_miss_rate_ge2": float(np.mean(err >= 2)),
    }


def decision_rules(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grades = np.arange(5, dtype=float)
    posterior_mean = p @ grades
    # Equivalent to argmin_k E[(Y-k)^2]; ties go to the lower grade.
    risk_pred = np.floor(posterior_mean + 0.5).astype(int).clip(0, 4)
    risk = np.stack([((grades[None, :] - k) ** 2 * p).sum(axis=1) for k in range(5)], axis=1)
    risk_argmin = risk.argmin(axis=1)
    if not np.array_equal(risk_pred, risk_argmin):
        raise RuntimeError("posterior-mean decision is not equal to squared-risk argmin")
    return p.argmax(axis=1), risk_pred, posterior_mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / "results" / "ordinal_risk_decision"
    out.mkdir(parents=True, exist_ok=True)

    # The source filename is retained for compatibility; predictions are not
    # untouched outer-test outputs because each prediction fold informed selection.
    pred = pd.read_csv(root / "results/raw/oof_predictions.csv")
    fold = pd.read_csv(root / "results/raw/fold_assignments.csv")
    df = pred.merge(fold[["image_id", "subject_id"]], on="image_id", validate="one_to_one")
    y = df["label"].to_numpy(dtype=int)
    p = df[PROB_COLS].to_numpy(dtype=float)
    p = np.clip(p, 1e-12, None)
    p /= p.sum(axis=1, keepdims=True)
    argmax_pred, risk_pred, posterior_mean = decision_rules(p)

    summary = pd.DataFrame([
        {"decision_rule": "argmax", **metrics(y, p, argmax_pred)},
        {"decision_rule": "ordinal_squared_distance_bayes", **metrics(y, p, risk_pred)},
    ])
    summary.to_csv(out / "summary_metrics.csv", index=False)
    pd.DataFrame({
        "image_id": df["image_id"],
        "subject_id": df["subject_id"],
        "label": y,
        "posterior_mean_grade": posterior_mean,
        "argmax_pred": argmax_pred,
        "ordinal_risk_pred": risk_pred,
        "decision_changed": argmax_pred != risk_pred,
    }).to_csv(out / "decision_predictions.csv", index=False)

    groups = [g.to_numpy() for _, g in df.groupby("subject_id", sort=True).groups.items()]
    rng = np.random.default_rng(args.seed)
    rows = []
    for b in range(args.replicates):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        raw = metrics(y[idx], p[idx], argmax_pred[idx])
        risk = metrics(y[idx], p[idx], risk_pred[idx])
        row = {"replicate": b}
        for metric in METRICS:
            row[f"{metric}_argmax"] = raw[metric]
            row[f"{metric}_ordinal_risk"] = risk[metric]
            row[f"{metric}_difference"] = risk[metric] - raw[metric]
        rows.append(row)
    boot = pd.DataFrame(rows)
    boot.to_csv(out / "paired_group_bootstrap_replicates.csv", index=False)
    ci = []
    for metric in METRICS:
        vals = boot[f"{metric}_difference"].to_numpy()
        ci.append({
            "metric": metric,
            "argmax_estimate": metrics(y, p, argmax_pred)[metric],
            "ordinal_risk_estimate": metrics(y, p, risk_pred)[metric],
            "difference_ordinal_risk_minus_argmax": float(vals.mean()),
            "lower_95": float(np.quantile(vals, 0.025)),
            "upper_95": float(np.quantile(vals, 0.975)),
            "seed": args.seed,
            "replicates": args.replicates,
        })
    pd.DataFrame(ci).to_csv(out / "paired_group_bootstrap_differences.csv", index=False)
    (out / "method.json").write_text(json.dumps({
        "decision_rule": "argmin_k sum_j (j-k)^2 p_j(x)",
        "equivalence": "nearest integer to posterior mean grade, with lower-grade tie breaking",
        "primary_rule": "argmax retained as primary manuscript result",
        "group_bootstrap": True,
        "replicates": args.replicates,
        "seed": args.seed,
        "external_validation": False,
        "retraining": False,
    }, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nChanged decisions:", int(np.sum(argmax_pred != risk_pred)))
    print(pd.DataFrame(ci).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
