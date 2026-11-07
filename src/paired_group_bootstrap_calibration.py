#!/usr/bin/env python3
"""Paired filename-group bootstrap comparison of raw and calibrated probabilities.

The raw and calibrated predictions are evaluated on the same images. Resampling is
performed at the filename-derived grouping-key level, preserving both eye-side records.
This produces uncertainty intervals for paired metric differences without relying on
an independence assumption across images.
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


def root_default() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for i in range(bins):
        mask = (confidence >= edges[i]) & (
            confidence <= edges[i + 1] if i == bins - 1 else confidence < edges[i + 1]
        )
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = normalize(p)
    pred = p.argmax(axis=1)
    grade_error = np.abs(y - pred)
    referable_y = (y >= 2).astype(int)
    referable_p = p[:, 2:].sum(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "log_loss": float(log_loss(y, p, labels=list(range(5)))),
        "macro_auc_ovr": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "brier_multiclass": float(np.mean(np.sum((p - np.eye(5)[y]) ** 2, axis=1))),
        "ece_15": ece(y, p),
        "ordinal_mae": float(grade_error.mean()),
        "within_one_grade": float(np.mean(grade_error <= 1)),
        "severe_miss_rate_ge2": float(np.mean(grade_error >= 2)),
        "referable_auc": float(roc_auc_score(referable_y, referable_p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=root_default())
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.out_dir or root / "results" / "ordinal_calibration").resolve()
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(root / "results/raw/oof_predictions.csv")
    fold = pd.read_csv(root / "results/raw/fold_assignments.csv")
    calibrated = pd.read_csv(out / "nested_calibrated_probabilities.csv")
    df = pred.merge(fold[["image_id", "subject_id"]], on="image_id", validate="one_to_one")
    df = df.merge(calibrated, on=["image_id"], validate="one_to_one")
    y = df["label"].to_numpy(dtype=int)
    raw_p = normalize(df[PROB_COLS].to_numpy(dtype=float))
    cal_p = normalize(df[[f"calibrated_{c}" for c in PROB_COLS]].to_numpy(dtype=float))
    groups = df["subject_id"].astype(str).to_numpy()
    unique_groups = sorted(set(groups), key=lambda x: int(x) if x.isdigit() else x)
    positions = [np.flatnonzero(groups == g) for g in unique_groups]

    raw = metrics(y, raw_p)
    cal = metrics(y, cal_p)
    rows = []
    rng = np.random.default_rng(args.seed)
    for rep in range(args.replicates):
        chosen = rng.integers(0, len(positions), size=len(positions))
        idx = np.concatenate([positions[i] for i in chosen])
        m_raw = metrics(y[idx], raw_p[idx])
        m_cal = metrics(y[idx], cal_p[idx])
        rows.append({"replicate": rep, **{f"delta_{k}": m_cal[k] - m_raw[k] for k in raw}})
    boot = pd.DataFrame(rows)
    summary = []
    for metric in raw:
        d = boot[f"delta_{metric}"]
        summary.append({
            "metric": metric,
            "raw_estimate": raw[metric],
            "calibrated_estimate": cal[metric],
            "delta_calibrated_minus_raw": cal[metric] - raw[metric],
            "lower_95": float(d.quantile(0.025)),
            "upper_95": float(d.quantile(0.975)),
            "bootstrap_direction_fraction": float(np.mean(d > 0)),
            "seed": args.seed,
            "replicates": args.replicates,
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "paired_group_bootstrap_differences.csv", index=False)
    boot.to_csv(out / "paired_group_bootstrap_replicates.csv", index=False)

    raw_sum = raw_p.sum(axis=1)
    cal_sum = cal_p.sum(axis=1)
    audit = {
        "rows": int(len(df)),
        "filename_derived_groups": int(len(unique_groups)),
        "raw_min_probability": float(raw_p.min()),
        "calibrated_min_probability": float(cal_p.min()),
        "raw_max_abs_sum_error": float(np.max(np.abs(raw_sum - 1.0))),
        "calibrated_max_abs_sum_error": float(np.max(np.abs(cal_sum - 1.0))),
        "raw_argmax_matches": bool(np.array_equal(raw_p.argmax(axis=1), df["oof_pred"].to_numpy(dtype=int))),
        "calibrated_argmax_class_counts": {str(k): int(v) for k, v in pd.Series(cal_p.argmax(axis=1)).value_counts().sort_index().items()},
        "method_note": "Paired filename-group bootstrap; confidence intervals are for calibrated minus raw metric differences and are not p-values. The grouping key is not independently verified as a clinical patient identifier.",
    }
    (out / "paired_comparison_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary_df.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
