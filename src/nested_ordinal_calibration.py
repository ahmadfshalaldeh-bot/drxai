#!/usr/bin/env python3
"""Group-disjoint ordinal calibration on fixed EfficientNet-B4 fold-assigned probabilities.

This is a second-stage calibration analysis. It does not retrain EfficientNet-B4 and
must not be described as end-to-end nested cross-validation or external validation.
For each meta-fold, cumulative grade probabilities are calibrated on four partitions
of filename-derived grouping keys and evaluated on the fifth. Isotonic regression is fitted separately
for each cumulative threshold, then a decreasing projection guarantees a valid
ordinal cumulative distribution and non-negative class probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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


def multiclass_ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for i in range(bins):
        mask = (confidence >= edges[i]) & (
            confidence <= edges[i + 1] if i == bins - 1 else confidence < edges[i + 1]
        )
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def brier_multiclass(y: np.ndarray, p: np.ndarray) -> float:
    one_hot = np.eye(5, dtype=float)[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def enforce_decreasing(q: np.ndarray) -> np.ndarray:
    """Project each row to a non-increasing cumulative sequence."""
    return np.minimum.accumulate(q, axis=1)


def probabilities_to_cumulative(p: np.ndarray) -> np.ndarray:
    # q[:, k-1] = P(Y >= k), k=1,...,4.
    return np.column_stack([p[:, 1:].sum(axis=1), p[:, 2:].sum(axis=1), p[:, 3:].sum(axis=1), p[:, 4]])


def cumulative_to_probabilities(q: np.ndarray) -> np.ndarray:
    q = enforce_decreasing(np.clip(q, 0.0, 1.0))
    out = np.empty((len(q), 5), dtype=float)
    out[:, 0] = 1.0 - q[:, 0]
    out[:, 1] = q[:, 0] - q[:, 1]
    out[:, 2] = q[:, 1] - q[:, 2]
    out[:, 3] = q[:, 2] - q[:, 3]
    out[:, 4] = q[:, 3]
    return normalize(out)


def fit_calibrator(y: np.ndarray, p: np.ndarray) -> list[IsotonicRegression]:
    q = probabilities_to_cumulative(p)
    calibrators = []
    for k in range(4):
        target = (y >= (k + 1)).astype(float)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        model.fit(q[:, k], target)
        calibrators.append(model)
    return calibrators


def apply_calibrator(calibrators: list[IsotonicRegression], p: np.ndarray) -> np.ndarray:
    q = probabilities_to_cumulative(p)
    calibrated_q = np.column_stack([m.predict(q[:, k]) for k, m in enumerate(calibrators)])
    return cumulative_to_probabilities(calibrated_q)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = normalize(p)
    pred = p.argmax(axis=1)
    grade_error = np.abs(y - pred)
    y_ref = (y >= 2).astype(int)
    pred_ref = (pred >= 2).astype(int)
    p_ref = p[:, 2:].sum(axis=1)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "log_loss": float(log_loss(y, p, labels=list(range(5)))),
        "macro_auc_ovr": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "brier_multiclass": brier_multiclass(y, p),
        "ece_15": multiclass_ece(y, p),
        "ordinal_mae": float(grade_error.mean()),
        "within_one_grade": float(np.mean(grade_error <= 1)),
        "severe_miss_rate_ge2": float(np.mean(grade_error >= 2)),
        "referable_auc": float(roc_auc_score(y_ref, p_ref)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=root_default())
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.out_dir or root / "results" / "ordinal_calibration").resolve()
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(root / "results/raw/oof_predictions.csv")
    fold = pd.read_csv(root / "results/raw/fold_assignments.csv")
    df = pred.merge(fold[["image_id", "subject_id"]], on="image_id", validate="one_to_one")
    y = df["label"].to_numpy(dtype=int)
    p_raw = normalize(df[PROB_COLS].to_numpy(dtype=float))
    # This legacy manifest column contains filename-derived grouping keys; it is
    # not an independently verified patient or subject identifier.
    groups = df["subject_id"].astype(str).to_numpy()
    unique_groups = sorted(set(groups), key=lambda x: int(x) if x.isdigit() else x)
    rng = np.random.default_rng(args.seed)
    # Deterministic group-level meta-folds, independent of the original fold column.
    shuffled = np.asarray(unique_groups, dtype=object)
    rng.shuffle(shuffled)
    meta_fold_by_group = {g: i % 5 for i, g in enumerate(shuffled)}
    meta_fold = np.asarray([meta_fold_by_group[g] for g in groups], dtype=int)

    rows = []
    calibrated_probabilities = np.zeros_like(p_raw)
    for k in range(5):
        train = meta_fold != k
        test = meta_fold == k
        calibrators = fit_calibrator(y[train], p_raw[train])
        calibrated_probabilities[test] = apply_calibrator(calibrators, p_raw[test])
        raw_m = metrics(y[test], p_raw[test])
        cal_m = metrics(y[test], calibrated_probabilities[test])
        rows.extend([
            {"meta_fold": k, "model": "raw", **raw_m},
            {"meta_fold": k, "model": "nested_ordinal_isotonic", **cal_m},
        ])

    raw = metrics(y, p_raw)
    calibrated = metrics(y, calibrated_probabilities)
    summary = pd.DataFrame([
        {"model": "raw", **raw},
        {"model": "nested_ordinal_isotonic", **calibrated},
    ])
    fold_metrics = pd.DataFrame(rows)
    pd.DataFrame({"image_id": df["image_id"], "meta_fold": meta_fold, **{f"raw_{c}": p_raw[:, i] for i, c in enumerate(PROB_COLS)}, **{f"calibrated_{c}": calibrated_probabilities[:, i] for i, c in enumerate(PROB_COLS)}}).to_csv(out / "nested_calibrated_probabilities.csv", index=False)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    fold_metrics.to_csv(out / "metrics_by_meta_fold.csv", index=False)
    (out / "method.json").write_text(json.dumps({
        "method": "group-disjoint ordinal isotonic calibration of fixed predictions",
        "base_predictions": "fixed EfficientNet-B4 fold-assigned probabilities",
        "meta_folds": 5,
        "group_column": "subject_id (filename-derived grouping key; not verified identity)",
        "seed": args.seed,
        "external_validation": False,
        "end_to_end_nested_retraining": False,
        "description": "Each cumulative probability P(Y>=k) is calibrated on four partitions of filename-derived grouping keys and evaluated on the fifth; rows are projected to a valid decreasing cumulative distribution before conversion back to five-class probabilities. This is not end-to-end nested cross-validation and is not external validation."
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
