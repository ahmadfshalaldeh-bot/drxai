#!/usr/bin/env python3
"""Recompute per-grade and derived grade->=2 bootstrap intervals.

Resampling uses filename-derived grouping keys and retains every associated
image record. The manifest column is called `subject_id` for compatibility,
but it is not a verified patient identifier. The script is prediction-level
and does not retrain EfficientNet-B4.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROB_COLS = [f"prob_class_{i}" for i in range(5)]


def weighted_auc(y_binary: np.ndarray, score: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    scores = score[order]
    y = y_binary[order].astype(np.int8)
    w = weights[order].astype(np.int64)
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    total = np.add.reduceat(w, starts)
    positive = np.add.reduceat(w * y, starts)
    negative = total - positive
    negative_before = np.cumsum(negative) - negative
    denom = positive.sum() * negative.sum()
    if denom == 0:
        return float("nan")
    return float(np.sum(positive * (negative_before + 0.5 * negative)) / denom)


def class_metrics(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagonal = np.diag(cm).astype(float)
    row_totals = cm.sum(axis=1).astype(float)
    col_totals = cm.sum(axis=0).astype(float)
    recall = np.divide(diagonal, row_totals, out=np.zeros_like(diagonal), where=row_totals > 0)
    precision = np.divide(diagonal, col_totals, out=np.zeros_like(diagonal), where=col_totals > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(diagonal), where=(precision + recall) > 0)
    return precision, recall, f1


def binary_metrics(cm: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = map(float, cm.ravel())
    n = tn + fp + fn + tp
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else 0.0
    return {"accuracy": (tn + tp) / n, "balanced_accuracy": (sensitivity + specificity) / 2,
            "sensitivity": sensitivity, "specificity": specificity, "ppv": ppv, "f1": f1}


def ci(values: list[float]) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975)), int(len(array))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = args.root.resolve()
    pred = pd.read_csv(root / "results/raw/oof_predictions.csv")
    split = pd.read_csv(root / "results/raw/fold_assignments.csv")
    group_col = "subject_id"  # Legacy schema; values are filename-derived keys.
    df = pred.merge(split[["image_id", group_col]], on="image_id", validate="one_to_one")
    if len(df) != 35126 or df["image_id"].nunique() != len(df):
        raise ValueError("Unexpected row count or duplicate image identifiers")
    if df[group_col].nunique() != 17563:
        raise ValueError("Unexpected number of filename-derived grouping keys")

    y = df["label"].to_numpy(dtype=int)
    p = df[PROB_COLS].to_numpy(dtype=float)
    pred_class = p.argmax(axis=1)
    if not np.isfinite(p).all() or not np.array_equal(pred_class, df["oof_pred"].to_numpy(dtype=int)):
        raise ValueError("Probability validity or argmax audit failed")
    group_values = df[group_col].astype(str).to_numpy()
    ordered_groups = sorted(set(group_values), key=lambda g: int(g) if g.isdigit() else g)
    group_index = {key: i for i, key in enumerate(ordered_groups)}
    codes = np.asarray([group_index[g] for g in group_values], dtype=int)
    n_groups = len(ordered_groups)

    cm5 = np.bincount(y * 5 + pred_class, minlength=25).reshape(5, 5)
    precision, recall, f1 = class_metrics(cm5)
    aucs = np.asarray([weighted_auc((y == c).astype(int), p[:, c], np.ones(len(y), dtype=int)) for c in range(5)])
    ref_y = (y >= 2).astype(int)
    ref_pred = (pred_class >= 2).astype(int)
    ref_prob = p[:, 2:].sum(axis=1)
    cm2 = np.bincount(ref_y * 2 + ref_pred, minlength=4).reshape(2, 2)
    point_binary = binary_metrics(cm2)
    point_binary["roc_auc"] = weighted_auc(ref_y, ref_prob, np.ones(len(y), dtype=int))

    class_values = {(grade, metric): [] for grade in range(5) for metric in ("precision", "recall", "f1", "roc_auc")}
    binary_values = {metric: [] for metric in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "ppv", "f1", "roc_auc")}
    rng = np.random.default_rng(args.seed)
    for b in range(args.replicates):
        chosen = rng.integers(0, n_groups, size=n_groups)
        group_weights = np.bincount(chosen, minlength=n_groups)
        weights = group_weights[codes]
        cm = np.bincount(y * 5 + pred_class, weights=weights, minlength=25).reshape(5, 5)
        pr, rc, f1s = class_metrics(cm)
        for grade in range(5):
            class_values[grade, "precision"].append(float(pr[grade]))
            class_values[grade, "recall"].append(float(rc[grade]))
            class_values[grade, "f1"].append(float(f1s[grade]))
            class_values[grade, "roc_auc"].append(weighted_auc((y == grade).astype(int), p[:, grade], weights))
        cm_ref = np.bincount(ref_y * 2 + ref_pred, weights=weights, minlength=4).reshape(2, 2)
        bm = binary_metrics(cm_ref)
        bm["roc_auc"] = weighted_auc(ref_y, ref_prob, weights)
        for metric, value in bm.items():
            binary_values[metric].append(float(value))
        if (b + 1) % 100 == 0:
            print(f"Bootstrap replicate {b + 1}/{args.replicates}", flush=True)

    out = root / "results/metrics"
    out.mkdir(parents=True, exist_ok=True)
    class_rows = []
    for grade in range(5):
        estimates = {"precision": precision[grade], "recall": recall[grade], "f1": f1[grade], "roc_auc": aucs[grade]}
        for metric, estimate in estimates.items():
            lo, hi, valid = ci(class_values[grade, metric])
            class_rows.append({"grade": grade, "metric": metric, "estimate": float(estimate), "lower_95": lo,
                               "upper_95": hi, "seed": args.seed, "replicates": args.replicates,
                               "valid_replicates": valid})
    pd.DataFrame(class_rows).to_csv(out / "class_metric_bootstrap_intervals.csv", index=False)

    binary_rows = []
    for metric, estimate in point_binary.items():
        lo, hi, valid = ci(binary_values[metric])
        binary_rows.append({"metric": metric, "estimate": estimate, "lower_95": lo, "upper_95": hi,
                            "seed": args.seed, "replicates": args.replicates, "valid_replicates": valid})
    pd.DataFrame(binary_rows).to_csv(out / "referable_metric_bootstrap_intervals.csv", index=False)
    print(f"Wrote per-grade and referable intervals to {out}")


if __name__ == "__main__":
    main()
