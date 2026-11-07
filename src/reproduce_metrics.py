#!/usr/bin/env python3
"""Recompute the EfficientNet-B4 Kaggle benchmark from raw fold-assigned output.

This script intentionally performs prediction-level reproduction. It does not claim
that the archived training snapshot can be retrained bit-for-bit: the supplied
training histories and software lockfile are incomplete. The script verifies the
probability file, fold assignments, labels, probabilities, primary metrics,
referable-DR metrics, calibration, and filename-group bootstrap intervals. The
legacy `oof_predictions.csv` filename and `oof_pred` column are retained; these are
not untouched outer-test predictions because the same fold informed checkpoint
selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)

CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
PROB_COLS = [f"prob_class_{i}" for i in range(5)]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_group_column(df: pd.DataFrame) -> str:
    for candidate in ("subject_id", "group_id", "subject", "patient_id"):
        if candidate in df.columns:
            return candidate
    for candidate in ("filename_group", "group", "filename_subject"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "No grouping column was found. Expected subject_id/group_id or a documented equivalent."
    )


def load_and_verify(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    pred_path = root / "results" / "raw" / "oof_predictions.csv"
    split_path = root / "results" / "raw" / "fold_assignments.csv"
    pred = pd.read_csv(pred_path)
    splits = pd.read_csv(split_path)

    required_pred = {"image_id", "label", "oof_pred", *PROB_COLS}
    required_split = {"image_id", "fold"}
    missing_pred = required_pred - set(pred.columns)
    missing_split = required_split - set(splits.columns)
    if missing_pred:
        raise ValueError(f"OOF file is missing columns: {sorted(missing_pred)}")
    if missing_split:
        raise ValueError(f"Fold file is missing columns: {sorted(missing_split)}")

    if pred["image_id"].duplicated().any():
        raise ValueError("OOF image identifiers are not unique")
    if splits["image_id"].duplicated().any():
        raise ValueError("Fold-assignment image identifiers are not unique")
    if len(pred) != len(splits):
        raise ValueError(f"OOF rows ({len(pred)}) != fold rows ({len(splits)})")

    merged = pred.merge(splits, on="image_id", how="left", validate="one_to_one", suffixes=("", "_split"))
    if merged["fold"].isna().any():
        raise ValueError("Some OOF rows have no fold assignment")
    merged["fold"] = merged["fold"].astype(int)

    groups = find_group_column(merged)
    probs = merged[PROB_COLS].to_numpy(dtype=float)
    labels = merged["label"].to_numpy(dtype=int)
    preds = merged["oof_pred"].to_numpy(dtype=int)
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("Non-finite or negative probabilities found")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"Probability sums outside tolerance: min={sums.min()} max={sums.max()}")
    recomputed = probs.argmax(axis=1)
    if not np.array_equal(recomputed, preds):
        raise ValueError("Stored oof_pred does not equal probability argmax")
    if not np.array_equal(labels, merged["label"].to_numpy(dtype=int)):
        raise ValueError("Label merge changed labels unexpectedly")
    if merged[groups].isna().any():
        raise ValueError(f"Grouping column {groups!r} contains missing values")
    leakage = merged.groupby(groups)["fold"].nunique()
    if (leakage > 1).any():
        bad = int((leakage > 1).sum())
        raise ValueError(f"Detected {bad} groups spanning multiple folds")
    return merged, splits, groups


def multiclass_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df["label"].to_numpy(dtype=int)
    pred = df["oof_pred"].to_numpy(dtype=int)
    p = df[PROB_COLS].to_numpy(dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    absolute_grade_error = np.abs(y - pred)
    out = {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=list(range(5)))),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "macro_auc_ovr": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "ordinal_mae": float(absolute_grade_error.mean()),
        "ordinal_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "within_one_grade": float(np.mean(absolute_grade_error <= 1)),
        "severe_miss_rate_ge2": float(np.mean(absolute_grade_error >= 2)),
        "undergrading_rate": float(np.mean(pred < y)),
        "overgrading_rate": float(np.mean(pred > y)),
    }
    return out


def referable_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = (df["label"].to_numpy(dtype=int) >= 2).astype(int)
    pred = (df["oof_pred"].to_numpy(dtype=int) >= 2).astype(int)
    p = df[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(dtype=float).sum(axis=1)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity_recall_referable": float(recall_score(y, pred, zero_division=0)),
        "specificity_no_dr": float(tn / (tn + fp)),
        "precision_ppv_referable": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def ece(df: pd.DataFrame, bins: int = 15) -> float:
    y = df["label"].to_numpy(dtype=int)
    p = df[PROB_COLS].to_numpy(dtype=float)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence <= hi if hi == 1 else confidence < hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def filename_group_bootstrap(df: pd.DataFrame, group_col: str, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_values = df[group_col].astype(str).to_numpy()
    unique_groups = sorted(set(group_values), key=lambda x: int(x) if x.isdigit() else x)
    groups = [np.flatnonzero(group_values == group) for group in unique_groups]
    rows = []
    for rep in range(replicates):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        sample = df.iloc[idx]
        m = multiclass_metrics(sample)
        rows.append({"replicate": rep, **{k: m[k] for k in (
            "accuracy", "balanced_accuracy", "macro_f1", "qwk", "macro_auc_ovr",
            "log_loss", "ordinal_mae", "within_one_grade", "severe_miss_rate_ge2"
        )}})
    boot = pd.DataFrame(rows)
    summary = []
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "qwk", "macro_auc_ovr", "log_loss", "ordinal_mae", "within_one_grade", "severe_miss_rate_ge2"):
        summary.append({
            "metric": metric,
            "mean": float(boot[metric].mean()),
            "lower_95": float(boot[metric].quantile(0.025)),
            "upper_95": float(boot[metric].quantile(0.975)),
            "seed": seed,
            "replicates": replicates,
        })
    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out_dir or root / "results" / "recomputed").resolve()
    out.mkdir(parents=True, exist_ok=True)

    df, splits, group_col = load_and_verify(root)
    primary = multiclass_metrics(df)
    referable = referable_metrics(df)
    calibration = {"ece_15": ece(df, bins=15)}
    fold_rows = []
    for fold, part in df.groupby("fold", sort=True):
        fold_rows.append({"fold": int(fold), **multiclass_metrics(part)})
    fold_metrics = pd.DataFrame(fold_rows)
    boot = filename_group_bootstrap(df, group_col, args.bootstrap_replicates, args.seed)

    pd.DataFrame([primary]).to_csv(out / "aggregate_metrics_recomputed.csv", index=False)
    pd.DataFrame([referable]).to_csv(out / "referable_metrics_recomputed.csv", index=False)
    pd.DataFrame([calibration]).to_csv(out / "calibration_metrics_recomputed.csv", index=False)
    fold_metrics.to_csv(out / "metrics_by_fold_recomputed.csv", index=False)
    boot.to_csv(out / "confidence_intervals_recomputed.csv", index=False)
    pd.DataFrame(confusion_matrix(df["label"], df["oof_pred"], labels=list(range(5))), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / "confusion_matrix_recomputed.csv")

    audit = {
        "rows": int(len(df)),
        "unique_image_ids": int(df["image_id"].nunique()),
        "unique_groups": int(df[group_col].nunique()),
        "fold_counts": {str(k): int(v) for k, v in df["fold"].value_counts().sort_index().items()},
        "group_column": group_col,
        "ece_15": calibration["ece_15"],
        "probability_sum_min": float(df[PROB_COLS].sum(axis=1).min()),
        "probability_sum_max": float(df[PROB_COLS].sum(axis=1).max()),
    }
    (out / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"primary": primary, "referable": referable, "calibration": calibration, "out_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
