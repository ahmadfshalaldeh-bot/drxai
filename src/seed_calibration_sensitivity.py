#!/usr/bin/env python3
"""Recompute calibration sensitivity across deterministic group meta-fold seeds.

The EfficientNet-B4 predictions remain fixed. Only filename-derived group
assignments to the five calibration meta-folds vary. This is a sensitivity
analysis, not independent model replication or external validation.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

METRICS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "macro_f1",
    "qwk": "qwk",
    "log_loss": "log_loss",
    "macro_auc_ovr": "macro_auc_ovr",
    "brier_multiclass": "brier_multiclass",
    "ece_15": "ece_15",
    "ordinal_mae": "ordinal_mae",
    "within_one_grade": "within_one_grade",
    "severe_miss_rate_ge2": "severe_miss_rate_ge2",
    "referable_auc": "referable_auc",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    calibrator = Path(__file__).with_name("nested_ordinal_calibration.py")
    rows = []

    for seed in args.seeds:
        with tempfile.TemporaryDirectory(prefix=f"dr_cal_seed_{seed}_") as temp_dir:
            out_dir = Path(temp_dir)
            subprocess.run(
                [sys.executable, str(calibrator), "--root", str(root), "--out-dir", str(out_dir), "--seed", str(seed)],
                check=True,
            )
            summary = pd.read_csv(out_dir / "summary_metrics.csv").set_index("model")
            raw = summary.loc["raw"]
            calibrated = summary.loc["nested_ordinal_isotonic"]
            row = {"seed": seed}
            for column, metric in METRICS.items():
                row[column] = float(calibrated[metric] - raw[metric])
            rows.append(row)

    result = pd.DataFrame(rows)
    output = args.output or root / "results" / "ordinal_calibration" / "seed_stability_recomputed.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {output}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
