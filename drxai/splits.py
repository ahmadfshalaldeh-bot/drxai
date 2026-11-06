"""Patient-grouped, grade-stratified five-fold partitioning with rotation.

Iteration k (k = 0..4): test = fold k, validation = fold (k+1) mod 5,
training = the remaining three folds. Every image is therefore tested exactly
once, never by a model that saw the same patient during training or model
selection. The resulting assignment is written to disk and released with the
paper (fold indices requested by Reviewer 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.Series:
    """Return a fold id (0..n_folds-1) per image.

    Stratification uses the image grade; grouping uses ``df.group`` so that
    fellow eyes / examination pairs share a fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, test_idx) in enumerate(sgkf.split(df, df["grade"], groups=df["group"])):
        fold[test_idx] = k
    assert (fold >= 0).all()
    return pd.Series(fold, index=df.index, name="fold")


def iteration_roles(fold: pd.Series, k: int, n_folds: int = 5) -> pd.Series:
    """Map each image to 'train' / 'val' / 'test' for rotation iteration k."""
    val = (k + 1) % n_folds
    role = np.where(fold == k, "test", np.where(fold == val, "val", "train"))
    return pd.Series(role, index=fold.index, name="role")


def check_no_leakage(df: pd.DataFrame, roles: pd.Series) -> None:
    """Raise if any patient/examination appears in more than one role."""
    per_group = roles.groupby(df["group"]).nunique()
    if (per_group > 1).any():
        raise AssertionError(f"{int((per_group > 1).sum())} groups cross partition boundaries")


def partition_table(df: pd.DataFrame, fold: pd.Series, n_folds: int = 5) -> pd.DataFrame:
    """Images per iteration x role x grade, plus unique patients (Table S1)."""
    rows = []
    for k in range(n_folds):
        roles = iteration_roles(fold, k, n_folds)
        check_no_leakage(df, roles)
        for role in ("train", "val", "test"):
            m = roles == role
            sub = df[m]
            row = {"iteration": k + 1, "role": role, "images": int(m.sum()),
                   "patients": int(sub["group"].nunique())}
            for g in range(5):
                row[f"grade{g}"] = int((sub["grade"] == g).sum())
            rows.append(row)
    return pd.DataFrame(rows)
