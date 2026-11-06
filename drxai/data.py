"""Dataset indexing with explicit patient / examination identifiers.

Every row of an index is one image and carries a ``group`` column. The group is
the unit used for partitioning, so that both eyes of one person (EyePACS) or
both images of one examination (Messidor-2) always fall in the same fold.

Index columns: image_id, path, grade, group, dataset
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GRADES = (0, 1, 2, 3, 4)


def index_eyepacs(labels_csv: str, image_dir: str, image_ext: str = ".jpeg") -> pd.DataFrame:
    """Kaggle 2015 EyePACS training set (35,126 images, 17,563 patients).

    ``trainLabels.csv`` has columns ``image`` (e.g. ``10_left``) and ``level``.
    The numeric prefix is the anonymised patient identifier.
    """
    df = pd.read_csv(labels_csv)
    df = df.rename(columns={"image": "image_id", "level": "grade"})
    df["group"] = "E" + df["image_id"].str.split("_").str[0]
    df["path"] = [str(Path(image_dir) / f"{i}{image_ext}") for i in df["image_id"]]
    df["dataset"] = "eyepacs"
    _check_grades(df)
    return df[["image_id", "path", "grade", "group", "dataset"]].reset_index(drop=True)


def index_messidor2(labels_csv: str, pairs_csv: str, image_dir: str,
                    drop_ungradable: bool = True) -> pd.DataFrame:
    """Messidor-2 with the adjudicated ICDR grades released by Krause et al. (2018).

    ``labels_csv`` columns: image_id, adjudicated_dr_grade, adjudicated_dme,
    adjudicated_gradable. Four of the 1,748 images have no adjudicated grade
    (ungradable) and are excluded, giving the 1,744-image analytical sample.
    ``pairs_csv`` maps each image to its examination (two images per exam).
    """
    lab = pd.read_csv(labels_csv)
    lab = lab.rename(columns={"adjudicated_dr_grade": "grade"})
    n_all = len(lab)
    if drop_ungradable:
        if "adjudicated_gradable" in lab:
            lab = lab[lab["adjudicated_gradable"].astype(float) == 1]
        lab = lab[lab["grade"].notna()]
    lab["grade"] = lab["grade"].astype(int)
    pairs = pd.read_csv(pairs_csv)
    if not {"image_id", "exam_id"} <= set(pairs.columns):
        raise ValueError("pairs_csv needs columns image_id, exam_id")
    df = lab.merge(pairs[["image_id", "exam_id"]], on="image_id", how="left", validate="1:1")
    if df["exam_id"].isna().any():
        missing = df.loc[df["exam_id"].isna(), "image_id"].tolist()[:5]
        raise ValueError(f"images without examination id, e.g. {missing}")
    df["group"] = "M" + df["exam_id"].astype(str)
    df["path"] = [str(Path(image_dir) / i) for i in df["image_id"]]
    df["dataset"] = "messidor2"
    df.attrs["n_excluded_ungradable"] = n_all - len(df)
    _check_grades(df)
    return df[["image_id", "path", "grade", "group", "dataset"]].reset_index(drop=True)


def messidor_pairs_from_spreadsheet(xls_path: str, out_csv: str) -> pd.DataFrame:
    """Build ``image_id,exam_id`` from the examination spreadsheet shipped with
    Messidor-2 (one row per examination, image file names in two cells)."""
    raw = pd.read_excel(xls_path, header=None, dtype=str)
    rows = []
    exam = 0
    for _, r in raw.iterrows():
        names = [c for c in r.dropna().tolist()
                 if str(c).lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        if len(names) >= 2:
            exam += 1
            rows += [{"image_id": n.strip(), "exam_id": exam} for n in names]
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Per-grade counts and percentages (Section 3.2, Figure 2)."""
    counts = df["grade"].value_counts().reindex(GRADES, fill_value=0)
    return pd.DataFrame({"n": counts, "pct": 100 * counts / counts.sum()})


def _check_grades(df: pd.DataFrame) -> None:
    bad = set(np.unique(df["grade"])) - set(GRADES)
    if bad:
        raise ValueError(f"unexpected grades {bad}")
