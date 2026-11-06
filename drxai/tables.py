"""Write the manuscript tables (LaTeX, MDPI tabularx style) from results/.

Generated files go to ../manuscript/generated/ and can be \\input{} in place of
the typed tables, or compared against them with ``drxai.verify``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .figures import LABELS
from .utils import ensure_dir, load_json


def _ms(d, key, nd=3):
    m, s = d[key]
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def table_performance(results: Path, out: Path):
    rows = ["vgg16", "vgg19", "resnet50", "inceptionv3", "dual_branch", "proposed"]
    S = {ds: load_json(results / ds / "summary.json")["fold_mean_sd"]
         for ds in ("messidor2", "eyepacs")}
    lines = [r"\begin{tabularx}{\textwidth}{lCCCC}", r"\toprule",
             r"\textbf{Model} & \textbf{M-2 Acc.} & \textbf{M-2 AUC} & \textbf{EP Acc.} & \textbf{EP AUC}\\",
             r"\midrule"]
    for m in rows:
        cells = [_ms(S[ds][m], c) for ds in ("messidor2", "eyepacs") for c in ("accuracy", "macro_auc")]
        lines.append(f"{LABELS[m]} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (out / "table4_performance.tex").write_text("\n".join(lines))


def table_partitions(results: Path, out: Path):
    for ds in ("messidor2", "eyepacs"):
        t = pd.read_csv(results / ds / "partition_table.csv")
        lines = [r"\begin{tabularx}{\textwidth}{ccCCCCCCC}", r"\toprule",
                 r"\textbf{Iter.} & \textbf{Role} & \textbf{Images} & \textbf{Patients} & "
                 r"\textbf{G0} & \textbf{G1} & \textbf{G2} & \textbf{G3} & \textbf{G4}\\", r"\midrule"]
        for r in t.itertuples():
            lines.append(f"{r.iteration} & {r.role} & {r.images:,} & {r.patients:,} & "
                         f"{r.grade0:,} & {r.grade1:,} & {r.grade2:,} & {r.grade3:,} & {r.grade4:,}\\\\")
        lines += [r"\bottomrule", r"\end{tabularx}"]
        (out / f"tableS1_partitions_{ds}.tex").write_text("\n".join(lines))


def table_pairwise(results: Path, out: Path):
    for ds in ("messidor2", "eyepacs"):
        t = pd.read_csv(results / ds / "pairwise_tests.csv")
        lines = [r"\begin{tabularx}{\textwidth}{lCCCCC}", r"\toprule",
                 r"\textbf{Comparator} & \textbf{$\Delta$AUC} & \textbf{DeLong $z$} & "
                 r"\textbf{$p_{\text{Holm}}$} & \textbf{McNemar $n_{01}/n_{10}$} & \textbf{$p_{\text{Holm}}$}\\",
                 r"\midrule"]
        for r in t.itertuples():
            lines.append(f"{LABELS.get(r.comparator, r.comparator)} & {r.auc_diff:+.3f} & "
                         f"{r.delong_z:.2f} & {_p(r.delong_p_holm)} & {r.mcnemar_n01}/{r.mcnemar_n10}"
                         f" & {_p(r.mcnemar_p_holm)}\\\\")
        lines += [r"\bottomrule", r"\end{tabularx}"]
        (out / f"tableS2_pairwise_{ds}.tex").write_text("\n".join(lines))


def _p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_all(results_root: str, manuscript_dir: str):
    results, out = Path(results_root), ensure_dir(Path(manuscript_dir) / "generated")
    for fn in (table_performance, table_partitions, table_pairwise):
        try:
            fn(results, out)
        except FileNotFoundError as e:
            print(f"[tables] skipped {fn.__name__}: {e}")
