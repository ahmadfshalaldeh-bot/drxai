"""Paired filename-group bootstrap for the QC sensitivity analysis (manuscript Section 3.9).

Usage: python src/qc_sensitivity_bootstrap.py --root . [--replicates 1000 --seed 2026]
Reads results/qc/qc_flagged_images.csv (column image_id) and writes
results/qc/qc_sensitivity_summary.csv and results/qc/qc_paired_bootstrap_replicates.csv.
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, log_loss
warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")
PC = [f"prob_class_{k}" for k in range(5)]

def metrics(y, p, P):
    return {"accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", labels=range(5), zero_division=0),
            "qwk": cohen_kappa_score(y, p, weights="quadratic", labels=range(5)),
            "macro_auc_ovr": roc_auc_score(y, P, multi_class="ovr", average="macro", labels=range(5)),
            "log_loss": log_loss(y, P, labels=range(5))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    d = pd.read_csv(a.root / "results/raw/oof_predictions.csv")
    f = pd.read_csv(a.root / "results/raw/fold_assignments.csv")[["image_id", "subject_id"]]
    d = d.merge(f, on="image_id", validate="one_to_one")
    flagged = set(pd.read_csv(a.root / "results/qc/qc_flagged_images.csv")["image_id"])
    assert len(flagged) == 4 and flagged <= set(d.image_id)
    y = d.label.values; p = d.oof_pred.values; P = d[PC].values
    keep = ~d.image_id.isin(flagged).values
    full = metrics(y, p, P); qc = metrics(y[keep], p[keep], P[keep])
    g = d.subject_id.astype(str).values
    ug = sorted(set(g), key=lambda x: int(x) if x.isdigit() else x)
    pos = [np.flatnonzero(g == u) for u in ug]
    rng = np.random.default_rng(a.seed); reps = []
    for b in range(a.replicates):
        idx = np.concatenate([pos[i] for i in rng.integers(0, len(pos), size=len(pos))])
        k = keep[idx]
        mf = metrics(y[idx], p[idx], P[idx]); mq = metrics(y[idx][k], p[idx][k], P[idx][k])
        reps.append({m: mq[m] - mf[m] for m in mf})
    R = pd.DataFrame(reps)
    out = a.root / "results/qc"; R.to_csv(out / "qc_paired_bootstrap_replicates.csv", index_label="replicate")
    rows = [{"metric": m, "primary_n35126": full[m], "qc_n35122": qc[m], "difference": qc[m] - full[m],
             "lower_95": R[m].quantile(0.025), "upper_95": R[m].quantile(0.975),
             "seed": a.seed, "replicates": a.replicates} for m in full]
    S = pd.DataFrame(rows); S.to_csv(out / "qc_sensitivity_summary.csv", index=False)
    print(S.to_string(index=False))

if __name__ == "__main__":
    main()
