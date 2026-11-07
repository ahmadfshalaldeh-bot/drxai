"""Independent prediction-level audit of the EfficientNet-B4 archive.

Written from scratch (does not import the package's own scripts) so that every
reported number is checked by a second implementation.

    python independent_audit.py <path to github_ready_dr_efficientnet_b4> <out_dir>
"""
import warnings; warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import cohen_kappa_score, f1_score, log_loss, roc_auc_score

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
K = 5
B, SEED = 1000, 2026

pred = pd.read_csv(ROOT / "results/raw/oof_predictions.csv")
fold = pd.read_csv(ROOT / "results/raw/fold_assignments.csv")
pc = [f"prob_class_{i}" for i in range(K)]

# ---------------------------------------------------------------- integrity
chk = {}
chk["rows_pred"], chk["rows_fold"] = len(pred), len(fold)
chk["unique_ids_pred"] = int(pred.image_id.nunique())
chk["unique_ids_fold"] = int(fold.image_id.nunique())
m = pred.merge(fold, on="image_id", suffixes=("", "_f"), how="inner", validate="1:1")
chk["merged_rows"] = len(m)
chk["label_agreement"] = bool((m.label == m.label_f).all())
P = m[pc].to_numpy(float)
y = m.label.to_numpy(int)
g = m.subject_id.to_numpy()
fd = m.fold.to_numpy(int)
chk["prob_finite"] = bool(np.isfinite(P).all())
chk["prob_nonneg"] = bool((P >= 0).all())
chk["max_rowsum_dev"] = float(np.abs(P.sum(1) - 1).max())
chk["argmax_equals_stored"] = bool((P.argmax(1) == m.oof_pred.to_numpy()).all())
chk["class_support"] = np.bincount(y, minlength=K).tolist()
chk["pred_counts"] = np.bincount(P.argmax(1), minlength=K).tolist()
chk["n_groups"] = int(pd.Series(g).nunique())
chk["groups_in_multiple_folds"] = int((m.groupby("subject_id").fold.nunique() > 1).sum())
chk["fold_sizes"] = m.fold.value_counts().sort_index().tolist()
chk["seeds"] = sorted(m.seed.unique().tolist())
chk["roles"] = sorted(m.role.unique().tolist())
# filename prefix equals subject_id?
pref = m.image_id.str.split("_").str[0]
chk["group_equals_filename_prefix"] = bool((pref.astype(str) == m.subject_id.astype(str)).all())
chk["images_per_group"] = pd.Series(g).value_counts().value_counts().sort_index().to_dict()


# ---------------------------------------------------------------- metrics
def auc_bin(s, t):
    r = rankdata(s)
    mpos = t.sum()
    nneg = len(t) - mpos
    return (r[t == 1].sum() - mpos * (mpos + 1) / 2) / (mpos * nneg)


def metrics(y, P, ece_bins=15):
    yh = P.argmax(1)
    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, yh), 1)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    rec = tp / (tp + fn)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    spec = tn / (tn + fp)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    aucs = np.array([auc_bin(P[:, c], (y == c).astype(int)) for c in range(K)])
    err = np.abs(yh - y)
    conf = P.max(1)
    edges = np.linspace(0, 1, ece_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, ece_bins - 1)
    ece = 0.0
    for b in range(ece_bins):
        s = idx == b
        if s.any():
            ece += s.mean() * abs((yh[s] == y[s]).mean() - conf[s].mean())
    Y = np.eye(K)[y]
    yb = (y >= 2).astype(int)
    dec = (yh >= 2).astype(int)
    rtp = ((dec == 1) & (yb == 1)).sum(); rfn = ((dec == 0) & (yb == 1)).sum()
    rtn = ((dec == 0) & (yb == 0)).sum(); rfp = ((dec == 1) & (yb == 0)).sum()
    rsens, rspec = rtp / (rtp + rfn), rtn / (rtn + rfp)
    rppv = rtp / (rtp + rfp)
    return {
        "accuracy": (yh == y).mean(),
        "balanced_accuracy": rec.mean(),
        "macro_f1": f1.mean(),
        "weighted_f1": f1_score(y, yh, average="weighted"),
        "macro_precision": prec.mean(),
        "macro_specificity": spec.mean(),
        "macro_auc": aucs.mean(),
        "log_loss": log_loss(y, np.clip(P, 1e-15, 1), labels=list(range(K))),
        "qwk": cohen_kappa_score(y, yh, weights="quadratic"),
        "mae": err.mean(),
        "rmse": np.sqrt((err ** 2).mean()),
        "within_one": (err <= 1).mean(),
        "large_error": (err >= 2).mean(),
        "ece15": ece,
        "brier": ((P - Y) ** 2).sum(1).mean(),
        "ref_accuracy": (dec == yb).mean(),
        "ref_balanced_accuracy": (rsens + rspec) / 2,
        "ref_sensitivity": rsens,
        "ref_specificity": rspec,
        "ref_ppv": rppv,
        "ref_npv": rtn / (rtn + rfn),
        "ref_f1": 2 * rppv * rsens / (rppv + rsens),
        "ref_auc": roc_auc_score(yb, P[:, 2:].sum(1)),
        "_recall": rec, "_precision": prec, "_f1": f1, "_auc": aucs, "_cm": cm,
    }


full = metrics(y, P)
scal = {k: float(v) for k, v in full.items() if not k.startswith("_")}

per_fold = pd.DataFrame([{**{k: float(v) for k, v in metrics(y[fd == k], P[fd == k]).items()
                             if not k.startswith("_")}, "fold": k, "n": int((fd == k).sum())}
                         for k in range(5)])
per_class = pd.DataFrame({"grade": range(K), "support": np.bincount(y, minlength=K),
                          "recall": full["_recall"], "precision": full["_precision"],
                          "f1": full["_f1"], "auc": full["_auc"]})
pd.DataFrame(full["_cm"]).to_csv(OUT / "confusion_matrix.csv", index_label="reference")

# ---------------------------------------------------------------- bootstrap
rng = np.random.default_rng(SEED)
ug, inv = np.unique(g, return_inverse=True)
members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_precision",
        "macro_specificity", "macro_auc", "log_loss", "qwk", "mae", "within_one", "large_error",
        "ref_sensitivity", "ref_specificity", "ref_auc", "ece15"]
reps = {k: [] for k in keys}
for _ in range(B):
    draw = rng.integers(0, len(ug), len(ug))
    idx = np.concatenate([members[d] for d in draw])
    mm = metrics(y[idx], P[idx])
    for k in keys:
        reps[k].append(float(mm[k]))
ci = pd.DataFrame([{"metric": k, "estimate": scal[k], "lo": np.percentile(reps[k], 2.5),
                    "hi": np.percentile(reps[k], 97.5)} for k in keys])

# ---------------------------------------------------------------- write
per_fold.to_csv(OUT / "metrics_by_fold.csv", index=False)
per_class.to_csv(OUT / "per_class.csv", index=False)
ci.to_csv(OUT / "bootstrap_ci.csv", index=False)
fold_summary = {k: [float(per_fold[k].mean()), float(per_fold[k].std(ddof=1))] for k in keys}
json.dump({"integrity": chk, "pooled": scal, "fold_mean_sd": fold_summary},
          open(OUT / "audit.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(chk, indent=1, default=str))
print(pd.Series(scal).round(6).to_string())
print(ci.round(4).to_string())
print(per_class.round(4).to_string())
print(per_fold[["fold", "n", "accuracy", "balanced_accuracy", "macro_f1", "qwk", "macro_auc"]].round(4).to_string())
