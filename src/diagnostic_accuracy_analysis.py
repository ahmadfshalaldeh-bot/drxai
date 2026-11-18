"""Diagnostic-accuracy analyses derived from the archived fold-assigned probabilities.

Usage:
    python src/diagnostic_accuracy_analysis.py --root . [--replicates 1000 --seed 2026]

Outputs (results/diagnostic/):
    endpoint_accuracy.csv        any DR (>=1), referable DR (>=2), vision-threatening DR (>=3):
                                 counts, prevalence, sensitivity, specificity, PPV, NPV, LR+, LR-,
                                 diagnostic odds ratio, Youden index, ROC-AUC, with 95%
                                 filename-group bootstrap intervals
    referable_missed_cases.csv   referable false negatives cross-tabulated by true and predicted grade
    prevalence_adjusted.csv      PPV/NPV of the referable decision at other referable prevalences (Bayes)
    crossfit_thresholds.csv      sensitivity-targeted referable thresholds chosen on four
                                 group-disjoint calibration meta-folds and applied to the fifth
    decision_curve.csv           net benefit of referral by raw and calibrated referable probability
No threshold, label or probability is chosen using the outcomes it is evaluated on.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PC = [f"prob_class_{k}" for k in range(5)]
CC = [f"calibrated_prob_class_{k}" for k in range(5)]
ENDPOINTS = {"any_DR_grade_ge1": 1, "referable_DR_grade_ge2": 2, "vision_threatening_DR_grade_ge3": 3}


def binary_metrics(y, d, s):
    tp = int(np.sum(y & d)); fp = int(np.sum(~y & d)); fn = int(np.sum(y & ~d)); tn = int(np.sum(~y & ~d))
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp else np.nan; npv = tn / (tn + fn) if tn + fn else np.nan
    lrp = sens / (1 - spec) if spec < 1 else np.inf
    lrn = (1 - sens) / spec if spec > 0 else np.inf
    dor = (tp * tn) / (fp * fn) if fp * fn else np.inf
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "prevalence": y.mean(), "referral_rate": d.mean(),
            "sensitivity": sens, "specificity": spec, "PPV": ppv, "NPV": npv, "LR_pos": lrp, "LR_neg": lrn,
            "DOR": dor, "youden": sens + spec - 1, "roc_auc": roc_auc_score(y, s)}


def group_index(groups):
    ug = sorted(set(groups), key=lambda x: int(x) if x.isdigit() else x)
    return [np.flatnonzero(groups == u) for u in ug]


def boot(fn, pos, B, seed):
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        idx = np.concatenate([pos[i] for i in rng.integers(0, len(pos), size=len(pos))])
        out.append(fn(idx))
    return pd.DataFrame(out)


def net_benefit(y, s, t):
    d = s >= t; n = len(y)
    return np.sum(y & d) / n - np.sum(~y & d) / n * t / (1 - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = a.root / "results/diagnostic"; out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(a.root / "results/raw/oof_predictions.csv")
    f = pd.read_csv(a.root / "results/raw/fold_assignments.csv")[["image_id", "subject_id"]]
    c = pd.read_csv(a.root / "results/ordinal_calibration/nested_calibrated_probabilities.csv")
    d = d.merge(f, on="image_id", validate="one_to_one").merge(c[["image_id", "meta_fold"] + CC], on="image_id", validate="one_to_one")
    lab = d.label.values; pred = d.oof_pred.values; P = d[PC].values.astype(float); Q = d[CC].values.astype(float)
    pos = group_index(d.subject_id.astype(str).values)

    # 1. endpoint accuracy
    rows = []
    for name, k in ENDPOINTS.items():
        y = lab >= k; dec = pred >= k; s = P[:, k:].sum(1)
        pt = binary_metrics(y, dec, s)
        R = boot(lambda i: binary_metrics(y[i], dec[i], s[i]), pos, a.replicates, a.seed)
        for m, v in pt.items():
            lo, hi = (R[m].quantile(0.025), R[m].quantile(0.975)) if m not in ("TP", "FP", "FN", "TN") else (np.nan, np.nan)
            rows.append({"endpoint": name, "metric": m, "estimate": v, "lower_95": lo, "upper_95": hi})
    pd.DataFrame(rows).to_csv(out / "endpoint_accuracy.csv", index=False)

    # 2. referable missed cases
    fnmask = (lab >= 2) & (pred < 2)
    ct = pd.crosstab(pd.Series(lab[fnmask], name="true_grade"), pd.Series(pred[fnmask], name="predicted_grade"))
    ct["total"] = ct.sum(axis=1)
    ct["n_true_grade"] = [int(np.sum(lab == g)) for g in ct.index]
    ct["missed_fraction"] = ct["total"] / ct["n_true_grade"]
    ct.to_csv(out / "referable_missed_cases.csv")

    # 3. prevalence-adjusted PPV/NPV for the referable argmax decision
    m = binary_metrics(lab >= 2, pred >= 2, P[:, 2:].sum(1)); se, sp = m["sensitivity"], m["specificity"]
    pv = []
    for prev in [m["prevalence"], 0.05, 0.10, 0.20, 0.30]:
        pv.append({"prevalence": prev, "PPV": se * prev / (se * prev + (1 - sp) * (1 - prev)),
                   "NPV": sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev),
                   "referrals_per_1000": 1000 * (se * prev + (1 - sp) * (1 - prev)),
                   "missed_referable_per_1000": 1000 * (1 - se) * prev})
    pd.DataFrame(pv).to_csv(out / "prevalence_adjusted.csv", index=False)

    # 4. cross-fitted sensitivity-targeted thresholds (raw referable score)
    y = lab >= 2; s = P[:, 2:].sum(1); mf = d.meta_fold.values
    ct_rows = []
    for target in (0.85, 0.90, 0.95):
        dec = np.zeros(len(y), bool); thr = {}
        for k in np.unique(mf):
            tr = mf != k
            st = np.sort(s[tr & y])  # positives in fitting folds
            t = st[int(np.floor((1 - target) * len(st)))]  # largest threshold with sens >= target on fitting folds
            thr[int(k)] = t; dec[mf == k] = s[mf == k] >= t
        pt = binary_metrics(y, dec, s)
        R = boot(lambda i: binary_metrics(y[i], dec[i], s[i]), pos, a.replicates, a.seed)
        row = {"target_sensitivity": target, "thresholds_by_meta_fold": ";".join(f"{k}:{v:.4f}" for k, v in thr.items())}
        for mname in ("sensitivity", "specificity", "PPV", "NPV", "referral_rate", "LR_pos", "LR_neg"):
            row[mname] = pt[mname]; row[mname + "_lower_95"] = R[mname].quantile(0.025); row[mname + "_upper_95"] = R[mname].quantile(0.975)
        row["FN"] = pt["FN"]; row["FN_grade3or4"] = int(np.sum((lab >= 3) & ~dec))
        ct_rows.append(row)
    pd.DataFrame(ct_rows).to_csv(out / "crossfit_thresholds.csv", index=False)

    # 5. decision curve (referable)
    qs = Q[:, 2:].sum(1); prev = y.mean(); dc = []
    for t in np.round(np.arange(0.05, 0.51, 0.01), 2):
        dc.append({"threshold": t, "net_benefit_raw": net_benefit(y, s, t), "net_benefit_calibrated": net_benefit(y, qs, t),
                   "net_benefit_refer_all": prev - (1 - prev) * t / (1 - t), "net_benefit_refer_none": 0.0})
    pd.DataFrame(dc).to_csv(out / "decision_curve.csv", index=False)
    print("written to", out)


if __name__ == "__main__":
    main()
