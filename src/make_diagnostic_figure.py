"""Figure: referable-DR decision curve (A) and operating points on the referable ROC curve (B).
Usage: python src/make_diagnostic_figure.py --root . --out submission_figures/Figure_10_Referable_Decision_Analysis.png
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path(".")); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); R = a.root
dc = pd.read_csv(R / "results/diagnostic/decision_curve.csv")
cf = pd.read_csv(R / "results/diagnostic/crossfit_thresholds.csv")
ea = pd.read_csv(R / "results/diagnostic/endpoint_accuracy.csv")
d = pd.read_csv(R / "results/raw/oof_predictions.csv")
y = d.label.values >= 2; s = d[[f"prob_class_{k}" for k in (2, 3, 4)]].sum(axis=1).values
fpr, tpr, _ = roc_curve(y, s)
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.9), constrained_layout=True)
A = ax[0]
A.plot(dc.threshold, dc.net_benefit_calibrated, color="#1f4e79", lw=2, label="Model (calibrated probability)")
A.plot(dc.threshold, dc.net_benefit_raw, color="#5b9bd5", lw=1.6, ls="--", label="Model (raw probability)")
A.plot(dc.threshold, dc.net_benefit_refer_all, color="#7f7f7f", lw=1.2, label="Refer all")
A.plot(dc.threshold, dc.net_benefit_refer_none, color="black", lw=1.0, label="Refer none")
A.set_ylim(-0.05, 0.22); A.set_xlim(0.05, 0.5)
A.set_xlabel("Threshold probability for referral"); A.set_ylabel("Net benefit"); A.set_title("A. Decision curve (referable DR)", fontsize=10, loc="left")
A.legend(fontsize=7.5, frameon=False, loc="upper right")
B = ax[1]
B.plot(fpr, tpr, color="#1f4e79", lw=1.8, label="Referable ROC (AUC 0.9305)")
g = lambda m: float(ea[(ea.endpoint == "referable_DR_grade_ge2") & (ea.metric == m)].estimate.iloc[0])
B.scatter([1 - g("specificity")], [g("sensitivity")], marker="s", color="#c00000", zorder=3, label="Argmax decision")
mk = {0.85: "o", 0.90: "^", 0.95: "D"}
for _, r in cf.iterrows():
    B.scatter([1 - r.specificity], [r.sensitivity], marker=mk[round(r.target_sensitivity, 2)], color="#ed7d31", zorder=3,
              label=f"Cross-fitted, target sens. {r.target_sensitivity:.2f}")
B.plot([0, 1], [0, 1], color="#bfbfbf", lw=0.8, ls=":")
B.set_xlim(0, 0.6); B.set_ylim(0.5, 1.0)
B.set_xlabel("1 − specificity"); B.set_ylabel("Sensitivity"); B.set_title("B. Referable operating points", fontsize=10, loc="left")
B.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.savefig(a.out, dpi=600); print("saved", a.out)
