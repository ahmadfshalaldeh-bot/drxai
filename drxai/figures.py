"""All data figures of the manuscript, generated from experiment outputs.

Each function reads files written by ``evaluate`` / ``xai_audit`` and writes a
vector PDF (plus a 600-dpi PNG) with the exact file name used in main.tex.
No figure contains drawn or synthetic retinal content: image panels are the
real held-out photographs, and every panel is labelled with its image_id.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics as M

# Fixed categorical order (validated palette, slots 1-8); colour follows the model.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {
    "vgg16": SERIES[0], "vgg19": SERIES[1], "resnet50": SERIES[2], "inceptionv3": SERIES[3],
    "vgg16_se": SERIES[4], "resnet50_cbam": SERIES[5], "inceptionv3_se": SERIES[6],
    "dual_branch": SERIES[7], "proposed": "#0b0b0b",
}
LABELS = {
    "vgg16_scratch": "VGG-16 (scratch)", "vgg16": "VGG-16", "vgg19": "VGG-19",
    "resnet50": "ResNet-50", "inceptionv3": "InceptionV3", "vgg16_se": "VGG-16+SE",
    "resnet50_cbam": "ResNet-50+CBAM", "inceptionv3_se": "InceptionV3+SE",
    "dual_branch": "Dual-branch CNN", "proposed": "Proposed ensemble",
    "simple_average": "Simple average", "majority_vote": "Majority vote",
    "no_attention": "Ensemble (no attention)",
}
GRADES = ["G0", "G1", "G2", "G3", "G4"]
TEXT = "#0b0b0b"
MUTED = "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_preprocessing(stages: dict, image_id: str, out: Path):
    keys = ["original", "fov_mask", "cropped", "resized", "clahe"]
    titles = ["(a) Original", "(b) Otsu FOV mask", "(c) Square FOV crop",
              "(d) Bilinear 224×224", "(e) CLAHE (L channel)"]
    fig, ax = plt.subplots(1, 5, figsize=(7.2, 1.8))
    for a, k, t in zip(ax, keys, titles):
        a.imshow(stages[k], cmap="gray" if stages[k].ndim == 2 else None)
        a.set_title(t); a.axis("off")
    fig.suptitle(f"Source image: {image_id}", y=0.02, fontsize=7, color=MUTED)
    _save(fig, out, "fig04_preprocessing")


def fig_confusion(cm: np.ndarray, dataset_label: str, out: Path, name: str):
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.grid(False)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{100 * norm[i, j]:.1f}%\n({cm[i, j]:,})", ha="center", va="center",
                    fontsize=6.5, color="white" if norm[i, j] > 0.55 else TEXT)
    ax.set_xticks(range(5), GRADES); ax.set_yticks(range(5), GRADES)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("Reference grade")
    ax.set_title(dataset_label)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised recall")
    _save(fig, out, name)


def fig_roc(oof: dict, out: Path, name: str, models=("vgg16", "resnet50", "inceptionv3",
                                                     "dual_branch", "proposed")):
    """One panel per ICDR grade (one-vs-rest) + referable DR: individual ROC curves."""
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey=True)
    for c, ax in enumerate(axes.ravel()):
        for m in models:
            if m not in oof:
                continue
            _, y, P = oof[m]
            if c < 5:
                yb, s, ttl = (y == c).astype(int), P[:, c], f"Grade {c} vs rest"
            else:
                yb, s, ttl = (y >= 2).astype(int), P[:, 2:].sum(1), "Referable DR (≥ G2)"
            fpr, tpr, _ = roc_curve(yb, s)
            auc = M.auc_binary(s[yb == 1], s[yb == 0])
            ax.plot(fpr, tpr, color=MODEL_COLORS.get(m, MUTED),
                    lw=2.0 if m == "proposed" else 1.2, label=f"{LABELS[m]} ({auc:.3f})")
        ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=0.8)
        ax.set_title(ttl); ax.legend(loc="lower right", fontsize=5.5, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("1 − specificity")
    for ax in axes[:, 0]:
        ax.set_ylabel("Sensitivity")
    _save(fig, out, name)


def fig_curves(histories: dict, out: Path, name: str):
    """Validation accuracy and loss per model (mean over iterations, own length).
    The ensemble has no curve: its members are trained independently."""
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for m, hs in histories.items():
        L = max(len(h) for h in hs)
        for a, col in zip(ax, ["val_accuracy", "val_loss"]):
            arr = np.full((len(hs), L), np.nan)
            for i, h in enumerate(hs):
                arr[i, :len(h)] = h[col].to_numpy()
            mu = np.nanmean(arr, 0)
            a.plot(np.arange(1, L + 1), mu, color=MODEL_COLORS.get(m, MUTED), label=LABELS[m])
    ax[0].set_ylabel("Validation accuracy"); ax[1].set_ylabel("Validation cross-entropy")
    for a, t in zip(ax, ["(a)", "(b)"]):
        a.set_xlabel("Epoch"); a.set_title(t, loc="left")
    ax[1].legend(frameon=False, fontsize=6.5)
    _save(fig, out, name)


def fig_folds_and_cis(fold_metrics: pd.DataFrame, boot: dict, out: Path, name: str,
                      models=("vgg16", "resnet50", "inceptionv3", "dual_branch", "proposed")):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for m in models:
        d = fold_metrics[fold_metrics.model == m]
        if len(d):
            ax[0].plot(d.iteration, d.macro_auc, marker="o", ms=4,
                       color=MODEL_COLORS.get(m, MUTED), label=LABELS[m])
    ax[0].set_xlabel("Test fold (rotation iteration)"); ax[0].set_ylabel("Macro AUC")
    ax[0].set_xticks(range(1, 6)); ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title("(a)", loc="left")
    keys = list(boot)
    est = [boot[k]["estimate"] for k in keys]
    lo = [boot[k]["estimate"] - boot[k]["lo"] for k in keys]
    hi = [boot[k]["hi"] - boot[k]["estimate"] for k in keys]
    ax[1].errorbar(range(len(keys)), est, yerr=[lo, hi], fmt="o", color=TEXT, capsize=4)
    ax[1].set_xticks(range(len(keys)), [k.replace("_", " ") for k in keys])
    ax[1].set_ylim(min(boot[k]["lo"] for k in keys) - 0.03, 1.0)
    ax[1].set_ylabel("Estimate [95% cluster-bootstrap CI]"); ax[1].set_title("(b)", loc="left")
    _save(fig, out, name)


def fig_calibration(oof: dict, out: Path, name: str, bins: int = 10,
                    models=("vgg16_se", "resnet50_cbam", "inceptionv3_se", "proposed")):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for m in models:
        if m not in oof:
            continue
        _, y, P = oof[m]
        cf, acc, _ = M.reliability_curve(y, P, bins)
        ece = M.expected_calibration_error(y, P, bins)
        kw = M.quadratic_weighted_kappa(y, P.argmax(1))
        ax[0].plot(cf, acc, marker="o", ms=3.5, color=MODEL_COLORS[m],
                   label=f"{LABELS[m]} (ECE {ece:.3f})")
        ax[1].bar(LABELS[m].replace("+", "+\n"), kw, color=MODEL_COLORS[m], width=0.6)
    ax[0].plot([0, 1], [0, 1], ls=":", color=MUTED)
    ax[0].set_xlabel("Mean top-label confidence"); ax[0].set_ylabel("Observed accuracy")
    ax[0].legend(frameon=False, fontsize=6.5); ax[0].set_title("(a)", loc="left")
    ax[1].set_ylabel("Quadratic weighted κ"); ax[1].set_ylim(0, 1); ax[1].set_title("(b)", loc="left")
    _save(fig, out, name)


def fig_xai_quant(xai: pd.DataFrame, out: Path, name: str):
    """Per-grade IoU (lesion-bearing grades only) and clinician plausibility."""
    g = xai.groupby("grade")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    grades = sorted(xai.grade.unique())
    w = 0.38
    for off, meth, col in ((-w / 2, "shap", SERIES[0]), (w / 2, "lime", SERIES[1])):
        les = [gr for gr in grades if gr > 0]
        mu = [100 * g.get_group(gr)[f"iou_{meth}"].mean() for gr in les]
        ax[0].bar(np.array(les) + off, mu, w, color=col, label=meth.upper())
        ag = [100 * g.get_group(gr)[f"plausible_{meth}"].mean() for gr in grades]
        ax[1].bar(np.array(grades) + off, ag, w, color=col, label=meth.upper())
    ax[0].set_xticks([1, 2, 3, 4], GRADES[1:]); ax[1].set_xticks(grades, [GRADES[i] for i in grades])
    ax[0].set_ylabel("Lesion IoU at P85 (%)"); ax[1].set_ylabel("Rated plausible by both readers (%)")
    for a, t in zip(ax, ["(a)", "(b)"]):
        a.set_title(t, loc="left"); a.legend(frameon=False)
    _save(fig, out, name)


def fig_xai_panel(rows: list[dict], image_id: str, grade_ref: int, grade_pred: int,
                  out: Path, name: str):
    """rows: [{'label', 'image', 'gradcam', 'lime', 'lime_segments', 'shap', 'mask'?, 'attn'?}]
    One real held-out image, one row per classification head (three branches + ensemble)."""
    from skimage.segmentation import find_boundaries

    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(7.2, 1.85 * n))
    for r, row in enumerate(rows):
        img = row["image"]
        ax[r, 0].imshow(img)
        if row.get("mask") is not None:
            ax[r, 0].contour(row["mask"], levels=[0.5], colors="cyan", linewidths=0.6)
        ax[r, 1].imshow(img); ax[r, 1].imshow(row["gradcam"], cmap="magma", alpha=0.5)
        if row.get("attn") is not None:
            ax[r, 1].contour(row["attn"], levels=[np.quantile(row["attn"], 0.9)],
                             colors="white", linewidths=0.6)
        lim = row["lime"]
        ax[r, 2].imshow(img)
        ax[r, 2].imshow(np.ma.masked_where(lim <= 0, lim), cmap="Greens", alpha=0.6)
        ax[r, 2].contour(find_boundaries(row["lime_segments"]), levels=[0.5], colors="yellow",
                         linewidths=0.3)
        s = row["shap"]; v = np.abs(s).max() + 1e-12
        ax[r, 3].imshow(img.mean(-1), cmap="gray", alpha=0.6)
        ax[r, 3].imshow(s, cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.75)
        ax[r, 0].set_ylabel(row["label"], fontsize=7)
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([]); ax[r, c].grid(False)
    for c, t in enumerate(["Input (+ expert lesions)", "Grad-CAM", "LIME (SLIC)", "SHAP"]):
        ax[0, c].set_title(t)
    fig.suptitle(f"{image_id} — reference G{grade_ref}, ensemble prediction G{grade_pred}",
                 y=0.005, fontsize=7, color=MUTED)
    _save(fig, out, name)
