"""End-to-end stages. Run ``python -m drxai.cli --help`` for the command list."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import splits as SP
from .utils import ensure_dir, environment_report, load_json, save_json, set_seed

DATASETS = ("eyepacs", "messidor2")
TRAINED = ["vgg16_scratch", "vgg16", "vgg19", "resnet50", "inceptionv3",
           "vgg16_se", "resnet50_cbam", "inceptionv3_se", "dual_branch"]


def stage_index(cfg):
    out = {}
    for ds in DATASETS:
        c = cfg["datasets"][ds]
        idx = (D.index_eyepacs(c["labels_csv"], c["image_dir"], c["image_ext"]) if ds == "eyepacs"
               else D.index_messidor2(c["labels_csv"], c["pairs_csv"], c["image_dir"],
                                      c["drop_ungradable"]))
        res = ensure_dir(Path(cfg["paths"]["results"]) / ds)
        idx["fold"] = SP.assign_folds(idx, cfg["cv"]["n_folds"], cfg["seed"])
        idx.to_csv(res / "index_folds.csv", index=False)
        SP.partition_table(idx, idx["fold"], cfg["cv"]["n_folds"]).to_csv(
            res / "partition_table.csv", index=False)
        D.describe(idx).to_csv(res / "class_distribution.csv")
        out[ds] = {"images": len(idx), "patients": int(idx.group.nunique()),
                   "excluded_ungradable": idx.attrs.get("n_excluded_ungradable", 0)}
    save_json({"datasets": out, "environment": environment_report()},
              Path(cfg["paths"]["results"]) / "dataset_summary.json")
    return out


def stage_preprocess(cfg):
    from .preprocess import cache_dataset

    for ds in DATASETS:
        res = Path(cfg["paths"]["results"]) / ds
        idx = pd.read_csv(res / "index_folds.csv")
        qc = cache_dataset(idx, Path(cfg["paths"]["cache"]) / ds, cfg)
        qc.to_csv(res / "qc.csv", index=False)


def _table(cfg, ds, k):
    res = Path(cfg["paths"]["results"]) / ds
    idx = pd.read_csv(res / "index_folds.csv")
    qc = pd.read_csv(res / "qc.csv")[["image_id", "cached"]]
    t = idx.merge(qc, on="image_id")
    t["role"] = SP.iteration_roles(t["fold"], k, cfg["cv"]["n_folds"]).to_numpy()
    SP.check_no_leakage(t, t["role"])
    return t


def stage_train(cfg, models=None, datasets=DATASETS, iterations=None):
    from .train import train_one

    models = models or TRAINED
    iterations = range(cfg["cv"]["n_folds"]) if iterations is None else iterations
    for ds in datasets:
        for k in iterations:
            t = _table(cfg, ds, k)
            for m in models:
                done = Path(cfg["paths"]["runs"]) / ds / m / f"iter{k}" / "probs_test.npz"
                if done.exists():
                    continue
                train_one(m, k, t, cfg, str(Path(cfg["paths"]["runs"]) / ds))


def stage_evaluate(cfg, datasets=DATASETS):
    from .evaluate import evaluate_dataset

    return {ds: evaluate_dataset(ds, cfg, cfg["paths"]["runs"], cfg["paths"]["results"])
            for ds in datasets}


def stage_complexity(cfg):
    import tensorflow as tf

    from .complexity import count_params, flops, latency_ms
    from .models import build_ensemble_graph, build_model

    rows = []
    for m in TRAINED:
        model = build_model(m, cfg)
        rows.append({"model": m, **count_params(model), "gflops": flops(model) / 1e9,
                     **latency_ms(model)})
        tf.keras.backend.clear_session()
    for name, spec in cfg["ensembles"].items():
        members = [build_model(m, cfg) for m in spec["members"]]
        ens = build_ensemble_graph(members, [1 / len(members)] * len(members))
        rows.append({"model": name, **count_params(ens), "gflops": flops(ens) / 1e9,
                     **latency_ms(ens)})
        tf.keras.backend.clear_session()
    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg["paths"]["results"]) / "complexity.csv", index=False)
    return df


def stage_xai(cfg, n_panel_image: str | None = None):
    """Quantitative XAI audit on the annotated Messidor-2 subset (Section 3.11)."""
    import cv2
    import tensorflow as tf

    from . import xai as X
    from .metrics import macro_auc
    from .models import build_model
    from .stats import paired_t

    xc = cfg["xai"]
    ds = xc["dataset"]
    res = Path(cfg["paths"]["results"]) / ds
    ann = pd.read_csv(xc["annotation_index"])           # image_id, rater1_*, rater2_* ...
    idx = pd.read_csv(res / "index_folds.csv").merge(pd.read_csv(res / "qc.csv"), on="image_id")
    members_names = cfg["ensembles"]["proposed"]["members"]
    rows = []
    for k in range(cfg["cv"]["n_folds"]):
        t = _table(cfg, ds, k)
        sub = ann.merge(t[t.role == "test"], on="image_id")
        if sub.empty:
            continue
        members, val_aucs = [], []
        for m in members_names:
            mdl = build_model(m, cfg)
            mdl.load_weights(str(Path(cfg["paths"]["runs"]) / ds / m / f"iter{k}" / "best.weights.h5"))
            members.append(mdl)
            z = np.load(Path(cfg["paths"]["runs"]) / ds / m / f"iter{k}" / "probs_val.npz")
            val_aucs.append(macro_auc(z["y"], z["p"]))
        w = np.asarray(val_aucs) / np.sum(val_aucs)
        bg_ids = t[t.role == "train"].sample(xc["shap_background"], random_state=cfg["seed"])
        background = np.stack([_rgb(p) for p in bg_ids.cached]).astype(np.float32)

        def predict(batch, members=members, w=w):
            b = np.asarray(batch, np.float32)
            return sum(wi * m.predict(b, verbose=0) for wi, m in zip(w, members))

        for r in sub.itertuples():
            x = _rgb(r.cached).astype(np.float32)
            P = predict(x[None])[0]
            c = int(P.argmax())
            (shap_map, methods), t_shap, sd_shap = X.timed(
                lambda: X.shap_ensemble(members, w, x, background, c, xc["shap_samples"]),
                xc["runtime_repeats"])
            (lime_map, segs), t_lime, sd_lime = X.timed(
                lambda: X.lime_map(predict, x, c, xc, cfg["seed"]), xc["runtime_repeats"])
            cam = X.ensemble_gradcam(members, w, x, c)
            gt = _mask(Path(xc["annotations_dir"]) / f"{Path(r.image_id).stem}.png")
            a_s = X.binarize_top(shap_map, xc["binarize_percentile"])
            a_l = X.binarize_top(lime_map, xc["binarize_percentile"])
            a_g = X.binarize_top(cam, xc["binarize_percentile"])
            has_lesion = gt is not None and gt.any()
            rows.append({
                "image_id": r.image_id, "iteration": k + 1, "grade": int(r.grade), "pred": c,
                "iou_shap": X.iou(a_s, gt) if has_lesion else np.nan,
                "iou_lime": X.iou(a_l, gt) if has_lesion else np.nan,
                "iou_gradcam": X.iou(a_g, gt) if has_lesion else np.nan,
                "iou_shap_vs_lime": X.iou(a_s, a_l),
                "t_shap": t_shap, "t_shap_sd": sd_shap, "t_lime": t_lime, "t_lime_sd": sd_lime,
                "shap_method": "+".join(sorted(set(methods))),
                "plausible_shap": int(getattr(r, "rater1_shap") >= 4 and getattr(r, "rater2_shap") >= 4),
                "plausible_lime": int(getattr(r, "rater1_lime") >= 4 and getattr(r, "rater2_lime") >= 4),
                "rho_randomized": _sanity(members, w, x, background, c, shap_map, xc),
            })
        tf.keras.backend.clear_session()
    df = pd.DataFrame(rows)
    df.to_csv(res / "xai_audit.csv", index=False)
    les = df.dropna(subset=["iou_shap"])
    summary = {"n_images": len(df), "n_lesion_images": len(les),
               "iou_shap_mean": les.iou_shap.mean(), "iou_lime_mean": les.iou_lime.mean(),
               "paired_t": paired_t(les.iou_shap, les.iou_lime),
               "t_shap_mean": df.t_shap.mean(), "t_lime_mean": df.t_lime.mean(),
               "rho_randomized_median": df.rho_randomized.median(),
               "rater_kappa": _rater_kappa(ann)}
    save_json(summary, res / "xai_summary.json")
    return summary


def _sanity(members, w, x, background, c, shap_map, xc):
    from . import xai as X

    rnd = [X.randomize_top(m, xc["randomization_layers"]) for m in members]
    phi_r, _ = X.shap_ensemble(rnd, w, x, background, c, xc["shap_samples"])
    return X.rank_similarity(shap_map, phi_r)


def _rater_kappa(ann):
    from sklearn.metrics import cohen_kappa_score

    out = {}
    for meth in ("shap", "lime"):
        a, b = f"rater1_{meth}", f"rater2_{meth}"
        if a in ann and b in ann:
            out[meth] = float(cohen_kappa_score(ann[a], ann[b], weights="quadratic"))
    return out


def _rgb(path):
    import cv2

    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _mask(path):
    import cv2

    if not Path(path).exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (224, 224), interpolation=cv2.INTER_NEAREST) > 0


def stage_figures(cfg):
    from . import figures as F

    out = Path(cfg["paths"]["figures"])
    for ds, lab in (("eyepacs", "EyePACS"), ("messidor2", "Messidor-2")):
        res = Path(cfg["paths"]["results"]) / ds
        if not (res / "summary.json").exists():
            continue
        s = load_json(res / "summary.json")
        oof = {}
        for f in res.glob("oof_*.npz"):
            z = np.load(f, allow_pickle=True)
            oof[f.stem[4:]] = (z["image_id"], z["y"], z["p"])
        F.fig_confusion(np.array(s["confusion_proposed"]), lab, out, f"fig05_confusion_{ds}")
        F.fig_roc(oof, out, f"fig06_roc_{ds}")
        fm = pd.read_csv(res / "fold_metrics.csv")
        F.fig_folds_and_cis(fm, s["bootstrap_proposed"], out, f"fig08_folds_ci_{ds}")
        F.fig_calibration(oof, out, f"fig09_calibration_{ds}", cfg["statistics"]["ece_bins"])
        hist = {}
        for m in ["vgg16", "resnet50", "inceptionv3", "vgg16_se", "resnet50_cbam",
                  "inceptionv3_se", "dual_branch"]:
            hs = [pd.read_csv(p) for p in sorted((Path(cfg["paths"]["runs"]) / ds / m).glob(
                "iter*/history.csv"))]
            if hs:
                hist[m] = hs
        if hist:
            F.fig_curves(hist, out, f"fig07_curves_{ds}")
    xa = Path(cfg["paths"]["results"]) / cfg["xai"]["dataset"] / "xai_audit.csv"
    if xa.exists():
        F.fig_xai_quant(pd.read_csv(xa), out, "fig10_xai_quant")


def stage_figure_panels(cfg, image_id: str):
    """Figures 4 and 11 for one real held-out image chosen by the authors."""
    from . import figures as F
    from . import xai as X
    from .metrics import macro_auc
    from .models import build_model
    from .preprocess import preprocess_image, read_rgb

    ds = cfg["xai"]["dataset"]
    res = Path(cfg["paths"]["results"]) / ds
    idx = pd.read_csv(res / "index_folds.csv").merge(pd.read_csv(res / "qc.csv"), on="image_id")
    r = idx[idx.image_id == image_id].iloc[0]
    _, _, stages = preprocess_image(read_rgb(r.path))
    out = Path(cfg["paths"]["figures"])
    F.fig_preprocessing(stages, image_id, out)
    k = int(r.fold)                                         # the iteration that tests it
    t = _table(cfg, ds, k)
    x = _rgb(r.cached).astype(np.float32)
    members, val_aucs = [], []
    for m in cfg["ensembles"]["proposed"]["members"]:
        mdl = build_model(m, cfg)
        mdl.load_weights(str(Path(cfg["paths"]["runs"]) / ds / m / f"iter{k}" / "best.weights.h5"))
        members.append(mdl)
        z = np.load(Path(cfg["paths"]["runs"]) / ds / m / f"iter{k}" / "probs_val.npz")
        val_aucs.append(macro_auc(z["y"], z["p"]))
    w = np.asarray(val_aucs) / np.sum(val_aucs)
    bg = np.stack([_rgb(p) for p in t[t.role == "train"].sample(
        cfg["xai"]["shap_background"], random_state=cfg["seed"]).cached]).astype(np.float32)
    P = sum(wi * m.predict(x[None], verbose=0)[0] for wi, m in zip(w, members))
    c = int(P.argmax())
    gt = _mask(Path(cfg["xai"]["annotations_dir"]) / f"{Path(image_id).stem}.png")
    rows = []
    heads = list(zip(["VGG-16+SE", "ResNet-50+CBAM", "InceptionV3+SE"], members))
    for lab, m in heads:
        pf = (lambda b, m=m: m.predict(np.asarray(b, np.float32), verbose=0))
        lime, segs = X.lime_map(pf, x, c, cfg["xai"], cfg["seed"])
        phi, _ = X.shap_branch(m, x, bg, c, cfg["xai"]["shap_samples"])
        rows.append({"label": lab, "image": x.astype(np.uint8), "gradcam": X.gradcam(m, x, c)[0],
                     "lime": lime, "lime_segments": segs, "shap": phi, "mask": gt})
    pe = (lambda b: sum(wi * m.predict(np.asarray(b, np.float32), verbose=0)
                        for wi, m in zip(w, members)))
    lime, segs = X.lime_map(pe, x, c, cfg["xai"], cfg["seed"])
    phi, _ = X.shap_ensemble(members, w, x, bg, c, cfg["xai"]["shap_samples"])
    rows.append({"label": "Ensemble", "image": x.astype(np.uint8),
                 "gradcam": X.ensemble_gradcam(members, w, x, c), "lime": lime,
                 "lime_segments": segs, "shap": phi, "mask": gt})
    F.fig_xai_panel(rows, image_id, int(r.grade), c, out, "fig11_xai_panel")
