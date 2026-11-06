"""Aggregate per-iteration predictions into every number reported in Section 4.

Inputs  : runs/<dataset>/<model>/iter<k>/probs_{val,test}.npz
          results/<dataset>/index_folds.csv  (image_id, grade, group, fold)
Outputs : results/<dataset>/
            fold_metrics.csv        model x iteration metrics      (Tables 4, 6)
            summary.json            mean +- SD, pooled OOF + bootstrap CIs (Table 5)
            pairwise_tests.csv      DeLong / McNemar, Holm-adjusted   (Table S2)
            fusion_rules.json       weighted vs mean vs majority + Prop. 1 bound
            oof_<model>.npz         pooled out-of-fold probabilities
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ensemble as E
from . import metrics as M
from . import stats as S
from .utils import ensure_dir, save_json

SINGLE = ["vgg16_scratch", "vgg16", "vgg19", "resnet50", "inceptionv3",
          "vgg16_se", "resnet50_cbam", "inceptionv3_se", "dual_branch"]


def _load(runs: Path, model: str, k: int, role: str):
    z = np.load(runs / model / f"iter{k}" / f"probs_{role}.npz", allow_pickle=True)
    return z["image_id"], z["y"], z["p"]


def ensemble_predictions(runs: Path, cfg: dict, name: str, k: int):
    """Fuse member probabilities for iteration k; weights from the validation fold."""
    spec = cfg["ensembles"][name]
    val_aucs, test_p, ids, y = [], [], None, None
    for mbr in spec["members"]:
        _, yv, pv = _load(runs, mbr, k, "val")
        val_aucs.append(M.macro_auc(yv, pv))
        ids_t, y_t, p_t = _load(runs, mbr, k, "test")
        if ids is None:
            ids, y = ids_t, y_t
        assert (ids == ids_t).all(), "members must score the test fold in the same order"
        test_p.append(p_t)
    probs = np.stack(test_p)
    w = E.auc_weights(val_aucs)
    fused = E.fuse(probs, spec["rule"], w)
    return ids, y, fused, {"val_aucs": val_aucs, "weights": w.tolist(), "member_probs": probs}


def evaluate_dataset(dataset: str, cfg: dict, runs_root: str, results_root: str) -> dict:
    runs = Path(runs_root) / dataset
    res = ensure_dir(Path(results_root) / dataset)
    folds = pd.read_csv(res / "index_folds.csv").set_index("image_id")
    n_folds = cfg["cv"]["n_folds"]
    names = [m for m in SINGLE if (runs / m).exists()] + list(cfg["ensembles"])
    rows, oof = [], {}
    fusion_log = []

    for name in names:
        ids_all, y_all, p_all = [], [], []
        for k in range(n_folds):
            if name in cfg["ensembles"]:
                ids, y, p, info = ensemble_predictions(runs, cfg, name, k)
                if name == "proposed":
                    cmp = E.compare_rules(info["member_probs"], y, info["val_aucs"])
                    cmp["iteration"] = k + 1
                    fusion_log.append(cmp)
            else:
                ids, y, p = _load(runs, name, k, "test")
            met = M.all_metrics(y, p, cfg["num_classes"], cfg["statistics"]["ece_bins"])
            ref = M.referable(y, p, threshold=cfg["referable_threshold"])
            rows.append({"model": name, "iteration": k + 1, "n_test": len(y),
                         **{kk: v for kk, v in met.items() if not isinstance(v, list)},
                         "ref_auc": ref["auc"], "ref_sens": ref["sensitivity"],
                         "ref_spec": ref["specificity"]})
            ids_all.append(ids); y_all.append(y); p_all.append(p)
        oof[name] = (np.concatenate(ids_all), np.concatenate(y_all), np.vstack(p_all))
        np.savez_compressed(res / f"oof_{name}.npz", image_id=oof[name][0], y=oof[name][1],
                            p=oof[name][2])

    fm = pd.DataFrame(rows)
    fm.to_csv(res / "fold_metrics.csv", index=False)

    # ---- pooled out-of-fold estimates with patient-cluster bootstrap CIs -----------
    st = cfg["statistics"]
    ids, y, P = oof["proposed"]
    groups = folds.loc[ids, "group"].to_numpy()
    yhat = P.argmax(1)
    boot = {
        "macro_auc": S.cluster_bootstrap(lambda i: M.macro_auc(y[i], P[i]), groups, y,
                                         st["bootstrap_replicates"], cfg["seed"]),
        "qwk": S.cluster_bootstrap(lambda i: M.quadratic_weighted_kappa(y[i], yhat[i]),
                                   groups, y, st["bootstrap_replicates"], cfg["seed"]),
        "accuracy": S.cluster_bootstrap(lambda i: float((yhat[i] == y[i]).mean()), groups, y,
                                        st["bootstrap_replicates"], cfg["seed"]),
    }
    ref = M.referable(y, P, threshold=cfg["referable_threshold"])
    ppv, npv = M.predictive_values(ref["sensitivity"], ref["specificity"],
                                   st["screening_prevalence"])
    summary = {
        "fold_mean_sd": {m: {c: S.fold_summary(fm.loc[fm.model == m, c])
                             for c in ["accuracy", "macro_auc", "macro_sensitivity",
                                       "macro_specificity", "macro_f1", "qwk", "ece", "brier"]}
                         for m in fm.model.unique()},
        "pooled_proposed": M.all_metrics(y, P, cfg["num_classes"], st["ece_bins"]),
        "bootstrap_proposed": boot,
        "referable_proposed": {**ref, "ppv": ppv, "npv": npv,
                               "prevalence": st["screening_prevalence"]},
        "confusion_proposed": M.confusion(y, yhat).tolist(),
        "n_images": int(len(y)), "n_patients": int(len(np.unique(groups))),
    }

    # ---- paired comparisons: proposed vs every other configuration ----------------
    tests = []
    for name, (ids_b, y_b, P_b) in oof.items():
        if name == "proposed":
            continue
        order = pd.Series(np.arange(len(ids_b)), index=ids_b).loc[ids].to_numpy()
        P_b = P_b[order]
        d = S.delong_macro(y, P, P_b, groups=groups)
        mc = S.mcnemar(yhat == y, P_b.argmax(1) == y)
        tests.append({"comparator": name, "auc_proposed": d["auc_a"], "auc_comparator": d["auc_b"],
                      "auc_diff": d["diff"], "delong_z": d["z"], "delong_p": d["p"],
                      "mcnemar_n01": mc["n01"], "mcnemar_n10": mc["n10"], "mcnemar_p": mc["p"]})
    tt = pd.DataFrame(tests)
    if len(tt):
        tt["delong_p_holm"] = S.holm(tt.delong_p)
        tt["mcnemar_p_holm"] = S.holm(tt.mcnemar_p)
    tt.to_csv(res / "pairwise_tests.csv", index=False)

    save_json(fusion_log, res / "fusion_rules.json")
    save_json(summary, res / "summary.json")
    return summary
