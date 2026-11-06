"""End-to-end smoke test of index -> folds -> evaluate -> figures -> tables -> verify
on SYNTHETIC predictions (no images, no TensorFlow). Outputs go to a temporary
directory and must never be copied into the manuscript.

    python tests/smoke_synthetic.py
"""
import copy
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from drxai import splits as SP
from drxai.evaluate import evaluate_dataset
from drxai.pipeline import stage_figures
from drxai.tables import write_all
from drxai.utils import load_config


def main():
    cfg = load_config("configs/default.yaml")
    tmp = Path(tempfile.mkdtemp(prefix="drxai_smoke_"))
    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {k: str(tmp / k) for k in ("runs", "results", "figures", "cache")}
    rng = np.random.default_rng(0)
    n_pat = 400
    grades = rng.choice(5, n_pat, p=[0.6, 0.15, 0.17, 0.05, 0.03])
    df = pd.DataFrame({"image_id": [f"{p}_{e}" for p in range(n_pat) for e in ("l", "r")],
                       "group": np.repeat(np.arange(n_pat), 2),
                       "grade": np.repeat(grades, 2)})
    df["fold"] = SP.assign_folds(df)
    for ds in ("messidor2", "eyepacs"):
        res = Path(cfg["paths"]["results"]) / ds
        res.mkdir(parents=True)
        df.to_csv(res / "index_folds.csv", index=False)
        SP.partition_table(df, df.fold).to_csv(res / "partition_table.csv", index=False)
        for m, sig in [("vgg16", 1.0), ("vgg19", 1.1), ("resnet50", 1.2), ("inceptionv3", 1.3),
                       ("vgg16_se", 1.4), ("resnet50_cbam", 1.5), ("inceptionv3_se", 1.6),
                       ("dual_branch", 1.7)]:
            for k in range(5):
                out = Path(cfg["paths"]["runs"]) / ds / m / f"iter{k}"
                out.mkdir(parents=True)
                roles = SP.iteration_roles(df.fold, k)
                for role in ("val", "test"):
                    sub = df[roles == role]
                    lg = rng.normal(size=(len(sub), 5)); lg[np.arange(len(sub)), sub.grade] += sig
                    p = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
                    np.savez_compressed(out / f"probs_{role}.npz", image_id=sub.image_id.to_numpy(),
                                        y=sub.grade.to_numpy(), p=p)
                h = pd.DataFrame({"val_accuracy": np.linspace(.5, .8, 30), "val_loss": np.linspace(1.2, .6, 30)})
                h.to_csv(out / "history.csv", index=False)
        s = evaluate_dataset(ds, cfg, cfg["paths"]["runs"], cfg["paths"]["results"])
        print(ds, "pooled macro AUC", round(s["pooled_proposed"]["macro_auc"], 3),
              "CI", {k: round(v, 3) for k, v in s["bootstrap_proposed"]["macro_auc"].items()})
    stage_figures(cfg)
    write_all(cfg["paths"]["results"], str(tmp / "manuscript"))
    print("figures:", sorted(p.name for p in Path(cfg["paths"]["figures"]).glob("*.pdf")))
    print("tables:", sorted(p.name for p in (tmp / "manuscript" / "generated").glob("*.tex")))
    print("smoke outputs in", tmp)


if __name__ == "__main__":
    main()
