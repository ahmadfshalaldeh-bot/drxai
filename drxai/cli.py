"""Command-line entry point.

    python -m drxai.cli index        # build indices + patient-grouped folds (Table S1)
    python -m drxai.cli preprocess   # deterministic preprocessing cache + QC flags
    python -m drxai.cli train [--models ...] [--datasets ...] [--iterations ...]
    python -m drxai.cli evaluate     # metrics, bootstrap CIs, DeLong/McNemar/Holm
    python -m drxai.cli complexity   # parameters, FLOPs, latency
    python -m drxai.cli xai          # Grad-CAM / LIME / SHAP audit on annotated images
    python -m drxai.cli panels --image IMAGE_ID   # Figures 4 and 11 from one real image
    python -m drxai.cli figures      # all data figures -> ../manuscript/figures
    python -m drxai.cli tables       # LaTeX tables -> ../manuscript/generated
    python -m drxai.cli verify       # compare manuscript numbers with results/
"""
from __future__ import annotations

import argparse
import sys

from . import pipeline as P
from .utils import load_config, set_seed


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drxai")
    ap.add_argument("stage")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--datasets", nargs="*", default=list(P.DATASETS))
    ap.add_argument("--iterations", nargs="*", type=int)
    ap.add_argument("--image")
    ap.add_argument("--claims", default="manuscript_claims.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    if a.stage == "index":
        print(P.stage_index(cfg))
    elif a.stage == "preprocess":
        P.stage_preprocess(cfg)
    elif a.stage == "train":
        P.stage_train(cfg, a.models, a.datasets, a.iterations)
    elif a.stage == "evaluate":
        P.stage_evaluate(cfg, a.datasets)
    elif a.stage == "complexity":
        print(P.stage_complexity(cfg))
    elif a.stage == "xai":
        print(P.stage_xai(cfg))
    elif a.stage == "panels":
        P.stage_figure_panels(cfg, a.image)
    elif a.stage == "figures":
        P.stage_figures(cfg)
    elif a.stage == "tables":
        from .tables import write_all

        write_all(cfg["paths"]["results"], "../manuscript")
    elif a.stage == "verify":
        from .verify import run

        sys.exit(run(a.claims, cfg["paths"]["results"]))
    else:
        ap.error(f"unknown stage {a.stage}")


if __name__ == "__main__":
    main()
