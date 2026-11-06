#!/usr/bin/env bash
# Full reproduction of every number, table and figure in the manuscript.
# Expected wall time: ~5 GPU-days on one 16 GB GPU (9 models x 5 iterations x 2 datasets).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

python -m pytest -q tests                      # statistics unit tests (CPU, seconds)
python -m drxai.cli index                      # folds + Table S1 + class distribution
python -m drxai.cli preprocess                 # deterministic cache + QC flags
python -m drxai.cli train                      # 9 configurations x 5 iterations x 2 datasets
python -m drxai.cli evaluate                   # Tables 4-6, S2; bootstrap CIs; DeLong/McNemar
python -m drxai.cli complexity                 # parameters / GFLOPs / latency (Table 6)
python -m drxai.cli xai                        # Table 7: runtime, SHAP-LIME agreement, sanity check
python -m drxai.cli figures                    # Figures 5-10 -> ../manuscript/figures
python -m drxai.cli panels --image "${PANEL_IMAGE:?set PANEL_IMAGE to a Messidor-2 image_id}"
python -m drxai.cli tables                     # ../manuscript/generated/*.tex
python -m drxai.cli verify                     # manuscript numbers vs. results (must be 0 FAIL)
