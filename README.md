# drxai — Explainable attention-augmented ensemble for five-class DR grading

Companion code for:

> Shalaldeh, A.; AlMarshoud, M.; Alhawarat, M.O.; Alnsour, A.; Safa, M.; Logan, C.; Othman, M.;
> Al-mahamid, S.; Rahman, M.M. **An Explainable Ensemble Framework
> Combining Transfer Learning, Attention Mechanisms, and Post-Hoc Interpretability for Diabetic
> Retinopathy Grading.** *Mathematics* (MDPI), submitted 2026.

Every number, table and figure in the paper is produced by this repository from the two public
datasets. Nothing is hand-entered: `python -m drxai.cli verify` compares the values typed in the
manuscript with the regenerated results and fails on any mismatch.

## 1. What the code implements (paper section → module)

| Paper | Content | Module |
|---|---|---|
| §3.2 | EyePACS / Messidor-2 indexing with patient (exam) identifiers; 4 ungradable Messidor-2 images excluded | `drxai/data.py` |
| §3.3 | Stratified, **patient-grouped** 5-fold partition; rotation *test = k, val = k+1, train = rest*; leakage assertion; Table S1 | `drxai/splits.py` |
| §3.4 | Otsu FOV crop, QC flags, bilinear 224×224, CLAHE (8×8, clip 2.0); online augmentation on training batches only | `drxai/preprocess.py`, `drxai/train.py` |
| §3.5 | SE (r = 16) and CBAM (r = 16, 7×7) on the terminal feature map; partial fine-tuning; GAP–Dropout–Dense(256)–Softmax head; dual-branch comparator | `drxai/attention.py`, `drxai/models.py` |
| §3.6 | Validation-AUC weighted soft voting (Algorithm 1), simple averaging, majority voting; Proposition 1 bound | `drxai/ensemble.py` |
| §3.8 | Adam 1e-4, batch 16, ≤ 50 epochs, early stopping (10), ReduceLROnPlateau (×0.5, 5), inverse-frequency class weights | `drxai/train.py` |
| §3.9 | Macro OvR AUC, sensitivity, specificity, F1, QWK, ECE (10 bins), multiclass Brier, referable DR, PPV/NPV | `drxai/metrics.py` |
| §3.10 | Generalised paired DeLong for macro-AUC (Proposition 2), McNemar, Holm, patient-cluster bootstrap | `drxai/stats.py` |
| §3.11 | Grad-CAM, SLIC-LIME, ensemble SHAP (Proposition 3), P85 binarisation, SHAP–LIME agreement, runtime, cascading randomisation | `drxai/xai.py`, `drxai/pipeline.py` |
| Table 6 | Parameters, GFLOPs, latency | `drxai/complexity.py` |
| Figs 4–11 | All data figures from experiment outputs | `drxai/figures.py` |

## 2. Data (not redistributed — licences forbid it)

```
data/
  eyepacs/trainLabels.csv, train/*.jpeg          # Kaggle "Diabetic Retinopathy Detection" (2015), training set
  messidor2/IMAGES/*                             # Messidor-2, obtained from ADCIS under its licence
  messidor2/messidor_data.csv                    # Krause et al. (2018) adjudicated grades (Kaggle: google-brain/messidor2-dr-grades)
  messidor2/messidor-2_pairs.csv                 # image_id,exam_id  (build with data.messidor_pairs_from_spreadsheet)
```

## 3. Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
python -m pytest -q tests          # 12 statistical unit tests, CPU only
PANEL_IMAGE=<messidor2 image_id> bash run_all.sh
```

`tests/smoke_synthetic.py` exercises evaluation → figures → tables on synthetic predictions
(written to a temporary folder; never used in the paper).

## 4. Prediction-level reproducibility archive (Supplementary File S1 of the article)

* `results/*/index_folds.csv` — analysed manifest with the fold of every image (seed 42), so the partition is exact.
* `runs/*/*/iter*/probs_{val,test}.npz` — fold-assigned probabilities behind every metric.
* `results/*/summary.json`, `fold_metrics.csv`, `pairwise_tests.csv`, `qc.csv`, `partition_table.csv`, `xai_audit.csv`, figures and the `verify` report.

Not included: raw images, competition archives and trained checkpoints. They derive from data whose
licences (Kaggle Competition Rules; Messidor-2 licence) forbid redistribution. Every evaluation step
runs from the archive alone: `evaluate`, `figures`, `tables` and `verify` need no images.

## 5. Determinism

Seeds: Python/NumPy/TensorFlow = 42 (+ iteration index for training); `TF_DETERMINISTIC_OPS=1`.
Bootstrap and LIME use seed 42. GPU kernels can still introduce last-digit variation (< 0.002 AUC).

## 6. Licence

MIT (code). Datasets remain under their own licences.
