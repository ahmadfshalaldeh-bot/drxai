# Ordinal Evaluation and Group-Aware Calibration of EfficientNet-B4 for Five-Class Diabetic Retinopathy Grading

**Supplementary File S1: prediction-level reproducibility archive (revision of 7 November 2026)**

## Read this first: scope and limitations

- **What this archive reproduces:** every metric, interval, table value and data-derived figure in the manuscript, recomputed from the archived five-class probabilities. It is a *prediction-level* archive.
- **What it does not reproduce:** EfficientNet-B4 training. No Kaggle JPEGs, preprocessed images, competition archives or model checkpoints are included (checkpoint SHA-256 values only, in `results/raw/checkpoint_metadata.csv`). Complete epoch-wise histories, a software lockfile and runtime logs were not available, so the archive makes no bit-for-bit retraining claim.
- **Fold-assigned, not untouched test predictions.** The fold later used for prediction also drove ReduceLROnPlateau, early stopping and checkpoint selection (macro-F1). The pooled estimates may therefore be optimistic, and they are not strict out-of-fold or external-validation estimates. The file name `oof_predictions.csv` is historical.
- **Grouping key.** The grouping key is derived from filenames (the numeric prefix, so `13_left` and `13_right` form group `13`). Its manifest column is still named `subject_id`, but the key is not a verified patient identifier. The bootstrap resamples these filename groups (1000 replicates, seed 2026); it is not a patient-cluster bootstrap.
- **Primary analysis.** The primary analysis uses all 35,126 images and the argmax decision. The squared-distance decision rule and the five-meta-fold ordinal isotonic calibration are secondary sensitivity analyses with stated trade-offs.
- **Quality-control sensitivity analysis.** Four low-variance images, listed in `results/qc/qc_flagged_images.csv`, are excluded only in the sensitivity analysis (Section 3.9). Their identifiers were recovered from the archived sensitivity metrics and reproduce the Section 3.9 sensitivity estimates exactly; see `results/qc/README.md`. The authors should still confirm the grayscale rule on the original JPEGs.
- **Not established by this archive:** external validity, clinical calibration, fairness or subgroup performance, lesion localization, explainability, clinician agreement, or readiness for screening or patient care.

## Environment

The archive was verified on 7 November 2026 with Python 3.13.18 and the following packages: numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1 and matplotlib 3.11.2 (Linux, CPU only, no GPU needed).

- `requirements.txt` gives the supported ranges.
- `requirements-tested.txt` pins the versions verified here.
- `configs/archived_environment.yml` and `configs/archived_requirements.txt` describe the original training environment as supplied. They were not re-verified.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-tested.txt       # or: -r requirements.txt
```

## Exact reproduction commands (run from the archive root)

| Step | Command | Writes | Approx. CPU time |
|---|---|---|---|
| 1. Integrity audit and primary/referable/ordinal metrics with 1000-replicate filename-group intervals | `python src/reproduce_metrics.py --bootstrap-replicates 1000 --seed 2026` | `results/recomputed/` | ~1 min |
| 2. Per-grade and referable intervals (Tables 5 and 8) | `python src/recompute_table_uncertainty.py --replicates 1000 --seed 2026` | `results/metrics/` | ~1 min |
| 3. Extended intervals (Tables 4 and 6) | `python src/complete_extended_bootstrap_intervals.py` | `results/metrics/confidence_intervals_extended.csv` | ~1 min |
| 4. Squared-distance decision rule (Table 7, Figure 7) | `python src/ordinal_risk_decision_analysis.py --root . --replicates 1000 --seed 2026` | `results/ordinal_risk_decision/` | ~1 min |
| 5. Five-meta-fold group-disjoint ordinal isotonic calibration (Table 9) | `python src/nested_ordinal_calibration.py --root . --seed 2026` | `results/ordinal_calibration/` | <1 min |
| 6. Paired calibrated-minus-raw intervals (Table 10) | `python src/paired_group_bootstrap_calibration.py --root . --replicates 1000 --seed 2026` | `results/ordinal_calibration/` | ~2 min |
| 7. Meta-fold seed sensitivity, seeds 2026–2030 | `python src/seed_calibration_sensitivity.py --seeds 2026 2027 2028 2029 2030` | `results/ordinal_calibration/seed_stability_recomputed.csv` | ~1 min |
| 8. Independent second implementation (does not import `src/`) | `python audit/independent_audit.py . audit/outputs` | `audit/outputs/` | ~1 min |
| 8b. QC sensitivity analysis (Section 3.9) | `python src/qc_sensitivity_bootstrap.py --root . --replicates 1000 --seed 2026` | `results/qc/` | ~2 min |
| 9. Regenerate Figure 6 (confusion matrices) | `python src/make_confusion_matrix_figure.py results/raw/oof_predictions.csv submission_figures/Figure_6_Confusion_Matrices.png` | PNG, 600 dpi | seconds |
| 10. Verify file hashes | `sha256sum -c checksums.sha256` | — | seconds |

`make all` runs steps 1 to 8b. Steps 2 to 7 overwrite curated reference outputs, so run them on a copy if you want to diff the results against the shipped files.

### Result of re-running every step on 7 November 2026

| Step | Outcome |
|---|---|
| 1 | Byte-identical to the shipped `results/recomputed/`. |
| 5 | Reproduced `nested_calibrated_probabilities.csv` exactly (maximum difference 0). |
| 3, 7 | Agree with the shipped files to ≤1.1 × 10⁻⁸. |
| 4 | Metric values are identical. Only descriptive keys in `method.json` differ. |
| 6 | Identical for every metric except calibrated macro ROC-AUC, explained below. |
| 8 | Agrees with all primary, ordinal, calibration and referable values in the manuscript. |
| 8b | Reproduces every Section 3.9 QC-sensitivity value and interval. |

**Calibrated macro ROC-AUC and tie resolution.** Isotonic calibration produces many tied probabilities. As a result, the calibrated macro ROC-AUC depends, at the fourth decimal, on how floating-point near-ties are resolved:

- about 0.8722 when near-ties are collapsed before ranking (0.87220–0.87221 for rounding at 10⁻⁸ to 10⁻¹²; the manuscript reports 0.8722);
- 0.87237 on the raw floating-point values in this environment.

Under the second convention the paired difference is −0.00131 [−0.00385, 0.00123], compared with the shipped −0.00145 [−0.00404, 0.00114]. The interval includes zero under both conventions, and the manuscript states this sensitivity in the Table 10 note.

## Contents

The complete file list, with sizes and SHA-256 values, is in `MANIFEST.csv`. The main locations are:

```text
results/raw/                 fold-assigned probabilities, fold/group manifest, checkpoint hashes
results/metrics/             reference metrics and bootstrap intervals
results/ordinal_risk_decision/  squared-distance decision sensitivity
results/ordinal_calibration/ five-meta-fold calibration outputs and paired intervals
results/recomputed/          output of step 1
results/qc/                  QC-flagged image IDs and Section 3.9 sensitivity outputs
results/figures/             data-derived figures (generic file names) and provenance
submission_figures/          Figures 1–9 at 600 dpi, numbered as in the revised manuscript
audit/                       independent audit script and outputs
src/                         analysis scripts
archived_pipeline/           supplied preprocessing/training/evaluation scripts (not re-run)
configs/                     supplied training configuration and environment (not re-verified)
docs/                        audit notes and release notes
```

## Changes in this revision

- **Figures renumbered** to the order in which they are first cited in the revised manuscript. The mapping from old to new numbers is: 1→1, 2→2, 3→3, 8→4, 4→5, 5→6, 9→7, 6→8, 7→9.
- **Figure 6 (confusion matrices) regenerated.** In the earlier panel, the colour bar covered three cells. The counts are unchanged.
- **QC identifiers recovered and the Section 3.9 sensitivity analysis made reproducible** (`results/qc/`, `src/qc_sensitivity_bootstrap.py`).
- **Independent audit added** (`audit/`), together with `requirements-tested.txt`, `MANIFEST.csv` and regenerated `checksums.sha256`.
- **Stale manuscript text removed.** `docs/mathematics_dr_endnote_integrated.md`, which carried the old figure numbering, was removed; the submitted manuscript is the reference text.

No label, prediction, probability, fold assignment, decision rule or reported estimate was changed.

---

## Original package description (retained)

This repository contains the prediction-level reproducibility package for the manuscript:

> **Ordinal Evaluation and Group-Aware Calibration of EfficientNet-B4 for Five-Class Diabetic Retinopathy Grading**

This repository documents one retrospective Kaggle-only analysis using EfficientNet-B4 and five severity grades. Its scope is limited to the audited images, labels, fold assignments, predictions, and post-processing analyses described here.

## What is reproducible

The included script recomputes the principal published results from the raw fold-assigned prediction file. It verifies the following before calculating metrics:

- 35,126 unique image identifiers and labels;
- one fold assignment per image;
- finite five-class probabilities that sum to one within tolerance;
- agreement between stored and recomputed argmax predictions;
- 17,563 filename-derived grouping keys;
- absence of a grouping key in more than one fold;
- primary multiclass metrics;
- grade-≥2 referable-DR metrics;
- descriptive 15-bin expected calibration error;
- ordinal mean absolute error, root-mean-square grade error, within-one-grade accuracy, and large ordinal error rate (absolute grade error ≥ 2);
- 1000-replicate filename-group bootstrap intervals with seed 2026.

The package also includes `src/nested_ordinal_calibration.py` (a historical script filename) for the secondary five-meta-fold group-disjoint post-processing analysis. It fits isotonic maps to the cumulative events P(Y ≥ k) on four partitions of filename-derived grouping keys and evaluates them on a fifth. The legacy manifest column is named `subject_id`, but its values are derived from image filenames and are not verified identities. A decreasing cumulative projection guarantees valid non-negative class probabilities. This method does not retrain EfficientNet-B4 and is not external validation or end-to-end nested cross-validation.

The script `src/paired_group_bootstrap_calibration.py` calculates paired calibrated-minus-raw confidence intervals by resampling filename-derived groups with replacement. The calibration layer was also repeated under five meta-fold seeds (2026–2030); reductions in ECE and increases in QWK, alongside a decrease in balanced accuracy, were consistent. These intervals are uncertainty summaries, not p-values.

Recompute the meta-fold seed sensitivity table with:

```bash
python src/seed_calibration_sensitivity.py --seeds 2026 2027 2028 2029 2030
```

This command reruns only the second-stage calibration assignments on the same fixed predictions. It does not retrain EfficientNet-B4 or provide independent model replication.

The script `src/ordinal_risk_decision_analysis.py` evaluates a secondary posterior squared-distance decision rule, `argmin_k sum_j (j-k)^2 p_j`. The rule is mathematically equivalent to choosing the integer grade nearest to the posterior mean grade, with lower-grade tie breaking. It changed 12,069 of 35,126 decisions (34.36%), reduced the large ordinal error rate from 0.071457 to 0.024796 and increased within-one-grade accuracy from 0.928543 to 0.975204, but reduced accuracy, macro-F1, QWK, and mean absolute grade error. The primary argmax rule is therefore retained. The analysis is prediction-level, uses 1000 filename-group bootstrap replicates, and is not a new training run, QWK optimizer, or clinical decision study.

The reference prediction-level results are:

| Metric | Estimate |
|---|---:|
| Accuracy | 0.752975 |
| Balanced accuracy | 0.583907 |
| Macro-F1 | 0.569848 |
| Quadratic weighted kappa | 0.736006 |
| Macro one-vs-rest ROC-AUC | 0.873664 |
| Log loss | 0.739464 |
| Mean absolute ordinal error | 0.324660 grades |
| Within-one-grade accuracy | 0.928543 |
| Grade-≥2 ROC-AUC | 0.930503 |
| Grade-≥2 sensitivity | 0.775644 |
| Grade-≥2 specificity | 0.935971 |
| Descriptive ECE, 15 bins | 0.141454 |

The five-meta-fold group-disjoint ordinal calibration analysis reduced ECE to 0.010331, reduced multiclass Brier score from 0.408894 to 0.259393, reduced mean absolute grade error from 0.324660 to 0.239111 grades, and increased QWK from 0.736006 to 0.763745. It also reduced balanced accuracy from 0.583907 to 0.521113, so it must be reported as a secondary probability/ordinal sensitivity analysis rather than as an overall accuracy improvement.

The posterior squared-distance sensitivity output is stored in `results/ordinal_risk_decision/`, including `summary_metrics.csv`, `paired_group_bootstrap_differences.csv`, `decision_predictions.csv`, and `method.json`.

## Reproduction command

Create a clean Python environment and install the analysis dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the prediction-level audit and recomputation:

```bash
python src/reproduce_metrics.py --bootstrap-replicates 1000 --seed 2026
```

Outputs are written to `results/recomputed/`, including:

- `aggregate_metrics_recomputed.csv`;
- `metrics_by_fold_recomputed.csv`;
- `referable_metrics_recomputed.csv`;
- `calibration_metrics_recomputed.csv`;
- `confidence_intervals_recomputed.csv`;
- `confusion_matrix_recomputed.csv`;
- `audit_summary.json`.

Recompute the per-grade and derived referable-DR intervals used in Tables 5 and 8 with:

```bash
python src/recompute_table_uncertainty.py --replicates 1000 --seed 2026
```

This writes `results/metrics/class_metric_bootstrap_intervals.csv` and `referable_metric_bootstrap_intervals.csv`.

The calculation uses the raw fold-assigned probabilities rather than rounded manuscript values. The file `oof_predictions.csv` retains a legacy filename; its predictions are not strict untouched outer-test predictions because the prediction fold also informed checkpoint selection.

## Repository structure

```text
src/reproduce_metrics.py             Prediction-level audit and metric recomputation
src/ordinal_risk_decision_analysis.py Posterior squared-distance decision sensitivity
results/raw/oof_predictions.csv      Raw five-class fold-assigned probabilities (legacy filename)
results/raw/fold_assignments.csv     Image, label, filename-derived group, fold, and seed manifest
results/raw/checkpoint_metadata.csv  Checkpoint paths and SHA-256 metadata
results/metrics/                      Audited reference metrics and intervals
results/figures/                      Data-derived publication figures
results/recomputed/                   Outputs created by the reproduction command
archived_pipeline/                    Original supplied preprocessing/training/evaluation scripts
configs/                              Archived configuration and dependency declarations
docs/                                 Supplementary documentation location
```

## Training provenance boundary

The `archived_pipeline/` directory preserves the supplied preprocessing, training, and evaluation scripts for transparency. The repository does not claim bit-for-bit retraining reproducibility. The archived package lacked complete epoch-wise histories, a complete software lockfile, and complete runtime logs. The supplied history files were not sufficient to support learning-curve or exact-stopping-epoch claims.

The prediction-level results were independently audited against the raw probability file, fold assignments, labels, probabilities, checkpoint metadata, and checksum information. The fold-assigned training protocol also selected checkpoints using macro-F1 on the same fold later used for prediction. This reduces the strength of the generalization claim and should be reported as a limitation.

No original-resolution Kaggle JPEGs or full competition archives are redistributed here. The manuscript and figure package include a small representative-image panel derived from Kaggle photographs; the authors must confirm that its publication and redistribution comply with the competition terms. Users must obtain the data through the official Kaggle page and comply with its terms. No private patient information is included in this repository.

## Responsible interpretation

This repository supports a retrospective computational benchmark. It does not establish external clinical validity, fairness, independent calibration, autonomous screening, clinician trust, lesion localization, or patient-care readiness. The four low-variance images identified in a prediction-blind quality-control audit were retained in the primary analysis; their removal was evaluated only as a sensitivity analysis.

## Suggested citation

Cite the associated manuscript and report the repository commit or release tag used for analysis. No public repository URL, DOI, or open-source license is asserted in this local archive; the corresponding author should add these only after creating and verifying an authorized public release.

## License and authorship

The code in this repository is released under the MIT License (see `LICENSE`). The Kaggle images are not redistributed and remain subject to the Kaggle competition terms.
