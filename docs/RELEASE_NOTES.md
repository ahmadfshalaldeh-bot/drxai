# Release notes — 7 November 2026 (revision)

- **Figures renumbered** to match first citation in the revised manuscript (old→new: 8→4, 4→5, 5→6, 9→7, 6→8, 7→9). `submission_figures/` holds the renumbered 600-dpi set.
- **Figure 6 regenerated** from `results/raw/oof_predictions.csv` (`src/make_confusion_matrix_figure.py`); the earlier colour bar obscured three cells. Counts are unchanged.
- **Independent second-implementation audit added** (`audit/independent_audit.py`, `audit/outputs/`).
- **Every analysis script re-run.** The results agree with the shipped outputs, apart from documented tie-resolution sensitivity of the calibrated macro ROC-AUC at the fourth decimal (see README).
- **Added:** `requirements-tested.txt`, `MANIFEST.csv`, regenerated `checksums.sha256`, and an updated Makefile.
- **Manuscript text replaced.** `docs/mathematics_dr_endnote_integrated.md`, which used superseded figure numbering, was removed; the submitted manuscript is the reference text.
- **QC sensitivity analysis is now reproducible.** The four QC-flagged image IDs (1986_left, 34689_left, 43457_left, 32253_right) were recovered from the archived sensitivity metrics. Adding `src/qc_sensitivity_bootstrap.py` and its outputs makes the Section 3.9 estimates exactly reproducible. Table 11 was removed from the manuscript; the values are reported in the Section 3.9 text.
- **Reference 12 updated:** Appl. Comput. Inform. 2026, 22, 210–223.
- **Unchanged:** no labels, probabilities, fold assignments, decision rules or reported estimates were changed.

# Release notes — 3 November 2026

This is a prediction-level reproducibility companion for the retrospective Kaggle EfficientNet-B4 five-class diabetic-retinopathy benchmark. It includes the final clean manuscript source, audited fold assignments and probabilities, reconstructed tables and figures, per-grade and derived referable-DR bootstrap intervals, secondary calibration and ordinal-decision analyses, source scripts, validation reports, and checksums.

The release does not claim bit-for-bit retraining reproducibility. The supplied package lacks complete epoch-wise histories, a full software lockfile, and complete runtime records. It does not redistribute the original Kaggle JPEGs, competition archives, or model checkpoints. The filename-derived grouping key is not independently verified as a clinical patient identifier.

The manuscript source and figures now use consistent fold-assigned terminology and 1000-replicate filename-group bootstrap descriptions. The primary raw probability-argmax results are unchanged. The posterior squared-distance decision rule remains a secondary sensitivity analysis: it reduces the rate of errors spanning at least two grades but worsens accuracy, QWK, macro-F1, and mean absolute grade error. The ordinal calibration analysis remains secondary and reports its trade-offs, including reduced balanced accuracy.
