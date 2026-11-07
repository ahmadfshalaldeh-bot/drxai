# Quality-control sensitivity analysis (manuscript Section 3.9)

`qc_flagged_images.csv` lists the four images that the prediction-blind image audit flagged as blank or nearly blank. The audit rule was a grayscale pixel-intensity standard deviation below 2.0.

The primary analysis retains all 35,126 images. These four images are excluded only in this sensitivity analysis.

## Recompute the Section 3.9 sensitivity estimates

```bash
python src/qc_sensitivity_bootstrap.py --root . --replicates 1000 --seed 2026
```

The command writes two files:

- `qc_sensitivity_summary.csv`: point estimates and paired 95% filename-group bootstrap intervals.
- `qc_paired_bootstrap_replicates.csv`: the replicate-level differences.

## Provenance of the identifiers

The image identifiers were not stored in the original archive. They were recovered from the archived QC-sensitivity metrics in `results/reconstruction_notes.json` in three steps:

1. **Confusion-matrix search.** A search over every way of removing four images from the confusion matrix found a single combination that matches the archived accuracy, balanced accuracy, macro-F1, weighted F1, macro precision and QWK. That combination is two grade-0 images predicted as grade 1, and two grade-1 images predicted as grade 1.
2. **Log-loss search.** Among those candidates, a search matching the archived log loss left 392 sets of four images.
3. **ROC-AUC match.** Exactly one of the 392 sets reproduces the archived macro ROC-AUC (difference 0.0).

Re-running the paired bootstrap with these four identifiers reproduces every value and interval reported in Section 3.9.

## Remaining check

The image files are not in this archive, so the grayscale standard deviations could not be recomputed here. The authors can confirm the rule by opening the four Kaggle JPEGs or computing their grayscale standard deviations.
