# Independent validation report: EfficientNet-B4 Kaggle DR package

## Overall verdict

**Prediction-level validation: PASS.** The package contains 35,126 unique image records with internally consistent labels, five-class probabilities, recomputed argmax predictions, fold assignments, filename-derived grouping keys, and checkpoint metadata.

**Training-provenance completeness: LIMITED.** The package does not provide complete epoch-wise histories, runtime logs, or a full software lockfile. The manuscript therefore makes no learning-curve or exact-stopping-epoch claim and identifies the training-record limitations.

## Checks passed

1. The fold-assigned probability file contains 35,126 rows; image identifiers are unique.
2. Label counts are 25,810, 2,443, 5,292, 873, and 708 for grades 0–4.
3. All probabilities are finite and non-negative. Probability sums range from 0.99999983 to 1.00000018.
4. Stored argmax labels equal independently recomputed argmax labels for all 35,126 rows.
5. Each image matches one fold-assignment row. Fold sizes are 7,026, 7,024, 7,026, 7,024, and 7,026 images.
6. Each of the 17,563 filename-derived grouping keys occurs in one fold only. This verifies consistency of the supplied grouping variable; it does not verify clinical patient identity.
7. Independent metric recomputation reproduces accuracy 0.752975, macro-F1 0.569848, QWK 0.736006, log loss 0.739464, and macro ROC-AUC 0.873664.
8. Checkpoint hashes match the checkpoint-metadata and checksum records.
9. The prediction-blind quality-control sensitivity analysis for four low-variance images does not materially alter the primary estimates.
10. Figures were generated from the audited labels and probabilities, bootstrap outputs, and the representative Kaggle image panel described in the figure manifest.
11. The posterior squared-distance sensitivity analysis was recomputed from fixed probabilities. It changed 12,069 of 35,126 decisions, reduced the large ordinal error rate (absolute grade error ≥ 2) from 0.071457 to 0.024796, and increased within-one-grade accuracy from 0.928543 to 0.975204, while reducing accuracy, macro-F1, QWK, and mean absolute grade error. The primary argmax rule remains unchanged.

## Primary results

| Metric | Estimate | 95% filename-group bootstrap interval |
|---|---:|---:|
| Accuracy | 0.7530 | 0.7478–0.7578 |
| Balanced accuracy | 0.5839 | 0.5731–0.5964 |
| Macro-F1 | 0.5698 | 0.5596–0.5807 |
| QWK | 0.7360 | 0.7264–0.7444 |
| Macro one-vs-rest ROC-AUC | 0.8737 | 0.8695–0.8778 |
| Log loss | 0.7395 | 0.7294–0.7503 |

## Limitations retained in the manuscript

This is a single-dataset retrospective benchmark without external validation. The filename-derived key is a practical grouping variable rather than an independently verified patient identifier. The fold used for prediction also informed checkpoint selection, so estimates may be optimistic. Training histories and environment records are incomplete. The evidence does not support claims about external clinical validity, attention modules, ensembles, SHAP/LIME/Grad-CAM, lesion localization, clinician agreement, fairness, autonomous screening, or deployment. The squared-distance decision is a prediction-level sensitivity analysis, not a QWK optimizer or clinical decision rule.

The package is suitable as a prediction-level reproducibility companion for a narrowly scoped retrospective benchmark, subject to author verification of authorship, affiliations, declarations, data-use permissions, and training provenance. Journal acceptance or a Q1 outcome cannot be guaranteed.
