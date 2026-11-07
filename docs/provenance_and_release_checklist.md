# Provenance and release checklist

## Audited evidence included

- `results/raw/oof_predictions.csv`: 35,126 rows of raw five-class fold-assigned probabilities; the filename is retained for compatibility.
- `results/raw/fold_assignments.csv`: image, label, filename-derived grouping key, eye side, fold, role, and seed. The grouping key is not a verified patient identifier.
- `results/raw/checkpoint_metadata.csv`: checkpoint paths and SHA-256 hashes from the supplied archive.
- `results/metrics/`: independently reconstructed primary, fold-level, per-class, calibration, referable-DR, quality-control sensitivity, and confidence-interval files.
- `results/figures/`: figures regenerated from verified labels, raw probabilities, reconstructed statistics, and genuine source images.
- `results/ordinal_risk_decision/`: fixed-probability posterior squared-distance decision outputs and paired filename-group bootstrap differences.
- `src/ordinal_risk_decision_analysis.py`: reproducible implementation of the secondary decision-rule sensitivity analysis.
- `results/final_validation_report.md`: independent validation and provenance limitations.

## Must be verified before public release

1. Replace the local manuscript citation with the final publication metadata when available.
2. Add the final public repository URL and, preferably, an archival DOI release.
3. Confirm that the corresponding author has permission to redistribute every source file, script, figure, and checkpoint metadata record.
4. Do not upload Kaggle images, competition archives, or checkpoints unless the applicable terms explicitly permit it.
5. Confirm all author names, affiliations, email display permissions, CRediT roles, funding, ethics, acknowledgments, and conflicts of interest.
6. Record the exact commit hash used to produce the manuscript tables and figures.
7. Keep the prediction-level reproduction claim separate from bit-for-bit training reproducibility.
8. Do not add attention, ensemble, XAI, clinician-agreement, or external-validation claims without new raw evidence and a new independent audit.
