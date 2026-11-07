# Statistical audit of the secondary ordinal-calibration analysis

## Scope and interpretation

This is a group-disjoint, second-stage calibration analysis applied to fixed EfficientNet-B4 fold-assigned probabilities. It does not retrain the network, constitute end-to-end nested cross-validation, or provide external validation. The grouping key is derived from the numeric image-filename prefix; the legacy manifest column is named `subject_id`, but the key was not verified against a clinical patient identifier.

The paired filename-group bootstrap is appropriate for comparing raw and calibrated outputs because both outputs correspond to the same images and paired eye-side records are retained together. The reported intervals are descriptive uncertainty intervals for differences, not p-values. No additional generic hypothesis test is needed for this sensitivity analysis.

## Method checks

Five deterministic meta-fold assignments were formed by shuffling the 17,563 filename-derived grouping keys and assigning them cyclically. In each iteration, four meta-folds were used to fit cumulative isotonic maps and the remaining meta-fold was used for evaluation. Thus, a group’s labels were not used to fit the calibrator evaluated on that group. This split applies only to calibration of fixed predictions; it does not address checkpoint selection on the original prediction folds.

The method calibrated the four cumulative events P(Y ≥ 1), P(Y ≥ 2), P(Y ≥ 3), and P(Y ≥ 4). The cumulative-minimum projection enforced a non-increasing sequence. If 1 ≥ q₁ ≥ q₂ ≥ q₃ ≥ q₄ ≥ 0, the reconstructed values p₀ = 1 − q₁, p₁ = q₁ − q₂, p₂ = q₂ − q₃, p₃ = q₃ − q₄, and p₄ = q₄ are non-negative and sum to one. The probability audit passed: the calibrated probabilities had minimum value approximately 1.0 × 10⁻¹² and maximum absolute row-sum error 4.44 × 10⁻¹⁶.

## Seed-2026 results

For the seed-2026 meta-fold assignment, raw and calibrated estimates were:

- ECE: 0.141454 to 0.010331.
- Multiclass Brier score: 0.408894 to 0.259393.
- Mean absolute grade error: 0.324660 to 0.239111.
- Within-one-grade accuracy: 0.928543 to 0.934408.
- Large ordinal error rate (absolute grade error ≥ 2): 0.071457 to 0.065592.
- QWK: 0.736006 to 0.763745.
- Grade-≥2 referable-DR ROC-AUC: 0.930503 to 0.933085.
- Accuracy: 0.752975 to 0.833200.
- Balanced accuracy: 0.583907 to 0.521113.
- Macro-F1: 0.569848 to 0.569735.
- Macro ROC-AUC: 0.873664 to 0.872211.
- Log loss: 0.739464 to 0.754557.

The changes are not uniformly favorable: probability and ordinal-distance summaries improve, while balanced accuracy falls. Macro-F1 changes minimally; log loss increases numerically.

## Paired filename-group bootstrap differences

A 1000-replicate paired filename-group bootstrap was performed with seed 2026. Each replicate sampled filename-derived grouping keys with replacement and retained all associated eye-side rows. Differences are calibrated minus raw:

- Accuracy: +0.080225; 95% interval +0.075927 to +0.084412.
- Balanced accuracy: −0.062794; 95% interval −0.070710 to −0.055914.
- Macro-F1: −0.000113; 95% interval −0.008082 to +0.006997.
- QWK: +0.027738; 95% interval +0.020664 to +0.034500.
- Log loss: +0.015093; 95% interval −0.011570 to +0.044787.
- Macro ROC-AUC: −0.001453; 95% interval −0.004039 to +0.001135.
- Multiclass Brier score: −0.149501; 95% interval −0.153576 to −0.145182.
- ECE: −0.131122; 95% interval −0.137226 to −0.123600.
- Mean absolute grade error: −0.085549; 95% interval −0.091644 to −0.079512.
- Within-one-grade accuracy: +0.005865; 95% interval +0.003132 to +0.008569.
- Large ordinal error rate: −0.005865; 95% interval −0.008569 to −0.003132.
- Referable-DR ROC-AUC: +0.002581; 95% interval +0.001940 to +0.003183.

The intervals for macro-F1, macro ROC-AUC, and log loss include zero. Avoid describing the calibration as an overall classifier improvement.

## Five-seed meta-fold sensitivity

The same fixed predictions were calibrated under five alternative group-disjoint meta-fold assignments, using seeds 2026–2030. Across those seeds, ECE changes ranged from −0.131866 to −0.131122; QWK changes ranged from +0.027693 to +0.029504; and balanced-accuracy changes ranged from −0.062867 to −0.060201. The direction of the three principal trade-offs was stable. Macro-F1 changes ranged from −0.000394 to +0.001400 and macro-AUC changes from −0.001910 to −0.001466. The five runs are repeated meta-fold assignments on the same fixed predictions, not independent model replications.

## Recommended manuscript interpretation

Describe this as a secondary calibration analysis on fixed fold-assigned probabilities. Report the paired filename-group bootstrap intervals, the seed-2026 results, and the five-seed sensitivity range. State that calibration reduced ECE, Brier score, and grade-distance error and increased QWK, while reducing balanced accuracy; do not claim uniform improvement, clinical calibration, external validation, or improved patient-level risk estimates.
