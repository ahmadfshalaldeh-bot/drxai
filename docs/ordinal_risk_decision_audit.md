# Ordinal squared-distance decision sensitivity audit

## Scope

This analysis evaluates a secondary decision rule on fixed EfficientNet-B4 fold-assigned probabilities. It does not retrain the model, change the primary argmax result, optimize QWK, or constitute external validation. The grouping key used for uncertainty is derived from the image filename and is not a verified patient identifier.

For grades 0–4 and probability vector p, the squared-distance decision is

`argmin_k sum_j (j-k)^2 p_j`.

Completing the square shows that this rule chooses the integer grade nearest to the posterior mean grade `sum_j j p_j`, with deterministic lower-grade tie breaking. The decision is Bayes-optimal only for the stated item-level squared-distance loss under the relevant posterior distribution; it does not optimize QWK.

## Results

The rule changed 12,069 of 35,126 decisions (34.36%). The primary argmax rule remains preferable for the reported classification benchmark because the alternative materially worsened accuracy, macro-F1, and QWK despite reducing errors spanning two or more grades.

| Metric | Primary argmax | Squared-distance rule | Difference (risk − argmax) | 95% paired filename-group bootstrap interval |
|---|---:|---:|---:|---:|
| Accuracy | 0.752975 | 0.511445 | −0.241411 | −0.246684 to −0.235892 |
| Balanced accuracy | 0.583907 | 0.539113 | −0.044740 | −0.051924 to −0.037784 |
| Macro-F1 | 0.569848 | 0.494255 | −0.075478 | −0.082205 to −0.068697 |
| QWK | 0.736006 | 0.664990 | −0.070892 | −0.076281 to −0.065555 |
| Mean absolute grade error | 0.324660 | 0.516740 | +0.191899 | +0.185815 to +0.197663 |
| Within-one-grade accuracy | 0.928543 | 0.975204 | +0.046716 | +0.044241 to +0.049052 |
| Large ordinal error rate (absolute grade error ≥ 2) | 0.071457 | 0.024796 | −0.046716 | −0.049052 to −0.044241 |

The intervals are paired 95% percentile intervals from 1000 bootstrap replicates (seed 2026), resampling filename-derived groups and retaining associated eye-side images. They quantify uncertainty conditional on the supplied fold assignments; they do not correct checkpoint-selection optimism.

## Interpretation boundary

The rule reduces large grade jumps but worsens the primary classification and QWK metrics on these raw probabilities. This is evidence against presenting it as a universal improvement. A Bayes decision expression requires an appropriate posterior and an explicitly chosen utility; simplex membership alone does not establish calibration.
