**Article**

**Ahmad Shalaldeh¹,\*, Mishri AlMarshoud², Mohammad O. Alhawarat¹, Ayman Alnsour¹, Majeed Safa³, Chris Logan⁴, Mohmmad Othman⁵˒⁶, Sara Al-mahamid⁷ and Mohammad Mahfujur Rahman⁸**

¹ Department of Data Science and Artificial Intelligence, Faculty of Information Technology, Al-Ahliyya Amman University, Al-Salt 19111, Jordan; m.hawarat@ammanu.edu.jo; a.alnsour@ammanu.edu.jo

² Department of Computer Science and Cyber Security, International University of Kuwait (IUK), Ardiya, Kuwait; mishri.almarshoud@IUK.edu.kw

³ Faculty of Agribusiness and Commerce, Lincoln University, Lincoln 7647, New Zealand; majeed.safa@lincoln.ac.nz

⁴ Faculty of Agriculture and Life Sciences, Lincoln University, Lincoln 7647, New Zealand; chris.logan@lincoln.ac.nz

⁵ School of Health Science, University of Canterbury, Christchurch 8041, New Zealand; mohmmad.othman@canterbury.ac.nz

⁶ Intensive Care Department, Christchurch Public Hospital, Christchurch 8011, New Zealand

⁷ Strathclyde Institute of Pharmacy & Biomedical Sciences, University of Strathclyde, Glasgow G1 1XQ, United Kingdom; sara.al-mahamid@strath.ac.uk

⁸ Department of Software Engineering, Ilem Group, Casablanca 20270, Morocco; mohammad.mahfujur-rahman@ilemgroup.com

\* Correspondence: a.shalaldeh@ammanu.edu.jo

# Abstract

Grading diabetic retinopathy (DR) into five classes is an ordinal problem with a strongly imbalanced label distribution, and accuracy by itself hides most of what matters. In this work, we trained and evaluated an ImageNet-initialized EfficientNet-B4 (512 × 512 input) on 35,126 labeled Kaggle photographs, using five folds that kept both eye-side images of each of 17,563 filename-derived groups in the same fold. Pooled fold-assigned predictions gave accuracy 0.7530, balanced accuracy 0.5839, macro-F1 0.5698, quadratic weighted kappa (QWK) 0.7360, and macro one-vs-rest ROC-AUC 0.8737 (95% filename-group bootstrap interval 0.8695–0.8778), and 92.85% of predictions fell within one grade. A grade-≥2 referable endpoint derived from the same outputs reached ROC-AUC 0.9305, sensitivity 0.7756, and specificity 0.9360. We show that the decision minimizing expected squared grade distance is the integer nearest the posterior mean grade; on the raw probabilities it cut errors of two or more grades from 0.0715 to 0.0248 but lowered QWK to 0.6650. Group-disjoint ordinal isotonic calibration reduced expected calibration error from 0.1415 to 0.0103 and raised QWK to 0.7637, at the cost of balanced accuracy (0.5211). The prediction fold also guided checkpoint selection, so these estimates may be optimistic, and external validation is needed before any clinical use.

## Keywords

diabetic retinopathy; retinal fundus photography; ordinal classification; EfficientNet-B4; transfer learning; group-aware cross-validation; quadratic weighted kappa; probability calibration; referable diabetic retinopathy; reproducibility

MSC 2020: 68T07; 68T10; 68U10; 62P10.

# Highlights

- Five-class DR grading with EfficientNet-B4 was evaluated on 35,126 Kaggle photographs in group-disjoint folds.

- Paired eye-side images of each filename-derived group remained in the same fold.

- Primary QWK was 0.7360 and macro ROC-AUC 0.8737, with filename-group bootstrap intervals.

- The expected-squared-distance decision equals rounding the posterior mean grade and trades large errors for exact agreement.

- Ordinal isotonic calibration lowered ECE from 0.1415 to 0.0103 but reduced balanced accuracy.

# 1. Introduction

Diabetic retinopathy (DR) is a microvascular complication of diabetes and one of the main causes of preventable vision loss. The retinal signs range from microaneurysms, haemorrhages, hard exudates, and cotton-wool spots to venous abnormalities and new vessel growth \[1,2\]. Since the clinical severity scale is ordered, predicting a neighbouring grade is a smaller mistake than skipping several grades, and an automated grader should be judged with class-specific, ordinal, and probability-based measures and not with accuracy alone.

Grading retinal photographs by hand needs trained graders, and this becomes a bottleneck in large screening programmes. Deep learning has been applied widely to retinal images, both for binary referral and for severity grading \[3,4\], and the Kaggle Diabetic Retinopathy Detection competition offers a well-known five-grade benchmark \[5\]. Transfer learning gives a practical starting point for this kind of task, and EfficientNet models have already been used for retinal analysis \[6,7\]. However, a strong score on one public dataset says little about how a model behaves with other cameras, populations, grading protocols, or clinical workflows.

Three methodological issues matter in particular for five-class DR grading. First, the Kaggle cohort is heavily imbalanced, with no-DR photographs forming the majority, so accuracy can stay high even when mild, severe, and proliferative disease are poorly recognized. Second, the two eyes of one person share acquisition conditions and disease information; if they are split between training and evaluation, the reported performance becomes optimistic. Third, published numbers are hard to compare because studies use different class definitions, datasets, preprocessing, split units, and endpoints. A binary referable-DR AUC, a five-class macro-AUC, a segmentation Dice score, a validation accuracy, and a quadratic weighted kappa each measure something different, and placing them on one ranking is misleading.

For this reason, Table 1 is presented as an evidence map and not as a leaderboard. It covers binary referral studies, a prospective clinical evaluation, a separately collected clinical grading cohort, five-class grading against an adjudicated reference standard, and work that used the Kaggle data for five-class grading and referable-DR detection. Cohorts, architectures, preprocessing, partitioning, checkpoint selection, and metric definitions all differ, so no direct superiority claim can be made. In our benchmark we state the data source, grouping unit, metric definitions, and uncertainty procedure explicitly, so that the results can be read in this context.

In this paper, we evaluate an ImageNet-initialized EfficientNet-B4 for five-class grading of 35,126 Kaggle photographs, with filename-derived groups used to keep paired eye-side images together. We ask three questions: how the classifier performs on nominal, class-balanced, ordinal, and probabilistic measures; whether its five-class probabilities can support a secondary grade-≥2 referable-DR endpoint; and how ordinal calibration, squared-distance decisions, and prediction-blind image-quality filtering change the interpretation. The contribution is a group-aware ordinal evaluation of a single transfer-learning model, with explicit uncertainty estimates and a direct look at the trade-offs among calibration, class-balanced performance, and grade-distance error. The Results first report the primary five-class benchmark and then the secondary analyses of the probability outputs and decision rules. Figure 1 summarizes the full workflow.

## Literature comparison and study positioning

As Table 1 shows, earlier DR studies differ in dataset composition, label definition, clinical endpoint, architecture, image resolution, partitioning strategy, and metric calculation, which makes a direct ranking inappropriate. Validation accuracy and validation loss, for instance, cannot be compared across datasets with different class counts, prevalences, loss functions, and split designs. In the same way, an encoder–decoder segmentation model such as U-Net should not be ranked against an image-level multiclass grader, and a binary referral AUC is not evidence of five-class severity-grading performance. What the literature does suggest is the need for a protocol-specific evaluation that reports class-balanced metrics, an ordinal measure, probability-based performance, leakage control, and uncertainty. We therefore position this study as a reproducible EfficientNet-B4 benchmark and not as proof that the architecture is superior to previous DR systems.

**Table 1.** Contextual comparison of selected peer-reviewed DR studies and the present study

| **Study**                  | **Cohort and evaluation design**                                                                                                                                                           | **Endpoint**                                        | **Reported result in the source study**                                                                                                                                                               |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Gulshan et al. \[8\]       | EyePACS-1: 9963 images from 4997 patients; Messidor-2: 1748 images from 874 patients. Two independent validation sets graded by ophthalmologists.                                          | Binary referable DR                                 | AUC 0.991 (95% CI 0.988–0.993) for EyePACS-1 and 0.990 (0.986–0.995) for Messidor-2. At the high-sensitivity operating point, sensitivity/specificity were 97.5%/93.4% and 96.1%/93.9%, respectively. |
| Abràmoff et al. \[9\]      | Messidor-2: 874 subjects (1748 images); consensus reference standard of retinal specialists.                                                                                               | Binary referable DR                                 | AUC 0.980 (95% CI 0.968–0.992); sensitivity 96.8%; specificity 87.0%.                                                                                                                                 |
| Ting et al. \[4\]          | Primary Singapore screening validation: 14,880 patients and 71,896 images; 10 additional multiethnic cohorts contributed 40,752 images.                                                    | Binary referable DR and vision-threatening DR       | Primary referable-DR AUC 0.936 (95% CI 0.925–0.943), sensitivity 90.5%, and specificity 91.6%; AUC across the 10 additional cohorts ranged from 0.889 to 0.983.                                       |
| Ipp et al. \[10\]          | Prospective evaluation at 15 primary- and eye-care facilities; 893 participants (1786 eyes); two-field undilated AI images compared with ETDRS reading-center grades.                      | Binary more-than-mild DR and vision-threatening DR  | Sensitivity/specificity: 95.5%/85.0% for more-than-mild DR and 95.1%/89.0% for vision-threatening DR.                                                                                                 |
| Zhang et al. \[11\]        | Separately assembled clinical cohort of 13,767 images from 1872 patients; reported 70%/10%/20% training/validation/testing split.                                                          | DR identification and severity grading              | Identification: sensitivity 97.5%, specificity 97.7%, AUC 97.7%; grading: sensitivity 98.1% and specificity 98.9% as defined by the authors.                                                          |
| Kusakunniran et al. \[12\] | Kaggle data for model development and a separate self-collected hospital cohort for additional evaluation.                                                                                 | Kaggle five-class grading; binary DR identification | Kaggle grading accuracy 0.55. Kaggle binary identification: accuracy 0.80, sensitivity 0.80, specificity 0.83. Separate hospital binary evaluation: sensitivity 98.18%, specificity 54.55%.           |
| Krause et al. \[13\]       | Development images from screening programmes including EyePACS; evaluation against a reference standard adjudicated by retina specialists; higher input resolution than the present study. | Five-class grading; moderate-or-worse DR            | Five-class QWK 0.84; for moderate-or-worse DR, AUC 0.986, sensitivity 97.1%, specificity 92.3%.                                                                                                       |
| Voets et al. \[14\]        | Reproduction of a published referable-DR pipeline trained on Kaggle EyePACS images; evaluated on held-out Kaggle images and on Messidor-2.                                                 | Binary referable DR                                 | AUC 0.951 on the Kaggle test images and 0.853 on Messidor-2.                                                                                                                                          |
| Present study              | Labeled Kaggle cohort: 35,126 images and 17,563 filename-derived groups; five group-disjoint folds. The prediction fold was also used for checkpoint selection.                            | Five-class severity grading                         | Accuracy 0.7530; balanced accuracy 0.5839; macro-F1 0.5698; QWK 0.7360; macro one-vs-rest ROC-AUC 0.8737.                                                                                             |

Abbreviations: AUC, area under the receiver operating characteristic curve; CI, confidence interval; DR, diabetic retinopathy; ETDRS, Early Treatment Diabetic Retinopathy Study; QWK, quadratic weighted kappa.

Table 1 note. Results are transcribed from each cited study and retain that study’s cohort, endpoint definition, reference standard, and metric conventions. They are presented for context, not as a direct ranking: binary screening sensitivity/specificity and AUC are not equivalent to five-class accuracy, macro-F1, QWK, or macro-AUC. For the present study, the filename-derived grouping key was not independently verified as a clinical patient identifier, and fold labels informed checkpoint selection as well as prediction; its estimates therefore require the qualification described in Section 2.4.

[Figure image: see submission_figures/]

**Figure 1.** Overview of the Kaggle-based study, showing the image audit, filename-derived grouping of paired eye-side records, five-fold model evaluation, and primary and secondary analyses.

# 2. Materials and Methods

## 2.1. Study design and scope

We carried out a retrospective computational evaluation of five-class DR grading on the Kaggle Diabetic Retinopathy Detection cohort \[5\]. Only publicly available, de-identified retinal photographs were used; no participants were recruited, and no new clinical grades or lesion annotations were produced. One ImageNet-initialized EfficientNet-B4 model with a five-class linear classifier was evaluated. The study was designed as a group-aware benchmark and not as a clinical validation study.

## 2.2. Dataset, grouping variable, and labels

The Kaggle label manifest lists 35,126 photographs. Each image identifier has a numerical prefix and an eye-side suffix (\_left or \_right), and we used the numerical prefix as a filename-derived grouping key. This key was not treated as a verified patient identifier. The analyzed manifest contained 17,563 unique keys, each with two eye-side records.

Labels were coded from 0 to 4 as follows: grade 0, no apparent DR; grade 1, mild NPDR; grade 2, moderate NPDR; grade 3, severe NPDR; and grade 4, proliferative DR. The class counts were 25,810, 2443, 5292, 873, and 708, which add up to 35,126. Table 2 and Figure 2 show the distribution.

**Table 2.** Audited class distribution in the Kaggle analysis

| **Grade** | **Clinical label** | **Images** | **Percentage** |
|-----------|--------------------|------------|----------------|
| 0         | No apparent DR     | 25,810     | 73.48%         |
| 1         | Mild NPDR          | 2,443      | 6.95%          |
| 2         | Moderate NPDR      | 5,292      | 15.07%         |
| 3         | Severe NPDR        | 873        | 2.49%          |
| 4         | Proliferative DR   | 708        | 2.02%          |
| Total     | —                  | 35,126     | 100.00%        |

Table 2 note. NPDR, non-proliferative diabetic retinopathy. Percentages were calculated from the exact class counts. The displayed grade-specific percentages sum to 100.01% because each value is rounded to two decimal places; the unrounded percentages sum to 100.00%.

[Figure image: see submission_figures/]

**Figure 2.** Class distribution of the 35,126 photographs in the primary Kaggle analysis. The imbalance motivates macro-averaged performance measures; inverse-frequency sampling is specified in the archived training configuration.

## 2.3. Image audit and quality-control sensitivity analysis

Before analysis, we audited the manifest for duplicate identifiers, missing labels, agreement between folds and predictions, file existence and decodability, image dimensions, finite pixel values, grayscale variance, and exact image hashes. No missing or unreadable images, invalid dimensions, or exact duplicates across groups were found. Four photographs (1986_left, 34689_left, 43457_left, and 32253_right; listed in Supplementary File S1) had pixel-intensity standard deviations below 2.0 and were flagged by an objective image-quality rule as blank or nearly blank. We kept them in the primary analysis and removed them only in a separate sensitivity analysis, and the flag was based on image properties alone, without reference to model probabilities, predictions, losses, or confidence values. Figure 3 shows representative photographs from the five grades.

[Figure image: see submission_figures/]

**Figure 3.** Representative Kaggle fundus photographs with grade labels from the analyzed label manifest. The images provide visual context and are not a separate validation set.

## 2.4. Fold assignment and leakage control

The fold-assignment manifest placed all 35,126 images into five folds and recorded the grouping key and eye side of every image. The fold sizes were 7,026, 7,024, 7,026, 7,024, and 7,026 images. No grouping key appeared in more than one fold, so paired eye-side records always stayed together.

In each fold, the remaining four folds supplied the training images. The held-out fold was used to monitor macro-F1, and this value drove learning-rate reduction (ReduceLROnPlateau), early stopping, and checkpoint selection; the selected checkpoint then produced the predictions for that fold. In other words, the model weights were fitted only on the other four folds, but the labels of the prediction fold influenced the learning-rate schedule, the stopping point, and the choice of checkpoint. For this reason, we describe the results as fold-assigned benchmark estimates under the implemented protocol. They are not fully nested estimates from an untouched outer test fold and may be optimistic, and grouping the paired eyes removes one source of leakage but not this selection effect. Stochastic augmentation was used only for training images, inference transforms were deterministic, and the predictions of the five fold-specific checkpoints were concatenated for the pooled metrics.

## 2.5. Preprocessing and augmentation

Each JPEG was read with OpenCV and converted from BGR to RGB. Near-black borders were cropped with a grayscale-intensity tolerance of 7, and the image was resized to 512 × 512 pixels. Illumination was then normalized with the weighted transform I′ = 4I − 4G(I) + 128, where G is a Gaussian blur with a 51 × 51 kernel. The preprocessed images were stored before training, and the network input was 512 × 512 × 3.

Augmentation was applied to training images only and included random 90-degree rotations (p = 0.5), horizontal and vertical flips (p = 0.5 each), shift-scale-rotation (shift limit 0.1, scale limit 0.15, rotation limit 60 degrees; p = 0.7), hue-saturation-value perturbations (limits 20, 30, and 20; p = 0.5), and brightness/contrast perturbations (limits 0.2 and 0.2; p = 0.5). At inference, images were normalized with the ImageNet means \[0.485, 0.456, 0.406\] and standard deviations \[0.229, 0.224, 0.225\] and converted to tensors, with no stochastic augmentation.

## 2.6. EfficientNet-B4 model and optimization

EfficientNet-B4 was initialized from the torchvision ImageNet weights \[6,7\], and its original classifier was replaced by a linear layer with five outputs. For an image x, the network gives logits z(x), and the class probabilities are

p<sub>c</sub>(x) = exp(z<sub>c</sub>(x)) / Σ<sub>j=0</sub><sup>4</sup> exp(z<sub>j</sub>(x)), c ∈ {0, 1, 2, 3, 4} (1)

According to the archived training script and configuration, the optimizer was AdamW with a learning rate of 1 × 10−4 and a weight decay of 1 × 10−4, and the loss was focal loss with α = 1 and γ = 2 \[15\],

L<sub>focal</sub> = −α(1 − p<sub>t</sub>)<sup>γ</sup> log p<sub>t</sub> (2)

where pₜ is the predicted probability of the observed class. The training code also uses a WeightedRandomSampler in which each training image receives a weight equal to the inverse frequency of its class in the training partition; sampling is with replacement, and one epoch draws as many samples as the training set contains. No complete record of runtime overrides was available.

The script defaults are up to 40 epochs, a training batch size of 16, an evaluation batch size of 32, early-stopping patience of five evaluations without macro-F1 improvement, and a ReduceLROnPlateau scheduler (factor 0.5; patience 2). Automatic mixed precision is enabled for training, while prediction inference runs in standard precision. Because the run package has no complete command-line capture, we can report these values as the documented configuration but cannot confirm that none was overridden at execution time. History CSVs exist only for folds 0 and 4, each with a single row, and are missing for the other folds; as a result, the actual stopping epochs, the epoch-wise learning curves, and the runtime cannot be reconstructed. Table 3 separates the audited results from the documented settings.

**Table 3.** EfficientNet-B4 evaluation protocol and documented training configuration

| **Component**                 | **Documented or observed setting**                                                                                                       | **Evidence status**                                                                      |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Dataset and target            | 35,126 labeled Kaggle images; five DR grades                                                                                             | Label manifest and fold-assigned probabilities audited                                   |
| Grouped folds                 | Five folds; 17,563 filename-derived groups; seed 2026                                                                                    | Fold assignments audited; no group crossed folds                                         |
| Backbone and output           | ImageNet-initialized EfficientNet-B4; five-logit classifier                                                                              | Training code and checkpoint metadata                                                    |
| Network input                 | 512 × 512 × 3                                                                                                                            | Preprocessing script                                                                     |
| Optimizer and loss            | AdamW; learning rate 1 × 10−4; weight decay 1 × 10−4; focal loss (α = 1, γ = 2)                                                          | Archived configuration/code; execution-time overrides unavailable                        |
| Imbalance handling            | Inverse-frequency weighted sampling with replacement                                                                                     | Archived configuration/code; run-level sampler trace unavailable                         |
| Numerical precision           | Mixed precision specified for training; standard precision for inference                                                                 | Training-code setting; runtime override record unavailable                               |
| Batch sizes and schedule      | Training batch size 16; evaluation batch size 32; up to 40 epochs; early-stopping patience 5; ReduceLROnPlateau (factor 0.5, patience 2) | Documented defaults; complete command-line/runtime capture unavailable                   |
| Checkpoint use                | Fold-specific checkpoint used to generate fold-assigned predictions; same fold used for checkpoint selection and prediction              | Prediction and validation artifacts audited; selection-induced optimism remains possible |
| Training-history availability | One-row histories were present for folds 0 and 4; histories for folds 1–3 were absent                                                    | Full epoch curves and actual stopping epochs cannot be reconstructed                     |

Table 3 note. “Documented” settings describe the supplied training code/configuration and should not be read as independently confirmed runtime values where execution-time records are absent. The table distinguishes these settings from the image, grouping, and prediction properties directly checked in the archived outputs.

## 2.7. Formal ordinal estimand and error geometry

Let Yᵢ ∈ {0, 1, 2, 3, 4} denote the reference grade for image i, let Pᵢ = (pᵢ₀, …, pᵢ₄) be the predicted probability vector, and let Ŷᵢ = arg max_c pᵢc be the five-class decision rule. The primary pooled fold-assigned estimand is the functional T({(Yᵢ, Pᵢ)}ᵢ₌₁ⁿ) evaluated on the concatenation of the five fold-assigned prediction partitions. To retain the ordinal structure, the analysis reports the absolute grade error Dᵢ = \|Yᵢ − Ŷᵢ\|, mean absolute grade error n⁻¹ΣᵢDᵢ, root-mean-square grade error \[n⁻¹ΣᵢDᵢ²\]¹ᐟ², within-one-grade accuracy n⁻¹Σᵢ 𝟙(Dᵢ ≤ 1), and the large ordinal error rate n⁻¹Σᵢ 𝟙(Dᵢ ≥ 2). Under this formulation, a grade-0 to grade-1 error and a grade-0 to grade-4 error are not treated as equally distant outcomes.

To estimate uncertainty, we resampled filename-derived grouping keys instead of individual photographs \[16\]. In bootstrap replicate b, a group sample G⁽ᵇ⁾ is drawn with replacement, all eye-side records of the sampled groups are kept, and T⁽ᵇ⁾ is recomputed on the resulting multiset. The reported interval is the empirical 2.5th to 97.5th percentile of {T⁽ᵇ⁾} over b = 1, …, 1000, with seed 2026. These intervals are conditional on the archived fold assignments, so they neither correct the checkpoint-selection optimism nor replace external validation.

## 2.8. Posterior squared-distance decision sensitivity

For an ordinal squared-distance loss, define the posterior decision risk for grade k as R(k \| x) = Σⱼ₌₀⁴ (j − k)²pⱼ(x). The corresponding decision is ỹ(x) = arg minₖ R(k \| x). Completing the square shows that R(k \| x) = constant + (k − m(x))², where m(x) = Σⱼ₌₀⁴ j pⱼ(x) is the posterior mean grade. Thus, with deterministic lower-grade tie breaking, ỹ(x) is the nearest integer grade to m(x). This rule is Bayes-optimal for the specified item-level squared-distance loss only when p(x) is interpreted as the relevant posterior distribution; it is not asserted to optimize quadratic weighted kappa, which is a chance-corrected confusion-matrix statistic. Because the archived softmax probabilities were descriptively miscalibrated, the rule was evaluated only as a secondary sensitivity analysis and did not replace the primary argmax results.

Proposition 1 (posterior-mean characterization of the squared-distance decision). For p(x) ∈ Δ⁴ and k ∈ {0, 1, 2, 3, 4}, the minimizer of R(k \| x) = Σⱼ₌₀⁴ (j − k)²pⱼ(x) is an integer nearest to m(x) = Σⱼ₌₀⁴ j pⱼ(x). If m(x) lies exactly halfway between two admissible integer grades, selecting the lower grade gives a deterministic minimizer.

Proof. Expanding the square gives R(k \| x) = Σⱼ j²pⱼ(x) − 2km(x) + k² = \[Σⱼ j²pⱼ(x) − m(x)²\] + \[k − m(x)\]². The first bracket is independent of k, so minimizing R over the ordered integer label set is equivalent to minimizing \[k − m(x)\]². The nearest admissible integer is therefore a minimizer; the stated tie rule makes the decision deterministic. □

## 2.9. Evaluation metrics

The primary five-class metrics were accuracy, balanced accuracy, macro-F1, macro precision, macro recall, log loss, quadratic weighted kappa (QWK), and macro one-vs-rest ROC-AUC. Accuracy was computed from the argmax predictions, balanced accuracy as the unweighted mean of the five class recalls, and macro-F1 as the unweighted mean of the five class F1 scores. Log loss used the full five-class probability vectors, and macro ROC-AUC was the unweighted mean of the five one-vs-rest AUC values. We included QWK because the grades are ordered and a distant-grade disagreement should count more than an adjacent one. With O the 5 × 5 confusion matrix of reference grade i and predicted grade j, and E its expectation under independent margins,

κ<sub>w</sub> = 1 − Σ<sub>i,j</sub> w<sub>ij</sub>O<sub>ij</sub> / Σ<sub>i,j</sub> w<sub>ij</sub>E<sub>ij</sub>, w<sub>ij</sub> = (i − j)<sup>2</sup> / (K − 1)<sup>2</sup>, E<sub>ij</sub> = O<sub>i·</sub> O<sub>·j</sub> / n (3)

In the secondary binary analysis, referable DR was defined as grade ≥2. The referable probability was p2 + p3 + p4, with no threshold tuning, and the binary prediction was taken from the original five-class argmax. From these we computed sensitivity, specificity, positive predictive value, F1 score, and binary ROC-AUC.

## 2.10. Calibration and uncertainty

In the primary analysis the raw EfficientNet-B4 probabilities were used as they are, without post-hoc calibration. Their calibration was described by the expected calibration error (ECE) of the maximum class probability over 15 equal-width confidence bins B₁, …, B₁₅ on \[0, 1\] \[17\]:

ECE = Σ<sub>b=1</sub><sup>15</sup> (\|B<sub>b</sub>\| / n) · \|acc(B<sub>b</sub>) − conf(B<sub>b</sub>)\| (4)

where, for each bin, acc(·) is the proportion of correct argmax decisions and conf(·) is the mean maximum class probability of the images in that bin. The separate group-held-out ordinal calibration sensitivity analysis is described in Section 2.11. The primary ECE is a descriptive property of the raw model outputs, not evidence of clinical calibration.

All uncertainty intervals come from 1000 filename-grouped bootstrap replicates with seed 2026. Grouping keys, not single photographs, were sampled with replacement, and every eye-side record of a sampled key was kept. Percentile intervals were computed for the overall, class-specific, ordinal, probabilistic, and derived binary metrics. They describe sampling variability for the fixed fold assignments and do not account for the checkpoint-selection optimism described in Section 2.4.

## 2.11. Five-meta-fold group-disjoint ordinal isotonic calibration (secondary analysis)

As a secondary analysis, we examined ordinal calibration of the fixed fold-assigned probability vectors without retraining the network. The 17,563 grouping keys were shuffled with seed 2026 and assigned cyclically to five calibration meta-folds. In each meta-fold, four group partitions were used to fit isotonic regression maps \[18\] for the cumulative events P(Y ≥ 1), P(Y ≥ 2), P(Y ≥ 3), and P(Y ≥ 4), and the fitted maps were applied to the fifth partition. Hence, the labels of a group were never used to fit the maps applied to that same group. This is a second-stage calibration of fixed predictions, not end-to-end nested cross-validation, and it leaves the checkpoint-selection limitation of the primary evaluation in place.

The four calibrated cumulative values were made non-increasing with the cumulative minimum operator and then converted to five class probabilities by differencing adjacent values, where q₁, …, q₄ denote the calibrated values of P(Y ≥ 1), …, P(Y ≥ 4):

p<sub>0</sub> = 1 − q<sub>1</sub>, p<sub>j</sub> = q<sub>j</sub> − q<sub>j+1</sub> (j = 1, 2, 3), p<sub>4</sub> = q<sub>4</sub> (5)

Proposition 2 (validity of ordinal reconstruction). If 1 ≥ q₁ ≥ q₂ ≥ q₃ ≥ q₄ ≥ 0, then the reconstructed vector p = (p₀, p₁, p₂, p₃, p₄) is a valid probability vector: every pⱼ is non-negative and Σⱼpⱼ = 1.

Proof. Since q₁ ≤ 1, p₀ = 1 − q₁ ≥ 0. Since q₁ ≥ q₂ ≥ q₃ ≥ q₄, each successive difference p₁, p₂, and p₃ is non-negative, and p₄ = q₄ ≥ 0. The terms telescope: (1 − q₁) + (q₁ − q₂) + (q₂ − q₃) + (q₃ − q₄) + q₄ = 1. □

The purpose of this analysis is to test whether an ordinal probability constraint, fitted without the evaluation labels of the corresponding meta-fold, improves probability quality and grade-distance behaviour. It does not retrain EfficientNet-B4 and is not an external validation.

Raw and calibrated outputs were compared with a paired filename-group bootstrap. In each of 1000 replicates, grouping keys were sampled with replacement and all their eye-side records were kept. The reported 95% intervals are percentile intervals for the calibrated-minus-raw differences; they are not p-values.

# 3. Results

## 3.1. Dataset and split audit

The pooled prediction file had 35,126 rows and 35,126 unique image identifiers, and every row held a finite five-class probability vector. The probabilities summed to one within a maximum absolute deviation of 1.8 × 10−7, and the stored argmax matched the recomputed argmax in every row. Fold sizes differed by at most two images, and each of the 17,563 grouping keys belonged to exactly one fold, so paired eye-side records never crossed fold boundaries.

The four low-variance photographs flagged by the prediction-blind quality-control audit were kept in all primary analyses; Section 3.9 reports the effect of removing them.

## 3.2. Primary five-class performance

On the pooled fold-assigned predictions, the primary five-class analysis gave accuracy 0.752975, balanced accuracy 0.583907, macro-F1 0.569848, weighted-F1 0.773421, macro precision 0.568889, macro recall 0.583907, log loss 0.739464, QWK 0.736006, and macro one-vs-rest ROC-AUC 0.873664. The 95% filename-grouped bootstrap intervals were 0.747792–0.757758 for accuracy, 0.573080–0.596434 for balanced accuracy, 0.559609–0.580731 for macro-F1, 0.768507–0.777819 for weighted-F1, 0.557506–0.580687 for macro precision, 0.573080–0.596434 for macro recall, 0.726361–0.744378 for QWK, 0.869512–0.877827 for macro ROC-AUC, and 0.729363–0.750267 for log loss. Table 4 lists these estimates, and Figure 4 plots the main ones with their intervals.

**Table 4.** Primary five-class performance from pooled fold-assigned predictions

| **Metric**                | **Estimate** | **95% filename-group bootstrap interval** |
|---------------------------|--------------|-------------------------------------------|
| Accuracy                  | 0.7530       | 0.7478–0.7578                             |
| Balanced accuracy         | 0.5839       | 0.5731–0.5964                             |
| Macro-F1                  | 0.5698       | 0.5596–0.5807                             |
| Weighted-F1               | 0.7734       | 0.7685–0.7778                             |
| Macro precision           | 0.5689       | 0.5575–0.5807                             |
| Macro recall              | 0.5839       | 0.5731–0.5964                             |
| Log loss                  | 0.7395       | 0.7294–0.7503                             |
| Quadratic weighted kappa  | 0.7360       | 0.7264–0.7444                             |
| Macro one-vs-rest ROC-AUC | 0.8737       | 0.8695–0.8778                             |

Table 4 note. Intervals are 95% percentile intervals from 1000 filename-group bootstrap replicates (seed 2026) under the archived fold-assigned protocol. Because checkpoint selection used the same fold later used for prediction, they are not independent external-validation intervals.

[Figure image: see submission_figures/]

**Figure 4.** Primary point estimates and 95% filename-group bootstrap intervals for the principal performance metrics. The intervals describe the archived fold-assigned protocol and are not external-validation intervals.

## 3.3. Grade-specific performance and ordinal errors

Table 5 gives the grade-specific results. Grade 0 reached the highest F1 score, 0.8698, with precision 0.9153 and recall 0.8287. Grade 1 was clearly the hardest category, with precision 0.1907, recall 0.3606, F1 score 0.2495, and one-vs-rest ROC-AUC 0.6765. Grade 2 had F1 score 0.6121 and recall 0.6270. Grades 3 and 4 reached ROC-AUC values of 0.9469 and 0.9721, yet their argmax recalls were only 0.4903 and 0.6130, which is why we report probability ranking and ordinal agreement next to the hard decisions. Figure 5 shows the per-class ROC curves, and Figure 6 the pooled confusion counts and row-normalized recalls.

**Table 5.** Grade-specific performance in the primary analysis

| **Grade and label** | **Precision (95% filename-group bootstrap CI)** | **Recall (95% filename-group bootstrap CI)** | **F1 score (95% filename-group bootstrap CI)** | **Support** | **One-vs-rest ROC-AUC (95% filename-group bootstrap CI)** |
|---------------------|-------------------------------------------------|----------------------------------------------|------------------------------------------------|-------------|-----------------------------------------------------------|
| 0, no DR            | 0.9153 (0.9110–0.9193)                          | 0.8287 (0.8230–0.8335)                       | 0.8698 (0.8659–0.8733)                         | 25,810      | 0.8878 (0.8821–0.8925)                                    |
| 1, mild NPDR        | 0.1907 (0.1789–0.2043)                          | 0.3606 (0.3410–0.3810)                       | 0.2495 (0.2356–0.2643)                         | 2,443       | 0.6765 (0.6636–0.6894)                                    |
| 2, moderate NPDR    | 0.5978 (0.5839–0.6129)                          | 0.6270 (0.6132–0.6407)                       | 0.6121 (0.5998–0.6236)                         | 5,292       | 0.8850 (0.8792–0.8906)                                    |
| 3, severe NPDR      | 0.4417 (0.4090–0.4791)                          | 0.4903 (0.4574–0.5301)                       | 0.4647 (0.4345–0.4977)                         | 873         | 0.9469 (0.9365–0.9559)                                    |
| 4, PDR              | 0.6989 (0.6579–0.7391)                          | 0.6130 (0.5740–0.6531)                       | 0.6531 (0.6184–0.6867)                         | 708         | 0.9721 (0.9657–0.9777)                                    |

Table 5 note. NPDR, non-proliferative diabetic retinopathy; PDR, proliferative diabetic retinopathy. All intervals are 95% percentile intervals from 1000 bootstrap replicates (seed 2026) that resampled filename-derived grouping keys and retained associated eye-side records. These intervals quantify uncertainty conditional on the supplied fold assignments and do not account for checkpoint-selection optimism. Raw fold-assigned probabilities are provided for independent recalculation.

[Figure image: see submission_figures/]

**Figure 5.** Pooled one-vs-rest ROC curves generated from the five-class fold-assigned probabilities. The class-specific ROC-AUC values were 0.8878, 0.6765, 0.8850, 0.9469, and 0.9721 for grades 0–4, respectively; the unweighted macro ROC-AUC was 0.8737.

[Figure image: see submission_figures/]

**Figure 6.** Pooled five-class confusion matrices. The left panel shows raw counts and the right panel shows row-normalized recalls. Both panels were generated from the verified labels and fold-assigned argmax predictions.

In the confusion matrix, 21,388 of the 25,810 grade-0 images were predicted correctly, and the remaining grade-0 images went mostly to grade 1 or grade 2. Grade-1 images were often assigned to grade 0 or grade 2. Grade 2 was predicted correctly in 3318 cases, with most errors falling on the neighbouring grades. Most grade-3 errors went to grade 2 (364 of 873 images), and grade-4 images were most often assigned to grade 2 (155) or grade 3 (94). This pattern is another reason to rely on QWK and grade-specific recall in addition to accuracy.

## 3.4. Ordinal error geometry

The ordinal error measures summarize how far the predictions were from the reference grade. The mean absolute grade error was 0.3247 grades, and the root-mean-square grade error was 0.7046 grades. In 92.85% of photographs the prediction was within one grade, while 7.15% were off by two or more grades. The model undergraded 9.37% of photographs and overgraded 15.34%, an asymmetry that should be kept in mind when reading the referable endpoint. Table 6 collects these measures.

**Table 6.** Ordinal error geometry of the five-class decision rule

| **Ordinal measure**                                | **Estimate**  | **95% filename-group bootstrap interval** |
|----------------------------------------------------|---------------|-------------------------------------------|
| Mean absolute grade error                          | 0.3247 grades | 0.3179–0.3323                             |
| Root-mean-square grade error                       | 0.7046 grades | 0.6942–0.7156                             |
| Within-one-grade accuracy                          | 0.9285        | 0.9257–0.9317                             |
| Large ordinal error rate (absolute grade error ≥2) | 0.0715        | 0.0683–0.0743                             |
| Undergrading rate                                  | 0.0937        | 0.0901–0.0969                             |
| Overgrading rate                                   | 0.1534        | 0.1495–0.1578                             |

Table 6 note. These measures are calculated from the five-class argmax decision and preserve ordinal distance. The absolute-error threshold denotes a disagreement of at least two grades; it is not a clinical definition of a severe disease miss. Intervals are 95% percentile intervals from 1000 filename-group bootstrap replicates (seed 2026) and do not replace class-specific recall, QWK, or probability-based metrics.

## 3.5. Ordinal squared-distance decision sensitivity

As a secondary sensitivity analysis, we applied the posterior-risk decision rule ỹᵢ = arg minₖ Σⱼ (j − k)²pᵢⱼ, while keeping the probability argmax as the primary rule. The posterior-risk rule changed 12,069 of 35,126 decisions (34.36%). It raised within-one-grade accuracy from 0.9285 to 0.9752 and reduced the large ordinal error rate from 0.0715 to 0.0248, but it also lowered accuracy from 0.7530 to 0.5114, balanced accuracy from 0.5839 to 0.5391, macro-F1 from 0.5698 to 0.4943, and QWK from 0.7360 to 0.6650, and it increased the mean absolute grade error from 0.3247 to 0.5167. The paired filename-group bootstrap intervals give the uncertainty of these differences under the supplied grouping. The squared-distance rule therefore removes many large grade jumps but, on probabilities that are not well calibrated, makes exact and average grade performance worse, so the argmax rule stays primary. Table 7 and Figure 7 summarize this trade-off.

**Table 7.** Primary argmax versus posterior squared-distance decision rule

| **Metric**                                         | **Primary argmax** | **Ordinal squared-distance rule** | **Difference (risk − argmax)** | **95% paired filename-group bootstrap interval** |
|----------------------------------------------------|--------------------|-----------------------------------|--------------------------------|--------------------------------------------------|
| Accuracy                                           | 0.7530             | 0.5114                            | −0.2414                        | −0.2467 to −0.2359                               |
| Balanced accuracy                                  | 0.5839             | 0.5391                            | −0.0447                        | −0.0519 to −0.0378                               |
| Macro-F1                                           | 0.5698             | 0.4943                            | −0.0755                        | −0.0822 to −0.0687                               |
| QWK                                                | 0.7360             | 0.6650                            | −0.0709                        | −0.0763 to −0.0656                               |
| Mean absolute grade error                          | 0.3247             | 0.5167                            | +0.1919                        | +0.1858 to +0.1977                               |
| Within-one-grade accuracy                          | 0.9285             | 0.9752                            | +0.0467                        | +0.0442 to +0.0491                               |
| Large ordinal error rate (absolute grade error ≥2) | 0.0715             | 0.0248                            | −0.0467                        | −0.0491 to −0.0442                               |

Table 7 note. The ordinal squared-distance rule is a prediction-level sensitivity analysis on fixed fold-assigned probabilities. It is not a new training run, does not optimize QWK, and does not establish clinical utility. Differences use paired 95% percentile intervals from 1000 filename-group bootstrap replicates (seed 2026). The results show why a mathematically motivated decision rule should not be adopted without calibrated probabilities and a prespecified utility function.

[Figure image: see submission_figures/]

**Figure 7.** Prediction-level comparison of the primary probability-argmax rule and the posterior squared-distance decision rule. The squared-distance rule reduces the proportion of errors spanning at least two grades and increases within-one-grade accuracy, but it worsens QWK, macro-F1, and mean absolute grade error on the evaluated probabilities; the argmax rule therefore remains primary.

## 3.6. Derived referable-DR analysis

For the secondary binary analysis, grades 2–4 were grouped as referable DR. The derived endpoint reached accuracy 0.9046, balanced accuracy 0.8558, sensitivity 0.7756, specificity 0.9360, positive predictive value 0.7466, F1 score 0.7609, and ROC-AUC 0.9305. The binary confusion matrix had 26,444 true negatives, 1809 false positives, 1542 false negatives, and 5331 true positives. This endpoint was obtained from the five-class outputs and is not a separately trained or externally validated binary model. Table 8 lists its metrics, and Figure 8 visualizes it.

**Table 8.** Derived grade-≥2 referable-DR analysis

| **Metric**                       | **Estimate (95% filename-group bootstrap CI)** |
|----------------------------------|------------------------------------------------|
| Accuracy                         | 0.9046 (0.9012–0.9079)                         |
| Balanced accuracy                | 0.8558 (0.8503–0.8613)                         |
| Sensitivity                      | 0.7756 (0.7650–0.7872)                         |
| Specificity                      | 0.9360 (0.9326–0.9391)                         |
| Positive predictive value        | 0.7466 (0.7346–0.7580)                         |
| F1 score                         | 0.7609 (0.7518–0.7693)                         |
| ROC-AUC                          | 0.9305 (0.9261–0.9346)                         |
| True negatives / false positives | 26,444 / 1,809                                 |
| False negatives / true positives | 1,542 / 5,331                                  |

Table 8 note. The endpoint was derived after five-class inference and was not a separately trained binary classifier. Intervals are 95% percentile intervals from 1000 filename-group bootstrap replicates (seed 2026), retaining both eye-side records for each sampled group. Predictive values are prevalence-dependent and should not be interpreted as clinical screening performance.

[Figure image: see submission_figures/]

**Figure 8.** Derived grade-≥2 referable-DR analysis. The left panel shows the ROC curve from the summed grade-2-to-grade-4 probability. The right panel shows the argmax-derived binary confusion matrix. No operating threshold was optimized on an independent clinical cohort.

## 3.7. Probability calibration

For the primary uncalibrated baseline, the 15-bin maximum-confidence ECE was 0.1415. Since the checkpoint-selection fold overlapped with the prediction fold, we treat this value as descriptive, and the probabilities should not be read as calibrated clinical risks. Figure 9 shows the 15-bin reliability diagram.

[Figure image: see submission_figures/]

**Figure 9.** Descriptive multiclass confidence calibration for the primary uncalibrated baseline using the maximum predicted class probability and 15 fixed bins. The ECE was 0.1415. The secondary ordinal calibration analysis is reported in Section 3.8.

## 3.8. Five-meta-fold group-disjoint ordinal isotonic calibration

The secondary calibration analysis was evaluated only through the five group-disjoint meta-folds. Compared with the raw fold-assigned probabilities, ordinal isotonic post-processing lowered the descriptive ECE from 0.1415 to 0.0103, the multiclass Brier score from 0.4089 to 0.2594, and the mean absolute grade error from 0.3247 to 0.2391 grades, and it raised within-one-grade accuracy from 0.9285 to 0.9344 and QWK from 0.7360 to 0.7637. The gains were not uniform, however: balanced accuracy fell from 0.5839 to 0.5211, macro-F1 moved from 0.5698 to 0.5697, and macro ROC-AUC from 0.8737 to 0.8722. With five alternative meta-fold assignments (seeds 2026–2030), the ECE reduction ranged from 0.1311 to 0.1319, the QWK gain from 0.0277 to 0.0295, and the balanced-accuracy change from −0.0629 to −0.0602; these repetitions reuse the same base predictions and are sensitivity checks, not independent replications. The reason for the mixed picture is a shift in the decisions: calibration increased grade-0 decisions from 23,367 to 28,653 and reduced grade-1 decisions from 4619 to 1268, which lifts accuracy in this grade-0-dominated cohort while lowering minority-grade recall. Table 9 gives the full comparison, and Table 10 the paired calibrated-minus-raw differences.

**Table 9.** Raw probabilities versus five-meta-fold group-disjoint ordinal isotonic calibration

| **Metric**                                         | **Raw fold-assigned probabilities** | **Group-disjoint ordinal isotonic probabilities** |
|----------------------------------------------------|-------------------------------------|---------------------------------------------------|
| Accuracy                                           | 0.7530                              | 0.8332                                            |
| Balanced accuracy                                  | 0.5839                              | 0.5211                                            |
| Macro-F1                                           | 0.5698                              | 0.5697                                            |
| QWK                                                | 0.7360                              | 0.7637                                            |
| Log loss                                           | 0.7395                              | 0.7546                                            |
| Macro ROC-AUC                                      | 0.8737                              | 0.8722                                            |
| Multiclass Brier score                             | 0.4089                              | 0.2594                                            |
| ECE, 15 bins                                       | 0.1415                              | 0.0103                                            |
| Mean absolute grade error                          | 0.3247                              | 0.2391                                            |
| Within-one-grade accuracy                          | 0.9285                              | 0.9344                                            |
| Large ordinal error rate (absolute grade error ≥2) | 0.0715                              | 0.0656                                            |
| Grade-≥2 ROC-AUC                                   | 0.9305                              | 0.9331                                            |

Table 9 note. The calibrated column is a secondary group-disjoint post-processing analysis of fixed fold-assigned probabilities using calibration meta-fold seed 2026. It is not a new EfficientNet training run, external validation, or a replacement for the primary raw-probability result. Accuracy and balanced accuracy move in different directions because the calibration layer changes the argmax decision distribution under class imbalance.

In the paired filename-group bootstrap, calibration increased QWK by 0.027738 (95% CI 0.020664–0.034500), reduced the multiclass Brier score by 0.149501 (95% CI −0.153576 to −0.145182), reduced ECE by 0.131122 (95% CI −0.137226 to −0.123600), reduced the mean absolute grade error by 0.085549 grades (95% CI −0.091644 to −0.079512), and increased within-one-grade accuracy by 0.005865 (95% CI 0.003132–0.008569). Balanced accuracy decreased by 0.062794 (95% CI −0.070710 to −0.055914). The intervals for macro-F1 (−0.008082 to 0.006997) and macro ROC-AUC (−0.004039 to 0.001135) both included zero. Calibration thus improved the probability and ordinal summaries under the fixed-prediction protocol, but it did not improve classification performance across the board.

**Table 10.** Paired filename-group bootstrap differences after group-disjoint ordinal calibration

| **Metric difference (calibrated − raw)**           | **Estimate** | **95% filename-group bootstrap interval** |
|----------------------------------------------------|--------------|-------------------------------------------|
| Accuracy                                           | +0.080225    | +0.075927 to +0.084412                    |
| Balanced accuracy                                  | −0.062794    | −0.070710 to −0.055914                    |
| Macro-F1                                           | −0.000113    | −0.008082 to +0.006997                    |
| QWK                                                | +0.027738    | +0.020664 to +0.034500                    |
| Log loss                                           | +0.015093    | −0.011570 to +0.044787                    |
| Macro ROC-AUC                                      | −0.001453    | −0.004039 to +0.001135                    |
| Multiclass Brier score                             | −0.149501    | −0.153576 to −0.145182                    |
| ECE, 15 bins                                       | −0.131122    | −0.137226 to −0.123600                    |
| Mean absolute grade error                          | −0.085549    | −0.091644 to −0.079512                    |
| Within-one-grade accuracy                          | +0.005865    | +0.003132 to +0.008569                    |
| Large ordinal error rate (absolute grade error ≥2) | −0.005865    | −0.008569 to −0.003132                    |
| Referable-DR ROC-AUC                               | +0.002581    | +0.001940 to +0.003183                    |

Table 10 note. The paired intervals preserve image-level pairing and resample filename-derived groups using 1000 replicates (seed 2026); the grouping key is not independently verified as a clinical patient identifier. These are percentile uncertainty intervals for effect differences, not hypothesis-test p-values. Positive differences indicate higher values after calibration; for error measures (log loss, Brier score, ECE, mean absolute grade error, and large ordinal error rate), negative differences indicate lower values. Isotonic calibration produces many tied probability values, so the calibrated macro ROC-AUC depends at the fourth decimal on how floating-point near-ties are resolved (0.8722 when near-ties are treated as ties; 0.8724 otherwise); the corresponding paired interval includes zero under either convention.

## 3.9. Quality-control sensitivity analysis

For the sensitivity analysis, the four low-variance images flagged without reference to model outputs were removed. Accuracy changed from 0.752975 to 0.753004, balanced accuracy from 0.583907 to 0.583815, macro-F1 from 0.569848 to 0.569784, QWK from 0.736006 to 0.736029, macro ROC-AUC from 0.873664 to 0.873619, and log loss from 0.739464 to 0.739421. In 1000 paired filename-group bootstrap replicates, the 95% intervals of all six differences included zero. We keep the full-data estimate as primary because these images belong to the analyzed cohort, and their inclusion does not materially change the point estimates. The flagged identifiers, the replicate-level differences, and the script that reproduces these values are included in Supplementary File S1.

# 4. Discussion

## 4.1. Principal findings

In this study, a single EfficientNet-B4 classifier was evaluated for five-class DR grading on 35,126 Kaggle photographs, with the two eye-side records of each filename-derived group kept in the same fold. The primary analysis gave accuracy 0.7530, balanced accuracy 0.5839, macro-F1 0.5698, QWK 0.7360, log loss 0.7395, and macro one-vs-rest ROC-AUC 0.8737, and the derived grade-≥2 referable endpoint gave accuracy 0.9046, F1 score 0.7609, ROC-AUC 0.9305, sensitivity 0.7756, and specificity 0.9360. Removing four very low-variance photographs made almost no difference. These numbers describe performance under the evaluated fold-assignment and checkpoint-selection protocol, and without an independent cohort they cannot be generalized beyond this benchmark.

The metrics are best read together rather than as one accuracy figure. Raw accuracy is pushed up by grade 0, which makes up 73.48% of the analyzed images, whereas balanced accuracy and macro-F1 give each grade the same weight and are therefore lower. QWK adds the distance between grades, so a near-neighbour error is separated from a prediction that skips several severity levels. For an imbalanced ordinal task like this one, macro-F1, QWK, grade-specific recall, log loss, and macro ROC-AUC together say much more than accuracy.

The ordinal error analysis gives a direct, distance-based view: the mean absolute error was 0.3247 grades, 92.85% of decisions were within one grade, and 7.15% were off by at least two grades. These figures do not replace clinical adjudication, but they make the error structure visible and avoid treating every five-class mistake as equal. Overgrading was more common than undergrading in the archived predictions. Such an asymmetry may be acceptable in some screening settings, but retrospective data of this kind cannot determine the right clinical trade-off or threshold.

## 4.2. Posterior squared-distance decision analysis

We see the squared-distance decision rule as a mathematical stress test, not as a way to improve performance. For a true posterior distribution, minimizing the expected squared grade distance is the same as choosing the integer grade closest to the posterior mean. On the archived raw probabilities, the rule changed 34.36% of decisions, reduced the large ordinal error rate from 7.15% to 2.48%, and increased within-one-grade accuracy from 92.85% to 97.52%. At the same time, accuracy dropped to 51.14%, macro-F1 to 49.43%, and QWK to 0.6650, and the mean absolute grade error rose to 0.5167 grades, with all paired group-bootstrap intervals for these changes excluding zero.

This result matters for two reasons. It shows, first, that an ordinal loss that is mathematically sound does not automatically improve every performance objective, and second, that a softmax vector cannot be treated as a calibrated posterior just because its entries lie on a simplex. For these reasons we did not adopt the posterior-risk rule as the primary decision. Testing such a rule on an independently calibrated model under a prespecified clinical utility function would be a natural next step, but the data for that experiment were not available here.

## 4.3. Class-specific behavior and ordinal interpretation

Grade 0 had the highest F1 score and precision, although its recall was 0.8287 and not perfect. Because grade 0 dominates the cohort, this result drives much of the pooled accuracy, but it does not mean that the classifier can reliably rule out disease in every setting. Grade 1 was the hardest category (F1 score 0.2495, ROC-AUC 0.6765), and the confusion matrix shows frequent movement between grade 1 and its neighbours. This is consistent with the difficulty of recognizing subtle or borderline findings, but our data cannot tell whether the errors come from lesion visibility, label noise, image quality, or grader disagreement.

For grades 3 and 4, the probability ranking was much better than the argmax recall suggests: the one-vs-rest ROC-AUC values were 0.9469 and 0.9721, while the recalls were 0.4903 and 0.6130. The probabilities therefore carry class-separation information that the fixed argmax decision does not fully use. This does not justify tuning thresholds after seeing the evaluation outputs, nor does it show that another operating point would be clinically appropriate. In future work, clinically motivated thresholds should be fixed in advance on an independent calibration set and then tested on a separate external cohort.

## 4.4. Relation to previous DR studies

Our results should be read alongside, and not ranked against, studies with different endpoints. Ting et al. \[4\] and Abràmoff et al. \[9\] evaluated binary referral outcomes, and Ipp et al. \[10\] tested a referral system prospectively in clinical practice. These studies give important context for screening research, but they do not measure five-class severity grading. Segmentation studies, likewise, optimize spatial overlap and not image-level grade prediction, so their metrics describe a different task.

Zhang et al. \[11\] is closer to our work because it included a DR severity-grading task, although with a separately assembled clinical cohort and a different train/validation/test design. Any numerical comparison would have to account for differences in model capacity, preprocessing, augmentation, partitioning, checkpoint selection, and metric definitions, so our analysis is a protocol-specific benchmark and not a superiority claim. What it adds is the release of raw fold-assigned probabilities, an explicit grouping key, defined performance estimands, filename-group bootstrap intervals, and objective sensitivity analyses.

Two other studies are closer in endpoint or data source. Krause et al. \[13\] reported five-class agreement with an adjudicated reference standard (QWK 0.84) using higher-resolution inputs and a different development cohort. Voets et al. \[14\] reproduced a published referable-DR pipeline on the Kaggle EyePACS data and obtained ROC-AUC 0.951 on held-out Kaggle images but only 0.853 on Messidor-2. Our QWK (0.7360) and derived referable ROC-AUC (0.9305) are lower, but the reference standards, referable definitions, input resolutions, and partition designs differ, and the drop seen by Voets et al. on Messidor-2 shows how much performance can change under dataset shift. Taken together, these comparisons give a plausible performance range for the task, not a ranking.

A comparison with the literature is useful only when the task and protocol context is preserved; validation accuracy, validation loss, ROC-AUC, Dice, and QWK describe different properties and are not interchangeable measures of model quality.

## 4.5. Interpretation of the computational design

EfficientNet-B4 offers a transfer-learning baseline built on ImageNet initialization and a scaled convolutional architecture. In the documented training configuration, focal loss is combined with inverse-frequency sampling. The two mechanisms act on different parts of the problem: focal loss changes how much difficult examples contribute, whereas weighted sampling changes how often minority-class images are seen. Neither guarantees balanced performance or calibrated probabilities, and the grade-specific results confirm that the less frequent grades are still recognized imperfectly.

The model used 512 × 512 inputs, and our analysis does not show that this resolution is optimal; preprocessing and augmentation may affect how visible small lesions are and how the probabilities are distributed. Since the training histories are incomplete, the results also cannot be linked to a particular epoch, learning-rate trajectory, or training event.

Group-aware fold assignment lowers the chance that the model profits from acquisition characteristics shared between training and evaluation images. The grouping key, however, comes from filenames and was not checked against a clinical registry, and macro-F1 on each prediction fold was used for checkpoint selection. The estimates may therefore include selection optimism and are not fully nested estimates of generalization.

## 4.6. Referable-DR interpretation and calibration

The derived grade-≥2 endpoint was an easier task than five-class grading, with accuracy 0.9046, F1 score 0.7609, ROC-AUC 0.9305, and specificity 0.9360. Sensitivity was 0.7756, which corresponds to 1542 false negatives in the pooled analysis, and this number alone rules out describing the model as an autonomous rule-out system. The positive predictive value was 0.7466 here, but predictive values depend strongly on prevalence, and a different population, referral policy, or camera mix could give quite different operating characteristics.

An ECE of 0.1415 means that the raw maximum-confidence probabilities are not clinically calibrated. No independent calibration cohort was available, and the five-meta-fold group-disjoint calibration analysis post-processes fixed predictions without establishing external or clinical calibration. The probability outputs should therefore not be used as validated individual risk estimates; that would require independent calibration, subgroup assessment, decision-curve analysis, and prespecified operating points.

In the ordinal calibration analysis, the calibration maps were never fitted on the groups to which they were applied. The paired filename-group bootstrap intervals support lower ECE, multiclass Brier score, and mean absolute grade error, and higher within-one-grade accuracy and QWK, and the direction of the ECE, QWK, and balanced-accuracy changes held across five alternative meta-fold assignments of the same EfficientNet predictions. Balanced accuracy decreased, while the intervals for macro-F1 and macro ROC-AUC included zero. Calibration therefore reshapes the decision distribution: better probability and ordinal-distance summaries do not translate into uniformly better class-balanced classification. We report this as a secondary sensitivity analysis and not as evidence of clinical calibration or of superiority over the raw classifier.

## 4.7. Reproducibility and methodological contribution

The analysis package includes the analyzed manifest, fold assignments, raw fold-assigned probabilities, metric files, figures, scripts, and checksums, so the principal prediction metrics and figures can be recomputed independently. It does not allow every training event to be reproduced, because a complete software lockfile and complete epoch-wise histories are not available.

The contribution of this work is a retrospective benchmark with a defined five-class target, an explicit filename-derived grouping key, released raw probabilities, ordinal and class-balanced metrics, filename-group bootstrap uncertainty, descriptive calibration, and an objective quality-control sensitivity analysis. It is an empirical evaluation and not a claim of architectural novelty.

For a mathematics-oriented readership, the methodological contribution lies in the explicit definition of a group-conditional ordinal evaluation functional, its filename-group bootstrap uncertainty estimator, and the comparison of decision rules under different ordinal objectives. Proposition 1 characterizes the decision that minimizes expected squared grade distance, and Proposition 2 proves simplex validity for the cumulative-probability reconstruction. The sensitivity analysis then shows that a rule minimizing expected squared grade error does not necessarily improve accuracy, QWK, or macro-F1 when it is applied to the evaluated probability vectors. These results form an empirical contribution to ordinal evaluation and decision analysis, without claiming a new architecture or an asymptotic theorem.

# 5. Limitations

First, only one public Kaggle cohort was used, and no independent external validation cohort was available. Differences in cameras, image quality, population, grading practice, disease prevalence, and referral pathways may lower performance on other data, so the results are retrospective benchmark estimates and not evidence of clinical utility.

Second, the grouping variable was taken from the numerical part of the filename. The audit confirmed that paired eye-side records did not cross fold boundaries, but this key cannot verify patient identity or capture every clinical dependency. Where a verified patient or examination identifier exists, future studies should use it.

Third, the fold-assigned protocol was not fully nested. The training code chose the best checkpoint by macro-F1 on the same fold that later produced the fold-assigned predictions, and the archive contains no independent inner validation partition, which may introduce checkpoint-selection optimism. Nested group-aware cross-validation, or a prespecified training schedule with an untouched outer fold, would give a more conservative design.

Fourth, the archive lacks complete epoch-wise training histories and a complete machine-readable environment lockfile, and it does not contain enough evidence to verify the exact stopping epoch, the GPU runtime, or every software version. The raw probabilities and manifests allow the principal prediction metrics to be recomputed, but the training trajectory itself cannot be fully verified.

Fifth, the five-class labels come from a public competition dataset and may contain grading variability, especially near the boundaries between adjacent grades. No independent adjudication or clinician-reader comparison was carried out, so we cannot estimate inter-grader agreement or separate model error from uncertainty in the reference labels.

Sixth, the strong class imbalance remains a central limitation, with far fewer images in grades 1, 3, and 4 than in grade 0. Weighted sampling and focal loss can help minority-class learning, but they also change the effective training distribution and may affect calibration. The grade-1 F1 score and the recalls for grades 3 and 4 show that the imbalance problem has not been fully solved.

Seventh, the prediction-blind quality-control audit flagged four very low-variance images, but no clinical gradability assessment was available. The primary analysis kept them and the sensitivity analysis showed negligible changes, which still does not prove that all images were clinically gradable or that the dataset is free of other quality artifacts.

Eighth, demographic, camera, site, and clinical covariates were not available, so subgroup performance, fairness, site shift, and interactions with systemic disease could not be studied. The model uses retinal photographs only and does not include HbA1c, diabetes duration, blood pressure, renal function, or other clinical data.

Finally, lesion-level explanations and clinician interpretation were not evaluated, so the findings say nothing about explanation quality or lesion localization. The referable-DR endpoint was derived from the five-class outputs and not from a separately trained or independently calibrated screening model, and the five-meta-fold group-disjoint calibration was a secondary analysis of fixed predictions that does not establish external calibration or patient-level risk validity. Before any use in patient care, prospective clinical evaluation, independent calibration, external testing, subgroup analysis, and prespecified operating points would be needed.

# 6. Conclusions

We evaluated EfficientNet-B4 for five-class DR grading on 35,126 Kaggle photographs organized into 17,563 filename-derived groups. Under the supplied five-fold assignment, the primary analysis reached accuracy 0.7530, balanced accuracy 0.5839, macro-F1 0.5698, QWK 0.7360, log loss 0.7395, and macro one-vs-rest ROC-AUC 0.8737, and the derived grade-≥2 referable endpoint reached accuracy 0.9046, F1 score 0.7609, ROC-AUC 0.9305, sensitivity 0.7756, and specificity 0.9360. The posterior squared-distance decision reduced errors of two or more grades but worsened accuracy, QWK, macro-F1, and mean absolute grade error, so the argmax rule was kept as primary. Removing four images flagged by a prediction-blind quality rule changed the results only negligibly, with all paired filename-group bootstrap intervals for the six primary metric differences including zero.

Overall, this work gives a prediction-level assessment of ordinal performance, calibration trade-offs, and decision-rule sensitivity, with explicit grouping and uncertainty procedures. Since checkpoint selection used the same fold that later produced the predictions, the estimates may be optimistic, and no independent external validation or calibration cohort was available. The results do not establish clinical readiness, explainability, fairness, or suitability for autonomous screening. Future evaluations should use untouched outer test folds, complete training records, external data, and prespecified clinical operating points.

# Author Contributions

Conceptualization, A.S.; methodology, A.S., M.S. and C.L.; software, A.S. and M.S.; validation, A.S., M.S. and C.L.; formal analysis, A.S., M.A., M.S., C.L., S.A. and M.M.R.; investigation, A.S.; resources, A.S., M.S. and C.L.; data curation, A.S. and M.S.; writing—original draft preparation, A.S. and M.A.; writing—review and editing, A.S., M.A., M.S., C.L., M.O., S.A. and M.M.R.; visualization, A.S. and M.O.; supervision, M.S. and C.L.; project administration, A.S. All authors have read and agreed to the published version of the manuscript.

# Funding

This research received no external funding.

# Institutional Review Board Statement

Not applicable. This study used the publicly available, de-identified Kaggle Diabetic Retinopathy Detection dataset. Because the dataset contains only anonymized retinal fundus photographs with no identifiable personal health information, formal ethics approval was not required for this secondary data analysis. The images were originally collected by EyePACS under the ethical guidelines and protocols of the contributing institutions. All data use complied with the dataset terms of use.

# Informed Consent Statement

Not applicable. This study used only publicly available, de-identified data. Any informed consent procedures for data collection were conducted by the source institutions at the time of image acquisition. The analyzed images are anonymized and contain no identifiable personal health information.

# Data Availability Statement

The Kaggle Diabetic Retinopathy Detection images are third-party data available through the official competition page (https://www.kaggle.com/competitions/diabetic-retinopathy-detection) under the applicable Kaggle terms. The prediction-level reproducibility archive is supplied as Supplementary File S1 and is also available at https://github.com/ahmadfshalaldeh-bot/drxai. It contains the analyzed manifest, fold assignments, raw fold-assigned probabilities, metric files, source scripts, figures, calibration outputs, and validation reports. It does not contain original-resolution Kaggle JPEGs, full competition archives, or model checkpoints. The manuscript and figure package include a representative-image panel derived from a small number of dataset photographs; dissemination of these illustrative examples remains subject to the applicable Kaggle terms. The archive does not contain a complete software lockfile or complete epoch-wise training histories; these limitations are declared in the manuscript.

# Conflicts of Interest

The authors declare no conflicts of interest.

# Acknowledgments

The authors acknowledge Kaggle and the competition data providers for making the labeled image cohort available, and Al-Ahliyya Amman University for computational infrastructure and technical resources.

# Supplementary Materials

Supplementary File S1: Prediction-level reproducibility archive, including raw fold-assigned probabilities, fold assignments, analysis scripts, reconstructed metrics, calibration analyses, figures, and validation reports. Original-resolution Kaggle JPEGs, full competition archives, and non-redistributable checkpoints are not included.

# GenAI disclosure

During the preparation of this manuscript, Google Gemini and Manus AI were used to assist with drafting, language editing, document formatting, code assistance, and consistency checking. Claude (Anthropic; Claude Opus 5.5) was used to recompute the reported metrics independently from the archived probabilities, to check agreement between the manuscript and the source outputs, and for language editing and formatting of the revised manuscript. These tools did not generate dataset labels, model predictions, bootstrap samples, or performance values. The authors reviewed and edited the output and remain responsible for the scientific accuracy, authorship, declarations, and final submitted content.

# References

1\. Zhou, J.; Chen, B. Retinal Cell Damage in Diabetic Retinopathy. Cells 2023, 12, 1342. https://doi.org/10.3390/cells12091342.

2\. Ansari, P.; Tabasumma, N.; Snigdha, N.N.; Siam, N.H.; Panduru, R.V.N.R.S.; Azam, S.; Hannan, J.M.A.; Abdel-Wahab, Y.H.A. Diabetic Retinopathy: An Overview on Mechanisms, Pathophysiology and Pharmacotherapy. Diabetology 2022, 3, 159–175. https://doi.org/10.3390/diabetology3010011.

3\. Tsiknakis, N.; Theodoropoulos, D.; Manikis, G.; Ktistakis, E.; Boutsora, O.; Berto, A.; Scarpa, F.; Scarpa, A.; Fotiadis, D.I.; Marias, K. Deep Learning for Diabetic Retinopathy Detection and Classification Based on Fundus Images: A Review. Comput. Biol. Med. 2021, 135, 104599. https://doi.org/10.1016/j.compbiomed.2021.104599.

4\. Ting, D.S.W.; Cheung, C.Y.-L.; Lim, G.; Tan, G.S.W.; Quang, N.D.; Gan, A.; Hamzah, H.; Garcia-Franco, R.; San Yeo, I.Y.; Lee, S.Y.; et al. Development and Validation of a Deep Learning System for Diabetic Retinopathy and Related Eye Diseases Using Retinal Images From Multiethnic Populations With Diabetes. JAMA 2017, 318, 2211–2223. https://doi.org/10.1001/jama.2017.18152.

5\. Cukierski, W. Diabetic Retinopathy Detection. Kaggle, 2015. Available online: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data (accessed on 29 October 2026).

6\. Elmoufidi, A.; Ammoun, H. Diabetic Retinopathy Prevention Using EfficientNetB3 Architecture and Fundus Photography. SN Comput. Sci. 2022, 4, 9. https://doi.org/10.1007/s42979-022-01482-6.

7\. Tan, M.; Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In Proceedings of the 36th International Conference on Machine Learning, Long Beach, CA, USA, 9–15 June 2019; PMLR 97, pp. 6105–6114.

8\. Gulshan, V.; Peng, L.; Coram, M.; Stumpe, M.C.; Wu, D.; Narayanaswamy, A.; Venugopalan, S.; Widner, K.; Madams, T.; Cuadros, J.; et al. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. JAMA 2016, 316, 2402–2410. https://doi.org/10.1001/jama.2016.17216.

9\. Abràmoff, M.D.; Lou, Y.; Erginay, A.; Clarida, W.; Amelon, R.; Folk, J.C.; Niemeijer, M. Improved Automated Detection of Diabetic Retinopathy on a Publicly Available Dataset Through Integration of Deep Learning. Investig. Ophthalmol. Vis. Sci. 2016, 57, 5200–5206. https://doi.org/10.1167/iovs.16-19964.

10\. Ipp, E.; Liljenquist, D.; Bode, B.; Shah, V.N.; Silverstein, S.; Regillo, C.D.; Lim, J.I.; Sadda, S.; Domalpally, A.; Gray, G.; et al. Pivotal Evaluation of an Artificial Intelligence System for Autonomous Detection of Referrable and Vision-Threatening Diabetic Retinopathy. JAMA Netw. Open 2021, 4, e2134254. https://doi.org/10.1001/jamanetworkopen.2021.34254.

11\. Zhang, W.; Zhong, J.; Yang, S.; Gao, Z.; Hu, J.; Chen, Y.; Yi, Z. Automated Identification and Grading System of Diabetic Retinopathy Using Deep Neural Networks. Knowl.-Based Syst. 2019, 175, 12–25. https://doi.org/10.1016/j.knosys.2019.03.016.

12\. Kusakunniran, W.; Karnjanapreechakorn, S.; Choopong, P.; Siriapisith, T.; Tesavibul, N.; Phasukkijwatana, N.; Prakhunhungsit, S.; Boonsopon, S. Detecting and Staging Diabetic Retinopathy in Retinal Images Using Multi-Branch CNN. Appl. Comput. Inform. 2026, 22, 210–223. https://doi.org/10.1108/ACI-06-2022-0150.

13\. Krause, J.; Gulshan, V.; Rahimy, E.; Karth, P.; Widner, K.; Corrado, G.S.; Peng, L.; Webster, D.R. Grader Variability and the Importance of Reference Standards for Evaluating Machine Learning Models for Diabetic Retinopathy. Ophthalmology 2018, 125, 1264–1272. https://doi.org/10.1016/j.ophtha.2018.01.034.

14\. Voets, M.; Møllersen, K.; Bongo, L.A. Reproduction Study Using Public Data of: Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. PLoS ONE 2019, 14, e0217541. https://doi.org/10.1371/journal.pone.0217541.

15\. Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; Dollár, P. Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision, Venice, Italy, 22–29 October 2017; pp. 2999–3007. https://doi.org/10.1109/ICCV.2017.324.

16\. Efron, B.; Tibshirani, R.J. An Introduction to the Bootstrap; Chapman & Hall: New York, NY, USA, 1993.

17\. Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K.Q. On Calibration of Modern Neural Networks. In Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, 6–11 August 2017; PMLR 70, pp. 1321–1330.

18\. Zadrozny, B.; Elkan, C. Transforming Classifier Scores into Accurate Multiclass Probability Estimates. In Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Edmonton, AB, Canada, 23–26 July 2002; pp. 694–699. https://doi.org/10.1145/775047.775151.
