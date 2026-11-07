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

Five-class grading of diabetic retinopathy (DR) from fundus photographs is an imbalanced ordinal problem, so accuracy alone is an incomplete summary. We evaluated an ImageNet-initialized EfficientNet-B4 (512 × 512 input) on 35,126 labeled Kaggle photographs, using five folds that kept the two eye-side images of each of 17,563 filename-derived groups together. Pooled fold-assigned predictions gave accuracy 0.7530, balanced accuracy 0.5839, macro-F1 0.5698, quadratic weighted kappa (QWK) 0.7360, and macro one-vs-rest ROC-AUC 0.8737 (95% filename-group bootstrap interval 0.8695–0.8778); 92.85% of predictions were within one grade. A derived grade-≥2 referable endpoint, not separately trained, reached ROC-AUC 0.9305, sensitivity 0.7756, and specificity 0.9360. The decision minimizing expected squared grade distance is shown to be the integer nearest the posterior mean grade; applied to the raw probabilities, it reduced errors of two or more grades from 0.0715 to 0.0248 but lowered QWK to 0.6650. Group-disjoint ordinal isotonic calibration reduced expected calibration error from 0.1415 to 0.0103 and raised QWK to 0.7637, but lowered balanced accuracy to 0.5211. Because the prediction fold also guided checkpoint selection, estimates may be optimistic; external validation is required before clinical use.

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

Diabetic retinopathy (DR) is a microvascular complication of diabetes and an important cause of preventable visual impairment. Retinal changes may include microaneurysms, haemorrhages, hard exudates, cotton-wool spots, venous abnormalities, and proliferative vascular changes \[1,2\]. Because the clinical severity scale is ordinal, a prediction that differs by one grade is not equivalent to an error that skips several grades. Automated systems should therefore be assessed with class-specific, ordinal, and probability-based measures rather than accuracy alone.

Manual assessment of retinal photographs requires trained graders and may be difficult to scale. Deep learning has been widely studied for retinal-image analysis, including binary referral and severity grading \[3,4\]. The Kaggle Diabetic Retinopathy Detection competition provides an established five-grade image classification task \[5\]. Transfer learning can provide a practical initialization for image classification, and EfficientNet-based models have been investigated for retinal analysis \[6,7\]. Nevertheless, strong performance on a public dataset does not by itself establish generalization to other cameras, populations, grading practices, or clinical workflows.

Three methodological issues are particularly relevant to five-class DR grading. First, the Kaggle cohort is strongly imbalanced, with no-DR photographs forming the majority of the analytical sample. Accuracy can therefore remain relatively high while performance on mild, severe, or proliferative disease remains limited. Second, the two eyes associated with one subject may share acquisition characteristics and disease-related information. If those records are separated across training and evaluation partitions, performance can be optimistic. Third, reported metrics are difficult to compare when studies use different class definitions, datasets, preprocessing pipelines, split units, and endpoints. For example, binary referable-DR AUC, five-class macro-AUC, segmentation Dice, validation accuracy, and quadratic weighted kappa measure different properties and should not be placed on a single performance ranking.

Table 1 provides a contextual evidence map rather than a leaderboard. It includes binary referral studies, a prospective clinical evaluation, a separately assembled clinical grading cohort, five-class grading against an adjudicated reference standard, and work using the Kaggle data for five-class grading and referable-DR detection. Differences in cohort, architecture, preprocessing, partitioning, checkpoint selection, and metric definitions preclude a direct superiority claim. The present benchmark reports its data source, grouping unit, metric definitions, and uncertainty procedure so that its findings can be interpreted within this methodological context.

We evaluate ImageNet-initialized EfficientNet-B4 for five-class grading of 35,126 Kaggle photographs using filename-derived groups to keep paired eye-side images together. The study asks how the classifier performs across nominal, class-balanced, ordinal, and probabilistic measures; whether its five-class probabilities support a secondary grade-≥2 referable-DR endpoint; and how ordinal calibration, squared-distance decisions, and prediction-blind image-quality filtering affect interpretation. Its contribution is a group-aware ordinal evaluation of a single transfer-learning model, with explicit uncertainty estimates and direct analysis of trade-offs among calibration, class-balanced performance, and grade-distance error. The Results present the primary five-class benchmark first, followed by secondary analyses of its probability outputs and decision rules. The complete analysis workflow is summarized in Figure 1.

## Literature comparison and study positioning

As summarized in Table 1, previous DR studies differ in dataset composition, label definition, clinical endpoint, architecture, image resolution, partitioning strategy, and metric calculation. These differences make direct ranking inappropriate. In particular, validation accuracy and validation loss cannot be compared across datasets with different class counts, class prevalences, loss functions, and split designs. An encoder–decoder segmentation model such as U-Net should not be ranked against an image-level multiclass grader, and a binary referral AUC should not be presented as evidence of five-class severity-grading performance. The literature instead motivates a protocol-specific evaluation that reports class-balanced metrics, an ordinal measure, probability-based performance, leakage control, and uncertainty. Accordingly, the present study is positioned as a reproducible EfficientNet-B4 benchmark rather than as proof of architectural superiority over all previous DR systems.

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

We conducted a retrospective computational evaluation of five-class diabetic retinopathy grading using the Kaggle Diabetic Retinopathy Detection cohort \[5\]. The analysis used publicly available, de-identified retinal photographs; no new participants were recruited, and no new clinical grades or lesion annotations were generated. We evaluated one ImageNet-initialized EfficientNet-B4 model with a five-class linear classifier. The study was designed as a group-aware benchmark, not a clinical validation study.

## 2.2. Dataset, grouping variable, and labels

The Kaggle label manifest contained 35,126 photographs. Each image identifier comprised a numerical prefix and an eye-side suffix, such as \_left or \_right. We used the numerical prefix as a filename-derived grouping key; it was not treated as a clinically verified patient identifier. The analyzed manifest contained 17,563 unique grouping keys, with two eye-side records per key.

Labels were coded from 0 to 4: grade 0, no apparent DR; grade 1, mild NPDR; grade 2, moderate NPDR; grade 3, severe NPDR; and grade 4, proliferative DR. The class counts were 25,810, 2443, 5292, 873, and 708, respectively. These counts sum to 35,126. The class distribution is shown in Table 2 and Figure 2.

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

We audited the analyzed manifest for duplicate image identifiers, missing labels, fold-to-prediction agreement, file existence and decodability, image dimensions, finite pixel values, grayscale variance, and exact image hashes. The audit found no missing or unreadable images, invalid dimensions, or exact cross-group duplicates. Four photographs had pixel-intensity standard deviations below 2.0 and were flagged as blank or nearly blank by an objective image-quality rule (image identifiers 1986_left, 34689_left, 43457_left, and 32253_right; listed in Supplementary File S1). We retained them in the primary analysis and excluded them only in a separate sensitivity analysis; exclusion was based on image properties, not model probabilities, predictions, losses, or confidence values. Representative photographs from the five grades are shown in Figure 3.

[Figure image: see submission_figures/]

**Figure 3.** Representative Kaggle fundus photographs with grade labels from the analyzed label manifest. The images provide visual context and are not a separate validation set.

## 2.4. Fold assignment and leakage control

The fold-assignment manifest allocated all 35,126 images to five folds and recorded each image’s filename-derived grouping key and eye side. Fold sizes were 7,026, 7,024, 7,026, 7,024, and 7,026 images. No grouping key appeared in more than one fold, so paired eye-side records remained together.

For each fold, the other four folds supplied the model-training images. The held-out fold was used to monitor macro-F1, which drove learning-rate reduction (ReduceLROnPlateau), early stopping, and checkpoint selection, and the selected checkpoint then generated the predictions for that fold. Thus, although model weights were fitted using images from the other four folds, labels in the prediction fold informed the learning-rate schedule, early stopping, and checkpoint selection. We therefore report fold-assigned benchmark estimates under the implemented protocol; they are not fully nested estimates from an untouched outer test fold, and may be optimistic. Grouping paired eye-side images addresses one source of leakage but does not remove this checkpoint-selection bias. Stochastic augmentation was applied only to training images, and inference transforms were deterministic. Predictions from the five fold-specific checkpoints were concatenated for pooled metric calculation.

## 2.5. Preprocessing and augmentation

Each JPEG was read with OpenCV and converted from BGR to RGB. Near-black borders were cropped using a grayscale-intensity tolerance of 7, and the image was resized to 512 × 512 pixels. Illumination was normalized by combining the resized image with a Gaussian-blurred version using the implemented weighted transform, I′ = 4I − 4G(I) + 128; the blur kernel was 51 × 51 pixels. The preprocessed images were saved before model training. The EfficientNet-B4 input was 512 × 512 × 3.

Training-only augmentation included random 90-degree rotations (p = 0.5), horizontal and vertical flips (p = 0.5 each), shift-scale-rotation (shift limit 0.1, scale limit 0.15, rotation limit 60 degrees; p = 0.7), hue-saturation-value perturbations (limits 20, 30, and 20; p = 0.5), and brightness/contrast perturbations (limits 0.2 and 0.2; p = 0.5). For fold-assigned inference, images were normalized with ImageNet means \[0.485, 0.456, 0.406\] and standard deviations \[0.229, 0.224, 0.225\], then converted to tensors. No stochastic augmentation was applied during inference.

## 2.6. EfficientNet-B4 model and optimization

EfficientNet-B4 was initialized with the torchvision ImageNet weights \[6,7\]. The original classifier was replaced with a linear layer producing five logits. For an image x, the model produced logits z(x), and the class probabilities were

p<sub>c</sub>(x) = exp(z<sub>c</sub>(x)) / Σ<sub>j=0</sub><sup>4</sup> exp(z<sub>j</sub>(x)), c ∈ {0, 1, 2, 3, 4} (1)

The archived training script and configuration specify AdamW with a learning rate of 1 × 10−4 and weight decay of 1 × 10−4. They specify focal loss with α = 1 and γ = 2 \[15\],

L<sub>focal</sub> = −α(1 − p<sub>t</sub>)<sup>γ</sup> log p<sub>t</sub> (2)

where pₜ is the predicted probability assigned to the observed class. The archived training code specifies a WeightedRandomSampler that assigns each training image a weight equal to the inverse frequency of its class within the training partition; sampling is configured with replacement and the same number of samples as the training set for one epoch. A complete runtime-override record was not available.

The archived training script specifies defaults of up to 40 epochs, a training batch size of 16, an evaluation batch size of 32, early-stopping patience of five evaluations without macro-F1 improvement, and a ReduceLROnPlateau scheduler (factor 0.5; patience 2). Automatic mixed precision is enabled in the training code; prediction inference uses standard precision. Because the run package does not include a complete command-line capture, these script defaults describe the documented configuration but do not establish whether values were overridden during execution. Only history CSVs for folds 0 and 4 are present, and each contains one row; histories for the other folds are absent. Consequently, actual stopping epochs, epoch-wise learning curves, and runtime cannot be reconstructed. The distinction between audited results and documented settings is summarized in Table 3.

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

The uncertainty analysis estimates the sampling distribution of the same functional by resampling filename-derived grouping keys rather than individual photographs \[16\]. For bootstrap replicate b, a group sample G⁽ᵇ⁾ is drawn with replacement, all associated eye-side records are retained, and T⁽ᵇ⁾ is recomputed from the resulting multiset. The reported interval is the empirical 2.5th to 97.5th percentile of {T⁽ᵇ⁾} for b = 1, …, 1000, with seed 2026. This procedure quantifies uncertainty conditional on the archived fold assignments; it does not correct checkpoint-selection optimism or substitute for external validation.

## 2.8. Posterior squared-distance decision sensitivity

For an ordinal squared-distance loss, define the posterior decision risk for grade k as R(k \| x) = Σⱼ₌₀⁴ (j − k)²pⱼ(x). The corresponding decision is ỹ(x) = arg minₖ R(k \| x). Completing the square shows that R(k \| x) = constant + (k − m(x))², where m(x) = Σⱼ₌₀⁴ j pⱼ(x) is the posterior mean grade. Thus, with deterministic lower-grade tie breaking, ỹ(x) is the nearest integer grade to m(x). This rule is Bayes-optimal for the specified item-level squared-distance loss only when p(x) is interpreted as the relevant posterior distribution; it is not asserted to optimize quadratic weighted kappa, which is a chance-corrected confusion-matrix statistic. Because the archived softmax probabilities were descriptively miscalibrated, the rule was evaluated only as a secondary sensitivity analysis and did not replace the primary argmax results.

Proposition 1 (posterior-mean characterization of the squared-distance decision). For p(x) ∈ Δ⁴ and k ∈ {0, 1, 2, 3, 4}, the minimizer of R(k \| x) = Σⱼ₌₀⁴ (j − k)²pⱼ(x) is an integer nearest to m(x) = Σⱼ₌₀⁴ j pⱼ(x). If m(x) lies exactly halfway between two admissible integer grades, selecting the lower grade gives a deterministic minimizer.

Proof. Expanding the square gives R(k \| x) = Σⱼ j²pⱼ(x) − 2km(x) + k² = \[Σⱼ j²pⱼ(x) − m(x)²\] + \[k − m(x)\]². The first bracket is independent of k, so minimizing R over the ordered integer label set is equivalent to minimizing \[k − m(x)\]². The nearest admissible integer is therefore a minimizer; the stated tie rule makes the decision deterministic. □

## 2.9. Evaluation metrics

The primary five-class metrics were accuracy, balanced accuracy, macro-F1, macro precision, macro recall, log loss, quadratic weighted kappa (QWK), and macro one-vs-rest ROC-AUC. Accuracy was calculated from the five-class argmax predictions. Balanced accuracy was the unweighted mean of the five class-specific recalls. Macro-F1 was the unweighted mean of the five class-specific F1 scores. Log loss was calculated from the five-class probability vectors. Macro ROC-AUC was the unweighted mean of the five one-vs-rest ROC-AUC values. QWK was used because the grades are ordered and distant-grade disagreements should be distinguished from adjacent-grade disagreements. With O the 5 × 5 confusion matrix of reference grade i and predicted grade j, and E its expectation under independence of the margins,

κ<sub>w</sub> = 1 − Σ<sub>i,j</sub> w<sub>ij</sub>O<sub>ij</sub> / Σ<sub>i,j</sub> w<sub>ij</sub>E<sub>ij</sub>, w<sub>ij</sub> = (i − j)<sup>2</sup> / (K − 1)<sup>2</sup>, E<sub>ij</sub> = O<sub>i·</sub> O<sub>·j</sub> / n (3)

For the secondary binary analysis, referable DR was defined as grade ≥2. The referable probability was calculated as p2 + p3 + p4 without threshold tuning. The binary class prediction was derived from the original five-class argmax prediction. Sensitivity, specificity, positive predictive value, F1 score, and binary ROC-AUC were then computed.

## 2.10. Calibration and uncertainty

For the primary analysis, we evaluated the raw EfficientNet-B4 probabilities without applying post-hoc calibration. We quantified their descriptive calibration using the expected calibration error (ECE) of the maximum class probability with 15 equal-width confidence bins B₁, …, B₁₅ on \[0, 1\] \[17\]:

ECE = Σ<sub>b=1</sub><sup>15</sup> (\|B<sub>b</sub>\| / n) · \|acc(B<sub>b</sub>) − conf(B<sub>b</sub>)\| (4)

where, for each bin, acc(·) is the proportion of correct argmax decisions and conf(·) is the mean maximum class probability of the images in that bin. The separate group-held-out ordinal calibration sensitivity analysis is described in Section 2.11. The primary ECE is a descriptive property of the raw model outputs, not evidence of clinical calibration.

Uncertainty intervals were calculated from 1000 filename-grouped bootstrap replicates using seed 2026. The filename-derived grouping keys, rather than individual photographs, were sampled with replacement, and all eye-side records associated with each sampled key were retained. Percentile intervals were calculated for primary overall, class-specific, ordinal, probabilistic, and derived binary metrics. These intervals quantify sampling variability conditional on the fixed fold assignments; they do not account for checkpoint-selection optimism described in Section 2.4.

## 2.11. Five-meta-fold group-disjoint ordinal isotonic calibration (secondary analysis)

As a secondary analysis, we assessed ordinal calibration of the fixed EfficientNet-B4 fold-assigned probability vectors without retraining the network. We shuffled the 17,563 filename-derived grouping keys using seed 2026 and assigned them cyclically to five calibration meta-folds. For each meta-fold, four group partitions were used to fit isotonic regression maps \[18\] for the cumulative events P(Y ≥ 1), P(Y ≥ 2), P(Y ≥ 3), and P(Y ≥ 4); the fitted maps were then applied to the fifth partition. Thus, labels from a group were not used to fit the calibration maps applied to that group. This procedure evaluates a second-stage calibration of fixed predictions; it is not end-to-end nested cross-validation and does not remove the checkpoint-selection limitation of the primary model evaluation.

The four calibrated cumulative values were projected to a non-increasing sequence by the cumulative minimum operator. They were then converted to five class probabilities by differencing adjacent cumulative values, with q₁, …, q₄ denoting the calibrated values of P(Y ≥ 1), …, P(Y ≥ 4):

p<sub>0</sub> = 1 − q<sub>1</sub>, p<sub>j</sub> = q<sub>j</sub> − q<sub>j+1</sub> (j = 1, 2, 3), p<sub>4</sub> = q<sub>4</sub> (5)

Proposition 2 (validity of ordinal reconstruction). If 1 ≥ q₁ ≥ q₂ ≥ q₃ ≥ q₄ ≥ 0, then the reconstructed vector p = (p₀, p₁, p₂, p₃, p₄) is a valid probability vector: every pⱼ is non-negative and Σⱼpⱼ = 1.

Proof. Since q₁ ≤ 1, p₀ = 1 − q₁ ≥ 0. Since q₁ ≥ q₂ ≥ q₃ ≥ q₄, each successive difference p₁, p₂, and p₃ is non-negative, and p₄ = q₄ ≥ 0. The terms telescope: (1 − q₁) + (q₁ − q₂) + (q₂ − q₃) + (q₃ − q₄) + q₄ = 1. □

This secondary analysis applies group-disjoint calibration to fixed fold-assigned predictions through five calibration meta-folds. It is not end-to-end nested cross-validation, does not retrain EfficientNet-B4, and does not constitute external validation. Its purpose is to test whether an ordinal probability constraint can improve probability quality and grade-distance behavior without using the evaluation labels for the corresponding calibration meta-fold.

Differences between raw and calibrated outputs were quantified with a paired filename-group bootstrap. In each of 1000 replicates, filename-derived grouping keys were sampled with replacement and all associated eye-side records were retained. The reported 95% intervals are percentile intervals for calibrated-minus-raw metric differences; they are not p-values.

# 3. Results

## 3.1. Dataset and split audit

The final pooled prediction file contained 35,126 rows and 35,126 unique image identifiers. Every row contained a finite five-class probability vector. The probabilities summed to one with a maximum absolute numerical deviation of 1.8 × 10−7, and the stored argmax agreed with the recomputed probability argmax for all rows. The five fold sizes differed by no more than two images. The 17,563 filename-derived grouping keys were each assigned to a single fold, so paired eye-side records did not cross fold boundaries.

The four low-variance photographs identified by the prediction-blind quality-control audit were retained in all primary analyses; the effect of removing them is reported in Section 3.9.

## 3.2. Primary five-class performance

On the pooled fold-assigned prediction file, the primary five-class analysis achieved accuracy 0.752975, balanced accuracy 0.583907, macro-F1 0.569848, weighted-F1 0.773421, macro precision 0.568889, macro recall 0.583907, log loss 0.739464, QWK 0.736006, and macro one-vs-rest ROC-AUC 0.873664. The 95% filename-grouped bootstrap intervals were 0.747792–0.757758 for accuracy, 0.573080–0.596434 for balanced accuracy, 0.559609–0.580731 for macro-F1, 0.768507–0.777819 for weighted-F1, 0.557506–0.580687 for macro precision, 0.573080–0.596434 for macro recall, 0.726361–0.744378 for QWK, 0.869512–0.877827 for macro ROC-AUC, and 0.729363–0.750267 for log loss. These estimates are summarized in Table 4, and Figure 4 displays the principal point estimates with their bootstrap intervals.

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

Grade-specific results are shown in Table 5. Grade 0 had the highest F1 score, 0.8698, with precision 0.9153 and recall 0.8287. Grade 1 was the most difficult category, with precision 0.1907, recall 0.3606, F1 score 0.2495, and one-vs-rest ROC-AUC 0.6765. Grade 2 had F1 score 0.6121 and recall 0.6270. Grade 3 and grade 4 had ROC-AUC values of 0.9469 and 0.9721, respectively, but their argmax recalls were 0.4903 and 0.6130. These results show why probability ranking and ordinal agreement should be reported alongside hard five-class decisions. Classwise ranking performance is plotted in Figure 5, and Figure 6 shows the pooled confusion counts and row-normalized recalls.

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

The confusion matrix showed 21,388 correct grade-0 predictions out of 25,810 images. Grade-0 images were most often misclassified as grade 1 or grade 2. Grade-1 images were frequently assigned to grade 0 or grade 2. Grade-2 images were correctly classified in 3318 cases, with additional errors toward neighboring grades. Most grade-3 errors were assigned to grade 2 (364 of 873 images), and grade-4 images were most often assigned to grade 2 (155) or grade 3 (94). The error pattern supports the use of QWK and grade-specific recall rather than accuracy alone.

## 3.4. Ordinal error geometry

The ordinal error geometry provided an additional summary of grade distance. The mean absolute grade error was 0.3247 grades and the root-mean-square grade error was 0.7046 grades. Predictions were within one grade of the reference in 92.85% of photographs, while 7.15% differed by two or more grades. The model undergraded 9.37% of photographs and overgraded 15.34%; this directional imbalance should be considered when interpreting the derived referable endpoint. Table 6 summarizes these ordinal error measures.

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

We evaluated the posterior-risk decision rule ỹᵢ = arg minₖ Σⱼ (j − k)²pᵢⱼ as a secondary sensitivity analysis, while retaining the probability-argmax rule as the primary decision rule. The posterior-risk rule changed 12,069 of 35,126 decisions (34.36%). It increased within-one-grade accuracy from 0.9285 to 0.9752 and reduced the large ordinal error rate from 0.0715 to 0.0248, but it reduced accuracy from 0.7530 to 0.5114, balanced accuracy from 0.5839 to 0.5391, macro-F1 from 0.5698 to 0.4943, QWK from 0.7360 to 0.6650, and increased mean absolute grade error from 0.3247 to 0.5167. The paired filename-group bootstrap intervals quantify uncertainty in these differences under the supplied grouping scheme. The result is methodologically informative: the squared-distance rule can reduce large grade jumps while worsening exact and average grade performance when applied to miscalibrated probabilities. The argmax rule therefore remains the primary decision rule for this analysis. Table 7 and Figure 7 summarize the decision-level trade-off.

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

As a secondary binary analysis, grades 2–4 were grouped as referable DR. The derived endpoint had accuracy 0.9046, balanced accuracy 0.8558, sensitivity 0.7756, specificity 0.9360, positive predictive value 0.7466, F1 score 0.7609, and ROC-AUC 0.9305. The binary confusion matrix contained 26,444 true negatives, 1809 false positives, 1542 false negatives, and 5331 true positives. This endpoint was derived from the five-class predictions and probabilities; it was not a separately trained or externally validated binary model. Table 8 reports its metrics, and Figure 8 visualizes the derived endpoint.

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

The 15-bin maximum-confidence ECE for the primary uncalibrated baseline was 0.1415. Because the checkpoint-selection fold overlapped with the fold-assigned prediction partition, this value is descriptive. The probability vectors should not be described as calibrated clinical risk probabilities. The 15-bin reliability diagram is shown in Figure 9.

[Figure image: see submission_figures/]

**Figure 9.** Descriptive multiclass confidence calibration for the primary uncalibrated baseline using the maximum predicted class probability and 15 fixed bins. The ECE was 0.1415. The secondary ordinal calibration analysis is reported in Section 3.8.

## 3.8. Five-meta-fold group-disjoint ordinal isotonic calibration

The secondary calibration analysis was evaluated only through five group-disjoint meta-folds. Relative to the raw fold-assigned probabilities, ordinal isotonic post-processing reduced descriptive ECE from 0.1415 to 0.0103, reduced multiclass Brier score from 0.4089 to 0.2594, reduced mean absolute grade error from 0.3247 to 0.2391 grades, and increased within-one-grade accuracy from 0.9285 to 0.9344. QWK increased from 0.7360 to 0.7637. These changes should not be interpreted as uniform improvement: balanced accuracy decreased from 0.5839 to 0.5211, macro-F1 changed from 0.5698 to 0.5697, and macro ROC-AUC changed from 0.8737 to 0.8722. Across five alternative group-disjoint calibration meta-fold assignments (seeds 2026–2030), ECE reductions ranged from 0.1311 to 0.1319, QWK gains from 0.0277 to 0.0295, and balanced-accuracy changes from −0.0629 to −0.0602. These repeated assignments use the same fixed base predictions and are sensitivity analyses, not independent model replications. The calibration layer therefore improves probability and ordinal-distance summaries under this secondary protocol but changes the class decision trade-off, particularly across imbalanced grades. Calibration increased the number of grade-0 decisions from 23,367 to 28,653 and reduced grade-1 decisions from 4619 to 1268, which raises accuracy in this grade-0-dominated cohort while lowering minority-grade recall. The full comparison is reported in Table 9, with paired calibrated-minus-raw differences in Table 10.

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

The paired filename-group bootstrap showed that calibration increased QWK by 0.027738 (95% CI 0.020664–0.034500), reduced multiclass Brier score by 0.149501 (95% CI −0.153576 to −0.145182), reduced ECE by 0.131122 (95% CI −0.137226 to −0.123600), reduced mean absolute grade error by 0.085549 grades (95% CI −0.091644 to −0.079512), and increased within-one-grade accuracy by 0.005865 (95% CI 0.003132–0.008569). Balanced accuracy decreased by 0.062794 (95% CI −0.070710 to −0.055914). Macro-F1 95% CI = \[−0.008082, 0.006997\] and macro ROC-AUC 95% CI = \[−0.004039, 0.001135\] both included zero. These intervals support changes in probability and ordinal summaries under the fixed-prediction protocol, but not a uniform classification-performance improvement.

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

The four low-variance images identified without reference to model outputs were removed only for the sensitivity analysis. Accuracy changed from 0.752975 to 0.753004, balanced accuracy from 0.583907 to 0.583815, macro-F1 from 0.569848 to 0.569784, QWK from 0.736006 to 0.736029, macro ROC-AUC from 0.873664 to 0.873619, and log loss from 0.739464 to 0.739421. In 1000 paired filename-group bootstrap replicates, the 95% intervals for all six differences included zero. The full-data estimate remains primary because the four images were part of the analyzed cohort; the sensitivity analysis indicates that their inclusion does not materially alter the reported point estimates. The flagged identifiers, the paired replicate-level differences, and the script that recomputes these estimates are provided in Supplementary File S1.

# 4. Discussion

## 4.1. Principal findings

This study evaluates five-class DR grading with a single EfficientNet-B4 classifier on 35,126 Kaggle photographs. Five fold assignments kept the two eye-side records associated with each filename-derived grouping key together. The primary five-class analysis achieved accuracy 0.7530, balanced accuracy 0.5839, macro-F1 0.5698, QWK 0.7360, log loss 0.7395, and macro one-vs-rest ROC-AUC 0.8737. The secondary derived grade-≥2 referable-DR analysis achieved accuracy 0.9046, F1 score 0.7609, ROC-AUC 0.9305, sensitivity 0.7756, and specificity 0.9360. Removing four very low-variance photographs produced negligible changes in these estimates. The findings characterize performance under the evaluated fold-assignment and checkpoint-selection protocol; the lack of an independent cohort limits inference beyond this benchmark.

The results are best understood as complementary rather than as a single accuracy claim. The raw accuracy was influenced by the predominance of grade 0, which represented 73.48% of the analyzed images. Balanced accuracy and macro-F1 give each grade equal influence and were therefore lower than weighted measures. QWK adds information about the distance between ordinal grades: it distinguishes a near-neighbor error from a prediction that skips several severity levels. The combination of macro-F1, QWK, grade-specific recall, log loss, and macro ROC-AUC is consequently more informative than accuracy alone for this imbalanced ordinal task.

The ordinal error analysis adds a direct distance-based interpretation. The mean absolute error was 0.3247 grades, 92.85% of decisions were within one grade, and 7.15% differed by at least two grades. These quantities are not substitutes for clinical adjudication, but they make the error geometry explicit and avoid treating all five-class mistakes as equivalent. The overgrading rate exceeded the undergrading rate in the archived predictions. That asymmetry may be preferable to systematic undergrading in some screening contexts, but the present retrospective data cannot establish an appropriate clinical risk trade-off or threshold.

## 4.2. Posterior squared-distance decision analysis

The squared-distance decision rule provides a useful mathematical stress test rather than a performance-improvement claim. Under a true posterior distribution, minimizing expected squared grade distance is equivalent to choosing the integer grade nearest to the posterior mean. Applying this rule to the archived raw probabilities changed 34.36% of decisions. It reduced the large ordinal error rate from 7.15% to 2.48% and increased within-one-grade accuracy from 92.85% to 97.52%, but it also reduced accuracy to 51.14%, macro-F1 to 49.43%, QWK to 0.6650, and increased mean absolute grade error to 0.5167 grades. The paired group-bootstrap intervals consistently excluded zero for these changes.

This result is important for two reasons. First, it demonstrates that a mathematically coherent ordinal loss does not automatically improve every performance objective. Second, it shows why the raw softmax vector cannot be treated as a calibrated posterior simply because its components lie on a simplex. The posterior-risk rule was therefore not adopted as the primary classifier decision. A future study could evaluate a decision rule on an independently calibrated probability model under a prespecified clinical utility function, but that experiment was not available here.

## 4.3. Class-specific behavior and ordinal interpretation

Grade 0 had the highest F1 score and precision, but its recall was 0.8287 rather than perfect. The large number of grade-0 images makes this performance important for the pooled accuracy, but it should not be interpreted as evidence that the classifier reliably excludes disease in every setting. Grade 1 was the most difficult category, with F1 score 0.2495 and ROC-AUC 0.6765. The confusion matrix showed movement between grade 1 and its neighboring grades. This pattern is compatible with the difficulty of distinguishing subtle or borderline findings, but the present data cannot determine whether the errors resulted from lesion visibility, label noise, acquisition quality, or grader disagreement.

Grades 3 and 4 had higher probability-ranking performance than their argmax recall alone might suggest. Their one-vs-rest ROC-AUC values were 0.9469 and 0.9721, whereas recall was 0.4903 and 0.6130. This difference indicates that the probability outputs contained class-separation information that was not fully converted into the fixed argmax decision. It does not justify threshold optimization after observing the evaluation outputs, and it does not demonstrate that a different operating point would be clinically appropriate. Future work should predefine clinically motivated thresholds on an independent calibration set and evaluate them on a separate external cohort.

## 4.4. Relation to previous DR studies

The present results should be interpreted alongside, not ranked directly against, studies using different endpoints. Ting et al. \[4\] and Abràmoff et al. \[9\] evaluated binary referral-oriented outcomes, whereas Ipp et al. \[10\] assessed a referral system prospectively in clinical settings. These studies provide important context for screening research but do not measure five-class severity-grading performance. Likewise, segmentation studies optimize spatial overlap rather than image-level grade prediction, so their metrics estimate a different task.

Zhang et al. \[11\] is more closely related because it included a DR severity-grading task, although it used a separately assembled clinical cohort and a different train/validation/test design. Any numerical comparison must account for differences in model capacity, preprocessing, augmentation, partitioning, checkpoint selection, and metric definitions. The present analysis is therefore a protocol-specific benchmark, not a claim of superiority. Its contribution is to report raw fold-assigned probabilities, an explicit grouping key, defined performance estimands, filename-group bootstrap intervals, and objective sensitivity analyses.

Two further studies are closer in endpoint or data source. Krause et al. \[13\] reported five-class agreement of a deep learning model with an adjudicated reference standard (QWK 0.84) using higher-resolution inputs and a different development cohort. Voets et al. \[14\] reproduced a published referable-DR pipeline on the Kaggle EyePACS data and reported ROC-AUC 0.951 on held-out Kaggle images but 0.853 on Messidor-2. The present QWK (0.7360) and derived referable ROC-AUC (0.9305) are lower; the reference standards, referable definitions, input resolutions, and partition designs differ, and the drop reported by Voets et al. on Messidor-2 illustrates how much performance can change under dataset shift. These comparisons indicate a plausible performance range for the task, not a ranking.

A literature comparison is informative when it preserves task and protocol context. Validation accuracy, validation loss, ROC-AUC, Dice, and QWK describe different properties and cannot be treated as interchangeable indicators of model quality.

## 4.5. Interpretation of the computational design

EfficientNet-B4 provides a transfer-learning benchmark based on ImageNet initialization and a scaled convolutional architecture. The documented training configuration pairs focal loss with inverse-frequency sampling. These mechanisms address different aspects of the task: focal loss changes the contribution of difficult examples, whereas weighted sampling changes how frequently minority-class examples are presented. Neither approach guarantees balanced performance or calibrated probabilities, and the grade-specific results show that recognition remains imperfect for less frequent grades.

The model used 512 × 512 inputs, but this analysis does not establish that this resolution is optimal. Preprocessing and augmentation may affect the visibility of small lesions and the resulting probability distribution. Because the available training histories were incomplete, the results cannot be attributed to a particular epoch, learning-rate trajectory, or training-time event.

The group-aware fold assignment reduces the risk that the model benefits from shared acquisition characteristics across training and evaluation. The grouping key, however, was derived from filenames and was not verified against a clinical registry. Moreover, macro-F1 on each prediction fold was used for checkpoint selection. The estimates may therefore contain selection optimism and should not be interpreted as fully nested estimates of generalization.

## 4.6. Referable-DR interpretation and calibration

The derived grade-≥2 endpoint was easier than the original five-class task, with accuracy 0.9046, F1 score 0.7609, ROC-AUC 0.9305, and specificity 0.9360. Sensitivity was 0.7756, corresponding to 1542 false-negative cases in the pooled analysis. This false-negative count is important because it prevents the model from being described as an autonomous rule-out system. The positive predictive value was 0.7466 in this dataset, but predictive values depend strongly on prevalence. A different population, referral policy, or camera distribution could produce different operating characteristics.

The ECE of 0.1415 indicates that the primary raw maximum-confidence probabilities should not be described as clinically calibrated. No independent calibration cohort was available. The secondary five-meta-fold, group-disjoint ordinal calibration analysis evaluates post-processing of fixed predictions, but it does not establish external or clinical calibration. Probability outputs should therefore not be interpreted as validated individual risk estimates. Deployment-oriented evaluation would require independent calibration, subgroup assessment, decision-curve analysis, and prespecified operating points.

The secondary ordinal calibration analysis assesses post-processing of fixed probabilities without fitting the calibration maps to the groups on which they are evaluated. Paired filename-group bootstrap intervals support reductions in ECE, multiclass Brier score, and mean absolute grade error, together with increases in within-one-grade accuracy and QWK. The direction of these ECE, QWK, and balanced-accuracy changes was consistent across five alternative meta-fold assignments, although the underlying EfficientNet predictions were unchanged. Balanced accuracy decreased, whereas the intervals for macro-F1 and macro ROC-AUC included zero. The calibration step therefore changes the decision distribution: improvements in probability and ordinal-distance summaries do not imply uniform gains in class-balanced classification. These results are a secondary sensitivity analysis, not evidence of clinical calibration or superiority over the raw classifier.

## 4.7. Reproducibility and methodological contribution

The analysis package contains the analyzed manifest, fold assignments, raw fold-assigned probabilities, metric files, figures, scripts, and checksum information. These artifacts permit independent recomputation of the principal prediction metrics and figures. They do not establish complete reproducibility of every training event because a complete software lockfile and complete epoch-wise histories are unavailable.

The contribution is a retrospective benchmark with a defined five-class target, an explicit filename-derived grouping key, raw probabilities, ordinal and class-balanced metrics, filename-group bootstrap uncertainty, descriptive calibration, and an objective quality-control sensitivity analysis. It is an empirical evaluation contribution rather than a claim of architectural novelty.

For a mathematics-oriented readership, the methodological contribution is the explicit definition of a group-conditional ordinal evaluation functional, its filename-group bootstrap uncertainty estimator, and the comparison of decision rules under distinct ordinal objectives. Proposition 1 characterizes the decision minimizing expected squared grade distance, while Proposition 2 establishes simplex validity for cumulative-probability reconstruction. The sensitivity analysis demonstrates that a rule minimizing expected squared grade error need not improve accuracy, QWK, or macro-F1 when applied to the evaluated probability vectors. These results support an empirical contribution in ordinal evaluation and decision analysis without claiming a new architecture or asymptotic theorem.

# 5. Limitations

First, this study uses one public Kaggle cohort and has no independent external validation cohort. Differences in camera systems, image quality, population composition, grading practices, disease prevalence, and referral pathways may reduce performance outside the analyzed data. The results should therefore be interpreted as retrospective benchmark estimates rather than evidence of clinical utility.

Second, the grouping variable was derived from the numerical component of the filename. The audit confirmed that paired eye-side records did not cross the supplied fold boundaries, but the filename-derived key cannot independently verify patient identity or capture all possible clinical dependencies. A future study should use a verified patient or examination identifier where available.

Third, the fold-assigned protocol was not fully nested. The training code selected the best checkpoint using macro-F1 on the same fold later used to generate fold-assigned predictions, and the archive did not contain an independent inner validation partition. This may introduce checkpoint-selection optimism. Nested group-aware cross-validation or a prespecified training schedule with an untouched outer evaluation fold would provide a more conservative design.

Fourth, the archive did not include complete epoch-wise training histories, a complete machine-readable environment lockfile, or sufficient evidence to verify the exact stopping epoch, GPU runtime, and every software-version detail. The raw probabilities and manifests support independent recomputation of the principal prediction metrics, but not complete verification of the training trajectory.

Fifth, the five-class labels originate from a public competition dataset and may contain grading variability, particularly near adjacent-grade boundaries. No independent adjudication or clinician-reader comparison was performed in the present analysis. The study therefore cannot estimate inter-grader agreement or separate model error from reference-label uncertainty.

Sixth, the severe class imbalance remains a central limitation. Grade 1, grade 3, and grade 4 had much smaller supports than grade 0. Weighted sampling and focal loss may improve minority-class learning, but they also modify the effective training distribution and can affect probability calibration. The model’s grade-1 F1 score and the recall values for grades 3 and 4 show that the imbalance problem was not solved completely.

Seventh, the prediction-blind quality-control audit identified four very low-variance images, but no clinical gradability assessment was available. The primary analysis retained them, and the sensitivity analysis showed negligible changes. This does not establish that all images were clinically gradable or that the dataset is free of other quality-related artifacts.

Eighth, demographic, camera, site, and clinical covariates were unavailable. Subgroup performance, fairness, site shift, and interactions with systemic disease factors could not be assessed. The model uses retinal photographs only and does not incorporate HbA1c, diabetes duration, blood pressure, renal function, or other clinical information.

Finally, the model used 512 × 512 inputs, but lesion-level explanations and clinician interpretation were not evaluated. The findings therefore provide no evidence about explanation quality or lesion localization. The referable-DR endpoint was derived from five-class outputs, not from a separately trained or independently calibrated screening model. Similarly, five-meta-fold group-disjoint calibration was a secondary analysis of fixed predictions; it does not establish external calibration or patient-level risk validity. Prospective clinical evaluation, independent calibration, external testing, subgroup analysis, and prespecified operating points would be required before considering patient-care use.

# 6. Conclusions

We evaluated EfficientNet-B4 for five-class DR grading on 35,126 Kaggle photographs grouped into 17,563 filename-derived groups. Under the supplied five-fold assignment, the primary five-class analysis achieved accuracy 0.7530, balanced accuracy 0.5839, macro-F1 0.5698, QWK 0.7360, log loss 0.7395, and macro one-vs-rest ROC-AUC 0.8737. The secondary derived grade-≥2 referable-DR analysis achieved accuracy 0.9046, F1 score 0.7609, ROC-AUC 0.9305, sensitivity 0.7756, and specificity 0.9360. The posterior squared-distance decision reduced errors spanning at least two grades but worsened accuracy, QWK, macro-F1, and mean absolute grade error; the argmax rule was therefore retained as primary. Removing four images identified by a prediction-blind quality rule produced negligible changes, with all paired filename-group bootstrap intervals for the six primary metric differences including zero.

This study provides a prediction-level assessment of ordinal performance, calibration trade-offs, and decision-rule sensitivity with explicit grouping and uncertainty procedures. Because checkpoint selection used the fold later used for fold-assigned prediction, estimates may be optimistic; the analysis also lacks an independent external validation and calibration cohort. These results do not establish clinical readiness, explainability, fairness, or autonomous screening. Future evaluation should use untouched outer test folds, complete training records, external data, and prespecified clinical operating points.

# Author Contributions

Conceptualization, A.S.; methodology, A.S., M.S. and C.L.; software, A.S. and M.S.; validation, A.S., M.S. and C.L.; formal analysis, A.S., M.A., M.S., C.L., S.A. and M.M.R.; investigation, A.S.; resources, A.S., M.S. and C.L.; data curation, A.S. and M.S.; writing—original draft preparation, A.S. and M.A.; writing—review and editing, A.S., M.A., M.S., C.L., M.O., S.A. and M.M.R.; visualization, A.S. and M.O.; supervision, M.S. and C.L.; project administration, A.S. All authors have read and agreed to the published version of the manuscript.

# Funding

This research received no external funding.

# Institutional Review Board Statement

Not applicable. This study used the publicly available, de-identified Kaggle Diabetic Retinopathy Detection dataset. Because the dataset contains only anonymized retinal fundus photographs with no identifiable personal health information, formal ethics approval was not required for this secondary data analysis. The images were originally collected by EyePACS under the ethical guidelines and protocols of the contributing institutions. All data use complied with the dataset terms of use.

# Informed Consent Statement

Not applicable. This study used only publicly available, de-identified data. Any informed consent procedures for data collection were conducted by the source institutions at the time of image acquisition. The analyzed images are anonymized and contain no identifiable personal health information.

# Data Availability Statement

The Kaggle Diabetic Retinopathy Detection images are third-party data available through the official competition page (https://www.kaggle.com/competitions/diabetic-retinopathy-detection) under the applicable Kaggle terms. The prediction-level reproducibility archive is supplied as Supplementary File S1. It contains the analyzed manifest, fold assignments, raw fold-assigned probabilities, metric files, source scripts, figures, calibration outputs, and validation reports. It does not contain original-resolution Kaggle JPEGs, full competition archives, or model checkpoints. The manuscript and figure package include a representative-image panel derived from a small number of dataset photographs; dissemination of these illustrative examples remains subject to the applicable Kaggle terms. The archive does not contain a complete software lockfile or complete epoch-wise training histories; these limitations are declared in the manuscript.

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
