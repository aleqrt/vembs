# One-Radiograph-per-Patient Sensitivity Analysis

This folder contains the sensitivity analysis performed to assess whether the primary per-radiograph evaluation is affected by patients contributing different numbers of radiographs.

## Motivation

In the main analysis, performance metrics are computed **per radiograph**. Therefore, a patient contributing (N_i) radiographs contributes (N_i) observations to the reported test-set metrics.

To assess the sensitivity of the results to this unequal patient weighting, we re-evaluated the already trained models on test sets restricted to **one randomly selected radiograph per patient**.

This analysis changes only the composition of the evaluation set:

* no model is refitted;
* no training or validation partition is modified;
* the original held-out test predictions are retained;
* only the radiographs contributing to each sensitivity-analysis evaluation are changed.

## Sensitivity-analysis procedure

For each dataset and embedding configuration:

1. Test-set radiographs are grouped by patient.
2. One radiograph is selected uniformly at random for each patient.
3. Accuracy, precision, recall, F1-score, and ROC-AUC are recomputed on the resulting patient-balanced subset.
4. The procedure is repeated over **200 independent random draws** using seed `42`.
5. Metrics are averaged across the 200 draws for each of the 10 cross-validation-trained models.
6. Results across the 10 models are summarized using the same 95% confidence-interval procedure used in the main paper.

The same sampled radiograph indices are reused across the 10 fold-trained models for a given dataset and embedding so that differences between folds are not driven by different random patient samples.

Importantly, this procedure is **not patient-level prediction aggregation**: predictions from multiple radiographs are never combined into a single patient-level prediction. 
The analysis instead preserves the single-radiograph prediction task while giving each patient equal representation within each random draw.

## Experimental coverage

The analysis covers:

* **Datasets**

  * MIMIC-CXR
  * CheXpert

* **Foundation-model embeddings**

  * CXR Foundation
  * MedCLIP
  * BiomedCLIP

* **Downstream classifiers**

  * Multi-Layer Perceptron (MLP)
  * Logistic Regression
  * Random Forest
  * XGBoost

* **Prediction tasks**

  * Sex
  * Age
  * Binary ethnicity
  * Multiclass ethnicity
  * Insurance type (MIMIC-CXR only)
  * No Finding

The complete analysis therefore evaluates all applicable dataset × embedding × classifier × prediction-task configurations.

## Comparison with the primary analysis

Two evaluation schemes are relevant to the paper:

**Per radiograph — primary analysis**

Every radiograph in the held-out test set contributes equally to the metrics. Consequently, patients with multiple radiographs contribute multiple observations.

**One radiograph per patient — sensitivity analysis**

Each patient contributes exactly one randomly selected radiograph in each draw. The procedure is repeated 200 times to reduce dependence on any particular image selection.


An additional inverse-cluster-size weighting check (`1/N_i` for each radiograph belonging to patient (i)) is implemented in the analysis script, but this should not be interpreted as aggregation of multiple predictions into a patient-level prediction.

## Files

### `sensitivity_analysis_one_cxr_per_patient.py`

Script used to perform the sensitivity analysis. It reconstructs the patient identifiers associated with the held-out test radiographs, evaluates the previously trained models, performs the repeated one-radiograph-per-patient sampling, and computes the corresponding performance metrics.

### Compiled MLP table

A compiled image of the same MLP sensitivity-analysis table is provided for convenient inspection. 
[View the compiled MLP sensitivity-analysis table](./mlp_one_radiograph_per_patient_table.png)_)

### `summary.csv`

Machine-readable summary of the complete sensitivity analysis. 

### `README.md`

Description of the motivation, methodology, and outputs of the sensitivity analysis.

## Reproducibility

The sensitivity analysis uses:

* **200 random draws**
* random seed: **42**
* the same 10 previously trained cross-validation models used in the primary experiments
* the same held-out test partitions used in the primary experiments

No model parameters are updated during this analysis.

The script also performs a quality-assurance check by recomputing the original per-radiograph metrics and comparing them with the previously stored test results. The sensitivity analysis is aborted if these values cannot be reproduced within the specified numerical tolerance.

## Interpretation

This analysis is intended specifically to test the effect of **test-set weighting caused by repeated radiographs from the same patient**.

It does not assess the potential effect of repeated radiographs during model fitting, since the training data and fitted models remain unchanged.

The results can therefore be interpreted as a sensitivity check of the reported performance estimates to the number of radiographs contributed by each patient.


## Reported results

The complete sensitivity analysis covers all applicable dataset × embedding × classifier × prediction-task configurations. However, for conciseness, the table shown in the repo reports only the results obtained with the **MLP classifier** across the three foundation-model embeddings (CXR Foundation, MedCLIP, and BiomedCLIP), both datasets, and all applicable downstream tasks.
