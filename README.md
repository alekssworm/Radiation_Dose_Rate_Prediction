

# Radiation Dose Rate Prediction

**Applied ML | Geospatial ML | Spatial Validation | Scenario Comparison**

Geospatial machine learning project for **radiation dose rate prediction** and **scenario comparison** after radioactive contamination.

This repository contains a compact portfolio version of the project built around two stages:

- **MVP-A** — research and validation stage
- **MVP-B** — operational training, inference, and scenario-comparison stage

The project combines radiation measurements, radionuclide features, soil properties, and terrain variables to model `target_dose_rate` in the Ivankiv district / southern edge of the CEZ area.

---

## Project Goal

The main objective of this project is to predict **radiation dose rate** from geospatial and contamination-related tabular features.

A second objective is to use the trained model as a **scenario comparison tool**: modify contamination-related inputs in a controlled way and compare how the predicted dose rate changes.

---

## What This Project Demonstrates

This project demonstrates practical skills in:

- tabular machine learning
- geospatial feature engineering
- data cleaning and dataset assembly
- leakage-aware feature selection
- spatial cross-validation
- feature ablation
- model selection and packaging
- prediction workflow design
- scenario generation and scenario comparison

---

## Project Stages

### MVP-A

MVP-A is the research stage of the project.  
It was used to answer the question:

> Can this task be meaningfully modeled on the available data, and which feature groups actually carry predictive signal?

Main MVP-A components:

- baseline models
- baseline analysis
- spatial validation
- spatial feature ablation

Main MVP-A result:

- contamination-related radionuclide features provide the strongest stable predictive signal
- `env_only` features are weaker, but still provide some contextual information
- spatial validation is much more informative than random split alone

---

### MVP-B

MVP-B is the operational stage of the project.

It turns the earlier research results into a usable pipeline for:

- training shortlisted models
- freezing selected configurations
- running prediction on new tables
- generating and comparing contamination scenarios

Main MVP-B components:

- `train_mvp_b.py`
- `predict_mvp_b.py`
- `make_mvp_b_scenarios.py`
- `compare_scenarios_mvp_b.py`

Main MVP-B result:

- the project can now be used not only for model training, but also for **controlled scenario comparison**

---

## Data Used

The project uses multiple data sources that were merged into tabular training sets.

### Radiation data

Used to define the target and contamination-related inputs:

- background radiation measurements
- radionuclide activity measurements

### Soil features

Used as environmental predictors:

- organic carbon
- clay fraction
- sand fraction
- bulk density
- soil pH

### Terrain features

Used as environmental predictors:

- elevation
- slope
- topographic wetness index (TWI)

---

## Repository Structure

```text
.
├─ MVP_B/
│  ├─ examples/
│  │  └─ scenarios/
│  ├─ outputs/
│  │  ├─ predictions/
│  │  ├─ scenario_comparisons/
│  │  └─ training_runs/
│  ├─ compare_scenarios_mvp_b.py
│  ├─ feature_sets_b.py
│  ├─ make_mvp_b_scenarios.py
│  ├─ model_registry_b.py
│  ├─ predict_mvp_b.py
│  ├─ scenario_report.md
│  ├─ spatial_cv.py
│  └─ train_mvp_b.py
│
├─ mvp_a/
│  ├─ data_mvp_a/
│  │  └─ data/
│  │     └─ processed/
│  ├─ analyze_baseline.py
│  ├─ baseline_models.py
│  ├─ spatial_baseline_models.py
│  └─ spatial_feature_ablation.py
````

---

## Training Data

The main processed training tables are located in:

```text
mvp_a/data_mvp_a/data/processed/
```

Key files:

* `train_env_v1.csv`
* `train_nuclide_v1.csv`

These are the main input tables used for the MVP-B training pipeline.

---

## Key Findings

The current project already supports several important conclusions:

* **Contamination-related radionuclide features carry the strongest stable predictive signal.**
* **Spatial validation is essential and more realistic than random split alone.**
* **The contamination-aware branch is consistently stronger than the env-only branch.**
* **`cs137` is the strongest practical scenario lever in the current model.**
* **`sr90` matters, but its isolated effect is weaker than the effect of `cs137`.**
* **The MVP-B workflow is already usable for structured scenario comparison.**

---

## MVP-B Working Configuration

The current working MVP-B setup selected:

* **env reference model:** `env_only + random_forest`
* **primary model:** `env_plus_no_ratio + extra_trees`

This reflects the strongest currently selected operational configuration from the shortlisted feature sets and models.

---

## Scenario Support

This repository includes a small scenario pack that can be used for interactive testing and comparison.

Scenario files are located in:

```text
MVP_B/examples/scenarios/
```

Included examples:

* `scenario_baseline.csv`
* `scenario_cs137_plus_20.csv`
* `scenario_cs137_minus_20.csv`
* `scenario_sr90_plus_20.csv`
* `scenario_remediation_light.csv`
* `scenario_manifest.csv`

These scenarios were created by applying controlled modifications to contamination-related inputs while keeping the same spatial support and environmental context.

### Example scenario types

* **baseline**
  No modification

* **cs137_plus_20**
  Increase `cs137_kBq_m2` by 20%

* **cs137_minus_20**
  Decrease `cs137_kBq_m2` by 20%

* **sr90_plus_20**
  Increase `sr90_kBq_m2` by 20%

* **remediation_light**
  Simple remediation-style contamination reduction

---

## Scenario Findings

The first scenario-comparison runs already show that:

* increasing `cs137` increases the predicted dose rate on average
* decreasing `cs137` decreases the predicted dose rate on average
* increasing `sr90` produces a smaller effect than increasing `cs137`
* remediation-style scenarios produce the strongest average reduction among the tested examples

This means the repository already contains not only a modeling workflow, but also a working **scenario-analysis pipeline**.

See:

* `MVP_B/scenario_report.md`

---

## How to Run

### 1. Train MVP-B

```powershell
python MVP_B/train_mvp_b.py `
  --env-dataset .\mvp_a\data_mvp_a\data\processed\train_env_v1.csv `
  --nuclide-dataset .\mvp_a\data_mvp_a\data\processed\train_nuclide_v1.csv `
  --output-dir .\MVP_B\outputs\training_runs `
  --defaults-only
```

### 2. Generate scenario inputs

```powershell
python MVP_B/make_mvp_b_scenarios.py `
  --input-csv .\mvp_a\data_mvp_a\data\processed\train_nuclide_v1.csv `
  --output-dir .\MVP_B\examples\scenarios `
  --compact
```

### 3. Run prediction for a scenario

```powershell
python MVP_B/predict_mvp_b.py `
  --input-csv .\MVP_B\examples\scenarios\scenario_baseline.csv `
  --output-csv .\MVP_B\outputs\predictions\scenario_baseline_pred.csv `
  --metadata-path .\MVP_B\outputs\training_runs\<RUN_NAME>\artifacts\mvp_b_metadata.json `
  --artifact-path .\MVP_B\outputs\training_runs\<RUN_NAME>\artifacts\mvp_b_primary_model.joblib `
  --mode nuclide_mode
```

### 4. Compare baseline vs scenario

```powershell
python MVP_B/compare_scenarios_mvp_b.py `
  --baseline-csv .\MVP_B\outputs\predictions\scenario_baseline_pred.csv `
  --scenario-csv .\MVP_B\outputs\predictions\scenario_cs137_plus_20_pred.csv `
  --output-dir .\MVP_B\outputs\scenario_comparisons
```

---

## Why Spatial Validation Matters

This project uses **spatial grouping / GroupKFold** as a core evaluation principle.

That matters because geospatial tabular data can produce overoptimistic results under ordinary random split due to local spatial similarity.

The project therefore treats spatial robustness as a first-class modeling requirement rather than a secondary check.

---

## Representative Outputs

Representative MVP-B outputs are included in:

* `MVP_B/outputs/training_runs/`
* `MVP_B/outputs/predictions/`
* `MVP_B/outputs/scenario_comparisons/`

Key examples include:

* selected model configuration summary
* model passport
* scenario prediction tables
* scenario comparison summaries
* top changed points under scenario perturbations

---

## Results

Initial scenario-comparison runs on **545 rows** show that the current MVP-B primary model already works as a **usable comparative scenario-analysis tool**.

- **Baseline mean prediction:** `0.10431`
- **`cs137 +20%` scenario:** mean prediction change **+1.29%**
- **`sr90 +20%` scenario:** mean prediction change **+0.21%**
- **`remediation_light` scenario:** strongest average reduction, **-1.65%**

These results suggest that **`cs137` is the dominant scenario driver** in the current model, while remediation-style contamination reduction produces the strongest average downward shift.


## Summary

This repository shows the progression from:

* exploratory baseline modeling,
* to spatially aware validation,
* to a compact inference and scenario-comparison pipeline.

At its current stage, the project can already be used as a meaningful applied ML portfolio project focused on:

* geospatial tabular modeling
* spatial validation
* contamination-aware prediction
* scenario comparison

---

## Author

Personal applied ML / geospatial ML portfolio project focused on practical pipeline design, validation, and structured experimentation.

````


