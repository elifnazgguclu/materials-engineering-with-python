# Steel Yield Strength Prediction with Machine Learning

A materials-engineering project exploring whether steel yield strength can be estimated from chemical-composition and heat-treatment features using a Random Forest regression model.

## Project status

The current model ran successfully on the local dataset. Results are from one group-based train/test split and should be treated as an initial evaluation, not as a validated engineering tool.

## Dataset

Expected local path: `data/steelbench_core_open.csv`

Inputs: `C`, `Mn`, `Cr`, `Mo`, `Ni`, `Si`, `V`, `Cu`, `Al`, `austenitize_T`, and `temper_T`. Target: `yield_strength`.

Dataset record: [SteelBench v1.0 on Zenodo](https://zenodo.org/records/18530558)

Check the dataset card and license before redistributing raw data. This repository template does not include the dataset itself.

## Method

1. Exclude rows without a target or `grade_id`.
2. Use `GroupShuffleSplit` for an 80/20 train/test split grouped by `grade_id`.
3. Impute missing numeric values with medians learned from training data and add missingness indicators.
4. Train a `RandomForestRegressor` and compare it with a baseline that predicts the training-target mean.

Settings: `n_estimators=300`, `min_samples_leaf=2`, `random_state=42`.

## Current results

| Metric | Mean baseline | Random Forest |
|---|---:|---:|
| MAE | 143.42 MPa | 89.34 MPa |
| RMSE | 204.62 MPa | 162.16 MPa |
| R² | -0.001 | 0.371 |

The Random Forest reduced test MAE by approximately 37.7% relative to the baseline. Performance varied by data source, and some high-strength samples had very large errors.

## Run locally

Place the dataset at `data/steelbench_core_open.csv`, then run:

```bash
python -m pip install -r requirements.txt
python train_strength_model_final.py
```

## Repository layout

```text
.
├── data/
│   └── steelbench_core_open.csv   # local data; check license before sharing
├── reports/
│   └── project_report.md
├── train_strength_model_final.py
├── density_calculator.py
├── requirements.txt
└── README.md
```

## Limitations

- Evaluation uses a single group-based split; variability across splits has not been assessed.
- Data sources differ in feature completeness and test performance.
- Some high-strength samples are poorly predicted.
- Feature importance describes model reliance, not causality.
- Verify the target definition and units against the original dataset documentation.
- Do not use this model alone for material selection, safety-critical design, certification, or engineering decisions. Independent experimental validation is required.

## Data citation

SteelBench v1.0, Zenodo record: https://zenodo.org/records/18530558

Consult the dataset record and its `DATASET_CARD.md` for the recommended citation and license details.

