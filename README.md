# Data and code for “Machine learning prediction of hardness and electrical conductivity in Cu–Cr-based alloys”

This repository contains the datasets, processed data, result tables, model state file, and complete manuscript-figure generation code used for the manuscript:

**Machine learning prediction of hardness and electrical conductivity in Cu–Cr-based alloys**

Authors: Jianzeng Du, Yanchao Chai, Jiao Man, Menghang Li, Yuxue Zhang, and Wenzheng An.

## Repository contents

```text
CuCr_ML_Data_Code_Repository/
├── README.md
├── CITATION.cff
├── LICENSE.md
├── requirements.txt
├── Data/
│   ├── raw/
│   │   └── Cu-Cr-X dataset.xlsx
│   ├── processed/
│   │   ├── cucrx_hardness_processed.csv
│   │   └── cucrx_conductivity_processed.csv
│   └── data_dictionary.csv
├── Code/
│   ├── check_repository.py
│   ├── figure_utils.py
│   ├── run_all_figures.py
│   └── full_figures/
│       └── generate_figures.py
├── Tables_and_Results/
│   ├── metrics_summary.csv
│   ├── metrics_summary_normalized.csv
│   ├── cross_validation_summary.csv
│   ├── predictions.csv
│   ├── shap_feature_importance_top10.csv
│   ├── screening_map_predictions.csv
│   ├── split_info.json
│   ├── Table1_optimized_hyperparameters.csv
│   ├── Table1_optimized_hyperparameters.xlsx
│   └── modeling_state.pkl
└── Figures/
    ├── manuscript_full_figures/
    └── supplementary_figures/
```

## Data files

- `Data/raw/Cu-Cr-X dataset.xlsx`: source dataset used to construct the ML-ready datasets.
- `Data/processed/cucrx_hardness_processed.csv`: processed dataset for hardness prediction, 1,340 records.
- `Data/processed/cucrx_conductivity_processed.csv`: processed dataset for electrical-conductivity prediction, 1,280 records.
- `Data/data_dictionary.csv`: column meanings, units, and raw/processed-file coverage.

## Results and model files

- `metrics_summary.csv` and `metrics_summary_normalized.csv`: train/test/validation metrics for ET, RF, GB, XGB, LGBM, and CAT models.
- `cross_validation_summary.csv`: cross-validation results.
- `predictions.csv`: measured and predicted values used to generate validation and parity plots.
- `shap_feature_importance_top10.csv`: top SHAP features for the LGBM models.
- `screening_map_predictions.csv`: values used for the hardness–conductivity screening map.
- `split_info.json`: train/test/validation split counts and random seed.
- `Table1_optimized_hyperparameters.csv` and `.xlsx`: optimized hyperparameters for the six models.
- `modeling_state.pkl`: serialized modelling state used by the figure-generation code for SHAP summary plots.

## Python environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

The figure-generation scripts use the packaged datasets and result tables. Fig. 4 requires the serialized model state and SHAP-related packages.

## Reproduce the complete manuscript figures

From the repository root, run:

```bash
python Code/check_repository.py
python Code/full_figures/generate_figures.py
```

Or run the full-figure runner:

```bash
python Code/run_all_figures.py
```

The complete manuscript figures will be written to:

```text
Code/output/full_figures/
```

This repository intentionally includes only the complete manuscript-figure generation code. 

