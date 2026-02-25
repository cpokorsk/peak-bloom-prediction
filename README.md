# Cherry Blossom Peak Bloom Prediction

This repository contains a reproducible forecasting pipeline for the GMU cherry blossom competition, including data preparation, multi-model training, model selection, stacked ensembling, and final 2026 predictions for five target locations.

The project is derived from the official competition template and has been extended to a Python-first workflow with Quarto reporting.

## Project contents

- `data/`: input data and generated model outputs
- `0*` to `5*` scripts: end-to-end pipeline steps
- `pipeline_walkthrough.qmd`: reproducible report that installs dependencies, runs the pipeline, and renders outputs
- `prediction_model.ipynb`: notebook version of the execution flow
- `requirements.txt`: Python dependencies used by the pipeline and Quarto execution

## Quickstart

### 1) Clone your competition repository

```sh
git clone <your-public-repo-url>
cd peak-bloom-prediction
```

### 2) Prerequisites

- Python 3.10+
- Quarto CLI installed and available on PATH
- Git

Optional (for original template/demo content):

- R 4.3+

### 3) Install dependencies

```sh
python -m pip install -r requirements.txt
```

### 4) Run the full pipeline and render report

```sh
quarto render pipeline_walkthrough.qmd
```

The report bootstraps dependencies, executes the pipeline, and shows key artifacts (model rankings, ensemble metrics, and final predictions).

## Method summary

The modeling approach combines complementary model families:

- Linear/statistical: OLS, weighted OLS, Ridge, Lasso, Bayesian Ridge
- Nonlinear ML: Gradient Boosting Quantile, Random Forest
- Time series: ARIMAX
- Process-based phenology: thermal model and DTS

Top models are selected by composite ranking (MAE, RMSE, and \(R^2\)) and combined via a RidgeCV stacked ensemble. Prediction intervals are calibrated using holdout/CV residuals.

Validation mode is controlled by `USE_CV_FOLDS` in `phenology_config.py`:

- `False`: simple last-`N`-years holdout
- `True`: year-block cross-validation

## Outputs and artifacts

Primary outputs are written under `data/model_outputs/`:

- `model_selection_metrics_summary.csv`
- `model_selection_recommended_for_ensemble.csv`
- `stacked_ensemble_model_metrics.csv`
- `stacked_ensemble_meta_model_weights.csv`
- `predictions/final_2026_predictions_stacked_ensemble.csv`

Additional holdout/CV files are produced per model depending on configuration.

## Competition submission checklist (2026)

Before submission, verify that all of the following are complete:

1. Submit before **February 28, 2026, 23:59 AOE**.
2. Include all **five** location predictions.
3. Provide a **public repository** with code and data needed to reproduce results.
4. Include a reproducible Quarto document (this repo uses `pipeline_walkthrough.qmd`).
5. Include a **blinded abstract** of at most **500 words** describing selected features and models.

Entries that cannot be reproduced or lack a coherent, sufficiently complete abstract may be rejected at organizer discretion.

## Notes

- Official competition portal and full rules: https://competition.statistics.gmu.edu
- Original template repository: https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction

## License

![CC-BYNCSA-4](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)

Unless otherwise noted, the content in this repository is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

For the data sets in the _data/_ folder, please see [_data/README.md_](data/README.md) for the applicable copyrights and licenses.
