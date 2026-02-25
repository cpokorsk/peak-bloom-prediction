"""Stacked ensemble built from Step-4 exported holdout predictions."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from phenology_config import (
    MODEL_OUTPUT_DIR,
    HOLDOUT_OUTPUT_DIR,
    PREDICTIONS_OUTPUT_DIR,
    HOLDOUT_LAST_N_YEARS,
    TARGET_PREDICTION_LOCATIONS,
    USE_CV_FOLDS,
    normalize_location,
)


RECOMMENDED_MODELS_FILE = os.path.join(MODEL_OUTPUT_DIR, "model_selection_recommended_for_ensemble.csv")

# Mapping from model selection names to CV holdout file paths (preferred)
CV_HOLDOUT_FILE_MAPPING = {
    "linear_ols": (
        os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_linear_ols.csv"),
        "predicted_doy",
    ),
    "weighted_lm": (
        os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_weighted_lm.csv"),
        "predicted_doy",
    ),
    "bayesian_ridge": (
        os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_bayesian_ridge.csv"),
        "predicted_doy",
    ),
    "ridge": (
        os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_ridge_lasso.csv"),
        "predicted_doy_ridge",
    ),
    "lasso": (
        os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_ridge_lasso.csv"),
        "predicted_doy_lasso",
    ),
    "gradient_boosting_quantile": (
        os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_gradient_boosting_quantile.csv"),
        "predicted_doy",
    ),
    "arimax": (
        os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_arimax.csv"),
        "predicted_doy",
    ),
}

# Mapping from model selection names to simple holdout file paths (fallback)
SIMPLE_HOLDOUT_FILE_MAPPING = {
    "linear_ols": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_linear_ols.csv"),
        "predicted_doy",
    ),
    "weighted_lm": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_weighted_lm.csv"),
        "predicted_doy",
    ),
    "bayesian_ridge": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_bayesian_ridge.csv"),
        "predicted_doy",
    ),
    "ridge": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_ridge_lasso.csv"),
        "predicted_doy_ridge",
    ),
    "lasso": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_ridge_lasso.csv"),
        "predicted_doy_lasso",
    ),
    "gradient_boosting_quantile": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_gradient_boosting_quantile.csv"),
        "predicted_doy",
    ),
    "arimax": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_arimax.csv"),
        "predicted_doy",
    ),
    "process_based_thermal": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_process_based_thermal.csv"),
        "predicted_doy",
    ),
    "dts": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_dts.csv"),
        "predicted_bloom_date",  # DTS uses date column
    ),
    "random_forest": (
        os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_random_forest.csv"),
        "predicted_doy",
    ),
}

def get_holdout_file_mapping():
    """Get holdout file mapping based on USE_CV_FOLDS flag."""
    mapping = {}
    
    if USE_CV_FOLDS:
        # Use CV holdouts for CV-enabled models, simple holdouts for others
        for model in CV_HOLDOUT_FILE_MAPPING:
            mapping[model] = CV_HOLDOUT_FILE_MAPPING[model]
        
        # Process-based models without CV support fall back to simple holdout
        for model in ["process_based_thermal", "dts", "random_forest"]:
            if model in SIMPLE_HOLDOUT_FILE_MAPPING:
                mapping[model] = SIMPLE_HOLDOUT_FILE_MAPPING[model]
    else:
        # Use simple holdouts for all models
        mapping = SIMPLE_HOLDOUT_FILE_MAPPING.copy()
    
    return mapping

HOLDOUT_FILE_MAPPING = get_holdout_file_mapping()

FUTURE_FILE_MAPPING = {
    "linear_ols": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions.csv"),
        "predicted_doy",
    ),
    "weighted_lm": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_weighted_lm.csv"),
        "predicted_doy",
    ),
    "bayesian_ridge": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_bayesian_ridge.csv"),
        "predicted_doy",
    ),
    "ridge": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_ridge_lasso.csv"),
        "predicted_doy_ridge",
    ),
    "lasso": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_ridge_lasso.csv"),
        "predicted_doy_lasso",
    ),
    "gradient_boosting_quantile": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_gradient_boosting_quantile.csv"),
        "predicted_doy",
    ),
    "arimax": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_arimax.csv"),
        "predicted_doy",
    ),
    "process_based_thermal": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_process_based_thermal.csv"),
        "predicted_doy",
    ),
    "dts": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_dts.csv"),
        "predicted_bloom_date",  # DTS uses date column
    ),
    "random_forest": (
        os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_random_forest.csv"),
        "predicted_doy",
    ),
}

OUTPUT_ENSEMBLE = os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_stacked_ensemble.csv")
OUTPUT_WEIGHTS = os.path.join(MODEL_OUTPUT_DIR, "stacked_ensemble_meta_model_weights.csv")
OUTPUT_METRICS = os.path.join(MODEL_OUTPUT_DIR, "stacked_ensemble_model_metrics.csv")
OUTPUT_OBS_PRED_FULL = os.path.join(MODEL_OUTPUT_DIR, "stacked_ensemble_observed_vs_predicted_full_available.csv")
OUTPUT_OBS_EXP_SCATTER = os.path.join(MODEL_OUTPUT_DIR, "stacked_ensemble_observed_vs_expected_scatter_by_location.png")
OUTPUT_OBS_EXP_TIMESERIES = os.path.join(MODEL_OUTPUT_DIR, "stacked_ensemble_observed_vs_expected_timeseries_by_location.png")
OUTPUT_RESIDUALS = os.path.join(MODEL_OUTPUT_DIR, "stacked_ensemble_residuals_by_location.png")
PI_ALPHA = 0.10


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{label} -> MAE: {mae:.2f} days | MSE: {mse:.2f} | RMSE: {rmse:.2f} days")
    return mae, mse, rmse


def doy_to_date(year, doy):
    if pd.isna(doy) or pd.isna(year):
        return None
    return (pd.to_datetime(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy) - 1)).strftime("%b %d")


def _subplot_grid(n_panels):
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, n_cols


def plot_observed_vs_expected_scatter_by_location(df):
    locations = sorted(df["location"].dropna().unique())
    if not locations:
        return

    n_rows, n_cols = _subplot_grid(len(locations))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    global_min = min(df["actual_bloom_doy"].min(), df["predicted_doy"].min()) - 3
    global_max = max(df["actual_bloom_doy"].max(), df["predicted_doy"].max()) + 3

    for idx, location in enumerate(locations):
        ax = axes_flat[idx]
        loc_df = df[df["location"] == location].sort_values("year")
        ax.scatter(loc_df["actual_bloom_doy"], loc_df["predicted_doy"], s=70, alpha=0.8)
        ax.plot([global_min, global_max], [global_min, global_max], "k--", linewidth=1.5, alpha=0.7)
        ax.set_title(location)
        ax.set_xlabel("Observed DOY")
        ax.set_ylabel("Expected DOY")
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.grid(True, alpha=0.3, linestyle="--")

    for idx in range(len(locations), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle("Stacked Ensemble: Observed vs Expected (Scatter by Location)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_OBS_EXP_SCATTER, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_observed_vs_expected_timeseries_by_location(df):
    locations = sorted(df["location"].dropna().unique())
    if not locations:
        return

    n_rows, n_cols = _subplot_grid(len(locations))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, location in enumerate(locations):
        ax = axes_flat[idx]
        loc_df = df[df["location"] == location].sort_values("year")
        ax.plot(loc_df["year"], loc_df["actual_bloom_doy"], marker="o", linewidth=2, label="Observed")
        ax.plot(loc_df["year"], loc_df["predicted_doy"], marker="s", linewidth=2, label="Expected")
        ax.fill_between(
            loc_df["year"],
            loc_df["predicted_doy"] - 3,
            loc_df["predicted_doy"] + 3,
            alpha=0.2,
            label="Expected ±3 days",
        )
        ax.set_title(location)
        ax.set_xlabel("Year")
        ax.set_ylabel("Bloom DOY")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)

    for idx in range(len(locations), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle("Stacked Ensemble: Observed vs Expected (Time Series by Location)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_OBS_EXP_TIMESERIES, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_by_location(df):
    """Plot residuals (actual - predicted) over time for each location."""
    locations = sorted(df["location"].dropna().unique())
    if not locations:
        return

    n_rows, n_cols = _subplot_grid(len(locations))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, location in enumerate(locations):
        ax = axes_flat[idx]
        loc_df = df[df["location"] == location].sort_values("year")
        residuals = loc_df["actual_bloom_doy"] - loc_df["predicted_doy"]
        
        ax.scatter(loc_df["year"], residuals, s=70, alpha=0.8, color="#1f77b4")
        ax.axhline(y=0, color="k", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_title(location)
        ax.set_xlabel("Year")
        ax.set_ylabel("Residual (Actual - Predicted) Days")
        ax.grid(True, alpha=0.3, linestyle="--")
        
        # Add text with mean and std of residuals
        mean_resid = residuals.mean()
        std_resid = residuals.std()
        ax.text(0.02, 0.98, f"Mean: {mean_resid:.2f}\nStd: {std_resid:.2f}",
                transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=8)

    for idx in range(len(locations), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle("Stacked Ensemble: Residuals Over Time by Location", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_RESIDUALS, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_recommended_models():
    """Load recommended models from model selection output."""
    if not os.path.exists(RECOMMENDED_MODELS_FILE):
        raise FileNotFoundError(
            f"Model recommendation file not found: {RECOMMENDED_MODELS_FILE}\n"
            "Run 5_model_selection.py first to select models for ensemble."
        )
    
    rec_df = pd.read_csv(RECOMMENDED_MODELS_FILE)
    models = rec_df["model"].tolist()
    print(f"Using {len(models)} recommended models for ensemble:")
    for model in models:
        print(f"  - {model}")
    return models


def load_holdout_matrix():
    """Dynamically load holdout predictions from recommended models."""
    recommended_models = load_recommended_models()
    
    # Get current holdout file mapping based on USE_CV_FOLDS
    current_mapping = get_holdout_file_mapping()
    
    print(f"\nHoldout mode: {'CV folds' if USE_CV_FOLDS else f'Simple (last {HOLDOUT_LAST_N_YEARS} years)'}")
    
    # Check all required files exist
    required_files = []
    for model in recommended_models:
        if model not in current_mapping:
            raise ValueError(f"No holdout file mapping found for recommended model: {model}")
        file_path, _ = current_mapping[model]
        
        # Print which type of holdout we're using based on USE_CV_FOLDS flag
        if USE_CV_FOLDS and model in CV_HOLDOUT_FILE_MAPPING:
            print(f"  {model}: CV holdout")
        else:
            print(f"  {model}: Simple holdout")
        
        required_files.append((model, file_path))
    
    missing = [(model, path) for model, path in required_files if not os.path.exists(path)]
    if missing:
        missing_text = "\n".join([f"{model}: {path}" for model, path in missing])
        raise FileNotFoundError(
            "Missing Step-4 holdout outputs:\n"
            f"{missing_text}"
        )
    
    # Load first model as base
    first_model = recommended_models[0]
    first_path, first_col = current_mapping[first_model]
    df = pd.read_csv(first_path)
    
    # Handle both CV and simple holdout column names for actual_bloom_doy
    if "bloom_doy" in df.columns and "actual_bloom_doy" not in df.columns:
        df["actual_bloom_doy"] = df["bloom_doy"]
    
    # Handle DTS special case
    if first_model == "dts":
        if "predicted_bloom_date" in df.columns:
            df["predicted_bloom_date"] = pd.to_datetime(df["predicted_bloom_date"], errors="coerce")
            df[f"pred_{first_model}"] = df["predicted_bloom_date"].dt.dayofyear
        elif "predicted_bloom_doy" in df.columns:
            df[f"pred_{first_model}"] = pd.to_numeric(df["predicted_bloom_doy"], errors="coerce")
        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()
        if "observed_bloom_date" in df.columns:
            df["observed_bloom_date"] = pd.to_datetime(df["observed_bloom_date"], errors="coerce")
            df["actual_bloom_doy"] = df["observed_bloom_date"].dt.dayofyear
        holdout = df[["location", "year", "actual_bloom_doy", f"pred_{first_model}"]].copy()
    else:
        holdout = df[["location", "year", "actual_bloom_doy", first_col]].rename(
            columns={first_col: f"pred_{first_model}"}
        ).copy()
    
    holdout["location"] = holdout["location"].apply(normalize_location)
    
    # Merge remaining models
    for model in recommended_models[1:]:
        file_path, pred_col = current_mapping[model]
        model_df = pd.read_csv(file_path)
        
        # Handle both CV and simple holdout column names
        if "bloom_doy" in model_df.columns and "actual_bloom_doy" not in model_df.columns:
            model_df["actual_bloom_doy"] = model_df["bloom_doy"]
        
        if model == "dts":
            if "predicted_bloom_date" in model_df.columns:
                model_df["predicted_bloom_date"] = pd.to_datetime(model_df["predicted_bloom_date"], errors="coerce")
                model_df[f"pred_{model}"] = model_df["predicted_bloom_date"].dt.dayofyear
            elif "predicted_bloom_doy" in model_df.columns:
                model_df[f"pred_{model}"] = pd.to_numeric(model_df["predicted_bloom_doy"], errors="coerce")
            if "status" in model_df.columns:
                model_df = model_df[model_df["status"] == "ok"].copy()
            model_df["location"] = model_df["location"].apply(normalize_location)
            holdout = holdout.merge(
                model_df[["location", "year", f"pred_{model}"]],
                on=["location", "year"],
                how="inner",
            )
        else:
            model_df["location"] = model_df["location"].apply(normalize_location)
            holdout = holdout.merge(
                model_df[["location", "year", pred_col]].rename(columns={pred_col: f"pred_{model}"}),
                on=["location", "year"],
                how="inner",
            )
    
    return holdout, recommended_models


def load_future_matrix(recommended_models):
    """Dynamically load future predictions from recommended models."""
    # Check all required files exist
    required_files = []
    for model in recommended_models:
        if model not in FUTURE_FILE_MAPPING:
            raise ValueError(f"No future file mapping found for recommended model: {model}")
        file_path, _ = FUTURE_FILE_MAPPING[model]
        required_files.append((model, file_path))
    
    missing = [(model, path) for model, path in required_files if not os.path.exists(path)]
    if missing:
        missing_text = "\n".join([f"{model}: {path}" for model, path in missing])
        raise FileNotFoundError(
            "Missing Step-4 future prediction outputs:\n"
            f"{missing_text}"
        )
    
    # Load first model as base
    first_model = recommended_models[0]
    first_path, first_col = FUTURE_FILE_MAPPING[first_model]
    df = pd.read_csv(first_path)
    
    # Handle DTS special case
    if first_model == "dts":
        if "predicted_bloom_doy" in df.columns:
            df[f"pred_{first_model}"] = pd.to_numeric(df["predicted_bloom_doy"], errors="coerce")
        elif "predicted_bloom_date" in df.columns:
            df["predicted_bloom_date"] = pd.to_datetime(df["predicted_bloom_date"], errors="coerce")
            df[f"pred_{first_model}"] = df["predicted_bloom_date"].dt.dayofyear
        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()
        future = df[["location", f"pred_{first_model}"]].copy()
    else:
        future = df[["location", first_col]].rename(columns={first_col: f"pred_{first_model}"}).copy()
    
    future["location"] = future["location"].apply(normalize_location)
    
    # Merge remaining models
    for model in recommended_models[1:]:
        file_path, pred_col = FUTURE_FILE_MAPPING[model]
        model_df = pd.read_csv(file_path)
        
        if model == "dts":
            if "predicted_bloom_doy" in model_df.columns:
                model_df[f"pred_{model}"] = pd.to_numeric(model_df["predicted_bloom_doy"], errors="coerce")
            elif "predicted_bloom_date" in model_df.columns:
                model_df["predicted_bloom_date"] = pd.to_datetime(model_df["predicted_bloom_date"], errors="coerce")
                model_df[f"pred_{model}"] = model_df["predicted_bloom_date"].dt.dayofyear
            if "status" in model_df.columns:
                model_df = model_df[model_df["status"] == "ok"].copy()
            model_df["location"] = model_df["location"].apply(normalize_location)
            future = future.merge(
                model_df[["location", f"pred_{model}"]],
                on="location",
                how="inner",
            )
        else:
            model_df["location"] = model_df["location"].apply(normalize_location)
            future = future.merge(
                model_df[["location", pred_col]].rename(columns={pred_col: f"pred_{model}"}),
                on="location",
                how="inner",
            )
    
    future = future[future["location"].isin(TARGET_PREDICTION_LOCATIONS)].copy()
    return future


def main():
    print("=" * 80)
    print("STACKED ENSEMBLE FROM STEP-4 HOLDOUT OUTPUTS")
    print("=" * 80)
    print(f"Config: USE_CV_FOLDS={USE_CV_FOLDS}")

    print("\n--- Loading Step-4 holdout predictions ---")
    holdout, recommended_models = load_holdout_matrix()

    meta_features = [f"pred_{model}" for model in recommended_models]

    holdout = holdout.dropna(subset=["actual_bloom_doy"] + meta_features).copy()
    if holdout.empty:
        raise ValueError("No usable rows in merged holdout matrix.")

    X_meta = holdout[meta_features].values
    y_meta = holdout["actual_bloom_doy"].values

    print(f"Merged holdout rows for meta-model: {len(holdout)}")
    print(f"Holdout year range: {int(holdout['year'].min())}-{int(holdout['year'].max())}")

    metrics_records = []
    for feature in meta_features:
        feature_mae, feature_mse, feature_rmse = evaluate(y_meta, holdout[feature].values, f"Base {feature}")
        metrics_records.append(
            {
                "model": feature,
                "mae_days": round(float(feature_mae), 4),
                "mse_days2": round(float(feature_mse), 4),
                "rmse_days": round(float(feature_rmse), 4),
            }
        )

    print("\n--- Training stacked meta-model (RidgeCV) ---")
    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)
    meta_model.fit(X_meta, y_meta)

    holdout["stacked_pred"] = meta_model.predict(X_meta)
    holdout["simple_avg"] = holdout[meta_features].mean(axis=1)
    holdout["abs_error_stacked"] = np.abs(holdout["stacked_pred"] - holdout["actual_bloom_doy"])

    observed_vs_pred_full = holdout[["location", "year", "actual_bloom_doy", "stacked_pred", "simple_avg"] + meta_features].copy()
    observed_vs_pred_full["predicted_doy"] = observed_vs_pred_full["stacked_pred"].round(1)
    observed_vs_pred_full["observed_bloom_date"] = observed_vs_pred_full.apply(
        lambda row: doy_to_date(row["year"], row["actual_bloom_doy"]),
        axis=1,
    )
    observed_vs_pred_full["predicted_bloom_date"] = observed_vs_pred_full.apply(
        lambda row: doy_to_date(row["year"], row["predicted_doy"]),
        axis=1,
    )
    observed_vs_pred_full["abs_error_days"] = (
        observed_vs_pred_full["actual_bloom_doy"] - observed_vs_pred_full["predicted_doy"]
    ).abs().round(1)

    stack_mae, stack_mse, stack_rmse = evaluate(y_meta, holdout["stacked_pred"].values, "Stacked ensemble")
    avg_mae, avg_mse, avg_rmse = evaluate(y_meta, holdout["simple_avg"].values, "Simple average")
    metrics_records.append(
        {
            "model": "stacked_ensemble",
            "mae_days": round(float(stack_mae), 4),
            "mse_days2": round(float(stack_mse), 4),
            "rmse_days": round(float(stack_rmse), 4),
        }
    )
    metrics_records.append(
        {
            "model": "simple_average",
            "mae_days": round(float(avg_mae), 4),
            "mse_days2": round(float(avg_mse), 4),
            "rmse_days": round(float(avg_rmse), 4),
        }
    )
    improvement = ((avg_mae - stack_mae) / avg_mae) * 100 if avg_mae != 0 else 0.0
    print(f"Improvement over simple average: {improvement:.1f}%")
    print(
        f"Stacked vs Simple Average -> MSE improvement: {avg_mse - stack_mse:.2f}, "
        f"RMSE improvement: {avg_rmse - stack_rmse:.2f} days"
    )

    if holdout["abs_error_stacked"].dropna().empty:
        raise ValueError("Cannot calibrate prediction interval: no holdout residuals available.")

    q_hat = float(np.quantile(holdout["abs_error_stacked"].dropna().values, 1 - PI_ALPHA))
    empirical_coverage = float((holdout["abs_error_stacked"] <= q_hat).mean())
    print(
        f"Calibrated {(1-PI_ALPHA)*100:.0f}% PI half-width from holdout residuals: {q_hat:.2f} days "
        f"(empirical holdout coverage: {empirical_coverage*100:.1f}%)"
    )

    print("\n--- Loading Step-4 2026 predictions ---")
    future = load_future_matrix(recommended_models)
    if future.empty:
        raise ValueError("No overlapping 2026 predictions across Step-4 model outputs.")

    X_future = future[meta_features].values
    future["stacked_ensemble"] = meta_model.predict(X_future)
    future["simple_average"] = future[meta_features].mean(axis=1)
    future["90_pi_lower"] = (future["stacked_ensemble"] - q_hat).round(1)
    future["90_pi_upper"] = (future["stacked_ensemble"] + q_hat).round(1)
    future["interval_halfwidth_days"] = q_hat
    future["predicted_doy"] = future["stacked_ensemble"].round(1)
    future["predicted_date"] = future["predicted_doy"].apply(lambda doy: doy_to_date(2026, doy))
    future["90_pi_lower_date"] = future["90_pi_lower"].apply(lambda doy: doy_to_date(2026, doy))
    future["90_pi_upper_date"] = future["90_pi_upper"].apply(lambda doy: doy_to_date(2026, doy))

    output_cols = [
        "location",
        "predicted_date",
        "predicted_doy",
        "90_pi_lower",
        "90_pi_upper",
        "interval_halfwidth_days",
        "90_pi_lower_date",
        "90_pi_upper_date",
        "simple_average",
        "stacked_ensemble",
    ] + meta_features
    future = future[output_cols].sort_values("location").reset_index(drop=True)

    os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    future.to_csv(OUTPUT_ENSEMBLE, index=False)
    print(f"Saved stacked 2026 predictions: {OUTPUT_ENSEMBLE}")
    observed_vs_pred_full = observed_vs_pred_full.sort_values(["location", "year"]).reset_index(drop=True)
    observed_vs_pred_full.to_csv(OUTPUT_OBS_PRED_FULL, index=False)
    print(f"Saved observed vs predicted (full available rows): {OUTPUT_OBS_PRED_FULL}")

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df["holdout_rows"] = len(holdout)
    metrics_df["holdout_start_year"] = int(holdout["year"].min())
    metrics_df["holdout_end_year"] = int(holdout["year"].max())
    metrics_df["pi_alpha"] = PI_ALPHA
    metrics_df["pi_halfwidth_days"] = round(float(q_hat), 4)
    metrics_df["empirical_coverage"] = round(float(empirical_coverage), 4)
    metrics_df["mae_improvement_pct_vs_simple_avg"] = round(float(improvement), 4)
    metrics_df["mse_improvement_vs_simple_avg"] = round(float(avg_mse - stack_mse), 4)
    metrics_df["rmse_improvement_vs_simple_avg_days"] = round(float(avg_rmse - stack_rmse), 4)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved model metrics: {OUTPUT_METRICS}")

    plot_observed_vs_expected_scatter_by_location(observed_vs_pred_full)
    plot_observed_vs_expected_timeseries_by_location(observed_vs_pred_full)
    plot_residuals_by_location(observed_vs_pred_full)
    print(f"Saved observed vs expected scatter plot by location: {OUTPUT_OBS_EXP_SCATTER}")
    print(f"Saved observed vs expected time series by location: {OUTPUT_OBS_EXP_TIMESERIES}")
    print(f"Saved residuals plot by location: {OUTPUT_RESIDUALS}")

    weight_df = pd.DataFrame(
        {
            "feature": meta_features,
            "coefficient": meta_model.coef_,
        }
    )
    coef_sum = weight_df["coefficient"].sum()
    if coef_sum != 0:
        weight_df["weight_percent"] = (weight_df["coefficient"] / coef_sum * 100).round(1)
    else:
        weight_df["weight_percent"] = 0.0
    weight_df.to_csv(OUTPUT_WEIGHTS, index=False)
    print(f"Saved meta-model weights: {OUTPUT_WEIGHTS}")

    print("\nFinal stacked predictions:")
    print(future[["location", "predicted_date", "predicted_doy"]].to_string(index=False))


if __name__ == "__main__":
    main()
