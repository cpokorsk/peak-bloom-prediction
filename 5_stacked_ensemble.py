"""Stacked ensemble built from Step-4 exported holdout predictions."""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from phenology_config import (
    MODEL_OUTPUT_DIR,
    HOLDOUT_OUTPUT_DIR,
    PREDICTIONS_OUTPUT_DIR,
    HOLDOUT_LAST_N_YEARS,
    TARGET_PREDICTION_LOCATIONS,
    normalize_location,
)


HOLDOUT_FILES = {
    "linear_ols": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_linear_ols.csv"),
    "bayesian_ridge": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_bayesian_ridge.csv"),
    "arimax": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_arimax.csv"),
}

FUTURE_FILES = {
    "linear_ols": os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions.csv"),
    "bayesian_ridge": os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_bayesian_ridge.csv"),
    "arimax": os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_arimax.csv"),
}

OUTPUT_ENSEMBLE = os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_stacked_ensemble.csv")
OUTPUT_WEIGHTS = os.path.join(MODEL_OUTPUT_DIR, "stacked_ensemble_meta_model_weights.csv")
PI_ALPHA = 0.10


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label} -> MAE: {mae:.2f} days | RMSE: {rmse:.2f} days")
    return mae, rmse


def doy_to_date(year, doy):
    if pd.isna(doy) or pd.isna(year):
        return None
    return (pd.to_datetime(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy) - 1)).strftime("%b %d")


def load_holdout_matrix():
    required = [
        HOLDOUT_FILES["linear_ols"],
        HOLDOUT_FILES["bayesian_ridge"],
        HOLDOUT_FILES["arimax"],
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "Missing Step-4 holdout outputs. Run all Step-4 scripts first:\n"
            f"{missing_text}"
        )

    lm = pd.read_csv(HOLDOUT_FILES["linear_ols"])
    br = pd.read_csv(HOLDOUT_FILES["bayesian_ridge"])
    ar = pd.read_csv(HOLDOUT_FILES["arimax"])

    holdout = lm[["location", "year", "actual_bloom_doy", "predicted_doy"]].rename(
        columns={"predicted_doy": "pred_linear_ols"}
    )
    holdout["location"] = holdout["location"].apply(normalize_location)

    holdout = holdout.merge(
        br[["location", "year", "predicted_doy"]].rename(columns={"predicted_doy": "pred_bayesian_ridge"}),
        on=["location", "year"],
        how="inner",
    )
    holdout = holdout.merge(
        ar[["location", "year", "predicted_doy"]].rename(columns={"predicted_doy": "pred_arimax"}),
        on=["location", "year"],
        how="inner",
    )

    return holdout


def load_future_matrix():
    required = [
        FUTURE_FILES["linear_ols"],
        FUTURE_FILES["bayesian_ridge"],
        FUTURE_FILES["arimax"],
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "Missing Step-4 future prediction outputs. Run all Step-4 scripts first:\n"
            f"{missing_text}"
        )

    lm = pd.read_csv(FUTURE_FILES["linear_ols"])
    br = pd.read_csv(FUTURE_FILES["bayesian_ridge"])
    ar = pd.read_csv(FUTURE_FILES["arimax"])

    future = lm[["location", "predicted_doy"]].rename(columns={"predicted_doy": "pred_linear_ols"})
    future["location"] = future["location"].apply(normalize_location)

    future = future.merge(
        br[["location", "predicted_doy"]].rename(columns={"predicted_doy": "pred_bayesian_ridge"}),
        on="location",
        how="inner",
    )
    future = future.merge(
        ar[["location", "predicted_doy"]].rename(columns={"predicted_doy": "pred_arimax"}),
        on="location",
        how="inner",
    )

    future = future[future["location"].isin(TARGET_PREDICTION_LOCATIONS)].copy()
    return future


def main():
    print("=" * 80)
    print("STACKED ENSEMBLE FROM STEP-4 HOLDOUT OUTPUTS")
    print("=" * 80)

    print("\n--- Loading Step-4 holdout predictions ---")
    holdout = load_holdout_matrix()

    meta_features = [
        "pred_linear_ols",
        "pred_bayesian_ridge",
        "pred_arimax",
    ]

    holdout = holdout.dropna(subset=["actual_bloom_doy"] + meta_features).copy()
    if holdout.empty:
        raise ValueError("No usable rows in merged holdout matrix.")

    X_meta = holdout[meta_features].values
    y_meta = holdout["actual_bloom_doy"].values

    print(f"Merged holdout rows for meta-model: {len(holdout)}")
    print(f"Holdout year range: {int(holdout['year'].min())}-{int(holdout['year'].max())}")

    for feature in meta_features:
        evaluate(y_meta, holdout[feature].values, f"Base {feature}")

    print("\n--- Training stacked meta-model (RidgeCV) ---")
    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)
    meta_model.fit(X_meta, y_meta)

    holdout["stacked_pred"] = meta_model.predict(X_meta)
    holdout["simple_avg"] = holdout[meta_features].mean(axis=1)
    holdout["abs_error_stacked"] = np.abs(holdout["stacked_pred"] - holdout["actual_bloom_doy"])

    stack_mae, _ = evaluate(y_meta, holdout["stacked_pred"].values, "Stacked ensemble")
    avg_mae, _ = evaluate(y_meta, holdout["simple_avg"].values, "Simple average")
    improvement = ((avg_mae - stack_mae) / avg_mae) * 100 if avg_mae != 0 else 0.0
    print(f"Improvement over simple average: {improvement:.1f}%")

    if holdout["abs_error_stacked"].dropna().empty:
        raise ValueError("Cannot calibrate prediction interval: no holdout residuals available.")

    q_hat = float(np.quantile(holdout["abs_error_stacked"].dropna().values, 1 - PI_ALPHA))
    empirical_coverage = float((holdout["abs_error_stacked"] <= q_hat).mean())
    print(
        f"Calibrated {(1-PI_ALPHA)*100:.0f}% PI half-width from holdout residuals: {q_hat:.2f} days "
        f"(empirical holdout coverage: {empirical_coverage*100:.1f}%)"
    )

    print("\n--- Loading Step-4 2026 predictions ---")
    future = load_future_matrix()
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

    weight_df = pd.DataFrame(
        {
            "base_model": meta_features,
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
