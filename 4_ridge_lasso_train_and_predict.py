import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from phenology_config import (
    MODEL_FEATURES_FILE,
    FINAL_PREDICTIONS_FILE,
    HOLDOUT_OUTPUT_DIR,
    HOLDOUT_LAST_N_YEARS,
    MIN_MODEL_YEAR,
    TARGET_PREDICTION_LOCATIONS,
    normalize_location,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
FEATURES_FILE = MODEL_FEATURES_FILE
OUTPUT_PREDICTIONS = FINAL_PREDICTIONS_FILE.replace(
    ".csv", "_ridge_lasso.csv"
)
OUTPUT_HOLDOUT = os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_ridge_lasso.csv")
MIN_YEAR = MIN_MODEL_YEAR

PREDICTOR_COLUMNS = [
    "mean_tmax_early_spring",
    "mean_tmin_early_spring",
    "max_tmax_early_spring",
    "total_prcp_early_spring",
    "chill_days_oct1_dec31",
    "observed_gdd_to_bloom",
    "species",
    "continent",
]

NUMERIC_FEATURES = [
    "mean_tmax_early_spring",
    "mean_tmin_early_spring",
    "max_tmax_early_spring",
    "total_prcp_early_spring",
    "chill_days_oct1_dec31",
    "observed_gdd_to_bloom",
]

CATEGORICAL_FEATURES = ["species", "continent"]

# ==========================================
# 2. MAIN EXECUTION
# ==========================================

def evaluate(model, x, y, label):
    if x.empty:
        return
    preds = model.predict(x)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"{label} -> MAE: {mae:.2f} days | RMSE: {rmse:.2f} days")


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main():
    print("--- Loading Data ---")
    features_df = pd.read_csv(FEATURES_FILE)
    features_df["location"] = features_df["location"].apply(normalize_location)
    if "is_future" not in features_df.columns:
        features_df["is_future"] = False

    df = features_df[(features_df["is_future"] == False) & (features_df["year"] >= MIN_YEAR)].copy()
    df = df.dropna(subset=["bloom_doy"] + PREDICTOR_COLUMNS)

    print(f"\n--- Splitting Data (Last {HOLDOUT_LAST_N_YEARS} Years Holdout) ---")
    years = sorted(df["year"].dropna().unique().tolist())
    if len(years) <= HOLDOUT_LAST_N_YEARS:
        raise ValueError(f"Need more than {HOLDOUT_LAST_N_YEARS} unique years for holdout split.")

    holdout_years = set(years[-HOLDOUT_LAST_N_YEARS:])
    train_years = set(years[:-HOLDOUT_LAST_N_YEARS])

    train = df[df["year"].isin(train_years)].copy()
    df_holdout = df[df["year"].isin(holdout_years)].copy()

    print(f"Training set: {len(train)} records (years {min(train_years)}-{max(train_years)})")
    print(f"Holdout set: {len(df_holdout)} records (years {min(holdout_years)}-{max(holdout_years)})")

    x_train = train[PREDICTOR_COLUMNS]
    y_train = train["bloom_doy"]
    x_holdout = df_holdout[PREDICTOR_COLUMNS]
    y_holdout = df_holdout["bloom_doy"]

    print("\n--- Training Ridge ---")
    ridge = build_pipeline(RidgeCV(alphas=np.logspace(-3, 3, 25)))
    ridge.fit(x_train, y_train)

    evaluate(ridge, x_train, y_train, "Ridge Train")
    evaluate(ridge, x_holdout, y_holdout, f"Ridge Holdout (Last {HOLDOUT_LAST_N_YEARS} Years)")

    print("\n--- Training Lasso ---")
    lasso = build_pipeline(LassoCV(alphas=np.logspace(-3, 1, 25), max_iter=5000))
    lasso.fit(x_train, y_train)

    evaluate(lasso, x_train, y_train, "Lasso Train")
    evaluate(lasso, x_holdout, y_holdout, f"Lasso Holdout (Last {HOLDOUT_LAST_N_YEARS} Years)")

    if not df_holdout.empty:
        holdout_output = df_holdout[["location", "year", "bloom_doy"]].copy()
        holdout_output["predicted_doy_ridge"] = ridge.predict(x_holdout).round(1)
        holdout_output["predicted_doy_lasso"] = lasso.predict(x_holdout).round(1)
        holdout_output["abs_error_days_ridge"] = (holdout_output["predicted_doy_ridge"] - holdout_output["bloom_doy"]).abs().round(1)
        holdout_output["abs_error_days_lasso"] = (holdout_output["predicted_doy_lasso"] - holdout_output["bloom_doy"]).abs().round(1)
        holdout_output = holdout_output.rename(columns={"bloom_doy": "actual_bloom_doy"})
        os.makedirs(os.path.dirname(OUTPUT_HOLDOUT), exist_ok=True)
        holdout_output.to_csv(OUTPUT_HOLDOUT, index=False)
        print(f"Holdout predictions saved to: {OUTPUT_HOLDOUT}")

    print("\n--- Generating 2026 Predictions ---")
    df_2026 = features_df[features_df["is_future"] == True].copy()
    df_2026 = df_2026[df_2026["location"].isin(TARGET_PREDICTION_LOCATIONS)]
    df_2026 = df_2026.dropna(subset=PREDICTOR_COLUMNS)

    if df_2026.empty:
        raise ValueError("No complete 2026 feature rows available for prediction.")

    x_future = df_2026[PREDICTOR_COLUMNS]

    df_2026 = df_2026.reset_index(drop=True)
    df_2026["predicted_doy_ridge"] = ridge.predict(x_future).round(1)
    df_2026["predicted_doy_lasso"] = lasso.predict(x_future).round(1)

    def doy_to_date(year, doy):
        if pd.isna(doy) or pd.isna(year):
            return None
        return (
            pd.to_datetime(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy) - 1)
        ).strftime("%b %d")

    df_2026["predicted_date_ridge"] = df_2026.apply(
        lambda x: doy_to_date(x["year"], x["predicted_doy_ridge"]), axis=1
    )
    df_2026["predicted_date_lasso"] = df_2026.apply(
        lambda x: doy_to_date(x["year"], x["predicted_doy_lasso"]), axis=1
    )

    final_cols = [
        "location",
        "predicted_date_ridge",
        "predicted_doy_ridge",
        "predicted_date_lasso",
        "predicted_doy_lasso",
    ]
    final_predictions = df_2026[final_cols]

    print(final_predictions.to_string(index=False))

    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS), exist_ok=True)
    final_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"\nFinal predictions saved to: {OUTPUT_PREDICTIONS}")


if __name__ == "__main__":
    main()
