import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from phenology_config import (
    MODEL_FEATURES_FILE,
    FINAL_PREDICTIONS_FILE,
    HOLDOUT_LOCATIONS,
    HOLDOUT_EXTRA_COUNTRIES,
    HOLDOUT_PER_COUNTRY,
    HOLDOUT_RANDOM_SEED,
    MIN_MODEL_YEAR,
    TARGET_PREDICTION_LOCATIONS,
    normalize_location,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
FEATURES_FILE = MODEL_FEATURES_FILE
OUTPUT_PREDICTIONS = FINAL_PREDICTIONS_FILE.replace(
    ".csv", "_bayesian_ridge.csv"
)
MIN_YEAR = MIN_MODEL_YEAR

PREDICTOR_COLUMNS = [
    "max_tmax_early_spring",
    "total_prcp_early_spring",
    "species",
    "continent",
]

NUMERIC_FEATURES = [
    "max_tmax_early_spring",
    "total_prcp_early_spring",
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

    print("\n--- Splitting Data ---")
    base_holdout = set(HOLDOUT_LOCATIONS)
    rng = np.random.default_rng(HOLDOUT_RANDOM_SEED)

    location_meta = (
        df[["location", "country_code"]]
        .dropna(subset=["location"])
        .drop_duplicates(subset=["location"])
        .copy()
    )

    extra_holdout = set()
    for country in HOLDOUT_EXTRA_COUNTRIES:
        candidates = location_meta[location_meta["country_code"] == country]["location"]
        candidates = [
            loc for loc in candidates
            if loc not in base_holdout and loc not in TARGET_PREDICTION_LOCATIONS
        ]
        if candidates:
            pick_count = min(HOLDOUT_PER_COUNTRY, len(candidates))
            extra_holdout.update(rng.choice(candidates, size=pick_count, replace=False).tolist())

    holdout_locations = base_holdout.union(extra_holdout)
    holdout_mask = df["location"].isin(holdout_locations)
    df_holdout = df[holdout_mask].copy()
    df_main = df[~holdout_mask].copy()

    years = sorted(df_main["year"].dropna().unique().tolist())
    if len(years) < 3:
        raise ValueError("Not enough unique years for time-based split.")

    train_cut = int(len(years) * 0.70)
    val_cut = int(len(years) * 0.85)

    train_years = set(years[:train_cut])
    val_years = set(years[train_cut:val_cut])
    test_years = set(years[val_cut:])

    train = df_main[df_main["year"].isin(train_years)].copy()
    val = df_main[df_main["year"].isin(val_years)].copy()
    test_main = df_main[df_main["year"].isin(test_years)].copy()

    print(f"Training set: {len(train)} records")
    print(f"Validation set: {len(val)} records")
    print(f"Main Test set: {len(test_main)} records")
    print(f"Holdout Test set (NYC/Vancouver): {len(df_holdout)} records")

    x_train = train[PREDICTOR_COLUMNS]
    y_train = train["bloom_doy"]
    x_val = val[PREDICTOR_COLUMNS]
    y_val = val["bloom_doy"]
    x_test = test_main[PREDICTOR_COLUMNS]
    y_test = test_main["bloom_doy"]
    x_holdout = df_holdout[PREDICTOR_COLUMNS]
    y_holdout = df_holdout["bloom_doy"]

    print("\n--- Training Bayesian Ridge ---")
    bayes = build_pipeline(BayesianRidge())
    bayes.fit(x_train, y_train)

    evaluate(bayes, x_train, y_train, "Bayesian Ridge Train")
    evaluate(bayes, x_val, y_val, "Bayesian Ridge Validation")
    evaluate(bayes, x_test, y_test, "Bayesian Ridge Main Test")
    evaluate(bayes, x_holdout, y_holdout, "Bayesian Ridge Holdout")

    print("\n--- Generating 2026 Predictions ---")
    df_2026 = features_df[features_df["is_future"] == True].copy()
    df_2026 = df_2026[df_2026["location"].isin(TARGET_PREDICTION_LOCATIONS)]
    df_2026 = df_2026.dropna(subset=PREDICTOR_COLUMNS)

    if df_2026.empty:
        raise ValueError("No complete 2026 feature rows available for prediction.")

    x_future = df_2026[PREDICTOR_COLUMNS]

    df_2026 = df_2026.reset_index(drop=True)
    preds_mean, preds_std = bayes.predict(x_future, return_std=True)
    df_2026["predicted_doy"] = np.round(preds_mean, 1)
    df_2026["predicted_doy_std"] = np.round(preds_std, 1)
    df_2026["90_ci_lower"] = np.round(preds_mean - 1.645 * preds_std, 1)
    df_2026["90_ci_upper"] = np.round(preds_mean + 1.645 * preds_std, 1)

    def doy_to_date(year, doy):
        if pd.isna(doy) or pd.isna(year):
            return None
        return (
            pd.to_datetime(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy) - 1)
        ).strftime("%b %d")

    df_2026["predicted_date"] = df_2026.apply(
        lambda x: doy_to_date(x["year"], x["predicted_doy"]), axis=1
    )

    final_cols = [
        "location",
        "predicted_date",
        "predicted_doy",
        "predicted_doy_std",
        "90_ci_lower",
        "90_ci_upper",
    ]
    final_predictions = df_2026[final_cols]

    print(final_predictions.to_string(index=False))

    final_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"\nFinal predictions saved to: {OUTPUT_PREDICTIONS}")


if __name__ == "__main__":
    main()
