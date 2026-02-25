import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from phenology_config import (
    MODEL_FEATURES_FILE,
    FINAL_PREDICTIONS_FILE,
    HOLDOUT_OUTPUT_DIR,
    HOLDOUT_LAST_N_YEARS,
    MIN_MODEL_YEAR,
    TARGET_PREDICTION_LOCATIONS,
    USE_CV_FOLDS,
    normalize_location,
)


FEATURES_FILE = MODEL_FEATURES_FILE
OUTPUT_PREDICTIONS = FINAL_PREDICTIONS_FILE.replace(
    ".csv", "_random_forest.csv"
)
OUTPUT_HOLDOUT = os.path.join(
    HOLDOUT_OUTPUT_DIR,
    f"holdout_last{HOLDOUT_LAST_N_YEARS}y_random_forest.csv",
)
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



def evaluate(model, x, y, label):
    if x.empty:
        return
    preds = model.predict(x)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"{label} -> MAE: {mae:.2f} days | RMSE: {rmse:.2f} days | R²: {r2:.3f}")



def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )



def predict_with_tree_quantiles(model_pipeline, x, lower_q=0.05, upper_q=0.95):
    preprocessed = model_pipeline.named_steps["preprocess"].transform(x)
    forest = model_pipeline.named_steps["model"]

    tree_preds = np.array([tree.predict(preprocessed) for tree in forest.estimators_])
    mean_pred = tree_preds.mean(axis=0)
    lower_pred = np.quantile(tree_preds, lower_q, axis=0)
    upper_pred = np.quantile(tree_preds, upper_q, axis=0)

    return mean_pred, lower_pred, upper_pred



def doy_to_date(year, doy):
    if pd.isna(doy) or pd.isna(year):
        return None
    return (
        pd.to_datetime(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy) - 1)
    ).strftime("%b %d")



def main():
    print("=" * 80)
    print("RANDOM FOREST MODEL")
    print("=" * 80)
    if USE_CV_FOLDS:
        print("Note: CV mode not supported for this model. Using simple holdout.")
    print(f"Holdout: Last {HOLDOUT_LAST_N_YEARS} years")
    print("=" * 80)
    
    print("\n--- Loading Data ---")
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

    print("\n--- Training Random Forest ---")
    rf = build_pipeline(
        RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    )
    rf.fit(x_train, y_train)

    evaluate(rf, x_train, y_train, "Random Forest Train")
    evaluate(rf, x_holdout, y_holdout, f"Random Forest Holdout (Last {HOLDOUT_LAST_N_YEARS} Years)")

    if not df_holdout.empty:
        holdout_mean, holdout_lower, holdout_upper = predict_with_tree_quantiles(rf, x_holdout)
        holdout_output = df_holdout[["location", "year", "bloom_doy"]].copy()
        holdout_output["predicted_doy"] = np.round(holdout_mean, 1)
        holdout_output["90_pi_lower"] = np.round(holdout_lower, 1)
        holdout_output["90_pi_upper"] = np.round(holdout_upper, 1)
        holdout_output["abs_error_days"] = (holdout_output["predicted_doy"] - holdout_output["bloom_doy"]).abs().round(1)
        holdout_output["model_name"] = "random_forest"
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
    pred_mean, pred_lower, pred_upper = predict_with_tree_quantiles(rf, x_future)

    df_2026["predicted_doy"] = np.round(pred_mean, 1)
    df_2026["90_pi_lower"] = np.round(pred_lower, 1)
    df_2026["90_pi_upper"] = np.round(pred_upper, 1)
    df_2026["interval_halfwidth_days"] = np.round(
        (df_2026["90_pi_upper"] - df_2026["90_pi_lower"]) / 2.0, 1
    )

    df_2026["predicted_date"] = df_2026.apply(
        lambda x: doy_to_date(x["year"], x["predicted_doy"]), axis=1
    )

    final_cols = [
        "location",
        "predicted_date",
        "predicted_doy",
        "90_pi_lower",
        "90_pi_upper",
        "interval_halfwidth_days",
    ]
    final_predictions = df_2026[final_cols]

    print(final_predictions.to_string(index=False))

    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS), exist_ok=True)
    final_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"\nFinal predictions saved to: {OUTPUT_PREDICTIONS}")


if __name__ == "__main__":
    main()
