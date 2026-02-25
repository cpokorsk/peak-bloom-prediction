import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from phenology_config import (
    MODEL_FEATURES_FILE,
    FINAL_PREDICTIONS_FILE,
    HOLDOUT_OUTPUT_DIR,
    MODEL_OUTPUT_DIR,
    HOLDOUT_LAST_N_YEARS,
    MIN_MODEL_YEAR,
    TARGET_PREDICTION_LOCATIONS,
    USE_CV_FOLDS,
    CV_FOLDS_FILE,
    CV_CONFIG_FILE,
    CV_ACTIVE_SPLIT,
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
OUTPUT_CV_METRICS_RIDGE = os.path.join(MODEL_OUTPUT_DIR, "cv_metrics_ridge.csv")
OUTPUT_CV_METRICS_LASSO = os.path.join(MODEL_OUTPUT_DIR, "cv_metrics_lasso.csv")
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
# 2. CV UTILITIES
# ==========================================
def load_cv_configuration():
    if not os.path.exists(CV_FOLDS_FILE):
        raise FileNotFoundError(f"CV folds file not found: {CV_FOLDS_FILE}\nRun 3c_year_block_folds.py first.")
    if not os.path.exists(CV_CONFIG_FILE):
        raise FileNotFoundError(f"CV config file not found: {CV_CONFIG_FILE}\nRun 3c_year_block_folds.py first.")
    return pd.read_csv(CV_FOLDS_FILE), pd.read_csv(CV_CONFIG_FILE)

def get_cv_splits(folds_df, config_df, active_split=None):
    splits = []
    split_indices = [active_split] if active_split else config_df['cv_split'].tolist()
    for split_idx in split_indices:
        config_row = config_df[config_df['cv_split'] == split_idx].iloc[0]
        test_fold = config_row['test_fold']
        train_folds = [int(f) for f in config_row['train_folds'].split(',')]
        train_years = folds_df[folds_df['fold'].isin(train_folds)]['year'].tolist()
        test_years = folds_df[folds_df['fold'] == test_fold]['year'].tolist()
        splits.append({
            'split_id': split_idx,
            'train_years': set(train_years),
            'test_years': set(test_years),
            'test_fold': test_fold,
        })
    return splits

# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================
def evaluate(model, x, y, label):
    if x.empty:
        return
    preds = model.predict(x)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"{label} -> MAE: {mae:.2f} days | RMSE: {rmse:.2f} days | R²: {r2:.3f}")

def train_and_evaluate_split(train_df, test_df, split_name=""):
    x_train = train_df[PREDICTOR_COLUMNS]
    y_train = train_df["bloom_doy"]
    x_test = test_df[PREDICTOR_COLUMNS]
    y_test = test_df["bloom_doy"]
    
    ridge = build_pipeline(RidgeCV(alphas=np.logspace(-3, 3, 25)))
    lasso = build_pipeline(LassoCV(alphas=np.logspace(-3, 1, 25), max_iter=5000))
    
    ridge.fit(x_train, y_train)
    lasso.fit(x_train, y_train)
    
    ridge_train_preds = ridge.predict(x_train)
    ridge_train_mae = mean_absolute_error(y_train, ridge_train_preds)
    ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_preds))
    ridge_train_r2 = r2_score(y_train, ridge_train_preds)
    
    ridge_test_preds = ridge.predict(x_test)
    ridge_test_mae = mean_absolute_error(y_test, ridge_test_preds)
    ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_preds))
    ridge_test_r2 = r2_score(y_test, ridge_test_preds)
    
    lasso_train_preds = lasso.predict(x_train)
    lasso_train_mae = mean_absolute_error(y_train, lasso_train_preds)
    lasso_train_rmse = np.sqrt(mean_squared_error(y_train, lasso_train_preds))
    lasso_train_r2 = r2_score(y_train, lasso_train_preds)
    
    lasso_test_preds = lasso.predict(x_test)
    lasso_test_mae = mean_absolute_error(y_test, lasso_test_preds)
    lasso_test_rmse = np.sqrt(mean_squared_error(y_test, lasso_test_preds))
    lasso_test_r2 = r2_score(y_test, lasso_test_preds)
    
    print(f"{split_name}:")
    print(f"  Ridge Train: MAE={ridge_train_mae:.2f}, RMSE={ridge_train_rmse:.2f}, R²={ridge_train_r2:.3f} (n={len(train_df)})")
    print(f"  Ridge Test:  MAE={ridge_test_mae:.2f}, RMSE={ridge_test_rmse:.2f}, R²={ridge_test_r2:.3f} (n={len(test_df)})")
    print(f"  Lasso Train: MAE={lasso_train_mae:.2f}, RMSE={lasso_train_rmse:.2f}, R²={lasso_train_r2:.3f} (n={len(train_df)})")
    print(f"  Lasso Test:  MAE={lasso_test_mae:.2f}, RMSE={lasso_test_rmse:.2f}, R²={lasso_test_r2:.3f} (n={len(test_df)})")
    
    holdout_output = test_df[["location", "year", "bloom_doy"]].copy()
    holdout_output["predicted_doy_ridge"] = ridge_test_preds.round(1)
    holdout_output["predicted_doy_lasso"] = lasso_test_preds.round(1)
    holdout_output["abs_error_days_ridge"] = (holdout_output["predicted_doy_ridge"] - holdout_output["bloom_doy"]).abs().round(1)
    holdout_output["abs_error_days_lasso"] = (holdout_output["predicted_doy_lasso"] - holdout_output["bloom_doy"]).abs().round(1)
    holdout_output = holdout_output.rename(columns={"bloom_doy": "actual_bloom_doy"})
    
    ridge_metrics = {
        'train_n': len(train_df),
        'test_n': len(test_df),
        'train_mae': round(ridge_train_mae, 3),
        'train_rmse': round(ridge_train_rmse, 3),
        'train_r2': round(ridge_train_r2, 3),
        'test_mae': round(ridge_test_mae, 3),
        'test_rmse': round(ridge_test_rmse, 3),
        'test_r2': round(ridge_test_r2, 3),
    }
    
    lasso_metrics = {
        'train_n': len(train_df),
        'test_n': len(test_df),
        'train_mae': round(lasso_train_mae, 3),
        'train_rmse': round(lasso_train_rmse, 3),
        'train_r2': round(lasso_train_r2, 3),
        'test_mae': round(lasso_test_mae, 3),
        'test_rmse': round(lasso_test_rmse, 3),
        'test_r2': round(lasso_test_r2, 3),
    }
    
    return (ridge, lasso), holdout_output, (ridge_metrics, lasso_metrics)

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

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print("--- Loading Data ---")
    features_df = pd.read_csv(FEATURES_FILE)
    features_df["location"] = features_df["location"].apply(normalize_location)
    if "is_future" not in features_df.columns:
        features_df["is_future"] = False

    df = features_df[(features_df["is_future"] == False) & (features_df["year"] >= MIN_YEAR)].copy()
    df = df.dropna(subset=["bloom_doy"] + PREDICTOR_COLUMNS)

    if USE_CV_FOLDS:
        print(f"\n{'='*80}")
        print("MODE: Year-Block Cross-Validation")
        print(f"{'='*80}")
        
        folds_df, config_df = load_cv_configuration()
        splits = get_cv_splits(folds_df, config_df, active_split=CV_ACTIVE_SPLIT)
        print(f"\nRunning {len(splits)} CV split(s)...")
        
        ridge_cv_metrics = []
        lasso_cv_metrics = []
        all_holdout_outputs = []
        
        for split_info in splits:
            split_id = split_info['split_id']
            train_years = split_info['train_years']
            test_years = split_info['test_years']
            
            train_df = df[df['year'].isin(train_years)].copy()
            test_df = df[df['year'].isin(test_years)].copy()
            
            print(f"\n--- CV Split {split_id} (Test Fold {split_info['test_fold']}) ---")
            print(f"Train years: {min(train_years)}-{max(train_years)}")
            print(f"Test years: {min(test_years)}-{max(test_years)}")
            
            models, holdout_output, (ridge_metrics, lasso_metrics) = train_and_evaluate_split(train_df, test_df, f"Split {split_id}")
            
            ridge_metrics['split_id'] = split_id
            ridge_metrics['test_fold'] = split_info['test_fold']
            ridge_cv_metrics.append(ridge_metrics)
            
            lasso_metrics['split_id'] = split_id
            lasso_metrics['test_fold'] = split_info['test_fold']
            lasso_cv_metrics.append(lasso_metrics)
            
            holdout_output['cv_split'] = split_id
            all_holdout_outputs.append(holdout_output)
        
        ridge_metrics_df = pd.DataFrame(ridge_cv_metrics)
        ridge_mean_metrics = ridge_metrics_df[['test_mae', 'test_rmse', 'test_r2']].mean()
        ridge_std_metrics = ridge_metrics_df[['test_mae', 'test_rmse', 'test_r2']].std()
        
        lasso_metrics_df = pd.DataFrame(lasso_cv_metrics)
        lasso_mean_metrics = lasso_metrics_df[['test_mae', 'test_rmse', 'test_r2']].mean()
        lasso_std_metrics = lasso_metrics_df[['test_mae', 'test_rmse', 'test_r2']].std()
        
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY - Ridge")
        print(f"{'='*80}")
        print(f"Test MAE:  {ridge_mean_metrics['test_mae']:.2f} ± {ridge_std_metrics['test_mae']:.2f} days")
        print(f"Test RMSE: {ridge_mean_metrics['test_rmse']:.2f} ± {ridge_std_metrics['test_rmse']:.2f} days")
        print(f"Test R²:   {ridge_mean_metrics['test_r2']:.3f} ± {ridge_std_metrics['test_r2']:.3f}")
        
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY - Lasso")
        print(f"{'='*80}")
        print(f"Test MAE:  {lasso_mean_metrics['test_mae']:.2f} ± {lasso_std_metrics['test_mae']:.2f} days")
        print(f"Test RMSE: {lasso_mean_metrics['test_rmse']:.2f} ± {lasso_std_metrics['test_rmse']:.2f} days")
        print(f"Test R²:   {lasso_mean_metrics['test_r2']:.3f} ± {lasso_std_metrics['test_r2']:.3f}")
        
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        ridge_metrics_df.to_csv(OUTPUT_CV_METRICS_RIDGE, index=False)
        lasso_metrics_df.to_csv(OUTPUT_CV_METRICS_LASSO, index=False)
        print(f"\nRidge CV metrics saved to: {OUTPUT_CV_METRICS_RIDGE}")
        print(f"Lasso CV metrics saved to: {OUTPUT_CV_METRICS_LASSO}")
        
        all_holdout_df = pd.concat(all_holdout_outputs, ignore_index=True)
        output_holdout_cv = os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_ridge_lasso.csv")
        os.makedirs(os.path.dirname(output_holdout_cv), exist_ok=True)
        all_holdout_df.to_csv(output_holdout_cv, index=False)
        print(f"Holdout predictions saved to: {output_holdout_cv}")
        
        print(f"\n--- Training Final Models on All Historical Data ---")
        x_all = df[PREDICTOR_COLUMNS]
        y_all = df["bloom_doy"]
        
        ridge = build_pipeline(RidgeCV(alphas=np.logspace(-3, 3, 25)))
        lasso = build_pipeline(LassoCV(alphas=np.logspace(-3, 1, 25), max_iter=5000))
        ridge.fit(x_all, y_all)
        lasso.fit(x_all, y_all)
    else:
        print(f"\n{'='*80}")
        print(f"MODE: Simple Holdout (Last {HOLDOUT_LAST_N_YEARS} Years)")
        print(f"{'='*80}")
        
        years = sorted(df["year"].dropna().unique().tolist())
        if len(years) <= HOLDOUT_LAST_N_YEARS:
            raise ValueError(f"Need more than {HOLDOUT_LAST_N_YEARS} unique years for holdout split.")

        holdout_years = set(years[-HOLDOUT_LAST_N_YEARS:])
        train_years = set(years[:-HOLDOUT_LAST_N_YEARS])

        train = df[df["year"].isin(train_years)].copy()
        df_holdout = df[df["year"].isin(holdout_years)].copy()

        print(f"\nTraining set: {len(train)} records (years {min(train_years)}-{max(train_years)})")
        print(f"Holdout set: {len(df_holdout)} records (years {min(holdout_years)}-{max(holdout_years)})")

        print("\n--- Training Ridge & Lasso ---")
        (ridge, lasso), holdout_output, _ = train_and_evaluate_split(train, df_holdout, f"Last {HOLDOUT_LAST_N_YEARS} Years")
        
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
