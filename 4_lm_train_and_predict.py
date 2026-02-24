import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from phenology_config import (
    MODEL_FEATURES_FILE,
    FINAL_PREDICTIONS_FILE,
    HOLDOUT_LOCATIONS,
    HOLDOUT_EXTRA_COUNTRIES,
    HOLDOUT_PER_COUNTRY,
    HOLDOUT_RANDOM_SEED,
    MIN_MODEL_YEAR,
    TARGET_YEAR,
    TARGET_PREDICTION_LOCATIONS,
    normalize_location,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
FEATURES_FILE = MODEL_FEATURES_FILE
OUTPUT_PREDICTIONS = FINAL_PREDICTIONS_FILE
MIN_YEAR = MIN_MODEL_YEAR

# ==========================================
# 2. MAIN EXECUTION
# ==========================================
def main():
    print("--- Loading Data ---")
    features_df = pd.read_csv(FEATURES_FILE)
    features_df['location'] = features_df['location'].apply(normalize_location)
    if 'is_future' not in features_df.columns:
        features_df['is_future'] = False

    required_predictors = [
        'max_tmax_early_spring',
        'total_prcp_early_spring',
        'species',
        'continent'
    ]

    df = features_df[(features_df['is_future'] == False) & (features_df['year'] >= MIN_YEAR)].copy()
    df = df.dropna(subset=['bloom_doy'] + required_predictors)
    
    print("\n--- Splitting Data ---")
    # 1. Separate the Holdout Test Set (Vancouver & NYC)
    # 1. Separate the Holdout Test Set (target holdouts + extra country holdouts)
    base_holdout = set(HOLDOUT_LOCATIONS)
    rng = np.random.default_rng(HOLDOUT_RANDOM_SEED)

    location_meta = (
        df[['location', 'country_code']]
        .dropna(subset=['location'])
        .drop_duplicates(subset=['location'])
        .copy()
    )

    extra_holdout = set()
    for country in HOLDOUT_EXTRA_COUNTRIES:
        candidates = location_meta[location_meta['country_code'] == country]['location']
        candidates = [loc for loc in candidates if loc not in base_holdout and loc not in TARGET_PREDICTION_LOCATIONS]
        if candidates:
            pick_count = min(HOLDOUT_PER_COUNTRY, len(candidates))
            extra_holdout.update(rng.choice(candidates, size=pick_count, replace=False).tolist())

    holdout_locations = base_holdout.union(extra_holdout)
    holdout_mask = df['location'].isin(holdout_locations)
    df_holdout = df[holdout_mask].copy()
    df_main = df[~holdout_mask].copy()
    
    # 2. Split Main pool into Train, Validation, and Main Test sets by time
    # Train 70%, Val 15%, Test 15% using year-based slices
    years = sorted(df_main['year'].dropna().unique().tolist())
    if len(years) < 3:
        raise ValueError("Not enough unique years for time-based split.")

    train_cut = int(len(years) * 0.70)
    val_cut = int(len(years) * 0.85)

    train_years = set(years[:train_cut])
    val_years = set(years[train_cut:val_cut])
    test_years = set(years[val_cut:])

    train = df_main[df_main['year'].isin(train_years)].copy()
    val = df_main[df_main['year'].isin(val_years)].copy()
    test_main = df_main[df_main['year'].isin(test_years)].copy()
    
    print(f"Training set: {len(train)} records")
    print(f"Validation set: {len(val)} records")
    print(f"Main Test set: {len(test_main)} records")
    print(f"Holdout Test set (NYC/Vancouver): {len(df_holdout)} records")

    print("\n--- Training Model ---")
    # Using Ordinary Least Squares (OLS) which provides exact 90% Prediction Intervals
    formula = "bloom_doy ~ observed_gdd_to_bloom + chill_days_oct1_dec31 + total_prcp_early_spring + C(species)"
    model = smf.ols(formula=formula, data=train).fit()
    
    print(model.summary().tables[1])

    # Feature importance via absolute t-values
    importance_df = pd.DataFrame({
        'feature': model.params.index,
        'coefficient': model.params.values,
        't_value': model.tvalues.values,
        'abs_t_value': np.abs(model.tvalues.values)
    })
    importance_df = importance_df[importance_df['feature'] != 'Intercept']
    importance_df = importance_df.sort_values('abs_t_value', ascending=False).reset_index(drop=True)

    print("\n--- Feature Importance (absolute t-values) ---")
    print(importance_df.head(15).to_string(index=False))

    importance_path = os.path.join(os.path.dirname(FEATURES_FILE), 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
    print("\n--- Evaluating Model ---")
    def evaluate(dataset, name):
        if dataset.empty: return
        preds = model.predict(dataset)
        mae = mean_absolute_error(dataset['bloom_doy'], preds)
        rmse = np.sqrt(mean_squared_error(dataset['bloom_doy'], preds))
        print(f"{name} -> MAE: {mae:.2f} days | RMSE: {rmse:.2f} days")

    evaluate(train, "Train Set")
    evaluate(val, "Validation Set")
    evaluate(test_main, "Main Test Set")
    evaluate(df_holdout, "Holdout Test Set (Generalization)")

    print("\n--- Generating 2026 Predictions with 90% Prediction Intervals ---")
    df_2026_features = features_df[features_df['is_future'] == True].copy()
    if df_2026_features.empty:
        raise ValueError("No 2026 feature rows found. Run 3_feature_engineering.py to generate future features.")

    missing_columns = [col for col in required_predictors if col not in df_2026_features.columns]
    if missing_columns:
        raise ValueError(f"Missing required predictors in 2026 features: {', '.join(missing_columns)}")

    df_2026_features = df_2026_features[df_2026_features['location'].isin(TARGET_PREDICTION_LOCATIONS)]
    if df_2026_features.empty:
        raise ValueError("No 2026 feature rows for target locations. Check TARGET_PREDICTION_LOCATIONS or feature generation.")

    missing_counts = df_2026_features[required_predictors].isna().sum()
    if missing_counts.any():
        print("Missing predictor counts in 2026 features:")
        print(missing_counts[missing_counts > 0].to_string())

    df_2026_features = df_2026_features.dropna(subset=required_predictors)

    if df_2026_features.empty:
        raise ValueError("No complete 2026 feature rows available for prediction.")

    df_2026_features = df_2026_features.reset_index(drop=True)
    
    # Generate Predictions & 90% Intervals (alpha=0.10)
    predictions = model.get_prediction(df_2026_features)
    pred_summary = predictions.summary_frame(alpha=0.10)
    
    df_2026_features['predicted_doy'] = pred_summary['mean'].round(1)
    df_2026_features['90_ci_lower'] = pred_summary['obs_ci_lower'].round(1)
    df_2026_features['90_ci_upper'] = pred_summary['obs_ci_upper'].round(1)
    df_2026_features['interval_halfwidth_days'] = ((df_2026_features['90_ci_upper'] - df_2026_features['90_ci_lower']) / 2).round(1)
    
    # Convert DOY to actual calendar dates
    def doy_to_date(year, doy):
        if pd.isna(doy) or pd.isna(year):
            return None
        return (pd.to_datetime(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy)-1)).strftime("%b %d")
        
    df_2026_features['predicted_date'] = df_2026_features.apply(lambda x: doy_to_date(x['year'], x['predicted_doy']), axis=1)

    # Clean up and save
    final_cols = ['location', 'predicted_date', 'predicted_doy', '90_ci_lower', '90_ci_upper', 'interval_halfwidth_days']
    final_predictions = df_2026_features[final_cols]
    
    print(final_predictions.to_string(index=False))
    
    final_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"\nFinal predictions saved to: {OUTPUT_PREDICTIONS}")

if __name__ == "__main__":
    main()