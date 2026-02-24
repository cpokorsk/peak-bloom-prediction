import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from phenology_config import (
    MODEL_FEATURES_FILE,
    FINAL_PREDICTIONS_FILE,
    HOLDOUT_LOCATIONS,
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
        'mean_tmax_early_spring',
        'mean_tmin_early_spring',
        'max_tmax_early_spring',
        'min_tmin_early_spring',
        'observed_gdd_to_bloom',
        'total_prcp_early_spring',
        'chill_days_oct1_dec31',
        'alt',
        'species',
        'continent'
    ]

    df = features_df[(features_df['is_future'] == False) & (features_df['year'] >= MIN_YEAR)].copy()
    df = df.dropna(subset=['bloom_doy'] + required_predictors)
    
    print("\n--- Splitting Data ---")
    # 1. Separate the Holdout Test Set (Vancouver & NYC)
    holdout_mask = df['location'].isin(set(HOLDOUT_LOCATIONS))
    df_holdout = df[holdout_mask].copy()
    df_main = df[~holdout_mask].copy()
    
    # 2. Split Main pool into Train, Validation, and Main Test sets
    # Train 70%, Val 15%, Test 15%
    train_val, test_main = train_test_split(df_main, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=(0.15/0.85), random_state=42)
    
    print(f"Training set: {len(train)} records")
    print(f"Validation set: {len(val)} records")
    print(f"Main Test set: {len(test_main)} records")
    print(f"Holdout Test set (NYC/Vancouver): {len(df_holdout)} records")

    print("\n--- Training Model ---")
    # Using Ordinary Least Squares (OLS) which provides exact 90% Prediction Intervals
    formula = "bloom_doy ~ mean_tmax_early_spring + mean_tmin_early_spring + max_tmax_early_spring + min_tmin_early_spring + observed_gdd_to_bloom + total_prcp_early_spring + chill_days_oct1_dec31 + alt + C(species) + C(continent)"
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

    print("\n--- Generating 2026 Predictions with 90% Confidence Intervals ---")
    df_2026_features = features_df[features_df['is_future'] == True].copy()
    df_2026_features = df_2026_features[df_2026_features['location'].isin(TARGET_PREDICTION_LOCATIONS)]
    df_2026_features = df_2026_features.dropna(subset=required_predictors)

    if df_2026_features.empty:
        raise ValueError("No complete 2026 feature rows available for prediction.")
    
    # Generate Predictions & 90% Intervals (alpha=0.10)
    predictions = model.get_prediction(df_2026_features)
    pred_summary = predictions.summary_frame(alpha=0.10)
    
    df_2026_features['predicted_doy'] = pred_summary['mean'].round(1)
    df_2026_features['90_ci_lower'] = pred_summary['obs_ci_lower'].round(1)
    df_2026_features['90_ci_upper'] = pred_summary['obs_ci_upper'].round(1)
    df_2026_features['interval_width_days'] = (df_2026_features['90_ci_upper'] - df_2026_features['90_ci_lower']).round(1)
    
    # Convert DOY to actual calendar dates
    def doy_to_date(year, doy):
        if pd.isna(doy) or pd.isna(year):
            return None
        return (pd.to_datetime(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy)-1)).strftime("%b %d")
        
    df_2026_features['predicted_date'] = df_2026_features.apply(lambda x: doy_to_date(x['year'], x['predicted_doy']), axis=1)

    # Clean up and save
    final_cols = ['location', 'predicted_date', 'predicted_doy', '90_ci_lower', '90_ci_upper', 'interval_width_days']
    final_predictions = df_2026_features[final_cols]
    
    print(final_predictions.to_string(index=False))
    
    final_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"\nFinal predictions saved to: {OUTPUT_PREDICTIONS}")

if __name__ == "__main__":
    main()