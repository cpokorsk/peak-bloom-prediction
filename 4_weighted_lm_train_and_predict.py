import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from phenology_config import (
    MODEL_FEATURES_FILE,
    HOLDOUT_OUTPUT_DIR,
    PREDICTIONS_OUTPUT_DIR,
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
OUTPUT_PREDICTIONS = os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions_weighted_lm.csv")
OUTPUT_HOLDOUT = os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_last{HOLDOUT_LAST_N_YEARS}y_weighted_lm.csv")
OUTPUT_CV_METRICS = os.path.join(MODEL_OUTPUT_DIR, "cv_metrics_weighted_lm.csv")
MIN_YEAR = MIN_MODEL_YEAR

# WEIGHTING PARAMETERS
WEIGHT_HALF_LIFE_YEARS = 20  # Observations lose half their weight after this many years
WEIGHT_METHOD = "exponential"  # Options: "exponential", "linear", "none"


# ==========================================
# 2. WEIGHTING FUNCTIONS
# ==========================================
def calculate_weights(years, reference_year=None, method="exponential", half_life=20):
    """
    Calculate time-based weights for observations.
    
    Parameters:
    -----------
    years : array-like
        Array of observation years
    reference_year : int, optional
        Reference year (most recent gets weight 1.0). If None, uses max(years)
    method : str
        Weighting method: "exponential", "linear", or "none"
    half_life : float
        Years until weight decreases by half (for exponential) or to zero (for linear)
    
    Returns:
    --------
    weights : np.array
        Weight for each observation (0 to 1)
    """
    years = np.array(years)
    if reference_year is None:
        reference_year = years.max()
    
    years_ago = reference_year - years
    
    if method == "exponential":
        # Exponential decay: w = exp(-λt) where λ = ln(2)/half_life
        decay_rate = np.log(2) / half_life
        weights = np.exp(-decay_rate * years_ago)
    elif method == "linear":
        # Linear decay: w = max(0, 1 - t/half_life)
        weights = np.maximum(0, 1 - years_ago / half_life)
    elif method == "none":
        # Uniform weights
        weights = np.ones_like(years, dtype=float)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights


# ==========================================
# 3. CV UTILITIES
# ==========================================
def load_cv_configuration():
    """Load year-block fold assignments for cross-validation."""
    if not os.path.exists(CV_FOLDS_FILE):
        raise FileNotFoundError(
            f"CV folds file not found: {CV_FOLDS_FILE}\n"
            "Run 3c_year_block_folds.py first to generate folds."
        )
    if not os.path.exists(CV_CONFIG_FILE):
        raise FileNotFoundError(
            f"CV config file not found: {CV_CONFIG_FILE}\n"
            "Run 3c_year_block_folds.py first to generate configuration."
        )
    
    folds_df = pd.read_csv(CV_FOLDS_FILE)
    config_df = pd.read_csv(CV_CONFIG_FILE)
    
    return folds_df, config_df


def get_cv_splits(folds_df, config_df, active_split=None):
    """Get train/test year sets for each CV split."""
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
# 4. TRAINING & EVALUATION
# ==========================================
def train_and_evaluate_split(train_df, test_df, formula, split_name="", weight_method="exponential", half_life=20):
    """Train weighted model on train_df and evaluate on test_df."""
    
    # Calculate weights for training data
    train_weights = calculate_weights(
        train_df['year'].values,
        reference_year=train_df['year'].max(),
        method=weight_method,
        half_life=half_life
    )
    
    # Train weighted least squares model
    model = smf.wls(formula=formula, data=train_df, weights=train_weights).fit()
    
    # Evaluate on train
    train_preds = model.predict(train_df)
    train_mae = mean_absolute_error(train_df['bloom_doy'], train_preds)
    train_rmse = np.sqrt(mean_squared_error(train_df['bloom_doy'], train_preds))
    train_r2 = r2_score(train_df['bloom_doy'], train_preds)
    
    # Evaluate on test
    test_preds = model.predict(test_df)
    test_mae = mean_absolute_error(test_df['bloom_doy'], test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_df['bloom_doy'], test_preds))
    test_r2 = r2_score(test_df['bloom_doy'], test_preds)
    
    # Calculate weighted metrics for train set
    train_weighted_mae = np.average(np.abs(train_df['bloom_doy'] - train_preds), weights=train_weights)
    train_weighted_rmse = np.sqrt(np.average((train_df['bloom_doy'] - train_preds)**2, weights=train_weights))
    
    print(f"{split_name}:")
    print(f"  Train: MAE={train_mae:.2f}, RMSE={train_rmse:.2f}, R²={train_r2:.3f} (n={len(train_df)})")
    print(f"  Train (weighted): MAE={train_weighted_mae:.2f}, RMSE={train_weighted_rmse:.2f}")
    print(f"  Test:  MAE={test_mae:.2f}, RMSE={test_rmse:.2f}, R²={test_r2:.3f} (n={len(test_df)})")
    print(f"  Weight range: [{train_weights.min():.3f}, {train_weights.max():.3f}]")
    
    # Create holdout output
    holdout_output = test_df[['location', 'year', 'bloom_doy']].copy()
    holdout_output['predicted_doy'] = test_preds.round(1)
    holdout_output['abs_error_days'] = (holdout_output['predicted_doy'] - holdout_output['bloom_doy']).abs().round(1)
    holdout_output['model_name'] = 'weighted_lm'
    holdout_output = holdout_output.rename(columns={'bloom_doy': 'actual_bloom_doy'})
    
    metrics = {
        'train_n': len(train_df),
        'test_n': len(test_df),
        'train_mae': round(train_mae, 3),
        'train_rmse': round(train_rmse, 3),
        'train_r2': round(train_r2, 3),
        'train_weighted_mae': round(train_weighted_mae, 3),
        'train_weighted_rmse': round(train_weighted_rmse, 3),
        'test_mae': round(test_mae, 3),
        'test_rmse': round(test_rmse, 3),
        'test_r2': round(test_r2, 3),
        'weight_min': round(train_weights.min(), 4),
        'weight_max': round(train_weights.max(), 4),
        'weight_mean': round(train_weights.mean(), 4),
    }
    
    return model, holdout_output, metrics


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    print("=" * 80)
    print("TIME-WEIGHTED LINEAR MODEL (WLS)")
    print("=" * 80)
    print(f"Weighting method: {WEIGHT_METHOD}")
    print(f"Half-life: {WEIGHT_HALF_LIFE_YEARS} years")
    print("=" * 80)
    
    print("\n--- Loading Data ---")
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
    
    formula = "bloom_doy ~ mean_tmax_early_spring + C(species) + C(continent)"
    
    # ==========================================
    # MODE 1: Year-Block Cross-Validation
    # ==========================================
    if USE_CV_FOLDS:
        print(f"\n{'='*80}")
        print("MODE: Year-Block Cross-Validation")
        print(f"{'='*80}")
        
        folds_df, config_df = load_cv_configuration()
        splits = get_cv_splits(folds_df, config_df, active_split=CV_ACTIVE_SPLIT)
        
        print(f"\nRunning {len(splits)} CV split(s)...")
        
        cv_metrics = []
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
            
            model, holdout_output, metrics = train_and_evaluate_split(
                train_df, test_df, formula, f"Split {split_id}",
                weight_method=WEIGHT_METHOD, half_life=WEIGHT_HALF_LIFE_YEARS
            )
            
            metrics['split_id'] = split_id
            metrics['test_fold'] = split_info['test_fold']
            cv_metrics.append(metrics)
            
            holdout_output['cv_split'] = split_id
            all_holdout_outputs.append(holdout_output)
        
        # Aggregate CV metrics
        cv_metrics_df = pd.DataFrame(cv_metrics)
        mean_metrics = cv_metrics_df[['test_mae', 'test_rmse', 'test_r2']].mean()
        std_metrics = cv_metrics_df[['test_mae', 'test_rmse', 'test_r2']].std()
        
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Test MAE:  {mean_metrics['test_mae']:.2f} ± {std_metrics['test_mae']:.2f} days")
        print(f"Test RMSE: {mean_metrics['test_rmse']:.2f} ± {std_metrics['test_rmse']:.2f} days")
        print(f"Test R²:   {mean_metrics['test_r2']:.3f} ± {std_metrics['test_r2']:.3f}")
        
        # Save CV metrics
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        cv_metrics_df.to_csv(OUTPUT_CV_METRICS, index=False)
        print(f"\nCV metrics saved to: {OUTPUT_CV_METRICS}")
        
        # Save aggregated holdout outputs
        all_holdout_df = pd.concat(all_holdout_outputs, ignore_index=True)
        output_holdout_cv = os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_weighted_lm.csv")
        os.makedirs(os.path.dirname(output_holdout_cv), exist_ok=True)
        all_holdout_df.to_csv(output_holdout_cv, index=False)
        print(f"Holdout predictions saved to: {output_holdout_cv}")
        
        # Train final model on all data for 2026 predictions
        print(f"\n--- Training Final Model on All Historical Data ---")
        final_weights = calculate_weights(
            df['year'].values,
            reference_year=df['year'].max(),
            method=WEIGHT_METHOD,
            half_life=WEIGHT_HALF_LIFE_YEARS
        )
        final_model = smf.wls(formula=formula, data=df, weights=final_weights).fit()
        print(f"Final model weight range: [{final_weights.min():.3f}, {final_weights.max():.3f}]")
    
    # ==========================================
    # MODE 2: Simple Last-N-Years Holdout
    # ==========================================
    else:
        print(f"\n{'='*80}")
        print(f"MODE: Simple Holdout (Last {HOLDOUT_LAST_N_YEARS} Years)")
        print(f"{'='*80}")
        
        years = sorted(df['year'].dropna().unique().tolist())
        if len(years) <= HOLDOUT_LAST_N_YEARS:
            raise ValueError(f"Need more than {HOLDOUT_LAST_N_YEARS} unique years for holdout split.")

        holdout_years = set(years[-HOLDOUT_LAST_N_YEARS:])
        train_years = set(years[:-HOLDOUT_LAST_N_YEARS])

        train = df[df['year'].isin(train_years)].copy()
        df_holdout = df[df['year'].isin(holdout_years)].copy()

        print(f"\nTraining set: {len(train)} records (years {min(train_years)}-{max(train_years)})")
        print(f"Holdout set: {len(df_holdout)} records (years {min(holdout_years)}-{max(holdout_years)})")

        print("\n--- Training Weighted Model ---")
        final_model, holdout_output, metrics = train_and_evaluate_split(
            train, df_holdout, formula, f"Last {HOLDOUT_LAST_N_YEARS} Years",
            weight_method=WEIGHT_METHOD, half_life=WEIGHT_HALF_LIFE_YEARS
        )
        
        print("\n" + str(final_model.summary().tables[1]))

        # Feature importance via absolute t-values
        importance_df = pd.DataFrame({
            'feature': final_model.params.index,
            'coefficient': final_model.params.values,
            't_value': final_model.tvalues.values,
            'abs_t_value': np.abs(final_model.tvalues.values)
        })
        importance_df = importance_df[importance_df['feature'] != 'Intercept']
        importance_df = importance_df.sort_values('abs_t_value', ascending=False).reset_index(drop=True)

        print("\n--- Feature Importance (absolute t-values) ---")
        print(importance_df.head(15).to_string(index=False))

        importance_path = os.path.join(MODEL_OUTPUT_DIR, 'feature_importance_weighted_lm.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to: {importance_path}")
        
        # Save holdout predictions
        os.makedirs(os.path.dirname(OUTPUT_HOLDOUT), exist_ok=True)
        holdout_output.to_csv(OUTPUT_HOLDOUT, index=False)
        print(f"Holdout predictions saved to: {OUTPUT_HOLDOUT}")

    # ==========================================
    # GENERATE 2026 PREDICTIONS
    # ==========================================
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
    predictions = final_model.get_prediction(df_2026_features)
    pred_summary = predictions.summary_frame(alpha=0.10)
    
    df_2026_features['predicted_doy'] = pred_summary['mean'].round(1)
    df_2026_features['90_pi_lower'] = pred_summary['obs_ci_lower'].round(1)
    df_2026_features['90_pi_upper'] = pred_summary['obs_ci_upper'].round(1)
    df_2026_features['interval_halfwidth_days'] = ((df_2026_features['90_pi_upper'] - df_2026_features['90_pi_lower']) / 2).round(1)
    
    # Convert DOY to actual calendar dates
    def doy_to_date(year, doy):
        if pd.isna(doy) or pd.isna(year):
            return None
        return (pd.to_datetime(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy)-1)).strftime("%b %d")
        
    df_2026_features['predicted_date'] = df_2026_features.apply(lambda x: doy_to_date(x['year'], x['predicted_doy']), axis=1)
    df_2026_features['90_pi_lower_date'] = df_2026_features.apply(lambda x: doy_to_date(x['year'], x['90_pi_lower']), axis=1)
    df_2026_features['90_pi_upper_date'] = df_2026_features.apply(lambda x: doy_to_date(x['year'], x['90_pi_upper']), axis=1)

    # Clean up and save
    final_cols = [
        'location', 'predicted_date', 'predicted_doy', 
        '90_pi_lower', '90_pi_upper', 'interval_halfwidth_days',
        '90_pi_lower_date', '90_pi_upper_date'
    ]
    final_predictions = df_2026_features[final_cols]
    
    print(final_predictions.to_string(index=False))
    
    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS), exist_ok=True)
    final_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"\nFinal predictions saved to: {OUTPUT_PREDICTIONS}")


if __name__ == "__main__":
    main()
