"""
Stacked Ensemble Model for Peak Bloom Prediction

Stacking (also called meta-learning):
1. Train multiple base models on training data
2. Generate predictions from base models on holdout data
3. Train a meta-model that learns optimal combination of base predictions
4. Use meta-model to make final predictions

This approach can learn complex interactions between models and is often
superior to simple weighted averaging.
"""

import numpy as np
import pandas as pd
import warnings
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from phenology_config import (
    MODEL_FEATURES_FILE,
    HOLDOUT_LOCATIONS,
    HOLDOUT_EXTRA_COUNTRIES,
    HOLDOUT_PER_COUNTRY,
    HOLDOUT_RANDOM_SEED,
    MIN_MODEL_YEAR,
    TARGET_PREDICTION_LOCATIONS,
    normalize_location,
)

# ==========================================
# CONFIGURATION
# ==========================================
FEATURES_FILE = MODEL_FEATURES_FILE
MIN_YEAR = MIN_MODEL_YEAR
STACKING_HOLDOUT_YEARS = 10  # Use last 10 years for meta-model training

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

ARIMAX_EXOG_COLUMNS = [
    "max_tmax_early_spring",
    "total_prcp_early_spring",
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def evaluate(y_true, y_pred, label):
    """Calculate and print evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label} -> MAE: {mae:.2f} days | RMSE: {rmse:.2f} days")
    return mae, rmse


def build_sklearn_pipeline(model):
    """Build sklearn pipeline with preprocessing"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def fit_arimax_for_location(location_df):
    """Fit ARIMAX model for a specific location"""
    location_df = location_df.sort_values("year").drop_duplicates(
        subset=["year"], keep="last"
    ).copy()
    
    if len(location_df) < 12:
        return None  # Not enough data for ARIMAX
    
    annual_index = pd.PeriodIndex(location_df["year"].astype(int), freq="Y")
    y_train = pd.Series(location_df["bloom_doy"].astype(float).values, index=annual_index)
    x_train = location_df[ARIMAX_EXOG_COLUMNS].astype(float).copy()
    x_train.index = annual_index
    
    try:
        model = SARIMAX(
            endog=y_train,
            exog=x_train,
            order=(1, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results = model.fit(disp=False, maxiter=100)
        return results
    except Exception:
        return None


def predict_arimax(arimax_models, data_df):
    """Generate predictions using location-specific ARIMAX models"""
    predictions = np.full(len(data_df), np.nan)
    
    for idx, row in data_df.iterrows():
        location = row['location']
        if location in arimax_models and arimax_models[location] is not None:
            try:
                # For ARIMAX, we need to predict one step ahead with exog
                exog_values = row[ARIMAX_EXOG_COLUMNS].values.reshape(1, -1)
                pred = arimax_models[location].forecast(steps=1, exog=exog_values)
                predictions[idx] = pred.iloc[0] if hasattr(pred, 'iloc') else pred[0]
            except Exception:
                pass  # Keep NaN if prediction fails
    
    return predictions


def doy_to_date(year, doy):
    """Convert day of year to calendar date"""
    if pd.isna(doy) or pd.isna(year):
        return None
    return (pd.to_datetime(f"{int(year)}-01-01") + pd.Timedelta(days=int(doy)-1)).strftime("%b %d")


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("=" * 80)
    print("STACKED ENSEMBLE MODEL WITH 10-YEAR HOLDOUT")
    print("=" * 80)
    
    # Load data
    print("\n--- Loading Data ---")
    features_df = pd.read_csv(FEATURES_FILE)
    features_df["location"] = features_df["location"].apply(normalize_location)
    if "is_future" not in features_df.columns:
        features_df["is_future"] = False

    df = features_df[
        (features_df["is_future"] == False) & (features_df["year"] >= MIN_YEAR)
    ].copy()
    df = df.dropna(subset=["bloom_doy"] + PREDICTOR_COLUMNS)
    
    # Split data: last 10 years for stacking, rest for base model training
    print(f"\n--- Splitting Data (Last {STACKING_HOLDOUT_YEARS} Years for Meta-Model) ---")
    years = sorted(df["year"].unique())
    holdout_year_threshold = max(years) - STACKING_HOLDOUT_YEARS + 1
    
    train_mask = df["year"] < holdout_year_threshold
    stacking_mask = df["year"] >= holdout_year_threshold
    
    # Also separate location-based holdout for final evaluation
    holdout_locations = set(HOLDOUT_LOCATIONS)
    location_holdout_mask = df["location"].isin(holdout_locations)
    
    # Training set: old years, not in holdout locations
    train = df[train_mask & ~location_holdout_mask].copy()
    
    # Stacking set: recent years, not in holdout locations
    stacking = df[stacking_mask & ~location_holdout_mask].copy()
    
    # Test set: holdout locations only
    test = df[location_holdout_mask].copy()
    
    print(f"Training set: {len(train)} records (years < {holdout_year_threshold})")
    print(f"Stacking set: {len(stacking)} records (years >= {holdout_year_threshold})")
    print(f"Test set (holdout locations): {len(test)} records")
    print(f"Year range - Train: {train['year'].min()}-{train['year'].max()}, "
          f"Stacking: {stacking['year'].min()}-{stacking['year'].max()}")
    
    # Prepare data splits
    X_train = train[PREDICTOR_COLUMNS]
    y_train = train["bloom_doy"]
    X_stack = stacking[PREDICTOR_COLUMNS]
    y_stack = stacking["bloom_doy"]
    X_test = test[PREDICTOR_COLUMNS]
    y_test = test["bloom_doy"]
    
    # ==========================================
    # STEP 1: Train Base Models (Ridge, Lasso, Bayesian Ridge, ARIMAX)
    # ==========================================
    print("\n" + "=" * 80)
    print("STEP 1: TRAINING BASE MODELS (Ridge, Lasso, Bayesian Ridge, ARIMAX)")
    print("=" * 80)
    
    base_models = {}
    
    # Model 1: Ridge
    print("\n1. Ridge Regression")
    ridge_model = build_sklearn_pipeline(Ridge(alpha=1.0))
    ridge_model.fit(X_train, y_train)
    base_models['ridge'] = ridge_model
    ridge_train_pred = ridge_model.predict(X_train)
    evaluate(y_train, ridge_train_pred, "  Ridge Train")
    
    # Model 2: Lasso
    print("\n2. Lasso Regression")
    lasso_model = build_sklearn_pipeline(Lasso(alpha=0.1, max_iter=5000))
    lasso_model.fit(X_train, y_train)
    base_models['lasso'] = lasso_model
    lasso_train_pred = lasso_model.predict(X_train)
    evaluate(y_train, lasso_train_pred, "  Lasso Train")
    
    # Model 3: Bayesian Ridge
    print("\n3. Bayesian Ridge")
    bayes_model = build_sklearn_pipeline(BayesianRidge())
    bayes_model.fit(X_train, y_train)
    base_models['bayesian'] = bayes_model
    bayes_train_pred = bayes_model.predict(X_train)
    evaluate(y_train, bayes_train_pred, "  Bayesian Train")
    
    # Model 4: ARIMAX (location-specific time series)
    print("\n4. ARIMAX (per-location time series)")
    arimax_models = {}
    locations_trained = 0
    for location in train['location'].unique():
        loc_data = train[train['location'] == location].copy()
        if len(loc_data) >= 12:  # Minimum samples for ARIMAX
            arimax_model = fit_arimax_for_location(loc_data)
            if arimax_model is not None:
                arimax_models[location] = arimax_model
                locations_trained += 1
    
    print(f"  Trained ARIMAX models for {locations_trained}/{len(train['location'].unique())} locations")
    base_models['arimax'] = arimax_models
    
    # ==========================================
    # STEP 2: Generate Base Model Predictions on Stacking Set
    # ==========================================
    print("\n" + "=" * 80)
    print("STEP 2: GENERATING BASE MODEL PREDICTIONS ON STACKING SET")
    print("=" * 80)
    
    stacking_predictions = pd.DataFrame()
    stacking_predictions['actual'] = y_stack.values
    
    print("\nBase model performance on stacking set:")
    
    # Ridge predictions
    stacking_predictions['ridge_pred'] = ridge_model.predict(X_stack)
    evaluate(y_stack, stacking_predictions['ridge_pred'], "  Ridge")
    
    # Lasso predictions
    stacking_predictions['lasso_pred'] = lasso_model.predict(X_stack)
    evaluate(y_stack, stacking_predictions['lasso_pred'], "  Lasso")
    
    # Bayesian predictions
    stacking_predictions['bayesian_pred'] = bayes_model.predict(X_stack)
    evaluate(y_stack, stacking_predictions['bayesian_pred'], "  Bayesian")
    
    # ARIMAX predictions
    stacking_predictions['arimax_pred'] = predict_arimax(arimax_models, stacking)
    
    # Handle locations without ARIMAX models (fill with mean of other predictions)
    arimax_nan_count = stacking_predictions['arimax_pred'].isna().sum()
    if arimax_nan_count > 0:
        print(f"  Warning: {arimax_nan_count} ARIMAX predictions are NaN (no model for location)")
        # Fill NaN with mean of other base model predictions for those rows
        other_cols = ['ridge_pred', 'lasso_pred', 'bayesian_pred']
        stacking_predictions.loc[stacking_predictions['arimax_pred'].isna(), 'arimax_pred'] = \
            stacking_predictions.loc[stacking_predictions['arimax_pred'].isna(), other_cols].mean(axis=1)
    
    # Evaluate ARIMAX performance (on non-NaN original predictions)
    arimax_mask = ~stacking_predictions['arimax_pred'].isna()
    if arimax_mask.any():
        evaluate(y_stack, stacking_predictions['arimax_pred'], "  ARIMAX")
    
    # ==========================================
    # STEP 3: Train Meta-Model
    # ==========================================
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING META-MODEL (Stacked Ensemble)")
    print("=" * 80)
    
    # Prepare meta-features (base model predictions)
    meta_features = ['ridge_pred', 'lasso_pred', 'bayesian_pred', 'arimax_pred']
    X_meta = stacking_predictions[meta_features].values
    y_meta = stacking_predictions['actual'].values
    
    # Train meta-model (Ridge regression to prevent overfitting)
    print("\nTraining Ridge meta-model on base predictions...")
    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)
    meta_model.fit(X_meta, y_meta)
    
    print(f"Selected alpha: {meta_model.alpha_:.4f}")
    print("\nMeta-model coefficients (how much weight each base model gets):")
    for feature, coef in zip(meta_features, meta_model.coef_):
        print(f"  {feature:15s}: {coef:7.4f}")
    print(f"  {'intercept':15s}: {meta_model.intercept_:7.4f}")
    
    # Evaluate meta-model on stacking set
    meta_pred_stack = meta_model.predict(X_meta)
    evaluate(y_meta, meta_pred_stack, "\n  Meta-Model on Stacking Set")
    
    # ==========================================
    # STEP 4: Evaluate on Test Set (Holdout Locations)
    # ==========================================
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATION ON TEST SET (Holdout Locations)")
    print("=" * 80)
    
    if len(test) > 0:
        # Generate base model predictions on test set
        test_predictions = pd.DataFrame()
        test_predictions['actual'] = y_test.values
        test_predictions['ridge_pred'] = ridge_model.predict(X_test)
        test_predictions['lasso_pred'] = lasso_model.predict(X_test)
        test_predictions['bayesian_pred'] = bayes_model.predict(X_test)
        test_predictions['arimax_pred'] = predict_arimax(arimax_models, test)
        
        # Handle NaN in ARIMAX predictions
        arimax_nan_count = test_predictions['arimax_pred'].isna().sum()
        if arimax_nan_count > 0:
            print(f"  Warning: {arimax_nan_count} ARIMAX predictions are NaN on test set")
            other_cols = ['ridge_pred', 'lasso_pred', 'bayesian_pred']
            test_predictions.loc[test_predictions['arimax_pred'].isna(), 'arimax_pred'] = \
                test_predictions.loc[test_predictions['arimax_pred'].isna(), other_cols].mean(axis=1)
        
        print("\nBase model performance on test set:")
        evaluate(y_test, test_predictions['ridge_pred'], "  Ridge")
        evaluate(y_test, test_predictions['lasso_pred'], "  Lasso")
        evaluate(y_test, test_predictions['bayesian_pred'], "  Bayesian")
        evaluate(y_test, test_predictions['arimax_pred'], "  ARIMAX")
        
        # Generate stacked ensemble prediction
        X_meta_test = test_predictions[meta_features].values
        test_predictions['stacked_pred'] = meta_model.predict(X_meta_test)
        
        print("\nStacked ensemble performance on test set:")
        stack_mae, stack_rmse = evaluate(y_test, test_predictions['stacked_pred'], "  Stacked Ensemble")
        
        # Compare to simple average
        test_predictions['simple_avg'] = test_predictions[meta_features].mean(axis=1)
        avg_mae, avg_rmse = evaluate(y_test, test_predictions['simple_avg'], "  Simple Average (baseline)")
        
        improvement = ((avg_mae - stack_mae) / avg_mae) * 100
        print(f"\n✓ Improvement over simple average: {improvement:.1f}%")
    else:
        print("\nNo test data available (holdout locations not found)")
    
    # ==========================================
    # STEP 5: Generate 2026 Predictions
    # ==========================================
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING 2026 PREDICTIONS")
    print("=" * 80)
    
    df_2026 = features_df[features_df["is_future"] == True].copy()
    df_2026 = df_2026[df_2026["location"].isin(TARGET_PREDICTION_LOCATIONS)]
    df_2026 = df_2026.dropna(subset=PREDICTOR_COLUMNS)
    
    if df_2026.empty:
        raise ValueError("No complete 2026 feature rows available for prediction.")
    
    X_2026 = df_2026[PREDICTOR_COLUMNS]
    
    # Generate base model predictions for 2026
    print("\nGenerating base model predictions for 2026...")
    pred_2026 = pd.DataFrame()
    pred_2026['location'] = df_2026['location'].values
    pred_2026['year'] = df_2026['year'].values
        
    pred_2026['ridge_pred'] = ridge_model.predict(X_2026)
    pred_2026['lasso_pred'] = lasso_model.predict(X_2026)
    pred_2026['bayesian_pred'] = bayes_model.predict(X_2026)
    pred_2026['arimax_pred'] = predict_arimax(arimax_models, df_2026)
    
    # Handle NaN in ARIMAX predictions
    arimax_nan_count = pred_2026['arimax_pred'].isna().sum()
    if arimax_nan_count > 0:
        print(f"  Warning: {arimax_nan_count} ARIMAX predictions are NaN for 2026")
        other_cols = ['ridge_pred', 'lasso_pred', 'bayesian_pred']
        pred_2026.loc[pred_2026['arimax_pred'].isna(), 'arimax_pred'] = \
            pred_2026.loc[pred_2026['arimax_pred'].isna(), other_cols].mean(axis=1)
    
    # Generate stacked ensemble prediction
    X_meta_2026 = pred_2026[meta_features].values
    
    # Double-check for NaNs before passing to meta-model
    if np.isnan(X_meta_2026).any():
        print("  Warning: Meta-features contain NaN, filling with column means")
        for i in range(X_meta_2026.shape[1]):
            col_mean = stacking_predictions[meta_features[i]].mean()
            X_meta_2026[:, i] = np.where(np.isnan(X_meta_2026[:, i]), col_mean, X_meta_2026[:, i])
    
    pred_2026['stacked_ensemble'] = meta_model.predict(X_meta_2026)
    
    # Calculate simple average for comparison
    pred_2026['simple_average'] = pred_2026[meta_features].mean(axis=1)
    
    # Convert to dates
    pred_2026['predicted_doy'] = pred_2026['stacked_ensemble'].round(1)
    pred_2026['predicted_date'] = pred_2026.apply(
        lambda x: doy_to_date(x['year'], x['predicted_doy']), axis=1
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("FINAL 2026 PREDICTIONS - STACKED ENSEMBLE")
    print("=" * 80)
    
    output_cols = ['location', 'predicted_date', 'predicted_doy', 
                   'ridge_pred', 'lasso_pred', 'bayesian_pred', 'arimax_pred', 'simple_average', 
                   'stacked_ensemble']
    print("\n" + pred_2026[output_cols].to_string(index=False))
    
    # Calculate prediction spread
    pred_2026['prediction_range'] = (
        pred_2026[['ridge_pred', 'lasso_pred', 'bayesian_pred', 'arimax_pred']].max(axis=1) -
        pred_2026[['ridge_pred', 'lasso_pred', 'bayesian_pred', 'arimax_pred']].min(axis=1)
    ).round(1)
    
    print("\n\nPrediction Uncertainty (Range across base models):")
    for _, row in pred_2026.iterrows():
        print(f"  {row['location']:15s}: ±{row['prediction_range']:.1f} days")
    
    # Save results
    output_path = 'data/model_outputs/final_2026_predictions_stacked_ensemble.csv'
    pred_2026.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    # Save meta-model details
    meta_details = pd.DataFrame({
        'base_model': meta_features,
        'coefficient': meta_model.coef_,
        'weight_percent': (meta_model.coef_ / meta_model.coef_.sum() * 100).round(1)
    })
    meta_details_path = 'data/model_outputs/stacked_ensemble_meta_model_weights.csv'
    meta_details.to_csv(meta_details_path, index=False)
    print(f"✓ Meta-model weights saved to: {meta_details_path}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 80)
    print("STACKING SUMMARY")
    print("=" * 80)
    
    print(f"\n✓ Trained {len(base_models)} base models on {len(train)} training samples")
    print(f"✓ Trained meta-model on {len(stacking)} stacking samples ({STACKING_HOLDOUT_YEARS} years)")
    if len(test) > 0:
        print(f"✓ Evaluated on {len(test)} test samples (holdout locations)")
        print(f"✓ Stacked ensemble MAE: {stack_mae:.2f} days")
        print(f"✓ Improvement over simple average: {improvement:.1f}%")
    print(f"✓ Generated predictions for {len(pred_2026)} target locations")
    
    print("\nMeta-Model Learning:")
    print("  The meta-model learned optimal weights for combining base models")
    print("  based on their complementary strengths and weaknesses.")
    print("\nKey Insight:")
    total_weight = meta_model.coef_.sum()
    if total_weight > 1.2:
        print("  Meta-model is amplifying predictions → suggests base models are conservative")
    elif total_weight < 0.8:
        print("  Meta-model is dampening predictions → suggests base models overestimate")
    else:
        print("  Meta-model weights sum near 1.0 → close to weighted average")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
