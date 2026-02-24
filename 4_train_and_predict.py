import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. CONFIGURATION
# ==========================================
BLOOM_FILE = os.path.join("data", "model_outputs", "aggregated_bloom_data.csv")
CLIMATE_FILE = os.path.join("data", "model_outputs", "aggregated_climate_data.csv")
FORECAST_FILE = os.path.join("data", "model_outputs", "projected_climate_2026.csv")
OUTPUT_PREDICTIONS = os.path.join("data", "model_outputs", "final_2026_predictions.csv")

HOLDOUT_LOCATIONS = ["vancouver", "nyc", "newyorkcity"]
EARLY_SPRING_END_DOY = 74  # Roughly March 15th (Predictors must be calculated before bloom)
CHILL_TEMP_C = 4.3
FORCING_BASE_C = 5.0
MIN_YEAR = 1974

# ==========================================
# 2. FEATURE ENGINEERING (STRICT NO-LEAKAGE)
# ==========================================
def build_predictive_features(bloom_df, climate_df, is_future=False):
    print("Extracting strictly predictive fixed-window features...")
    features = []
    
    climate_by_loc = {loc: group for loc, group in climate_df.groupby('location')}
    
    for _, row in bloom_df.iterrows():
        loc = row['location']
        year = row['year']
        b_doy = row.get('bloom_doy', np.nan) # NaN for 2026 forecast
        
        if loc not in climate_by_loc:
            continue
            
        loc_climate = climate_by_loc[loc]
        
        # 1. Early Spring Window (Jan 1 to March 15)
        jan1 = pd.to_datetime(f"{year}-01-01")
        mar15 = pd.to_datetime(f"{year}-03-15")
        spring_window = loc_climate[(loc_climate['date'] >= jan1) & (loc_climate['date'] <= mar15)]
        
        if spring_window.empty:
            continue
            
        early_temp = spring_window['tmax_c'].mean()
        early_prcp = spring_window['prcp_mm'].sum()
        early_gdd = np.maximum(spring_window['tmean_c'] - FORCING_BASE_C, 0).sum()
        
        # 2. Winter Chill Window (Oct 1 to Dec 31 of previous year)
        chill_start = pd.to_datetime(f"{year-1}-10-01")
        chill_end = pd.to_datetime(f"{year-1}-12-31")
        chill_window = loc_climate[(loc_climate['date'] >= chill_start) & (loc_climate['date'] <= chill_end)]
        
        chill_days = (chill_window['tmean_c'] <= CHILL_TEMP_C).sum() if not chill_window.empty else np.nan
        
        features.append({
            'location': loc,
            'year': year,
            'species': row.get('species', 'Unknown').replace(" ", "_"), # Format for formula API
            'alt': row['alt'],
            'early_spring_temp': early_temp,
            'early_spring_prcp': early_prcp,
            'early_spring_gdd': early_gdd,
            'chill_days': chill_days,
            'bloom_doy': b_doy
        })
        
    return pd.DataFrame(features).dropna(subset=['early_spring_temp', 'chill_days'])

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    print("--- Loading Data ---")
    bloom_df = pd.read_csv(BLOOM_FILE)
    bloom_df = bloom_df[bloom_df['year'] >= MIN_YEAR]
    climate_df = pd.read_csv(CLIMATE_FILE)
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    
    # Generate features for historical data
    df = build_predictive_features(bloom_df, climate_df)
    df = df.dropna(subset=['bloom_doy'])
    
    print("\n--- Splitting Data ---")
    # 1. Separate the Holdout Test Set (Vancouver & NYC)
    holdout_mask = df['location'].isin(HOLDOUT_LOCATIONS)
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
    formula = "bloom_doy ~ early_spring_temp + early_spring_gdd + early_spring_prcp + chill_days + alt + C(species)"
    model = smf.ols(formula=formula, data=train).fit()
    
    print(model.summary().tables[1])
    
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
    # Load our 2026 projected climate
    forecast_climate = pd.read_csv(FORECAST_FILE)
    forecast_climate['date'] = pd.to_datetime(forecast_climate['date'])
    
    # Create dummy bloom records for 2026 to feed into our feature extractor
    target_locations = forecast_climate['location'].unique()
    dummy_bloom = []
    for loc in target_locations:
        # Get species and alt from historical data
        loc_history = df[df['location'] == loc]
        species = loc_history['species'].iloc[0] if not loc_history.empty else "Unknown"
        alt = loc_history['alt'].iloc[0] if not loc_history.empty else 0
        
        dummy_bloom.append({
            'location': loc,
            'year': 2026,
            'species': species,
            'alt': alt
        })
        
    df_2026_input = pd.DataFrame(dummy_bloom)
    
    # Generate 2026 features
    df_2026_features = build_predictive_features(df_2026_input, forecast_climate, is_future=True)
    
    # Generate Predictions & 90% Intervals (alpha=0.10)
    predictions = model.get_prediction(df_2026_features)
    pred_summary = predictions.summary_frame(alpha=0.10)
    
    df_2026_features['predicted_doy'] = pred_summary['mean'].round(1)
    df_2026_features['90_ci_lower'] = pred_summary['obs_ci_lower'].round(1)
    df_2026_features['90_ci_upper'] = pred_summary['obs_ci_upper'].round(1)
    df_2026_features['interval_width_days'] = (df_2026_features['90_ci_upper'] - df_2026_features['90_ci_lower']).round(1)
    
    # Convert DOY to actual calendar dates
    def doy_to_date(year, doy):
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