import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.ar_model import AutoReg
from phenology_config import (
    AGGREGATED_CLIMATE_FILE,
    PROJECTED_CLIMATE_FILE,
    TARGET_YEAR,
    WINTER_START_MONTH_DAY,
    FORECAST_END_MONTH_DAY,
    AR_LAGS,
    TARGET_PREDICTION_LOCATIONS,
    BASELINE_START_YEAR,
    normalize_location,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CLIMATE_FILE = AGGREGATED_CLIMATE_FILE
OUTPUT_FORECAST_FILE = PROJECTED_CLIMATE_FILE

PREDICTION_START_DATE = pd.to_datetime(f"{TARGET_YEAR-1}-{WINTER_START_MONTH_DAY}")
FORECAST_END_DATE = pd.to_datetime(f"{TARGET_YEAR}-{FORECAST_END_MONTH_DAY}")
TARGET_LOCATIONS = TARGET_PREDICTION_LOCATIONS

# ==========================================
# 2. TIME SERIES FORECASTING FUNCTION
# ==========================================
def forecast_2026_climate():
    print("1. Loading historical aggregated climate data...")
    if not os.path.exists(CLIMATE_FILE):
        raise FileNotFoundError(f"Cannot find {CLIMATE_FILE}. Run Step 1 first.")

    df = pd.read_csv(CLIMATE_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df['location'] = df['location'].apply(normalize_location)
    df['doy'] = df['date'].dt.dayofyear
    
    # Sort chronologically
    df = df.sort_values(by=['location', 'date']).reset_index(drop=True)

    print("2. Calculating Deterministic Seasonal Components (Historical DOY Averages)...")
    baseline_df = df[(df['year'] >= BASELINE_START_YEAR) & (df['year'] < TARGET_YEAR)]
    
    seasonal_components = baseline_df.groupby(['location', 'doy']).agg({
        'tmax_c': 'mean',
        'tmin_c': 'mean',
        'prcp_mm': 'mean' 
    }).rename(columns={'tmax_c': 'S_tmax', 'tmin_c': 'S_tmin', 'prcp_mm': 'S_prcp'}).reset_index()

    print(f"3. Forecasting {TARGET_YEAR} daily weather via AR(p) on anomalies...")
    forecasted_records = []

    available_locations = set(df['location'].unique())
    locations = [loc for loc in TARGET_LOCATIONS if loc in available_locations]
    missing_locations = [loc for loc in TARGET_LOCATIONS if loc not in available_locations]
    if missing_locations:
        print(f"Warning: Missing climate data for target locations: {', '.join(missing_locations)}")

    for loc in tqdm(locations, desc="Modeling Locations"):
        loc_df = df[df['location'] == loc].copy()
        loc_seasonal = seasonal_components[seasonal_components['location'] == loc]
        
        # Merge seasonal component to calculate anomalies
        loc_df = loc_df.merge(loc_seasonal, on=['location', 'doy'], how='left')
        
        # Calculate Y_t (Anomalies)
        loc_df['Y_tmax'] = loc_df['tmax_c'] - loc_df['S_tmax']
        loc_df['Y_tmin'] = loc_df['tmin_c'] - loc_df['S_tmin']
        loc_df['is_forecast'] = False

        # Keep only target prediction locations in the output
        if loc not in TARGET_LOCATIONS:
            continue

        # Historical rows needed for 2026 prediction window output
        loc_2026_window = loc_df[
            (loc_df['date'] >= PREDICTION_START_DATE) &
            (loc_df['date'] <= FORECAST_END_DATE)
        ].copy()
        
        # Identify the last date of actual data we have for this location
        last_actual_date = loc_df['date'].max()
        
        if last_actual_date >= FORECAST_END_DATE:
            # We already have data past May 31st, no need to forecast
            forecasted_records.append(loc_2026_window)
            continue
            
        # ---------------------------------------------------------
        # Fit AR(p) models to the anomalies (Shumway & Stoffer Ch 3)
        # ---------------------------------------------------------
        train_anom = loc_df.dropna(subset=['Y_tmax', 'Y_tmin']).copy()
        
        # Fit models if we have enough data
        if len(train_anom) > AR_LAGS:
            ar_tmax = AutoReg(train_anom['Y_tmax'].values, lags=AR_LAGS).fit()
            ar_tmin = AutoReg(train_anom['Y_tmin'].values, lags=AR_LAGS).fit()
            
            # ---------------------------------------------------------
            # Forecast Future Days
            # ---------------------------------------------------------
            future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), end=FORECAST_END_DATE)
            days_to_predict = len(future_dates)
            
            if days_to_predict > 0:
                pred_Y_tmax = ar_tmax.predict(start=len(train_anom), end=len(train_anom) + days_to_predict - 1)
                pred_Y_tmin = ar_tmin.predict(start=len(train_anom), end=len(train_anom) + days_to_predict - 1)
                
                # Build the forecasted dataframe
                future_df = pd.DataFrame({
                    'location': loc,
                    'station_id': loc_df['station_id'].iloc[0] if 'station_id' in loc_df.columns else "Forecast",
                    'date': future_dates,
                    'year': future_dates.year,
                    'doy': future_dates.dayofyear,
                    'is_forecast': True
                })
                
                # Attach Seasonal Averages
                future_df = future_df.merge(loc_seasonal, on=['location', 'doy'], how='left')
                future_df = future_df.bfill() # Handle leap years
                
                # Reconstruct: X_t = S_t + Y_t
                future_df['tmax_c'] = future_df['S_tmax'] + pred_Y_tmax
                future_df['tmin_c'] = future_df['S_tmin'] + pred_Y_tmin
                future_df['prcp_mm'] = future_df['S_prcp'] 
                
                # Ensure TMAX is always >= TMIN
                future_df['tmax_c'] = np.maximum(future_df['tmax_c'], future_df['tmin_c'] + 0.1)
                
                # Calculate standard metrics
                future_df['tmean_c'] = (future_df['tmax_c'] + future_df['tmin_c']) / 2.0
                
                # Append to records
                combined_loc_df = pd.concat([loc_df, future_df], ignore_index=True)
                combined_2026_window = combined_loc_df[
                    (combined_loc_df['date'] >= PREDICTION_START_DATE) &
                    (combined_loc_df['date'] <= FORECAST_END_DATE)
                ].copy()
                forecasted_records.append(combined_2026_window)
        else:
            # Not enough data to model, keep only available 2026 prediction window
            forecasted_records.append(loc_2026_window)

    # ==========================================
    # 4. CLEANUP AND SAVE
    # ==========================================
    final_df = pd.concat(forecasted_records, ignore_index=True)

    # Safety filter: keep only target locations and 2026 prediction window
    final_df = final_df[
        final_df['location'].isin(TARGET_LOCATIONS)
        & (final_df['date'] >= PREDICTION_START_DATE)
        & (final_df['date'] <= FORECAST_END_DATE)
    ].copy()
    
    # Sort and clean columns
    cols_to_keep = ['location', 'date', 'year', 'doy', 'is_forecast', 'tmax_c', 'tmin_c', 'tmean_c', 'prcp_mm']
    if 'station_id' in final_df.columns:
        cols_to_keep.insert(1, 'station_id')
        
    final_df = final_df[cols_to_keep].sort_values(by=['location', 'date']).reset_index(drop=True)

    prcp_missing = final_df['prcp_mm'].isna().sum()
    if prcp_missing > 0:
        print(f"Filling {prcp_missing} missing precipitation values with 0.0")
        final_df['prcp_mm'] = final_df['prcp_mm'].fillna(0.0)

    numeric_cols = ['tmax_c', 'tmin_c', 'tmean_c', 'prcp_mm']
    final_df[numeric_cols] = final_df[numeric_cols].round(3)
    
    print(f"\n4. Saving Projected Climate prediction window to {OUTPUT_FORECAST_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FORECAST_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FORECAST_FILE, index=False)
    
    print("\n--- 2026 Climate Forecasting Complete ---")
    forecast_only = final_df[final_df['is_forecast'] == True]
    print(f"Total Rows in 2026 prediction dataset: {len(final_df)}")
    print(f"Forecasted Rows in 2026 prediction dataset: {len(forecast_only)}")
    if not forecast_only.empty:
        print("\nSample Forecast Preview:")
        print(forecast_only[['location', 'date', 'tmax_c', 'tmin_c', 'prcp_mm']].head())

if __name__ == "__main__":
    forecast_2026_climate()