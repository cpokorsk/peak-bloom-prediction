import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.ar_model import AutoReg

# ==========================================
# 1. CONFIGURATION
# ==========================================
CLIMATE_FILE = os.path.join("data", "model_outputs", "aggregated_climate_data.csv")
OUTPUT_FORECAST_FILE = os.path.join("data", "model_outputs", "projected_climate_2026.csv")

TARGET_YEAR = 2026
FORECAST_END_DATE = pd.to_datetime(f"{TARGET_YEAR}-05-31")
FORCING_BASE_TEMP_C = 5.0
AR_LAGS = 3  # Use the last 3 days of anomalies to predict the next day

# Only generate future forecasts for these specific locations
TARGET_LOCATIONS = ["washingtondc", "kyoto", "liestal", "vancouver", "nyc"]

# ==========================================
# 2. TIME SERIES FORECASTING FUNCTION
# ==========================================
def forecast_2026_climate():
    print("1. Loading historical aggregated climate data...")
    if not os.path.exists(CLIMATE_FILE):
        raise FileNotFoundError(f"Cannot find {CLIMATE_FILE}. Run Step 1 first.")

    df = pd.read_csv(CLIMATE_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df['doy'] = df['date'].dt.dayofyear
    
    # Sort chronologically
    df = df.sort_values(by=['location', 'date']).reset_index(drop=True)

    print("2. Calculating Deterministic Seasonal Components (Historical DOY Averages)...")
    # We use a 20-year window to capture modern climate normals (2005-2025)
    baseline_df = df[(df['year'] >= 2005) & (df['year'] < TARGET_YEAR)]
    
    seasonal_components = baseline_df.groupby(['location', 'doy']).agg({
        'tmax_c': 'mean',
        'tmin_c': 'mean',
        'prcp_mm': 'mean' 
    }).rename(columns={'tmax_c': 'S_tmax', 'tmin_c': 'S_tmin', 'prcp_mm': 'S_prcp'}).reset_index()

    print(f"3. Forecasting {TARGET_YEAR} daily weather via AR(p) on anomalies...")
    forecasted_records = []

    locations = df['location'].unique()
    
    for loc in tqdm(locations, desc="Modeling Locations"):
        loc_df = df[df['location'] == loc].copy()
        loc_seasonal = seasonal_components[seasonal_components['location'] == loc]
        
        # Merge seasonal component to calculate anomalies
        loc_df = loc_df.merge(loc_seasonal, on=['location', 'doy'], how='left')
        
        # Calculate Y_t (Anomalies)
        loc_df['Y_tmax'] = loc_df['tmax_c'] - loc_df['S_tmax']
        loc_df['Y_tmin'] = loc_df['tmin_c'] - loc_df['S_tmin']
        loc_df['is_forecast'] = False
        
        # If this is not one of our target cities, just keep historical data and skip forecasting
        if loc not in TARGET_LOCATIONS:
            forecasted_records.append(loc_df)
            continue
            
        # Identify the last date of actual data we have for this location
        last_actual_date = loc_df['date'].max()
        
        if last_actual_date >= FORECAST_END_DATE:
            # We already have data past May 31st, no need to forecast
            forecasted_records.append(loc_df)
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
                future_df['forcing_gdd'] = np.maximum(future_df['tmean_c'] - FORCING_BASE_TEMP_C, 0)
                
                # Append to records
                combined_loc_df = pd.concat([loc_df, future_df], ignore_index=True)
                forecasted_records.append(combined_loc_df)
        else:
            # Not enough data to model, just append history
            forecasted_records.append(loc_df)

    # ==========================================
    # 4. CLEANUP AND SAVE
    # ==========================================
    final_df = pd.concat(forecasted_records, ignore_index=True)
    
    # Sort and clean columns
    cols_to_keep = ['location', 'date', 'year', 'doy', 'is_forecast', 'tmax_c', 'tmin_c', 'tmean_c', 'prcp_mm', 'forcing_gdd']
    if 'station_id' in final_df.columns:
        cols_to_keep.insert(1, 'station_id')
        
    final_df = final_df[cols_to_keep].sort_values(by=['location', 'date']).reset_index(drop=True)
    
    print(f"\n4. Saving Hybrid Projected Climate data to {OUTPUT_FORECAST_FILE}...")
    final_df.to_csv(OUTPUT_FORECAST_FILE, index=False)
    
    print("\n--- 2026 Climate Forecasting Complete ---")
    forecast_only = final_df[final_df['is_forecast'] == True]
    print(f"Total Forecasted Days for target locations: {len(forecast_only)}")
    if not forecast_only.empty:
        print("\nSample Forecast Preview:")
        print(forecast_only[['location', 'date', 'tmax_c', 'tmin_c', 'prcp_mm']].head())

if __name__ == "__main__":
    forecast_2026_climate()