import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from phenology_config import (
    AGGREGATED_BLOOM_FILE,
    AGGREGATED_CLIMATE_FILE,
    MODEL_FEATURES_FILE,
    MIN_MODEL_YEAR,
    EARLY_SPRING_END_MONTH_DAY,
    WINTER_START_MONTH_DAY,
    WINTER_END_MONTH_DAY,
    get_species_thresholds,
    normalize_location,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
BLOOM_FILE = AGGREGATED_BLOOM_FILE
CLIMATE_FILE = AGGREGATED_CLIMATE_FILE
OUTPUT_FEATURES_FILE = MODEL_FEATURES_FILE
MIN_YEAR = MIN_MODEL_YEAR

# ==========================================
# 2. FEATURE ENGINEERING FUNCTION
# ==========================================
def engineer_features():
    print("1. Loading aggregated datasets...")
    if not os.path.exists(BLOOM_FILE) or not os.path.exists(CLIMATE_FILE):
        raise FileNotFoundError("Aggregated data files missing. Please run the Step 1 scripts first.")

    bloom_df = pd.read_csv(BLOOM_FILE)
    climate_df = pd.read_csv(CLIMATE_FILE)

    # Ensure dates are proper datetime objects
    bloom_df['bloom_date'] = pd.to_datetime(bloom_df['bloom_date'])
    bloom_df['location'] = bloom_df['location'].apply(normalize_location)
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    climate_df['location'] = climate_df['location'].apply(normalize_location)

    # Filter bloom data to our target modeling timeframe
    bloom_df = bloom_df[bloom_df['year'] >= MIN_YEAR].copy()

    print("2. Indexing climate data for faster processing...")
    climate_by_loc = {loc: group for loc, group in climate_df.groupby('location')}

    print("3. Extracting fixed winter and early-spring species-specific features...")
    features = []

    for _, row in tqdm(bloom_df.iterrows(), total=len(bloom_df), desc="Processing Bloom Events"):
        loc = row['location']
        year = row['year']
        b_doy = row['bloom_doy']
        species = row.get('species', 'Unknown')

        # Skip if we don't have climate data for this location
        if loc not in climate_by_loc:
            continue
            
        loc_climate = climate_by_loc[loc]
        
        # Get species-specific parameters
        thresholds = get_species_thresholds(species)
        chill_thresh = thresholds["chill_temp_c"]
        forcing_base = thresholds["forcing_base_c"]

        # --------------------------------------------------
        # WINDOW 1: Early Spring (Jan 1 to Mar 15)
        # --------------------------------------------------
        jan1 = pd.to_datetime(f"{year}-01-01")
        early_spring_end = pd.to_datetime(f"{year}-{EARLY_SPRING_END_MONTH_DAY}")
        early_spring = loc_climate[(loc_climate['date'] >= jan1) & (loc_climate['date'] <= early_spring_end)]

        if early_spring.empty:
            continue

        mean_tmax_early_spring = early_spring['tmax_c'].mean()
        mean_tmin_early_spring = early_spring['tmin_c'].mean()
        total_prcp_early_spring = early_spring['prcp_mm'].sum()
        
        observed_gdd = np.maximum(early_spring['tmean_c'] - forcing_base, 0).sum()

        # --------------------------------------------------
        # WINDOW 2: Winter Chill (Oct 1 to Dec 31 of previous year)
        # --------------------------------------------------
        chill_start = pd.to_datetime(f"{year-1}-{WINTER_START_MONTH_DAY}")
        chill_end = pd.to_datetime(f"{year-1}-{WINTER_END_MONTH_DAY}")
        chill_window = loc_climate[(loc_climate['date'] >= chill_start) & (loc_climate['date'] <= chill_end)]

        # Calculate Chill Days dynamically based on species chill threshold
        if not chill_window.empty:
            chill_days = (chill_window['tmean_c'] <= chill_thresh).sum()
        else:
            chill_days = np.nan

        features.append({
            'location': loc,
            'country_code': row.get('country_code', 'UNK'),
            'species': species,
            'lat': row['lat'],
            'long': row['long'],
            'alt': row['alt'],
            'year': year,
            'bloom_date': row['bloom_date'],
            'bloom_doy': b_doy,
            'mean_tmax_early_spring': mean_tmax_early_spring,
            'mean_tmin_early_spring': mean_tmin_early_spring,
            'total_prcp_early_spring': total_prcp_early_spring,
            'chill_days_oct1_dec31': chill_days,
            'observed_gdd_to_bloom': observed_gdd,
            'chill_threshold_used': chill_thresh,
            'forcing_base_used': forcing_base
        })

    # ==========================================
    # 3. CLEANUP AND SAVE
    # ==========================================
    features_df = pd.DataFrame(features)
    
    # Drop rows missing complete weather data for the windows
    initial_len = len(features_df)
    features_df = features_df.dropna(subset=['mean_tmax_early_spring', 'chill_days_oct1_dec31'])
    final_len = len(features_df)
    
    if initial_len != final_len:
        print(f"\nDropped {initial_len - final_len} rows due to incomplete historic climate windows.")

    print(f"\n4. Saving engineered features to {OUTPUT_FEATURES_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FEATURES_FILE), exist_ok=True)
    features_df.to_csv(OUTPUT_FEATURES_FILE, index=False)

    print("\n--- Feature Engineering Complete ---")
    print(f"Total Usable Records: {len(features_df)}")
    print(features_df[['location', 'year', 'species', 'chill_days_oct1_dec31', 'observed_gdd_to_bloom']].head())

if __name__ == "__main__":
    engineer_features()