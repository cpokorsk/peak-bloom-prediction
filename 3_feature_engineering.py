import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from phenology_config import (
    AGGREGATED_BLOOM_FILE,
    AGGREGATED_CLIMATE_FILE,
    PROJECTED_CLIMATE_FILE,
    MODEL_FEATURES_FILE,
    MIN_MODEL_YEAR,
    EARLY_SPRING_END_MONTH_DAY,
    WINTER_START_MONTH_DAY,
    WINTER_END_MONTH_DAY,
    TARGET_YEAR,
    TARGET_PREDICTION_LOCATIONS,
    get_species_thresholds,
    SPECIES_THRESHOLDS,
    get_continent_for_country,
    normalize_location,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
BLOOM_FILE = AGGREGATED_BLOOM_FILE
CLIMATE_FILE = AGGREGATED_CLIMATE_FILE
FORECAST_FILE = PROJECTED_CLIMATE_FILE
OUTPUT_FEATURES_FILE = MODEL_FEATURES_FILE
MIN_YEAR = MIN_MODEL_YEAR

# ==========================================
# 2. FEATURE ENGINEERING FUNCTION
# ==========================================
def build_features(bloom_df, climate_df, is_future=False):
    climate_by_loc = {loc: group for loc, group in climate_df.groupby('location')}
    climate_locations = set(climate_by_loc.keys())

    before_count = len(bloom_df)
    bloom_df = bloom_df[bloom_df['location'].isin(climate_locations)].copy()
    dropped = before_count - len(bloom_df)
    if dropped:
        label = "future" if is_future else "historical"
        print(f"Dropped {dropped} {label} bloom rows without climate coverage.")

    features = []
    unknown_species = set()

    for _, row in tqdm(bloom_df.iterrows(), total=len(bloom_df), desc="Processing Bloom Events"):
        loc = row['location']
        year = row['year']
        b_doy = row['bloom_doy']
        species = row.get('species', 'Unknown')
        if species not in SPECIES_THRESHOLDS:
            unknown_species.add(species)

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
        max_tmax_early_spring = early_spring['tmax_c'].max()
        min_tmin_early_spring = early_spring['tmin_c'].min()
        total_prcp_early_spring = early_spring['prcp_mm'].fillna(0).sum()
        
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
            'continent': get_continent_for_country(row.get('country_code', 'UNK')),
            'species': species,
            'lat': row['lat'],
            'long': row['long'],
            'alt': row['alt'],
            'year': year,
            'bloom_date': row['bloom_date'],
            'bloom_doy': b_doy,
            'mean_tmax_early_spring': mean_tmax_early_spring,
            'mean_tmin_early_spring': mean_tmin_early_spring,
            'max_tmax_early_spring': max_tmax_early_spring,
            'min_tmin_early_spring': min_tmin_early_spring,
            'total_prcp_early_spring': total_prcp_early_spring,
            'chill_days_oct1_dec31': chill_days,
            'observed_gdd_to_bloom': observed_gdd,
            'chill_threshold_used': chill_thresh,
            'forcing_base_used': forcing_base,
            'is_future': is_future
        })

    if unknown_species:
        print("Warning: Missing species thresholds for:")
        print(", ".join(sorted(unknown_species)))

    return pd.DataFrame(features)


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

    print("2. Extracting fixed winter and early-spring species-specific features (historical)...")
    historical_features = build_features(bloom_df, climate_df, is_future=False)

    print("3. Extracting fixed winter and early-spring species-specific features (forecast)...")
    if os.path.exists(FORECAST_FILE):
        forecast_climate = pd.read_csv(FORECAST_FILE)
        forecast_climate['date'] = pd.to_datetime(forecast_climate['date'])
        forecast_climate['location'] = forecast_climate['location'].apply(normalize_location)

        location_meta = (
            bloom_df.sort_values(['location', 'year'])
            .groupby('location', as_index=False)
            .agg(
                country_code=('country_code', 'last'),
                species=('species', 'last'),
                lat=('lat', 'last'),
                long=('long', 'last'),
                alt=('alt', 'last')
            )
        )

        target_locations = [loc for loc in TARGET_PREDICTION_LOCATIONS if loc in set(location_meta['location'])]
        location_meta = location_meta[location_meta['location'].isin(target_locations)].copy()
        location_meta['year'] = TARGET_YEAR
        location_meta['bloom_date'] = pd.NaT
        location_meta['bloom_doy'] = np.nan

        future_features = build_features(location_meta, forecast_climate, is_future=True)
    else:
        print(f"Forecast climate file missing: {FORECAST_FILE}. Skipping future features.")
        future_features = pd.DataFrame()

    # ==========================================
    # 3. CLEANUP AND SAVE
    # ==========================================
    features_df = pd.concat([historical_features, future_features], ignore_index=True)
    
    # Drop rows missing complete weather data for the windows
    initial_len = len(features_df)
    features_df = features_df.dropna(subset=['mean_tmax_early_spring', 'chill_days_oct1_dec31'])
    final_len = len(features_df)
    
    if initial_len != final_len:
        print(f"\nDropped {initial_len - final_len} rows due to incomplete climate windows.")

    print(f"\n4. Saving engineered features to {OUTPUT_FEATURES_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FEATURES_FILE), exist_ok=True)
    features_df.to_csv(OUTPUT_FEATURES_FILE, index=False)

    print("\n--- Feature Engineering Complete ---")
    print(f"Total Usable Records: {len(features_df)}")
    print(features_df[['location', 'year', 'species', 'chill_days_oct1_dec31', 'observed_gdd_to_bloom', 'is_future']].head())

if __name__ == "__main__":
    engineer_features()