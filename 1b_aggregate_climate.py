import os
import math
import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
BLOOM_FILE = os.path.join("data", "model_outputs", "aggregated_bloom_data.csv")

NOAA_DIR = os.path.join("data", "noaa") # Folder where your station CSVs are located
STATION_METADATA_FILE = os.path.join("data", "NOAA_station_metadata.csv")
OUTPUT_CLIMATE_FILE = os.path.join("data", "aggregated_climate_data.csv")

FORCING_BASE_TEMP_C = 5.0
LAPSE_RATE_C_PER_M = 0.0065  # 6.5°C per 1000m

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius of earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def find_closest_station(row, stations_df):
    distances = stations_df.apply(
        lambda s: haversine(row['lat'], row['long'], s['lat'], s['lon']), 
        axis=1
    )
    closest_idx = distances.idxmin()
    closest_station = stations_df.loc[closest_idx]
    
    return pd.Series({
        'closest_station_id': closest_station['station_id'],
        'source_file': closest_station['source_file'],
        'distance_km': distances.min(),
        'station_elevation_m': closest_station['elevation_m']
    })

# ==========================================
# 3. AGGREGATE CLIMATE DATA
# ==========================================
def build_aggregated_climate():
    print("1. Loading bloom locations...")
    if not os.path.exists(BLOOM_FILE):
        raise FileNotFoundError(f"Bloom file not found: {BLOOM_FILE}")

    bloom_df = pd.read_csv(BLOOM_FILE)
    unique_locations = bloom_df[['location', 'lat', 'long', 'alt']].drop_duplicates()
    
    print("2. Mapping locations to closest NOAA stations...")
    stations_df = pd.read_csv(STATION_METADATA_FILE)
    required_station_cols = {'station_id', 'source_file', 'lat', 'lon', 'elevation_m'}
    missing_station_cols = required_station_cols - set(stations_df.columns)
    if missing_station_cols:
        missing_str = ", ".join(sorted(missing_station_cols))
        raise ValueError(
            f"Station metadata is missing required columns: {missing_str}. "
            f"Regenerate {STATION_METADATA_FILE} using 0_generate_metadata.py."
        )

    mapping_results = unique_locations.apply(lambda row: find_closest_station(row, stations_df), axis=1)
    location_mapping = pd.concat([unique_locations, mapping_results], axis=1)
    location_mapping['alt_diff_m'] = location_mapping['alt'] - location_mapping['station_elevation_m']
    
    print("3. Loading and adjusting raw station data...")
    aggregated_records = []
    
    # Iterate through each unique bloom location
    for _, loc_row in location_mapping.iterrows():
        station_id = loc_row['closest_station_id']
        source_file = loc_row['source_file']
        loc_name = loc_row['location']
        alt_diff = loc_row['alt_diff_m']

        if pd.isna(source_file) or not str(source_file).strip():
            source_file = f"{station_id}.csv"

        file_path = os.path.join(NOAA_DIR, str(source_file))
        if not os.path.exists(file_path):
            print(
                f"Warning: Missing data file {source_file} for station {station_id} "
                f"(Location: {loc_name})"
            )
            continue
            
        # Read the raw station data
        df = pd.read_csv(file_path, low_memory=False)

        if 'STATION' in df.columns:
            df = df[df['STATION'].astype(str) == str(station_id)]
        if df.empty:
            print(
                f"Warning: No rows found for station {station_id} in {source_file} "
                f"(Location: {loc_name})"
            )
            continue
        
        # Keep only the rows we need to save memory
        df['date'] = pd.to_datetime(df['DATE'])
        df['year'] = df['date'].dt.year
        df['location'] = loc_name  # Tag with the friendly bloom location name
        df['station_id'] = station_id
        
        # Convert NOAA units (tenths of degrees/mm)
        tmax_raw = pd.to_numeric(df['TMAX'], errors='coerce') / 10.0
        tmin_raw = pd.to_numeric(df['TMIN'], errors='coerce') / 10.0
        
        if 'PRCP' in df.columns:
            df['prcp_mm'] = pd.to_numeric(df['PRCP'], errors='coerce') / 10.0
        else:
            df['prcp_mm'] = np.nan
            
        # Apply Altitude Adjustment (Lapse Rate)
        df['tmax_c'] = tmax_raw - (LAPSE_RATE_C_PER_M * alt_diff)
        df['tmin_c'] = tmin_raw - (LAPSE_RATE_C_PER_M * alt_diff)
        df['tmean_c'] = (df['tmax_c'] + df['tmin_c']) / 2.0
        
        # Calculate Growing Degree Days (GDD)
        df['forcing_gdd'] = np.maximum(df['tmean_c'] - FORCING_BASE_TEMP_C, 0)
        
        # Filter down to just the columns we need for the models
        clean_df = df[['location', 'station_id', 'date', 'year', 'tmax_c', 'tmin_c', 'tmean_c', 'prcp_mm', 'forcing_gdd']]
        aggregated_records.append(clean_df)

    # Combine all locations into one master dataframe
    master_climate_df = pd.concat(aggregated_records, ignore_index=True)
    
    print(f"4. Saving aggregated climate data to {OUTPUT_CLIMATE_FILE}...")
    # Sort chronologically by location
    master_climate_df = master_climate_df.sort_values(by=['location', 'date']).reset_index(drop=True)
    master_climate_df.to_csv(OUTPUT_CLIMATE_FILE, index=False)
    
    print("\n--- Aggregation Complete ---")
    print(f"Total Rows: {len(master_climate_df)}")
    print(master_climate_df.head())

if __name__ == "__main__":
    build_aggregated_climate()