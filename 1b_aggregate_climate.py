import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from phenology_config import (
    BLOSSOM_SITE_METADATA_FILE,
    NOAA_DIR,
    NOAA_STATION_METADATA_FILE,
    AGGREGATED_CLIMATE_FILE,
    LAPSE_RATE_C_PER_M,
    MIN_CLIMATE_YEAR,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
# We use the metadata file generated in the previous step
SITE_METADATA_FILE = BLOSSOM_SITE_METADATA_FILE
STATION_METADATA_FILE = NOAA_STATION_METADATA_FILE
OUTPUT_CLIMATE_FILE = AGGREGATED_CLIMATE_FILE
MIN_YEAR = MIN_CLIMATE_YEAR

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
        'station_id': closest_station['station_id'],
        'source_file': closest_station['source_file'],
        'distance_km': distances.min(),
        'station_elevation_m': closest_station['elevation_m']
    })

# ==========================================
# 3. AGGREGATE CLIMATE DATA
# ==========================================
def build_aggregated_climate():
    print("1. Loading blossom site metadata...")
    if not os.path.exists(SITE_METADATA_FILE):
        raise FileNotFoundError(f"Site metadata not found: {SITE_METADATA_FILE}")

    site_df = pd.read_csv(SITE_METADATA_FILE)
    
    print("2. Mapping locations to closest NOAA stations...")
    stations_df = pd.read_csv(STATION_METADATA_FILE)
    
    # Haversine mapping
    tqdm.pandas(desc="Calculating Spatial Distances")
    mapping_results = site_df.progress_apply(lambda row: find_closest_station(row, stations_df), axis=1)
    
    # Combine site data with closest station data
    location_mapping = pd.concat([site_df, mapping_results], axis=1)
    location_mapping['alt_diff_m'] = location_mapping['alt'] - location_mapping['station_elevation_m']
    
    # Identify unique stations required
    unique_stations = location_mapping[['station_id', 'source_file']].drop_duplicates()
    print(f"\nIdentified {len(unique_stations)} unique stations needed for {len(site_df)} locations.")

    print(f"\n3. Loading and adjusting raw station data (Filtering >= {MIN_YEAR})...")
    aggregated_records = []

    # Loop over the unique stations instead of locations
    for _, station_row in tqdm(unique_stations.iterrows(), total=len(unique_stations), desc="Processing Stations"):
        station_id = station_row['station_id']
        source_file = station_row['source_file']
        
        if pd.isna(source_file) or not str(source_file).strip():
            source_file = f"{station_id}.csv"

        file_path = os.path.join(NOAA_DIR, str(source_file))
        if not os.path.exists(file_path):
            tqdm.write(f"Warning: Missing data file {source_file} for station {station_id}")
            continue

        # Load and filter the station data ONCE using chunks
        valid_chunks = []
        for chunk in pd.read_csv(file_path, chunksize=100000, low_memory=False):
            if 'STATION' in chunk.columns:
                chunk = chunk[chunk['STATION'].astype(str) == str(station_id)]
            
            if chunk.empty:
                continue
            
            # Extract Year and Filter
            chunk['date'] = pd.to_datetime(chunk['DATE'], errors='coerce')
            chunk = chunk.dropna(subset=['date'])
            chunk['year'] = chunk['date'].dt.year
            chunk = chunk[chunk['year'] >= MIN_YEAR]
            
            if not chunk.empty:
                # Keep only necessary columns to minimize memory footprint
                cols_to_keep = ['STATION', 'date', 'year', 'TMAX', 'TMIN']
                if 'PRCP' in chunk.columns:
                    cols_to_keep.append('PRCP')
                
                cols_to_keep = [c for c in cols_to_keep if c in chunk.columns]
                valid_chunks.append(chunk[cols_to_keep])

        if not valid_chunks:
            tqdm.write(f"Warning: No valid records >= {MIN_YEAR} for station {station_id}")
            continue

        # Combine chunks for this specific station
        station_df = pd.concat(valid_chunks, ignore_index=True)
        
        # Pre-process generic NOAA values (tenths to standard)
        tmax_raw = pd.to_numeric(station_df['TMAX'], errors='coerce') / 10.0
        tmin_raw = pd.to_numeric(station_df['TMIN'], errors='coerce') / 10.0
        
        if 'PRCP' in station_df.columns:
            prcp_mm = pd.to_numeric(station_df['PRCP'], errors='coerce') / 10.0
        else:
            prcp_mm = np.nan

        station_df['tmax_raw'] = tmax_raw
        station_df['tmin_raw'] = tmin_raw
        station_df['prcp_mm'] = prcp_mm

        # Now apply the data to each location that maps to this station
        locations_for_station = location_mapping[location_mapping['station_id'] == station_id]
        
        for _, loc_row in locations_for_station.iterrows():
            loc_name = loc_row['location']
            alt_diff = loc_row['alt_diff_m']
            
            # Create a copy for this specific location
            loc_df = station_df[['date', 'year', 'prcp_mm', 'tmax_raw', 'tmin_raw']].copy()
            loc_df['location'] = loc_name
            loc_df['station_id'] = station_id
            
            # Apply Altitude Adjustment (Lapse Rate)
            loc_df['tmax_c'] = loc_df['tmax_raw'] - (LAPSE_RATE_C_PER_M * alt_diff)
            loc_df['tmin_c'] = loc_df['tmin_raw'] - (LAPSE_RATE_C_PER_M * alt_diff)
            loc_df['tmean_c'] = (loc_df['tmax_c'] + loc_df['tmin_c']) / 2.0

            # Reorder columns and drop raw temps
            loc_df = loc_df[['location', 'station_id', 'date', 'year', 'tmax_c', 'tmin_c', 'tmean_c', 'prcp_mm']]
            aggregated_records.append(loc_df)

    # Combine all locations into one master dataframe
    if aggregated_records:
        master_climate_df = pd.concat(aggregated_records, ignore_index=True)
        
        print(f"\n4. Saving aggregated climate data to {OUTPUT_CLIMATE_FILE}...")
        os.makedirs(os.path.dirname(OUTPUT_CLIMATE_FILE), exist_ok=True)
        
        # Sort chronologically by location
        master_climate_df = master_climate_df.sort_values(by=['location', 'date']).reset_index(drop=True)
        master_climate_df.to_csv(OUTPUT_CLIMATE_FILE, index=False)
        
        print("\n--- Aggregation Complete ---")
        print(f"Total Rows: {len(master_climate_df)}")
        print(f"Total Unique Locations: {master_climate_df['location'].nunique()}")
        print(master_climate_df.head())
    else:
        print("\nNo valid climate records were found to aggregate.")

if __name__ == "__main__":
    build_aggregated_climate()