import os
import glob
import pandas as pd
from phenology_config import (
    NOAA_DIR,
    NOAA_STATION_METADATA_FILE,
    STATION_PREFIX_TO_COUNTRY_CODE,
    STATION_SUFFIX_TO_COUNTRY_CODE,
)

# ==========================================
# CONFIGURATION
# ==========================================
METADATA_OUTPUT_FILE = NOAA_STATION_METADATA_FILE


def infer_country_code(station_id, station_name):
    station_id_str = str(station_id)
    prefix = station_id_str[:2].upper()
    if prefix in STATION_PREFIX_TO_COUNTRY_CODE:
        return STATION_PREFIX_TO_COUNTRY_CODE[prefix]

    if pd.isna(station_name):
        return "UNK"

    suffix = str(station_name).split(",")[-1].strip().upper()
    return STATION_SUFFIX_TO_COUNTRY_CODE.get(suffix, "UNK")


def extract_station_records(file_path):
    source_file = os.path.basename(file_path)
    source_name = os.path.splitext(source_file)[0]
    required_cols = {'STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION'}

    # Read only header first to validate schema without loading the full file
    header_df = pd.read_csv(file_path, nrows=0)
    columns = set(header_df.columns)
    missing_cols = sorted(required_cols - columns)
    if missing_cols:
        return [], f"Missing required columns: {', '.join(missing_cols)}"

    selected_cols = ['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION']
    if 'NAME' in columns:
        selected_cols.append('NAME')
    if 'DATE' in columns:
        selected_cols.append('DATE')

    records_by_station = {}
    for chunk in pd.read_csv(file_path, usecols=selected_cols, chunksize=200000):
        chunk = chunk.dropna(subset=['STATION'])
        chunk['date'] = pd.to_datetime(chunk.get('DATE'), errors='coerce')

        for station_id, station_chunk in chunk.groupby(chunk['STATION'].astype(str)):
            station_info = records_by_station.get(station_id)

            if station_info is None:
                first_row = station_chunk.iloc[0]
                station_info = {
                    'station_id': station_id,
                    'country_code': infer_country_code(station_id, first_row.NAME if 'NAME' in station_chunk.columns else None),
                    'source_name': source_name,
                    'source_file': source_file,
                    'lat': first_row.LATITUDE,
                    'lon': first_row.LONGITUDE,
                    'elevation_m': first_row.ELEVATION,
                    'name': first_row.NAME if 'NAME' in station_chunk.columns else "Unknown",
                    'first_date': None,
                    'last_date': None,
                }

            if 'date' in station_chunk.columns:
                min_date = station_chunk['date'].min()
                max_date = station_chunk['date'].max()
                if pd.notna(min_date):
                    current_min = station_info['first_date']
                    station_info['first_date'] = min_date if current_min is None or min_date < current_min else current_min
                if pd.notna(max_date):
                    current_max = station_info['last_date']
                    station_info['last_date'] = max_date if current_max is None or max_date > current_max else current_max

            records_by_station[station_id] = station_info

    return list(records_by_station.values()), None

def generate_metadata():
    print(f"Scanning '{NOAA_DIR}/' for station CSV files...")
    
    # Grab all CSV files in the directory
    station_files = sorted(glob.glob(os.path.join(NOAA_DIR, "*.csv")))
    
    metadata_records = []
    processed_files = []
    skipped_files = []
    
    for file_path in station_files:
        file_name = os.path.basename(file_path)
        try:
            records, error = extract_station_records(file_path)
            if error:
                skipped_files.append((file_name, error))
                print(f"Skipped {file_name} ({error})")
                continue

            metadata_records.extend(records)
            processed_files.append(file_name)
            print(f"Processed {file_name} -> {len(records)} unique station(s)")
        except Exception as e:
            skipped_files.append((file_name, f"Read error: {e}"))
            print(f"Skipped {file_name} (read error)")

    # Save to CSV if we found any stations
    if metadata_records:
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df = metadata_df.drop_duplicates(subset=['station_id']).sort_values('station_id').reset_index(drop=True)
        metadata_df = metadata_df[['station_id', 'country_code', 'source_name', 'source_file', 'lat', 'lon', 'elevation_m', 'name', 'first_date', 'last_date']]
        metadata_df['first_date'] = pd.to_datetime(metadata_df['first_date'], errors='coerce').dt.date
        metadata_df['last_date'] = pd.to_datetime(metadata_df['last_date'], errors='coerce').dt.date
        metadata_df['source_file'] = metadata_df['source_file'].astype(str).str.strip()
        metadata_df.to_csv(METADATA_OUTPUT_FILE, index=False)

        missing_source = int((metadata_df['source_file'] == "").sum())
        print(f"\nSuccess! Extracted metadata for {len(metadata_df)} stations.")
        print(f"Saved to: {METADATA_OUTPUT_FILE}")
        print(f"source_file completeness: {len(metadata_df) - missing_source}/{len(metadata_df)}")
        print(f"Processed files: {len(processed_files)} / {len(station_files)}")
        if skipped_files:
            print("\nSkipped files:")
            for file_name, reason in skipped_files:
                print(f"- {file_name}: {reason}")
        print("\n--- Preview ---")
        print(metadata_df.head())
    else:
        print("\nNo valid NOAA station CSV files found. Please check your directory path.")
        if skipped_files:
            print("\nSkipped files:")
            for file_name, reason in skipped_files:
                print(f"- {file_name}: {reason}")

if __name__ == "__main__":
    generate_metadata()