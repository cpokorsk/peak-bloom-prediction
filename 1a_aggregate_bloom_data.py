import glob
import os

import pandas as pd
from phenology_config import (
    AGGREGATED_BLOOM_FILE,
    BLOOM_DIR,
    infer_country_code_from_location,
    get_species_for_country,
    normalize_location,
)


OUTPUT_FILE = AGGREGATED_BLOOM_FILE


def aggregate_bloom_data() -> pd.DataFrame:
    bloom_files = sorted(glob.glob(os.path.join(BLOOM_DIR, "*.csv")))
    if not bloom_files:
        raise FileNotFoundError(f"No bloom CSV files found in: {BLOOM_DIR}")

    required_columns = {"location", "lat", "long", "alt", "year", "bloom_date", "bloom_doy"}
    bloom_frames = []

    print(f"Found {len(bloom_files)} bloom CSV files.")
    for file_path in bloom_files:
        df = pd.read_csv(file_path)
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(f"File '{file_path}' is missing required columns: {missing_str}")

        df = df.copy()
        df["source_file"] = os.path.basename(file_path)
        bloom_frames.append(df)
        print(f"Loaded {os.path.basename(file_path)} ({len(df)} rows)")

    combined = pd.concat(bloom_frames, ignore_index=True)
    combined["location"] = combined["location"].apply(normalize_location)
    combined["country_code"] = combined["location"].apply(infer_country_code_from_location)
    combined["species"] = combined["country_code"].apply(get_species_for_country)

    ordered_columns = [
        "location",
        "country_code",
        "species",
        "lat",
        "long",
        "alt",
        "year",
        "bloom_date",
        "bloom_doy",
        "source_file",
    ]
    return combined[ordered_columns].sort_values(["country_code", "location", "year"]).reset_index(drop=True)


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    aggregated = aggregate_bloom_data()
    aggregated.to_csv(OUTPUT_FILE, index=False)

    print("\nAggregation complete.")
    print(f"Rows: {len(aggregated)}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nCountry coverage:")
    print(aggregated["country_code"].value_counts(dropna=False))


if __name__ == "__main__":
    main()