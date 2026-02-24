import glob
import os

import pandas as pd
from phenology_config import (
    BLOOM_DIR,
    BLOSSOM_SITE_METADATA_FILE,
    dedupe_bloom_sources,
    infer_country_code_from_location,
    get_species_for_country,
    normalize_location,
)


OUTPUT_FILE = BLOSSOM_SITE_METADATA_FILE


def load_all_blossom_rows() -> pd.DataFrame:
    bloom_files = sorted(glob.glob(os.path.join(BLOOM_DIR, "*.csv")))
    if not bloom_files:
        raise FileNotFoundError(f"No bloom CSV files found in: {BLOOM_DIR}")

    required_columns = {"location", "lat", "long", "alt", "year", "bloom_date", "bloom_doy"}
    frames = []

    print(f"Found {len(bloom_files)} bloom CSV files.")
    for file_path in bloom_files:
        df = pd.read_csv(file_path)
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(f"File '{file_path}' is missing required columns: {missing_str}")

        df = df.copy()
        df["source_file"] = os.path.basename(file_path)
        frames.append(df)
        print(f"Loaded {os.path.basename(file_path)} ({len(df)} rows)")

    combined = pd.concat(frames, ignore_index=True)
    combined["location"] = combined["location"].apply(normalize_location)
    combined = dedupe_bloom_sources(combined)
    combined["country_code"] = combined["location"].apply(infer_country_code_from_location)
    combined["species"] = combined["country_code"].apply(get_species_for_country)
    return combined


def build_site_metadata(df: pd.DataFrame) -> pd.DataFrame:
    site_df = (
        df.groupby("location", as_index=False)
        .agg(
            country_code=("country_code", "first"),
            species=("species", "first"),
            lat=("lat", "first"),
            long=("long", "first"),
            alt=("alt", "first"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            observation_count=("year", "size"),
            source_files=("source_file", lambda s: ";".join(sorted(set(s)))),
        )
        .sort_values(["country_code", "location"])
        .reset_index(drop=True)
    )

    return site_df


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    combined = load_all_blossom_rows()
    site_metadata = build_site_metadata(combined)
    site_metadata.to_csv(OUTPUT_FILE, index=False)

    print("\nSite metadata generation complete.")
    print(f"Unique sites: {len(site_metadata)}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nSite count by country:")
    print(site_metadata["country_code"].value_counts(dropna=False))
    print("\nPreview:")
    print(site_metadata.head(10).to_string(index=False))


if __name__ == "__main__":
    main()