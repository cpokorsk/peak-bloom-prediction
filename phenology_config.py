import os

# ==========================================
# FILES & DIRECTORIES
# ==========================================
DATA_DIR = "data"
NOAA_DIR = os.path.join(DATA_DIR, "noaa")
BLOOM_DIR = os.path.join(DATA_DIR, "blossoms")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
MODEL_INPUT_DIR = os.path.join(DATA_DIR, "model_inputs")
MODEL_OUTPUT_DIR = os.path.join(DATA_DIR, "model_outputs")
HOLDOUT_OUTPUT_DIR = os.path.join(MODEL_OUTPUT_DIR, "holdout")
PREDICTIONS_OUTPUT_DIR = os.path.join(MODEL_OUTPUT_DIR, "predictions")

NOAA_STATION_METADATA_FILE = os.path.join(METADATA_DIR, "NOAA_station_metadata.csv")
BLOSSOM_SITE_METADATA_FILE = os.path.join(METADATA_DIR, "blossom_site_metadata.csv")
AGGREGATED_BLOOM_FILE = os.path.join(MODEL_INPUT_DIR, "aggregated_bloom_data.csv")
AGGREGATED_CLIMATE_FILE = os.path.join(MODEL_INPUT_DIR, "aggregated_climate_data.csv")
PROJECTED_CLIMATE_FILE = os.path.join(MODEL_INPUT_DIR, "projected_climate_2026.csv")
MODEL_FEATURES_FILE = os.path.join(MODEL_INPUT_DIR, "model_features.csv")
FINAL_PREDICTIONS_FILE = os.path.join(PREDICTIONS_OUTPUT_DIR, "final_2026_predictions.csv")

# ==========================================
# PIPELINE PARAMS
# ==========================================
TARGET_YEAR = 2026
MIN_CLIMATE_YEAR = 1960
MIN_MODEL_YEAR = 1961
BASELINE_START_YEAR = 2005

LAPSE_RATE_C_PER_M = 0.0065
AR_LAGS = 3
MAX_STATION_DISTANCE_KM = 50.0

EARLY_SPRING_END_MONTH_DAY = "03-15"
WINTER_START_MONTH_DAY = "10-01"
WINTER_END_MONTH_DAY = "12-31"
FORECAST_END_MONTH_DAY = "05-31"

TARGET_PREDICTION_LOCATIONS = ["washingtondc", "kyoto", "liestal", "vancouver", "newyorkcity"]
LOCATION_ALIASES = {
    "nyc": "newyorkcity",
    "japan/kyoto": "kyoto",
    "switzerland/liestal": "liestal",
}
HOLDOUT_LOCATIONS = ["vancouver", "newyorkcity"]
HOLDOUT_EXTRA_COUNTRIES = ["KR", "CH", "JP"]
HOLDOUT_PER_COUNTRY = 3
HOLDOUT_LAST_N_YEARS = 10
HOLDOUT_RANDOM_SEED = 42

# ==========================================
# LOCATION / SPECIES MAPPINGS
# ==========================================
COUNTRY_NAME_TO_CODE = {
    "japan": "JP",
    "south korea": "KR",
    "switzerland": "CH",
    "usa": "US",
    "canada": "CA",
}

LOCATION_TO_COUNTRY_CODE = {
    "kyoto": "JP",
    "washingtondc": "US",
    "newyorkcity": "US",
    "liestal": "CH",
    "vancouver": "CA",
}

COUNTRY_CODE_TO_SPECIES = {
    "US": "Prunus x yedoensis",
    "CA": "Prunus x yedoensis",
    "CH": "Prunus avium",
    "JP": "Prunus x jamasakura",
    "KR": "Prunus x yedoensis",
}

COUNTRY_CODE_TO_CONTINENT = {
    "US": "North America",
    "CA": "North America",
    "CH": "Europe",
    "JP": "Asia",
    "KR": "Asia",
}

LOCATION_SOURCE_PREFERENCE = {
    "kyoto": ["kyoto.csv", "japan.csv"],
    "liestal": ["liestal.csv", "meteoswiss.csv"],
}

STATION_PREFIX_TO_COUNTRY_CODE = {
    "JA": "JP",
    "KS": "KR",
    "US": "US",
    "CA": "CA",
    "SZ": "CH",
    "GM": "DE",
    "IT": "IT",
}

STATION_SUFFIX_TO_COUNTRY_CODE = {
    "JA": "JP",
    "KS": "KR",
    "SZ": "CH",
    "US": "US",
    "CA": "CA",
    "IT": "IT",
    "GM": "DE",
}

# ==========================================
# PHENOLOGY THRESHOLDS
# ==========================================
SPECIES_THRESHOLDS = {
    "Prunus x yedoensis": {"chill_temp_c": 5.0, "forcing_base_c": 5.0},
    "Prunus avium": {"chill_temp_c": 4.3, "forcing_base_c": 4.0},
    "Prunus x jamasakura": {"chill_temp_c": 5.0, "forcing_base_c": 5.0},
    "Unknown": {"chill_temp_c": 5.0, "forcing_base_c": 5.0},
}

DEFAULT_SPECIES = "Unknown"
DEFAULT_CHILL_TEMP_C = SPECIES_THRESHOLDS[DEFAULT_SPECIES]["chill_temp_c"]
DEFAULT_FORCING_BASE_C = SPECIES_THRESHOLDS[DEFAULT_SPECIES]["forcing_base_c"]


def normalize_location(location_name):
    location = str(location_name).strip().lower()
    return LOCATION_ALIASES.get(location, location)


def get_species_thresholds(species_name):
    return SPECIES_THRESHOLDS.get(species_name, SPECIES_THRESHOLDS[DEFAULT_SPECIES])


def infer_country_code_from_location(location_name):
    if location_name is None:
        return "UNK"

    location = str(location_name).strip()
    if not location:
        return "UNK"

    location_norm = normalize_location(location)
    if location_norm in LOCATION_TO_COUNTRY_CODE:
        return LOCATION_TO_COUNTRY_CODE[location_norm]

    if "/" in location:
        country_name = location.split("/", 1)[0].strip().lower()
        if country_name in COUNTRY_NAME_TO_CODE:
            return COUNTRY_NAME_TO_CODE[country_name]

    return "UNK"


def get_species_for_country(country_code):
    return COUNTRY_CODE_TO_SPECIES.get(country_code, DEFAULT_SPECIES)


def get_continent_for_country(country_code):
    return COUNTRY_CODE_TO_CONTINENT.get(country_code, "Unknown")


def dedupe_bloom_sources(df, location_col="location", year_col="year", source_col="source_file"):
    if source_col not in df.columns:
        return df

    df = df.copy()
    df["_location_norm"] = df[location_col].apply(normalize_location)

    def priority(row):
        prefs = LOCATION_SOURCE_PREFERENCE.get(row["_location_norm"], [])
        if not prefs:
            return len(prefs)
        try:
            return prefs.index(str(row[source_col]))
        except ValueError:
            return len(prefs)

    df["_source_priority"] = df.apply(priority, axis=1)
    df = df.sort_values(["_location_norm", year_col, "_source_priority", source_col])

    deduped = df.drop_duplicates(subset=["_location_norm", year_col], keep="first")
    return deduped.drop(columns=["_location_norm", "_source_priority"])
