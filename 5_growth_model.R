# Load the Data
library(dplyr)
library(readr)
library(lubridate)

load(file.path("data", "bloom_history.RData"))
load(file.path("data", "climate_data.RData"))
load(file.path("data", "climate_ts_stationary.RData"))

# ---- Growth model fitting ----
MODEL_INPUT_FILE <- file.path("data", "growth_model_inputs.RData")
ENV_LAPSE_RATE_C_PER_KM <- 6.5

clean_noaa_numeric <- function(x) {
    readr::parse_number(as.character(x), na = c("", "NA"))
}

required_climate_cols <- c("location", "DATE", "ELEVATION", "TMAX", "TMIN", "PRCP")
missing_climate_cols <- setdiff(required_climate_cols, names(climate_data))
if (length(missing_climate_cols) > 0) {
    stop(
        sprintf(
            "Missing required climate columns: %s",
            paste(missing_climate_cols, collapse = ", ")
        )
    )
}

required_climate_coord_cols <- c("LATITUDE", "LONGITUDE")
missing_climate_coord_cols <- setdiff(required_climate_coord_cols, names(climate_data))
if (length(missing_climate_coord_cols) > 0) {
    stop(
        sprintf(
            "Missing required climate coordinate columns: %s",
            paste(missing_climate_coord_cols, collapse = ", ")
        )
    )
}

required_bloom_cols <- c("location", "year", "bloom_date", "bloom_doy", "alt")
missing_bloom_cols <- setdiff(required_bloom_cols, names(bloom_history))
if (length(missing_bloom_cols) > 0) {
    stop(
        sprintf(
            "Missing required bloom columns: %s",
            paste(missing_bloom_cols, collapse = ", ")
        )
    )
}

required_bloom_coord_cols <- c("lat", "long")
missing_bloom_coord_cols <- setdiff(required_bloom_coord_cols, names(bloom_history))
if (length(missing_bloom_coord_cols) > 0) {
    stop(
        sprintf(
            "Missing required bloom coordinate columns: %s",
            paste(missing_bloom_coord_cols, collapse = ", ")
        )
    )
}

haversine_km <- function(lat1, lon1, lat2, lon2) {
    earth_radius_km <- 6371
    to_rad <- pi / 180

    lat1_r <- lat1 * to_rad
    lon1_r <- lon1 * to_rad
    lat2_r <- lat2 * to_rad
    lon2_r <- lon2 * to_rad

    dlat <- lat2_r - lat1_r
    dlon <- lon2_r - lon1_r

    a <- sin(dlat / 2)^2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2)^2
    c <- 2 * atan2(sqrt(a), sqrt(1 - a))

    earth_radius_km * c
}

# Altitude mapping: station altitude (from NOAA ELEVATION) vs bloom-site altitude (alt)
station_altitude <- climate_data %>%
    transmute(
        location,
        station_alt_m = clean_noaa_numeric(ELEVATION)
    ) %>%
    group_by(location) %>%
    summarize(
        station_alt_m = median(station_alt_m, na.rm = TRUE),
        .groups = "drop"
    )

station_coords <- climate_data %>%
    transmute(
        location,
        station_lat = clean_noaa_numeric(LATITUDE),
        station_long = clean_noaa_numeric(LONGITUDE)
    ) %>%
    group_by(location) %>%
    summarize(
        station_lat = median(station_lat, na.rm = TRUE),
        station_long = median(station_long, na.rm = TRUE),
        .groups = "drop"
    )

bloom_altitude <- bloom_history %>%
    transmute(
        location,
        bloom_alt_m = as.numeric(alt)
    ) %>%
    group_by(location) %>%
    summarize(
        bloom_alt_m = median(bloom_alt_m, na.rm = TRUE),
        .groups = "drop"
    )

bloom_coords <- bloom_history %>%
    transmute(
        location,
        bloom_lat = as.numeric(lat),
        bloom_long = as.numeric(long)
    ) %>%
    group_by(location) %>%
    summarize(
        bloom_lat = median(bloom_lat, na.rm = TRUE),
        bloom_long = median(bloom_long, na.rm = TRUE),
        .groups = "drop"
    )

altitude_adjustment <- station_altitude %>%
    inner_join(bloom_altitude, by = "location") %>%
    inner_join(station_coords, by = "location") %>%
    inner_join(bloom_coords, by = "location") %>%
    mutate(
        altitude_delta_m = station_alt_m - bloom_alt_m,
        temp_adjustment_c = ENV_LAPSE_RATE_C_PER_KM * altitude_delta_m / 1000,
        distance_km = haversine_km(station_lat, station_long, bloom_lat, bloom_long)
    )

# Apply lapse-rate correction so climate temperatures reflect bloom-site altitude
climate_adjusted_daily <- climate_data %>%
    mutate(
        DATE = as.Date(DATE),
        year = lubridate::year(DATE),
        TMAX = clean_noaa_numeric(TMAX) / 10,
        TMIN = clean_noaa_numeric(TMIN) / 10,
        PRCP = clean_noaa_numeric(PRCP) / 10
    ) %>%
    left_join(altitude_adjustment, by = "location") %>%
    mutate(
        tmax_adj = TMAX + temp_adjustment_c,
        tmin_adj = TMIN + temp_adjustment_c
    )

bloom_history_yearly <- bloom_history %>%
    mutate(bloom_date = as.Date(bloom_date)) %>%
    group_by(location, year) %>%
    summarize(
        bloom_date = min(bloom_date, na.rm = TRUE),
        bloom_doy = min(bloom_doy, na.rm = TRUE),
        .groups = "drop"
    )

prebloom_features <- climate_adjusted_daily %>%
    inner_join(
        bloom_history_yearly,
        by = c("location", "year"),
        relationship = "many-to-one"
    ) %>%
    filter(
        DATE <= bloom_date
    ) %>%
    group_by(location, year) %>%
    summarize(
        mean_tmax_adj_prebloom = mean(tmax_adj, na.rm = TRUE),
        mean_tmin_adj_prebloom = mean(tmin_adj, na.rm = TRUE),
        total_prcp_prebloom = sum(PRCP, na.rm = TRUE),
        n_daily_obs = n(),
        .groups = "drop"
    )

model_inputs <- bloom_history %>%
    mutate(bloom_date = as.Date(bloom_date)) %>%
    select(location, year, bloom_date, bloom_doy, alt) %>%
    left_join(altitude_adjustment, by = "location") %>%
    left_join(prebloom_features, by = c("location", "year"))

save(
    altitude_adjustment,
    climate_adjusted_daily,
    prebloom_features,
    model_inputs,
    file = MODEL_INPUT_FILE
)

message(sprintf("Saved growth model inputs to %s", MODEL_INPUT_FILE))
