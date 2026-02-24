library(tidyr)
library(dplyr)

# Load the data
# Bloom history files used to build the core `cherry` dataset.
CHERRY_BLOOM_FILES <- c(
    file.path("data", "washingtondc.csv"),
    file.path("data", "liestal.csv"),
    file.path("data", "kyoto.csv"),
    file.path("data", "vancouver.csv"),
    file.path("data", "nyc.csv")
)


# Climate data files used to build the core `climate` dataset.
CLIMATE_DATA_FILES <- c(
    file.path("data", "USW00013743.csv"),
    file.path("data", "CA001108395.csv"),
    file.path("data", "USW00014732.csv"),
    file.path("data", "SZ000001940.csv"),
    file.path("data", "JA000047759.csv")
)

# Manual mapping from NOAA station IDs to bloom-history locations
STATION_LOCATION_MAP <- tibble(
    STATION = c("USW00013743", "CA001108395", "USW00014732", "SZ000001940", "JA000047759"),
    location = c("washingtondc", "vancouver", "newyorkcity", "liestal", "kyoto")
)

# Load the bloom history data
bloom_history <- CHERRY_BLOOM_FILES %>%
    lapply(read.csv) %>%
    bind_rows()

# Load the climate data
climate_data <- CLIMATE_DATA_FILES %>%
    lapply(read.csv) %>%
    bind_rows()

NOAA_DAILY_DATATYPES <- c("TMAX", "TMIN", "PRCP")
MIN_YEAR <- 1973L
# Filter the climate data to include only the relevant datatypes
RELEVANT_CLIMATE_COLUMNS <- c("STATION", "DATE", "NAME", "ELEVATION", "LATITUDE", "LONGITUDE", NOAA_DAILY_DATATYPES)

climate_data <- climate_data %>%
    select(any_of(RELEVANT_CLIMATE_COLUMNS))

# Associate each climate station with the bloom-history location
climate_data <- climate_data %>%
    left_join(STATION_LOCATION_MAP, by = "STATION")

# Helper to parse NOAA numeric fields stored as padded strings
clean_noaa_numeric <- function(x) {
    suppressWarnings(as.numeric(trimws(as.character(x))))
}

# Fill only short internal NA gaps (<= 2 days) for temperature series
fill_small_temp_gaps <- function(x, max_gap = 2L) {
    x_num <- clean_noaa_numeric(x)

    if (all(is.na(x_num))) {
        return(x_num)
    }

    non_missing <- !is.na(x_num)
    if (sum(non_missing) < 2) {
        return(x_num)
    }

    idx <- seq_along(x_num)
    interpolated <- stats::approx(
        x = idx[non_missing],
        y = x_num[non_missing],
        xout = idx,
        method = "linear",
        rule = 1,
        ties = "ordered"
    )$y

    missing_runs <- rle(is.na(x_num))
    run_end <- cumsum(missing_runs$lengths)
    run_start <- run_end - missing_runs$lengths + 1

    for (i in seq_along(missing_runs$values)) {
        run_is_missing <- missing_runs$values[i]
        run_len <- missing_runs$lengths[i]
        start_i <- run_start[i]
        end_i <- run_end[i]

        if (!run_is_missing || run_len > max_gap) {
            next
        }

        if (start_i == 1 || end_i == length(x_num)) {
            next
        }

        x_num[start_i:end_i] <- interpolated[start_i:end_i]
    }

    x_num
}

# Convert the DATE column to Date format
climate_data$DATE <- as.Date(climate_data$DATE, format = "%Y-%m-%d")

# Calculate the bloom day of year (DOY) for each bloom history record
bloom_history$bloom_date <- as.Date(bloom_history$bloom_date, format = "%Y-%m-%d")

# Keep only records from MIN_YEAR onward
bloom_history <- bloom_history %>%
    mutate(year = as.integer(year)) %>%
    filter(year >= MIN_YEAR)

climate_data <- climate_data %>%
    filter(!is.na(DATE), as.integer(format(DATE, "%Y")) >= MIN_YEAR)

# Truncate climate data to years available in bloom history for each location
available_bloom_years <- bloom_history %>%
    transmute(location, year = as.integer(year)) %>%
    distinct()

climate_data <- climate_data %>%
    mutate(year = as.integer(format(DATE, "%Y"))) %>%
    semi_join(available_bloom_years, by = c("location", "year")) %>%
    select(-year)

# Fill small (< 3 day) gaps for temperature columns only
TEMP_COLUMNS <- c("TMAX", "TMIN", "TAVG")

climate_data <- climate_data %>%
    arrange(location, DATE) %>%
    group_by(location) %>%
    mutate(across(any_of(TEMP_COLUMNS), ~ fill_small_temp_gaps(.x, max_gap = 2L))) %>%
    ungroup()

# Save the loaded data for use in the next steps
save(bloom_history, file = file.path("data", "bloom_history.RData"))
save(climate_data, file = file.path("data", "climate_data.RData"))


# Clean up the environment
rm(list = setdiff(ls(), c("bloom_history", "climate_data")))
gc()
