# Load libraries
library(dplyr)
library(readr)
library(lubridate)
library(tidyr)
library(purrr)

# Load data
load(file.path("data", "bloom_history.RData"))
load(file.path("data", "climate_data.RData"))

# ---- Thermal growth model config ----
THERMAL_MODEL_RESULTS_FILE <- file.path("data", "thermal_growth_model_results.RData")
MIN_YEAR <- 1973L
CHILL_TEMP_C <- 4.3
FORCING_BASE_TEMP_C <- 5.0
FORCING_START_SEARCH_END_MM_DD <- "06-30"
FORCING_START_ROLL_DAYS <- 7L
FORCING_START_ROLL_THRESHOLD_C <- 5.0

clean_noaa_numeric <- function(x) {
    readr::parse_number(as.character(x), na = c("", "NA"))
}

required_climate_cols <- c("location", "DATE", "TMAX", "TMIN")
missing_climate_cols <- setdiff(required_climate_cols, names(climate_data))
if (length(missing_climate_cols) > 0) {
    stop(
        sprintf(
            "Missing required climate columns: %s",
            paste(missing_climate_cols, collapse = ", ")
        )
    )
}

required_bloom_cols <- c("location", "year", "bloom_date", "bloom_doy")
missing_bloom_cols <- setdiff(required_bloom_cols, names(bloom_history))
if (length(missing_bloom_cols) > 0) {
    stop(
        sprintf(
            "Missing required bloom columns: %s",
            paste(missing_bloom_cols, collapse = ", ")
        )
    )
}

# 1. PROCESS ALL LOCATIONS (Removed the single location filter)
daily_climate <- climate_data %>%
    transmute(
        location,
        date = as.Date(DATE),
        year = lubridate::year(as.Date(DATE)),
        tmax_c = clean_noaa_numeric(TMAX) / 10,
        tmin_c = clean_noaa_numeric(TMIN) / 10
    ) %>%
    filter(!is.na(location), !is.na(date), year >= MIN_YEAR) %>%
    mutate(
        tmean_c = (tmax_c + tmin_c) / 2,
        forcing_gdd = pmax(tmean_c - FORCING_BASE_TEMP_C, 0)
    )

bloom_events <- bloom_history %>%
    transmute(
        location,
        year = as.integer(year),
        bloom_date = as.Date(bloom_date),
        bloom_doy = as.numeric(bloom_doy)
    ) %>%
    filter(!is.na(location), !is.na(year), !is.na(bloom_date), year >= MIN_YEAR) %>%
    group_by(location, year) %>%
    summarize(
        bloom_date = min(bloom_date, na.rm = TRUE),
        bloom_doy = min(bloom_doy, na.rm = TRUE),
        .groups = "drop"
    )

compute_event_features <- function(location_i, year_i, bloom_date_i, climate_tbl) {
    chill_start <- as.Date(sprintf("%d-10-01", year_i - 1L))
    forcing_search_start <- as.Date(sprintf("%d-01-01", year_i))
    forcing_search_end <- as.Date(sprintf("%d-%s", year_i, FORCING_START_SEARCH_END_MM_DD))

    local_climate <- climate_tbl %>% filter(location == location_i)

    forcing_window <- local_climate %>%
        filter(date >= forcing_search_start, date <= forcing_search_end) %>%
        arrange(date)

    if (nrow(forcing_window) >= FORCING_START_ROLL_DAYS) {
        rolling_mean <- stats::filter(
            forcing_window$tmean_c,
            rep(1 / FORCING_START_ROLL_DAYS, FORCING_START_ROLL_DAYS),
            sides = 1
        )
        candidate_idx <- which(!is.na(rolling_mean) & rolling_mean >= FORCING_START_ROLL_THRESHOLD_C)
        if (length(candidate_idx) > 0) {
            forcing_start <- forcing_window$date[min(candidate_idx)]
        } else {
            forcing_start <- forcing_search_start
        }
    } else {
        forcing_start <- forcing_search_start
    }

    chill_end <- forcing_start - 1

    chill_days <- local_climate %>%
        filter(date >= chill_start, date <= chill_end) %>%
        summarize(chill_days = sum(.data$tmean_c <= CHILL_TEMP_C, na.rm = TRUE)) %>%
        pull(chill_days)

    forcing_rows <- local_climate %>%
        filter(date >= forcing_start, date <= bloom_date_i)

    observed_gdd_to_bloom <- sum(forcing_rows$forcing_gdd, na.rm = TRUE)
    n_forcing_days <- nrow(forcing_rows)

    tibble(
        location = location_i,
        year = year_i,
        bloom_date = bloom_date_i,
        chill_start_date = chill_start,
        chill_end_date = chill_end,
        forcing_start_date = forcing_start,
        chill_days_oct1_dec31 = chill_days,
        observed_gdd_to_bloom = observed_gdd_to_bloom,
        n_forcing_days = n_forcing_days
    )
}

thermal_features <- pmap_dfr(
    list(bloom_events$location, bloom_events$year, bloom_events$bloom_date),
    \(location_i, year_i, bloom_date_i) {
        compute_event_features(location_i, year_i, bloom_date_i, daily_climate)
    }
) %>%
    left_join(bloom_events %>% select(location, year, bloom_doy), by = c("location", "year")) %>%
    filter(
        !is.na(chill_days_oct1_dec31),
        !is.na(observed_gdd_to_bloom),
        !is.na(bloom_doy),
        n_forcing_days > 0
    )

if (nrow(thermal_features) < 20) {
    stop("Not enough thermal feature rows to train/validate/test the thermal model.")
}

# 2. CUSTOM SPLIT FUNCTION: Only DC and Kyoto are allowed in the training/validation sets.
assign_split <- function(years, is_train_loc, train_frac = 0.70, val_frac = 0.15) {
    n <- length(years)

    # If it is NOT a training location (e.g., Liestal, Vancouver), force 100% to test set.
    if (!is_train_loc) {
        return(rep("test", n))
    }

    n_train <- max(1L, floor(n * train_frac))
    n_val <- max(1L, floor(n * val_frac))
    n_test <- n - n_train - n_val

    if (n_test < 1L) {
        n_test <- 1L
        if (n_train > n_val) {
            n_train <- n_train - 1L
        } else {
            n_val <- n_val - 1L
        }
    }

    c(rep("train", n_train), rep("validation", n_val), rep("test", n_test))
}

# Apply the custom split
thermal_features <- thermal_features %>%
    arrange(location, year) %>%
    group_by(location) %>%
    mutate(
        is_train_loc = location %in% c("washingtondc", "kyoto"),
        split = assign_split(year, is_train_loc[1])
    ) %>%
    ungroup() %>%
    select(-is_train_loc)

train_data <- thermal_features %>% filter(split == "train")
validation_data <- thermal_features %>% filter(split == "validation")
test_data <- thermal_features %>% filter(split == "test")

# 3. GENERALIZED MODEL: No "+ location" parameter, allowing it to extrapolate to the test locations.
gdd_threshold_model <- lm(
    observed_gdd_to_bloom ~ chill_days_oct1_dec31,
    data = train_data
)

predict_bloom_date_from_threshold <- function(location_i, year_i, threshold_gdd, climate_tbl) {
    forcing_search_start <- as.Date(sprintf("%d-01-01", year_i))
    forcing_search_end <- as.Date(sprintf("%d-%s", year_i, FORCING_START_SEARCH_END_MM_DD))
    forcing_end <- as.Date(sprintf("%d-12-31", year_i))

    forcing_window <- climate_tbl %>%
        filter(location == location_i, date >= forcing_search_start, date <= forcing_search_end) %>%
        arrange(date)

    if (nrow(forcing_window) >= FORCING_START_ROLL_DAYS) {
        rolling_mean <- stats::filter(
            forcing_window$tmean_c,
            rep(1 / FORCING_START_ROLL_DAYS, FORCING_START_ROLL_DAYS),
            sides = 1
        )
        candidate_idx <- which(!is.na(rolling_mean) & rolling_mean >= FORCING_START_ROLL_THRESHOLD_C)
        if (length(candidate_idx) > 0) {
            forcing_start <- forcing_window$date[min(candidate_idx)]
        } else {
            forcing_start <- forcing_search_start
        }
    } else {
        forcing_start <- forcing_search_start
    }

    forcing_path <- climate_tbl %>%
        filter(location == location_i, date >= forcing_start, date <= forcing_end) %>%
        arrange(date) %>%
        mutate(cum_gdd = cumsum(replace_na(.data$forcing_gdd, 0)))

    reached <- forcing_path %>% filter(.data$cum_gdd >= threshold_gdd)
    if (nrow(reached) == 0) {
        return(as.Date(NA))
    }

    reached$date[1]
}

score_split <- function(split_df, model_obj, climate_tbl) {
    if (nrow(split_df) == 0) {
        return(
            list(
                predictions = tibble(),
                metrics = tibble(n = 0L, mae = NA_real_, rmse = NA_real_)
            )
        )
    }

    threshold_pred <- predict(model_obj, newdata = split_df)
    predicted_dates <- pmap(
        list(split_df$location, split_df$year, as.numeric(threshold_pred)),
        \(location_i, year_i, threshold_i) {
            predict_bloom_date_from_threshold(location_i, year_i, threshold_i, climate_tbl)
        }
    )

    predictions <- split_df %>%
        mutate(predicted_gdd_threshold = as.numeric(threshold_pred)) %>%
        mutate(
            predicted_bloom_date = as.Date(unlist(predicted_dates), origin = "1970-01-01"),
            predicted_bloom_doy = as.numeric(lubridate::yday(.data$predicted_bloom_date)),
            abs_error = abs(.data$predicted_bloom_doy - .data$bloom_doy),
            sq_error = (.data$predicted_bloom_doy - .data$bloom_doy)^2
        )

    metrics <- predictions %>%
        summarize(
            n = sum(!is.na(.data$predicted_bloom_doy)),
            mae = mean(.data$abs_error, na.rm = TRUE),
            rmse = sqrt(mean(.data$sq_error, na.rm = TRUE))
        )

    list(predictions = predictions, metrics = metrics)
}

validation_scored <- score_split(validation_data, gdd_threshold_model, daily_climate)
test_scored <- score_split(test_data, gdd_threshold_model, daily_climate)

thermal_model_metrics <- bind_rows(
    validation_scored$metrics %>% mutate(split = "validation"),
    test_scored$metrics %>% mutate(split = "test")
)

thermal_model_predictions <- bind_rows(
    validation_scored$predictions %>% mutate(split = "validation"),
    test_scored$predictions %>% mutate(split = "test")
) %>%
    select(
        location,
        year,
        split,
        chill_start_date,
        chill_end_date,
        forcing_start_date,
        chill_days_oct1_dec31,
        observed_gdd_to_bloom,
        predicted_gdd_threshold,
        bloom_date,
        predicted_bloom_date,
        bloom_doy,
        predicted_bloom_doy,
        abs_error,
        sq_error
    )

# 2026 forecast from chill days in Oct-Dec 2025
future_2026 <- bloom_events %>%
    group_by(location) %>%
    summarize(year = 2026L, .groups = "drop") %>%
    mutate(
        chill_days_oct1_dec31 = map2_dbl(location, year, \(location_i, year_i) {
            chill_start <- as.Date(sprintf("%d-10-01", year_i - 1L))
            chill_end <- as.Date(sprintf("%d-12-31", year_i - 1L))

            daily_climate %>%
                filter(location == location_i, date >= chill_start, date <= chill_end) %>%
                summarize(chill_days = sum(.data$tmean_c <= CHILL_TEMP_C, na.rm = TRUE)) %>%
                pull(chill_days)
        })
    )

pred_2026_threshold_ci <- predict(
    gdd_threshold_model,
    newdata = future_2026,
    interval = "confidence",
    level = 0.90
)

pred_dates_fit <- pmap(
    list(future_2026$location, future_2026$year, as.numeric(pred_2026_threshold_ci[, "fit"])),
    \(location_i, year_i, threshold_i) {
        predict_bloom_date_from_threshold(location_i, year_i, threshold_i, daily_climate)
    }
)

pred_dates_low <- pmap(
    list(future_2026$location, future_2026$year, as.numeric(pred_2026_threshold_ci[, "lwr"])),
    \(location_i, year_i, threshold_i) {
        predict_bloom_date_from_threshold(location_i, year_i, threshold_i, daily_climate)
    }
)

pred_dates_high <- pmap(
    list(future_2026$location, future_2026$year, as.numeric(pred_2026_threshold_ci[, "upr"])),
    \(location_i, year_i, threshold_i) {
        predict_bloom_date_from_threshold(location_i, year_i, threshold_i, daily_climate)
    }
)

future_2026 <- future_2026 %>%
    mutate(
        predicted_gdd_threshold = as.numeric(pred_2026_threshold_ci[, "fit"]),
        predicted_gdd_threshold_low_90 = as.numeric(pred_2026_threshold_ci[, "lwr"]),
        predicted_gdd_threshold_high_90 = as.numeric(pred_2026_threshold_ci[, "upr"]),
        predicted_bloom_date = as.Date(unlist(pred_dates_fit), origin = "1970-01-01"),
        conf_low_date_90 = as.Date(unlist(pred_dates_low), origin = "1970-01-01"),
        conf_high_date_90 = as.Date(unlist(pred_dates_high), origin = "1970-01-01"),
        predicted_bloom_doy = as.numeric(lubridate::yday(.data$predicted_bloom_date)),
        conf_low_doy_90 = as.numeric(lubridate::yday(.data$conf_low_date_90)),
        conf_high_doy_90 = as.numeric(lubridate::yday(.data$conf_high_date_90)),
        conf_pm_days_90 = (.data$conf_high_doy_90 - .data$conf_low_doy_90) / 2,
        conf_minus_days_90 = .data$predicted_bloom_doy - .data$conf_low_doy_90,
        conf_plus_days_90 = .data$conf_high_doy_90 - .data$predicted_bloom_doy
    )

save(
    daily_climate,
    bloom_events,
    thermal_features,
    train_data,
    validation_data,
    test_data,
    gdd_threshold_model,
    thermal_model_metrics,
    thermal_model_predictions,
    future_2026,
    file = THERMAL_MODEL_RESULTS_FILE
)

message(sprintf("Saved thermal growth model results to %s", THERMAL_MODEL_RESULTS_FILE))
print(thermal_model_metrics)
print(future_2026)
