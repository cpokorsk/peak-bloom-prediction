# load the libraries
library(tidyverse)
library(tseries)

# Load the data
load(file.path("data", "bloom_history.RData"))
load(file.path("data", "climate_data.RData"))

# ---- Climate data prep for time-series analysis ----
CLIMATE_TS_OUTPUT_FILE <- file.path("data", "climate_ts_stationary.RData")
ACF_FIGURES_DIR <- file.path("figures", "acf")
STATIONARITY_ALPHA <- 0.05
MAX_DIFFERENCE_ORDER <- 2
MIN_ADF_OBS <- 10

REQUIRED_CLIMATE_COLUMNS <- c("location", "DATE", "TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "SNWD")
missing_climate_cols <- setdiff(REQUIRED_CLIMATE_COLUMNS, names(climate_data))
if (length(missing_climate_cols) > 0) {
    stop(
        sprintf(
            "Missing required climate columns: %s",
            paste(missing_climate_cols, collapse = ", ")
        )
    )
}

clean_noaa_numeric <- function(x) {
    readr::parse_number(as.character(x), na = c("", "NA"))
}

make_stationary <- function(values, max_diff = MAX_DIFFERENCE_ORDER,
                            alpha = STATIONARITY_ALPHA, min_obs = MIN_ADF_OBS) {
    current_values <- values
    current_diff_order <- 0L
    adf_p_value <- NA_real_
    is_stationary <- FALSE
    note <- ""

    repeat {
        if (length(current_values) < min_obs) {
            note <- sprintf("insufficient_obs(<%d)", min_obs)
            break
        }

        if (stats::var(current_values, na.rm = TRUE) == 0) {
            adf_p_value <- 0
            is_stationary <- TRUE
            note <- "constant_series"
            break
        }

        adf_p_value <- tryCatch(
            {
                tseries::adf.test(current_values, alternative = "stationary")$p.value
            },
            error = function(e) {
                NA_real_
            }
        )

        if (!is.na(adf_p_value) && adf_p_value <= alpha) {
            is_stationary <- TRUE
            if (current_diff_order == 0L) {
                note <- "already_stationary"
            } else {
                note <- "stationary_after_differencing"
            }
            break
        }

        if (current_diff_order >= max_diff) {
            note <- "max_differencing_reached"
            break
        }

        current_values <- diff(current_values)
        current_diff_order <- current_diff_order + 1L
    }

    list(
        stationary_values = current_values,
        diff_order = current_diff_order,
        adf_p_value = adf_p_value,
        is_stationary = is_stationary,
        note = note
    )
}

climate_ts_yearly <- climate_data %>%
    mutate(
        DATE = as.Date(DATE),
        TMAX = clean_noaa_numeric(TMAX) / 10,
        TMIN = clean_noaa_numeric(TMIN) / 10,
        TAVG = clean_noaa_numeric(TAVG) / 10,
        PRCP = clean_noaa_numeric(PRCP) / 10,
        SNOW = clean_noaa_numeric(SNOW),
        SNWD = clean_noaa_numeric(SNWD)
    ) %>%
    select(location, DATE, TMAX, TMIN, TAVG, PRCP, SNOW, SNWD) %>%
    filter(!is.na(location), !is.na(DATE)) %>%
    inner_join(
        bloom_history %>%
            mutate(bloom_date = as.Date(bloom_date)) %>%
            select(location, year, bloom_date),
        by = "location"
    ) %>%
    filter(
        lubridate::year(DATE) == year,
        DATE <= bloom_date
    ) %>%
    group_by(location, year) %>%
    summarize(
        bloom_date = first(bloom_date),
        mean_tmax_prebloom = mean(TMAX, na.rm = TRUE),
        mean_tmin_prebloom = mean(TMIN, na.rm = TRUE),
        mean_tavg_prebloom = mean(TAVG, na.rm = TRUE),
        total_prcp_prebloom = sum(PRCP, na.rm = TRUE),
        total_snow_prebloom = sum(SNOW, na.rm = TRUE),
        mean_snwd_prebloom = mean(SNWD, na.rm = TRUE),
        n_daily_obs = n(),
        .groups = "drop"
    ) %>%
    arrange(location, year)

climate_ts_long <- climate_ts_yearly %>%
    pivot_longer(
        cols = c(
            mean_tmax_prebloom,
            mean_tmin_prebloom,
            mean_tavg_prebloom,
            total_prcp_prebloom,
            total_snow_prebloom,
            mean_snwd_prebloom
        ),
        names_to = "metric",
        values_to = "value"
    ) %>%
    filter(!is.na(value))

stationarity_results_nested <- climate_ts_long %>%
    arrange(location, metric, year) %>%
    group_by(location, metric) %>%
    nest() %>%
    mutate(
        ts_result = purrr::map(data, \(df) {
            values <- df$value
            years <- df$year
            result <- make_stationary(values)
            kept_years <- years[(result$diff_order + 1):length(years)]

            list(
                diff_order = result$diff_order,
                adf_p_value = result$adf_p_value,
                is_stationary = result$is_stationary,
                note = result$note,
                years = kept_years,
                stationary_values = result$stationary_values
            )
        })
    )

stationarity_diagnostics <- stationarity_results_nested %>%
    transmute(
        location,
        metric,
        diff_order = purrr::map_int(ts_result, "diff_order"),
        adf_p_value = purrr::map_dbl(ts_result, "adf_p_value"),
        is_stationary = purrr::map_lgl(ts_result, "is_stationary"),
        note = purrr::map_chr(ts_result, "note")
    )

climate_ts_stationary <- stationarity_results_nested %>%
    transmute(location, metric, ts_points = purrr::map(ts_result, \(x) {
        tibble(
            year = x$years,
            value_stationary = x$stationary_values
        )
    })) %>%
    unnest(ts_points)

# ---- ACF charts for stationary climate series ----
dir.create(ACF_FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

acf_plot_data <- climate_ts_stationary %>%
    group_by(location, metric) %>%
    arrange(year, .by_group = TRUE) %>%
    summarize(
        acf_tbl = list({
            x <- value_stationary[!is.na(value_stationary)]
            if (length(x) < 3) {
                tibble(lag = numeric(0), acf = numeric(0))
            } else {
                lag_max <- min(20, length(x) - 1)
                acf_obj <- stats::acf(x, lag.max = lag_max, plot = FALSE)
                tibble(
                    lag = as.numeric(acf_obj$lag),
                    acf = as.numeric(acf_obj$acf)
                ) %>%
                    filter(lag > 0)
            }
        }),
        .groups = "drop"
    ) %>%
    unnest(acf_tbl)

acf_metric_plots <- acf_plot_data %>%
    group_by(metric) %>%
    group_split()

purrr::walk(acf_metric_plots, \(metric_df) {
    metric_name <- unique(metric_df$metric)
    p <- ggplot(metric_df, aes(x = lag, y = acf)) +
        geom_col(width = 0.7, fill = "steelblue") +
        geom_hline(yintercept = 0, color = "gray40") +
        facet_wrap(vars(location), scales = "free_y") +
        labs(
            title = sprintf("ACF of stationary %s by location", metric_name),
            x = "Lag",
            y = "Autocorrelation"
        ) +
        theme_minimal()

    ggsave(
        filename = file.path(ACF_FIGURES_DIR, sprintf("acf_%s.png", metric_name)),
        plot = p,
        width = 10,
        height = 6,
        dpi = 150
    )
})

message(sprintf("Saved ACF charts to %s", ACF_FIGURES_DIR))

save(
    climate_ts_yearly,
    climate_ts_long,
    stationarity_diagnostics,
    climate_ts_stationary,
    file = CLIMATE_TS_OUTPUT_FILE
)

message(sprintf("Saved climate time-series objects to %s", CLIMATE_TS_OUTPUT_FILE))
