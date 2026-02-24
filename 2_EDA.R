library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyr)

CLIMATE_FILE <- file.path("data", "historical_temps", "historic_climate_metrics.csv")
CHERRY_BLOOM_FILES <- c(
    file.path("data", "washingtondc.csv"),
    file.path("data", "liestal.csv"),
    file.path("data", "kyoto.csv"),
    file.path("data", "vancouver.csv"),
    file.path("data", "nyc.csv")
)

load_cherry_data <- function(file_paths = CHERRY_BLOOM_FILES) {
    missing_files <- file_paths[!file.exists(file_paths)]
    if (length(missing_files) > 0) {
        stop(
            sprintf(
                "Missing cherry bloom data file(s): %s",
                paste(missing_files, collapse = ", ")
            )
        )
    }

    purrr::map(file_paths, \(path) utils::read.csv(path, stringsAsFactors = FALSE)) |>
        dplyr::bind_rows()
}

if (!file.exists(CLIMATE_FILE)) {
    stop(sprintf("Climate data not found: %s. Run 1_gettingStarted.R first.", CLIMATE_FILE))
}

cherry <- load_cherry_data() |>
    mutate(bloom_date = as.Date(bloom_date))

climate <- utils::read.csv(CLIMATE_FILE, stringsAsFactors = FALSE) |>
    mutate(date = as.Date(date))

plot_start_year <- 2000

# ---- Summary tables ----
bloom_summary <- cherry |>
    group_by(location) |>
    summarize(
        n_years = n(),
        min_year = min(year, na.rm = TRUE),
        max_year = max(year, na.rm = TRUE),
        missing_bloom_doy = sum(is.na(bloom_doy)),
        min_bloom_doy = min(bloom_doy, na.rm = TRUE),
        max_bloom_doy = max(bloom_doy, na.rm = TRUE),
        .groups = "drop"
    )

climate_summary <- climate |>
    group_by(location) |>
    summarize(
        n_days = n(),
        min_date = min(date, na.rm = TRUE),
        max_date = max(date, na.rm = TRUE),
        pct_missing_avg = mean(is.na(avg_temp)) * 100,
        pct_missing_min = mean(is.na(min_temp)) * 100,
        pct_missing_max = mean(is.na(max_temp)) * 100,
        pct_missing_prcp = mean(is.na(precipitation)) * 100,
        pct_missing_tsun = mean(is.na(sunshine_duration)) * 100,
        .groups = "drop"
    )

message("Bloom summary:")
print(bloom_summary)
message("Climate summary:")
print(climate_summary)

# ---- Plots ----
ggplot(cherry, aes(x = year, y = bloom_doy)) +
    geom_point(alpha = 0.6) +
    geom_smooth(se = FALSE, method = "loess", span = 0.4) +
    facet_wrap(vars(location), scales = "free_y") +
    labs(
        title = "Peak bloom timing by location",
        x = "Year",
        y = "Peak bloom (days since Jan 1)"
    ) +
    theme_minimal()

climate_yearly <- climate |>
    mutate(year = year(date)) |>
    group_by(location, year) |>
    summarize(
        mean_avg_temp = mean(avg_temp, na.rm = TRUE),
        mean_max_temp = mean(max_temp, na.rm = TRUE),
        mean_min_temp = mean(min_temp, na.rm = TRUE),
        total_prcp = sum(precipitation, na.rm = TRUE),
        .groups = "drop"
    )

bloom_vs_climate <- cherry |>
    select(location, year, bloom_doy) |>
    left_join(climate_yearly, by = c("location", "year"))

ggplot(bloom_vs_climate, aes(x = mean_avg_temp, y = bloom_doy)) +
    geom_point(alpha = 0.6) +
    geom_smooth(se = FALSE, method = "loess", span = 0.6) +
    facet_wrap(vars(location), scales = "free") +
    labs(
        title = "Bloom timing vs. mean annual temperature",
        x = "Mean annual avg temp (C)",
        y = "Peak bloom (days since Jan 1)"
    ) +
    theme_minimal()

ggplot(climate_yearly, aes(x = year, y = mean_avg_temp)) +
    geom_line(alpha = 0.7) +
    facet_wrap(vars(location), scales = "free_y") +
    labs(
        title = "Mean annual temperature over time",
        x = "Year",
        y = "Mean avg temp (C)"
    ) +
    theme_minimal()

climate_yearly_long <- climate_yearly |>
    pivot_longer(
        cols = c(mean_max_temp, mean_min_temp),
        names_to = "metric",
        values_to = "value"
    )

ggplot(climate_yearly_long, aes(x = year, y = value, color = metric)) +
    geom_line(alpha = 0.7) +
    facet_wrap(vars(location), scales = "free_y") +
    labs(
        title = "Mean annual max/min temperature over time",
        x = "Year",
        y = "Temperature (C)",
        color = "Metric"
    ) +
    theme_minimal()

ggplot(climate_yearly, aes(x = year, y = total_prcp)) +
    geom_line(alpha = 0.7) +
    facet_wrap(vars(location), scales = "free_y") +
    labs(
        title = "Total annual precipitation over time",
        x = "Year",
        y = "Total precipitation (mm)"
    ) +
    theme_minimal()

dc_bloom_ts <- cherry |>
    filter(location == "washingtondc") |>
    mutate(
        bloom_doy_filled = if_else(
            !is.na(bloom_doy),
            bloom_doy,
            as.integer(yday(bloom_date))
        )
    ) |>
    filter(!is.na(year), !is.na(bloom_doy_filled)) |>
    arrange(year) |>
    filter(year >= plot_start_year)

reference_year <- 2000

ggplot(dc_bloom_ts, aes(x = year, y = bloom_doy_filled)) +
    geom_line(alpha = 0.7) +
    geom_point(size = 2, alpha = 0.8) +
    scale_y_continuous(
        name = "Peak bloom date (month-day)",
        breaks = seq(60, 140, by = 10),
        labels = function(x) {
            format(as.Date(x - 1, origin = sprintf("%d-01-01", reference_year)), "%b %d")
        }
    ) +
    labs(
        title = "Washington, DC: peak bloom date by year",
        x = "Year"
    ) +
    theme_minimal()

dc_climate_yearly <- climate_yearly |>
    filter(location == "washingtondc") |>
    select(year, mean_max_temp, mean_min_temp) |>
    filter(!is.na(year), year >= plot_start_year)

dc_overlay <- dc_climate_yearly |>
    left_join(
        dc_bloom_ts |> select(year, bloom_doy_filled),
        by = "year"
    )

temp_range <- range(
    c(dc_overlay$mean_max_temp, dc_overlay$mean_min_temp),
    na.rm = TRUE
)

dc_overlay <- dc_overlay |>
    mutate(
        bloom_event = if_else(!is.na(bloom_doy_filled), 1, 0),
        bloom_scaled = temp_range[1] + bloom_event * diff(temp_range)
    )

ggplot(dc_overlay, aes(x = year)) +
    geom_line(aes(y = mean_max_temp, color = "mean_max_temp"), alpha = 0.7) +
    geom_line(aes(y = mean_min_temp, color = "mean_min_temp"), alpha = 0.7) +
    geom_point(aes(y = bloom_scaled), color = "#2c3e50", size = 1.8) +
    scale_y_continuous(
        name = "Mean annual temperature (C)",
        sec.axis = sec_axis(
            ~ (. - temp_range[1]) / diff(temp_range),
            name = "Bloom event (0/1)",
            breaks = c(0, 1)
        )
    ) +
    scale_color_manual(
        values = c("mean_max_temp" = "#c0392b", "mean_min_temp" = "#2980b9"),
        labels = c("mean_max_temp" = "Mean max temp", "mean_min_temp" = "Mean min temp")
    ) +
    labs(
        title = "Washington, DC: temps with bloom time series overlay",
        x = "Year",
        color = "Metric"
    ) +
    theme_minimal()
