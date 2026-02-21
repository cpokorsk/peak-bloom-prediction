# Load the Data
library(dplyr)
library(readr)
library(lubridate)

load(file.path("data", "bloom_history.RData"))
load(file.path("data", "climate_data.RData"))
load(file.path("data", "climate_ts_stationary.RData"))
load(file.path("data", "growth_model_inputs.RData"))

# Starting the Model Fitting
MODEL_RESULTS_FILE <- file.path("data", "growth_model_results.RData")
set.seed(42)

required_model_cols <- c(
    "location",
    "year",
    "bloom_doy",
    "mean_tmax_adj_prebloom",
    "mean_tmin_adj_prebloom",
    "total_prcp_prebloom",
    "bloom_alt_m"
)

missing_model_cols <- setdiff(required_model_cols, names(model_inputs))
if (length(missing_model_cols) > 0) {
    stop(
        sprintf(
            "Missing required model input columns: %s",
            paste(missing_model_cols, collapse = ", ")
        )
    )
}

model_data <- model_inputs %>%
    select(all_of(required_model_cols)) %>%
    mutate(
        year = as.integer(year),
        bloom_doy = as.numeric(bloom_doy),
        bloom_alt_m = as.numeric(bloom_alt_m)
    ) %>%
    group_by(location, year) %>%
    summarize(
        bloom_doy = min(bloom_doy, na.rm = TRUE),
        mean_tmax_adj_prebloom = mean(mean_tmax_adj_prebloom, na.rm = TRUE),
        mean_tmin_adj_prebloom = mean(mean_tmin_adj_prebloom, na.rm = TRUE),
        total_prcp_prebloom = mean(total_prcp_prebloom, na.rm = TRUE),
        bloom_alt_m = mean(bloom_alt_m, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    filter(if_all(-c(location, year), ~ !is.na(.x)))

if (nrow(model_data) < 12) {
    stop("Not enough non-missing rows in model_data to create train/validation/test splits.")
}

assign_split <- function(years, train_frac = 0.70, val_frac = 0.15) {
    n <- length(years)
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

model_data <- model_data %>%
    arrange(location, year) %>%
    group_by(location) %>%
    mutate(split = assign_split(year)) %>%
    ungroup()

train_data <- model_data %>% filter(split == "train")
validation_data <- model_data %>% filter(split == "validation")
test_data <- model_data %>% filter(split == "test")

growth_model <- lm(
    bloom_doy ~ mean_tmin_adj_prebloom + mean_tmax_adj_prebloom + total_prcp_prebloom + bloom_alt_m,
    data = train_data
)

score_predictions <- function(df, model_obj) {
    preds <- predict(model_obj, newdata = df)
    out <- df %>%
        mutate(
            pred_bloom_doy = as.numeric(preds),
            abs_error = abs(.data$pred_bloom_doy - .data$bloom_doy),
            sq_error = (.data$pred_bloom_doy - .data$bloom_doy)^2,
            pred_bloom_date = as.Date(round(.data$pred_bloom_doy) - 1, origin = paste0(.data$year, "-01-01")),
            actual_bloom_date = as.Date(round(.data$bloom_doy) - 1, origin = paste0(.data$year, "-01-01"))
        )

    metrics <- out %>%
        summarize(
            n = n(),
            mae = mean(.data$abs_error, na.rm = TRUE),
            rmse = sqrt(mean(.data$sq_error, na.rm = TRUE))
        )

    list(predictions = out, metrics = metrics)
}

validation_scored <- score_predictions(validation_data, growth_model)
test_scored <- score_predictions(test_data, growth_model)

validation_metrics <- validation_scored$metrics %>% mutate(split = "validation")
test_metrics <- test_scored$metrics %>% mutate(split = "test")
model_metrics <- bind_rows(validation_metrics, test_metrics)

model_predictions <- bind_rows(
    validation_scored$predictions,
    test_scored$predictions
)

# ---- 2026 prediction by location (90% confidence bounds) ----
future_2026_inputs <- model_data %>%
    arrange(location, desc(year)) %>%
    group_by(location) %>%
    slice(1) %>%
    ungroup() %>%
    mutate(
        source_year = year,
        year = 2026L,
        split = "forecast_2026"
    )

pred_2026_matrix <- predict(
    growth_model,
    newdata = future_2026_inputs,
    interval = "confidence",
    level = 0.90
)

predictions_2026 <- future_2026_inputs %>%
    mutate(
        pred_bloom_doy = as.numeric(pred_2026_matrix[, "fit"]),
        conf_low_doy_90 = as.numeric(pred_2026_matrix[, "lwr"]),
        conf_high_doy_90 = as.numeric(pred_2026_matrix[, "upr"]),
        conf_pm_days_90 = (conf_high_doy_90 - conf_low_doy_90) / 2,
        conf_minus_days_90 = pred_bloom_doy - conf_low_doy_90,
        conf_plus_days_90 = conf_high_doy_90 - pred_bloom_doy,
        pred_bloom_date = as.Date(round(pred_bloom_doy) - 1, origin = "2026-01-01"),
        conf_low_date_90 = as.Date(round(conf_low_doy_90) - 1, origin = "2026-01-01"),
        conf_high_date_90 = as.Date(round(conf_high_doy_90) - 1, origin = "2026-01-01")
    ) %>%
    select(
        location,
        year,
        source_year,
        pred_bloom_doy,
        conf_low_doy_90,
        conf_high_doy_90,
        conf_pm_days_90,
        conf_minus_days_90,
        conf_plus_days_90,
        pred_bloom_date,
        conf_low_date_90,
        conf_high_date_90
    )

save(
    model_data,
    train_data,
    validation_data,
    test_data,
    growth_model,
    model_metrics,
    model_predictions,
    predictions_2026,
    file = MODEL_RESULTS_FILE
)

message(sprintf("Saved growth model results to %s", MODEL_RESULTS_FILE))
print(model_metrics)
print(predictions_2026)
