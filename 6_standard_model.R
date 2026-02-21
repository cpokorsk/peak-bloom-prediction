# Load the Data
library(dplyr)
library(readr)
library(lubridate)
library(glmnet)

load(file.path("data", "bloom_history.RData"))
load(file.path("data", "climate_data.RData"))
load(file.path("data", "climate_ts_stationary.RData"))
load(file.path("data", "growth_model_inputs.RData"))

# Starting the Model Fitting
MODEL_RESULTS_FILE <- file.path("data", "growth_model_results.RData")
MODEL_OUTPUT_DIR <- file.path("data", "model_outputs")
PRED_2026_LINEAR_CSV <- file.path(MODEL_OUTPUT_DIR, "predictions_2026_linear.csv")
PRED_2026_LASSO_CSV <- file.path(MODEL_OUTPUT_DIR, "predictions_2026_lasso.csv")
PRED_2026_COMPARISON_CSV <- file.path(MODEL_OUTPUT_DIR, "predictions_2026_comparison.csv")
TRAIN_LOCATIONS <- c("kyoto", "washingtondc", "liestal")
LASSO_CI_LEVEL <- 0.90
LASSO_BOOTSTRAP_REPS <- 500L
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

assign_holdout_split <- function(years) {
    n <- length(years)
    if (n <= 1L) {
        return(rep("validation", n))
    }

    n_validation <- max(1L, floor(n / 2))
    n_test <- n - n_validation

    if (n_test < 1L) {
        n_validation <- n - 1L
        n_test <- 1L
    }

    c(rep("validation", n_validation), rep("test", n_test))
}

model_data <- model_data %>%
    mutate(
        data_role = if_else(location %in% TRAIN_LOCATIONS, "train_pool", "holdout_pool"),
        split = NA_character_
    )

model_data <- model_data %>%
    arrange(location, year) %>%
    group_by(location) %>%
    mutate(
        split = if (first(data_role) == "train_pool") {
            rep("train", n())
        } else {
            assign_holdout_split(year)
        }
    ) %>%
    ungroup()

missing_train_locations <- setdiff(TRAIN_LOCATIONS, unique(model_data$location))
if (length(missing_train_locations) > 0) {
    stop(
        sprintf(
            "Training location(s) not present in model_data: %s",
            paste(missing_train_locations, collapse = ", ")
        )
    )
}

train_data <- model_data %>% filter(split == "train")
validation_data <- model_data %>% filter(split == "validation")
test_data <- model_data %>% filter(split == "test")

if (nrow(validation_data) == 0 || nrow(test_data) == 0) {
    stop("Holdout locations do not provide enough rows for both validation and test sets.")
}

growth_model <- lm(
    bloom_doy ~ mean_tmin_adj_prebloom + mean_tmax_adj_prebloom + total_prcp_prebloom + bloom_alt_m,
    data = train_data
)

lasso_feature_formula <- ~ mean_tmin_adj_prebloom + mean_tmax_adj_prebloom + total_prcp_prebloom + bloom_alt_m
x_train <- model.matrix(lasso_feature_formula, data = train_data)[, -1, drop = FALSE]
y_train <- train_data$bloom_doy

lasso_cv_model <- cv.glmnet(
    x = x_train,
    y = y_train,
    alpha = 1,
    family = "gaussian",
    nfolds = min(10, nrow(train_data))
)

lasso_lambda <- lasso_cv_model$lambda.min

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

score_predictions_lasso <- function(df, cv_model_obj) {
    x_df <- model.matrix(lasso_feature_formula, data = df)[, -1, drop = FALSE]
    preds <- as.numeric(predict(cv_model_obj, newx = x_df, s = "lambda.min"))

    out <- df %>%
        mutate(
            pred_bloom_doy = preds,
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

validation_scored_lasso <- score_predictions_lasso(validation_data, lasso_cv_model)
test_scored_lasso <- score_predictions_lasso(test_data, lasso_cv_model)

validation_metrics <- validation_scored$metrics %>% mutate(split = "validation", model = "linear")
test_metrics <- test_scored$metrics %>% mutate(split = "test", model = "linear")

validation_metrics_lasso <- validation_scored_lasso$metrics %>% mutate(split = "validation", model = "lasso")
test_metrics_lasso <- test_scored_lasso$metrics %>% mutate(split = "test", model = "lasso")

model_metrics <- bind_rows(
    validation_metrics,
    test_metrics,
    validation_metrics_lasso,
    test_metrics_lasso
)

model_predictions <- bind_rows(
    validation_scored$predictions %>% mutate(split = "validation", model = "linear"),
    test_scored$predictions %>% mutate(split = "test", model = "linear"),
    validation_scored_lasso$predictions %>% mutate(split = "validation", model = "lasso"),
    test_scored_lasso$predictions %>% mutate(split = "test", model = "lasso")
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
        pred_bloom_date,
        conf_low_date_90,
        conf_high_date_90,
        pred_bloom_doy,
        conf_pm_days_90,
        conf_minus_days_90,
        conf_plus_days_90
    )

x_future_2026 <- model.matrix(lasso_feature_formula, data = future_2026_inputs)[, -1, drop = FALSE]
pred_lasso_2026 <- as.numeric(predict(lasso_cv_model, newx = x_future_2026, s = "lambda.min"))

alpha_tail <- (1 - LASSO_CI_LEVEL) / 2
lasso_boot_pred_matrix <- replicate(LASSO_BOOTSTRAP_REPS, {
    idx <- sample(seq_len(nrow(train_data)), size = nrow(train_data), replace = TRUE)
    x_boot <- x_train[idx, , drop = FALSE]
    y_boot <- y_train[idx]

    lasso_boot_model <- glmnet(
        x = x_boot,
        y = y_boot,
        alpha = 1,
        family = "gaussian",
        lambda = lasso_lambda
    )

    as.numeric(predict(lasso_boot_model, newx = x_future_2026, s = lasso_lambda))
})

lasso_conf_low <- apply(lasso_boot_pred_matrix, 1, quantile, probs = alpha_tail, na.rm = TRUE)
lasso_conf_high <- apply(lasso_boot_pred_matrix, 1, quantile, probs = 1 - alpha_tail, na.rm = TRUE)

predictions_2026_lasso <- future_2026_inputs %>%
    transmute(
        location,
        year,
        source_year,
        pred_bloom_doy_lasso = pred_lasso_2026,
        conf_low_doy_90_lasso = as.numeric(lasso_conf_low),
        conf_high_doy_90_lasso = as.numeric(lasso_conf_high),
        conf_pm_days_90_lasso = (conf_high_doy_90_lasso - conf_low_doy_90_lasso) / 2,
        conf_minus_days_90_lasso = pred_bloom_doy_lasso - conf_low_doy_90_lasso,
        conf_plus_days_90_lasso = conf_high_doy_90_lasso - pred_bloom_doy_lasso,
        pred_bloom_date_lasso = as.Date(round(pred_bloom_doy_lasso) - 1, origin = "2026-01-01"),
        conf_low_date_90_lasso = as.Date(round(conf_low_doy_90_lasso) - 1, origin = "2026-01-01"),
        conf_high_date_90_lasso = as.Date(round(conf_high_doy_90_lasso) - 1, origin = "2026-01-01")
    )

predictions_2026_comparison <- predictions_2026 %>%
    select(location, year, source_year, pred_bloom_doy, pred_bloom_date) %>%
    rename(
        pred_bloom_doy_linear = pred_bloom_doy,
        pred_bloom_date_linear = pred_bloom_date
    ) %>%
    left_join(predictions_2026_lasso, by = c("location", "year", "source_year")) %>%
    mutate(
        doy_diff_lasso_minus_linear = pred_bloom_doy_lasso - pred_bloom_doy_linear
    )

dir.create(MODEL_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
readr::write_csv(predictions_2026, PRED_2026_LINEAR_CSV)
readr::write_csv(predictions_2026_lasso, PRED_2026_LASSO_CSV)
readr::write_csv(predictions_2026_comparison, PRED_2026_COMPARISON_CSV)

save(
    model_data,
    train_data,
    validation_data,
    test_data,
    growth_model,
    lasso_cv_model,
    lasso_lambda,
    model_metrics,
    model_predictions,
    predictions_2026,
    predictions_2026_lasso,
    predictions_2026_comparison,
    file = MODEL_RESULTS_FILE
)

message(sprintf("Saved growth model results to %s", MODEL_RESULTS_FILE))
message(sprintf("Saved CSV outputs to %s", MODEL_OUTPUT_DIR))
print(model_metrics)
print(predictions_2026)
print(predictions_2026_comparison)
