NOAA_WEB_API_TOKEN <- Sys.getenv("NOAA_WEB_API_TOKEN")

# Stop early if no NOAA API token is configured in the environment.
if (NOAA_WEB_API_TOKEN == "") {
    stop("Please set the NOAA_WEB_API_TOKEN environment variable with your API token.")
}

# install.packages("httr2")
# install.packages("jsonlite")


library(httr2)
library(jsonlite)
library(purrr)
library(tibble)
library(dplyr)
library(lubridate)
library(tidyr)

NOAA_API_BASE_URL <- "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
# Requested NOAA daily metrics: max/min/avg temp, precipitation, sunshine duration.
NOAA_DAILY_DATATYPES <- c("TMAX", "TMIN", "TAVG", "PRCP", "TSUN")
# NOAA API docs: token quotas are 10,000 requests/day and 5 requests/second.
NOAA_MAX_REQUESTS_PER_RUN <- as.integer(Sys.getenv("NOAA_MAX_REQUESTS_PER_RUN", "1000"))
NOAA_RESPONSE_LIMIT <- 1000
NOAA_WINDOW_SIZE_DAYS <- as.integer(Sys.getenv("NOAA_WINDOW_SIZE_DAYS", "180"))
# Dry-run mode: when TRUE, only estimate requests and skip API calls/file writes.
NOAA_DRY_RUN <- FALSE
# Progress logging: when TRUE, prints window/page status to the console.
NOAA_PROGRESS <- TRUE

# Persisted output location for re-use in downstream scripts.
HISTORICAL_CLIMATE_DIR <- file.path("data", "historical_temps")
HISTORICAL_CLIMATE_FILE <- file.path(HISTORICAL_CLIMATE_DIR, "historic_climate_metrics.csv")

# Mapping of project locations to NOAA GHCND station identifiers.
stations <- c(
    "foggybottom"  = "GHCND:USW00093725",
    "washingtondc" = "GHCND:USW00013743",
    "vancouver"    = "GHCND:CA001108395",
    "newyorkcity"  = "GHCND:USW00014732",
    "liestal"      = "GHCND:SZ000001940",
    "kyoto"        = "GHCND:JA000047759"
)

# Bloom history files used to build the core `cherry` dataset.
CHERRY_BLOOM_FILES <- c(
    file.path("data", "washingtondc.csv"),
    file.path("data", "liestal.csv"),
    file.path("data", "kyoto.csv"),
    file.path("data", "vancouver.csv"),
    file.path("data", "nyc.csv")
)

# Load bloom observations from CSV files and bind them into one table.
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

# Auto-load `cherry` if not already present in the current environment.
if (!exists("cherry", inherits = TRUE)) {
    cherry <- load_cherry_data()
}

nested_to_tibble <- function(x) {
    # Determine all unique field names across nested NOAA result objects.
    variable_names <- purrr::map(x, names) |>
        unlist(use.names = FALSE) |>
        unique()

    names(variable_names) <- variable_names

    # Reshape the response from nested lists into a tabular tibble.
    purrr::map(variable_names, \(i) {
        purrr::map(x, \(y) {
            if (is.null(y[[i]])) {
                NA_character_
            } else {
                y[[i]]
            }
        }) |>
            unlist(use.names = FALSE)
    }) |>
        tibble::as_tibble()
}

# Mutable request budget tracker for a single script run.
new_request_tracker <- function(max_requests = NOAA_MAX_REQUESTS_PER_RUN) {
    tracker <- new.env(parent = emptyenv())
    tracker$count <- 0L
    tracker$max_requests <- as.integer(max_requests)
    tracker
}

# Increments request count and fails fast before exceeding the configured budget.
consume_request_budget <- function(tracker) {
    if (is.null(tracker)) {
        return(invisible(NULL))
    }

    if (tracker$count >= tracker$max_requests) {
        stop(
            sprintf(
                "NOAA request budget reached for this run (%d requests). Re-run later or increase NOAA_MAX_REQUESTS_PER_RUN.",
                tracker$max_requests
            )
        )
    }

    tracker$count <- tracker$count + 1L
    invisible(NULL)
}

# Fetches all pages for a single station/date window (NOAA page size max is 1000 rows).
fetch_noaa_window_pages <- function(station_id, from, to, api_key, base_url, request_tracker = NULL) {
    offset <- 1L
    page_limit <- NOAA_RESPONSE_LIMIT
    pages <- list()
    page_index <- 0L

    repeat {
        page_index <- page_index + 1L
        if (isTRUE(NOAA_PROGRESS)) {
            message(
                sprintf(
                    "NOAA request: station %s, window %s to %s (page %d)",
                    station_id,
                    as.character(from),
                    as.character(to),
                    page_index
                )
            )
        }

        page_response <- tryCatch(
            {
                consume_request_budget(request_tracker)

                request(base_url) |>
                    req_headers(token = api_key) |>
                    req_url_query(
                        datasetid = "GHCND",
                        stationid = station_id,
                        datatypeid = paste(NOAA_DAILY_DATATYPES, collapse = ","),
                        startdate = from,
                        enddate = min(lubridate::as_date(to), Sys.Date()),
                        units = "metric",
                        includemetadata = "false",
                        limit = page_limit,
                        offset = offset
                    ) |>
                    req_retry(max_tries = 10) |>
                    req_perform() |>
                    resp_body_json()
            },
            httr2_http = \(cnd) {
                rlang::warn(
                    sprintf(
                        "Failed to retrieve data for station %s in time window %s--%s",
                        station_id, from, to
                    ),
                    parent = cnd
                )
                NULL
            }
        )

        if (is.null(page_response) || is.null(page_response$results) || length(page_response$results) == 0) {
            break
        }

        pages[[length(pages) + 1]] <- page_response$results

        if (length(page_response$results) < page_limit) {
            break
        }

        offset <- offset + page_limit
    }

    pages
}

get_daily_climate_metrics <- function(station_id, start_date, end_date,
                                      api_key, base_url, window_size = NOAA_WINDOW_SIZE_DAYS,
                                      request_tracker = NULL) {
    # Split large date ranges into windows and paginate each window.
    windows <- seq(lubridate::as_date(start_date),
        lubridate::as_date(end_date) + lubridate::days(window_size + 1),
        by = sprintf("%d days", window_size)
    )

    window_starts <- windows[-length(windows)]
    window_ends <- windows[-1] - lubridate::days(1)
    window_total <- length(window_starts)

    batches <- purrr::pmap(
        list(window_starts, window_ends, seq_len(window_total)),
        \(from, to, window_index) {
            # Skip windows that begin in the future.
            if (from > Sys.Date()) {
                return(NULL)
            }
            if (isTRUE(NOAA_PROGRESS)) {
                message(
                    sprintf(
                        "Fetching station %s window %d/%d (%s to %s)",
                        station_id,
                        window_index,
                        window_total,
                        as.character(from),
                        as.character(to)
                    )
                )
            }
            fetch_noaa_window_pages(
                station_id = station_id,
                from = from,
                to = to,
                api_key = api_key,
                base_url = base_url,
                request_tracker = request_tracker
            )
        }
    )

    # Keep only successful, non-empty API payloads.
    batch_tables <- purrr::map(batches, \(x) {
        if (is.null(x) || length(x) == 0) {
            return(NULL)
        }

        all_results <- unlist(x, recursive = FALSE)
        nested_to_tibble(all_results)
    }) |>
        purrr::compact()

    if (length(batch_tables) == 0) {
        return(tibble::tibble())
    }

    # Combine all windows, then pivot NOAA datatype rows into one row per day.
    daily_metrics <- dplyr::bind_rows(batch_tables) |>
        dplyr::mutate(
            date = lubridate::as_date(.data$date),
            datatype = toupper(.data$datatype),
            value = as.numeric(.data$value)
        ) |>
        dplyr::select("date", "station", "datatype", "value") |>
        dplyr::distinct() |>
        tidyr::pivot_wider(names_from = "datatype", values_from = "value")

    # Ensure all requested fields exist even when NOAA has missing observations.
    for (datatype_name in NOAA_DAILY_DATATYPES) {
        if (!datatype_name %in% names(daily_metrics)) {
            daily_metrics[[datatype_name]] <- NA_real_
        }
    }

    daily_metrics |>
        dplyr::transmute(
            date = .data$date,
            station_id = .data$station,
            max_temp = .data$TMAX,
            min_temp = .data$TMIN,
            avg_temp = .data$TAVG,
            precipitation = .data$PRCP,
            sunshine_duration = .data$TSUN
        )
}

# Estimate minimum request count from date windows (one request per window minimum).
estimate_requests_from_windows <- function(refresh_plan, end_date = Sys.Date(), window_size = NOAA_WINDOW_SIZE_DAYS) {
    if (nrow(refresh_plan) == 0) {
        return(tibble::tibble(location = character(), fetch_start_date = as.Date(character()), estimated_requests = integer()))
    }

    refresh_plan |>
        dplyr::mutate(
            fetch_start_date = lubridate::as_date(.data$fetch_start_date),
            end_date = lubridate::as_date(end_date),
            days_to_fetch = pmax(0L, as.integer(.data$end_date - .data$fetch_start_date) + 1L),
            estimated_requests = as.integer(ceiling(.data$days_to_fetch / window_size))
        ) |>
        dplyr::select("location", "fetch_start_date", "estimated_requests")
}


# Read the cached historical climate data in downstream files.
load_historic_climate_metrics <- function(path = HISTORICAL_CLIMATE_FILE) {
    if (!file.exists(path)) {
        stop(sprintf("Cached historical climate file not found: %s", path))
    }

    utils::read.csv(path, stringsAsFactors = FALSE) |>
        dplyr::mutate(date = lubridate::as_date(date))
}


# Build a per-location refresh plan with incremental fetch starts from cached data.
build_refresh_plan <- function(cherry_data, station_lookup, existing_climate = NULL, force_refresh = FALSE) {
    base_plan <- cherry_data |>
        dplyr::group_by(.data$location) |>
        dplyr::summarize(
            first_bloom_year = min(.data$year, na.rm = TRUE),
            start_year = pmax(1921, .data$first_bloom_year),
            base_start_date = sprintf("%d-01-01", .data$start_year),
            .groups = "drop"
        ) |>
        dplyr::select(-"first_bloom_year", -"start_year") |>
        dplyr::left_join(
            tibble::tibble(
                location = names(station_lookup),
                station_id = station_lookup
            ),
            by = "location"
        )

    if (force_refresh || is.null(existing_climate) || nrow(existing_climate) == 0) {
        return(base_plan |>
            dplyr::mutate(fetch_start_date = .data$base_start_date))
    }

    cache_last_date <- existing_climate |>
        dplyr::filter(!is.na(.data$date)) |>
        dplyr::group_by(.data$location) |>
        dplyr::summarize(last_cached_date = max(.data$date, na.rm = TRUE), .groups = "drop")

    base_plan |>
        dplyr::left_join(cache_last_date, by = "location") |>
        dplyr::mutate(
            fetch_start_date = dplyr::if_else(
                is.na(.data$last_cached_date),
                .data$base_start_date,
                as.character(.data$last_cached_date + lubridate::days(1))
            )
        ) |>
        dplyr::select(-"last_cached_date")
}


# Main refresh workflow:
# - Reuse cached data by default.
# - Fetch only missing dates per location.
# - Enforce per-run request budget for predictable API usage.
force_full_refresh <- identical(tolower(Sys.getenv("NOAA_FORCE_FULL_REFRESH", "false")), "true")
noaa_dry_run <- isTRUE(NOAA_DRY_RUN)
request_tracker <- new_request_tracker()

existing_climate <- if (file.exists(HISTORICAL_CLIMATE_FILE) && !force_full_refresh) {
    load_historic_climate_metrics(HISTORICAL_CLIMATE_FILE)
} else {
    tibble::tibble()
}

refresh_plan <- build_refresh_plan(
    cherry_data = cherry,
    station_lookup = stations,
    existing_climate = existing_climate,
    force_refresh = force_full_refresh
) |>
    dplyr::mutate(fetch_start_date = lubridate::as_date(.data$fetch_start_date))

to_fetch <- refresh_plan |>
    dplyr::filter(.data$fetch_start_date <= Sys.Date())

request_estimate <- estimate_requests_from_windows(to_fetch)
estimated_total_requests <- sum(request_estimate$estimated_requests)

if (estimated_total_requests > request_tracker$max_requests) {
    warning(
        sprintf(
            "Estimated minimum NOAA requests (%d) exceed NOAA_MAX_REQUESTS_PER_RUN (%d).",
            estimated_total_requests,
            request_tracker$max_requests
        )
    )
}

if (noaa_dry_run) {
    message("NOAA dry-run mode enabled. No API calls were made and no files were written.")
    message(sprintf("Estimated minimum NOAA requests for this refresh: %d", estimated_total_requests))
    print(request_estimate)
    message(sprintf("Configured run budget: %d", request_tracker$max_requests))
} else {
    new_climate <- to_fetch |>
        dplyr::group_by(.data$location) |>
        dplyr::group_modify(\(x, gr) {
            get_daily_climate_metrics(
                station_id = x$station_id,
                start_date = x$fetch_start_date,
                end_date = Sys.Date(),
                api_key = NOAA_WEB_API_TOKEN,
                base_url = NOAA_API_BASE_URL,
                request_tracker = request_tracker
            )
        })

    historic_climate_metrics <- dplyr::bind_rows(existing_climate, new_climate) |>
        dplyr::distinct(.data$location, .data$date, .keep_all = TRUE) |>
        dplyr::arrange(.data$location, .data$date)

    # Save refreshed historical climate metrics so other scripts can read from disk.
    dir.create(HISTORICAL_CLIMATE_DIR, recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(historic_climate_metrics, HISTORICAL_CLIMATE_FILE, row.names = FALSE)
    message(sprintf("Historical climate metrics saved to: %s", HISTORICAL_CLIMATE_FILE))
    message(sprintf("NOAA requests used in this run: %d / %d", request_tracker$count, request_tracker$max_requests))
}
