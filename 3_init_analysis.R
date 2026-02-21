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

NOAA_DAILY_DATATYPES <- c("TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "SNWD")
# Filter the climate data to include only the relevant datatypes
RELEVANT_CLIMATE_COLUMNS <- c("STATION", "DATE", "NAME", "ELEVATION", NOAA_DAILY_DATATYPES)

climate_data <- climate_data %>%
    select(any_of(RELEVANT_CLIMATE_COLUMNS))

# Associate each climate station with the bloom-history location
climate_data <- climate_data %>%
    left_join(STATION_LOCATION_MAP, by = "STATION")

# Convert the DATE column to Date format
climate_data$DATE <- as.Date(climate_data$DATE, format = "%Y-%m-%d")

# Calculate the bloom day of year (DOY) for each bloom history record
bloom_history$bloom_date <- as.Date(bloom_history$bloom_date, format = "%Y-%m-%d")

# Save the loaded data for use in the next steps
save(bloom_history, file = file.path("data", "bloom_history.RData"))
save(climate_data, file = file.path("data", "climate_data.RData"))


# Clean up the environment
rm(list = setdiff(ls(), c("bloom_history", "climate_data")))
gc()
