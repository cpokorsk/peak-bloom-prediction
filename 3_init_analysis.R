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
    file.path("data","USW00013743.csv"),
    file.path("data","CA001108395.csv"),
    file.path("data","USW00014732.csv"),
    file.path("data","SZ000001940.csv"),
    file.path("data","JA000047759.csv")
)

# Load the bloom history data
bloom_history <- CHERRY_BLOOM_FILES %>%
    lapply(read.csv) %>%
    bind_rows()

# Load the climate data
climate_data <- CLIMATE_DATA_FILES %>%
    lapply(read.csv) %>%
    bind_rows()

NOAA_DAILY_DATATYPES <- c("TMAX", "TMIN", "TAVG", "PRCP", "TSUN")
# Filter the climate data to include only the relevant datatypes
climate_data <- climate_data %>%
    filter(DATATYPE %in% NOAA_DAILY_DATATYPES)
    
