SPECIES_THRESHOLDS = {
    "Prunus x yedoensis": {"chill_temp_c": 5.0, "forcing_base_c": 5.0},
    "Prunus avium": {"chill_temp_c": 4.3, "forcing_base_c": 4.0},
    "Prunus x jamasakura": {"chill_temp_c": 5.0, "forcing_base_c": 5.0},
    "Unknown": {"chill_temp_c": 5.0, "forcing_base_c": 5.0},
}

DEFAULT_SPECIES = "Unknown"
DEFAULT_CHILL_TEMP_C = SPECIES_THRESHOLDS[DEFAULT_SPECIES]["chill_temp_c"]
DEFAULT_FORCING_BASE_C = SPECIES_THRESHOLDS[DEFAULT_SPECIES]["forcing_base_c"]


def get_species_thresholds(species_name):
    return SPECIES_THRESHOLDS.get(species_name, SPECIES_THRESHOLDS[DEFAULT_SPECIES])
