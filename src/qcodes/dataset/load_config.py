import qcodes

DATASET_CONFIG_SECTION = "dataset"
LOAD_FROM_FILE = "load_from_file"


def get_data_load_from_file() -> bool:
    """Get the flag to import data from sqlite db."""
    return qcodes.config[DATASET_CONFIG_SECTION][LOAD_FROM_FILE]
