import enum

from os.path import normpath, expanduser, exists
from typing import Union

from qcodes import config


DATASET_CONFIG_SECTION = "dataset"
EXPORT_TYPE = "export_type"
EXPORT_PATH = "export_path"
EXPORT_PREFIX = "export_prefix"


class DataExportType(enum.Enum):
    """File extensions for supported data types to export data"""
    NETCDF = "nc"
    CSV = "csv"


def set_data_export_type(export_type: str) -> None:
    """Set data export type

    Args:
        export_type (str): Export type to use
        Currently supported values: netcdf, csv.
    """
    if hasattr(DataExportType, export_type.upper()):
        config[DATASET_CONFIG_SECTION][EXPORT_TYPE] = export_type.upper()


def set_data_export_path(export_path: str) -> None:
    """Set path to export data to at the end of a measurement

    Args:
        export_path (str): An existing file path on disk

    Raises:
        ValueError: If the path does not exist, this raises an error
    """
    if not exists(export_path):
        raise ValueError(f"Cannot set export path to '{export_path}' \
        because it does not exist.")
    config[DATASET_CONFIG_SECTION][EXPORT_PATH] = export_path


def get_data_export_type(
    export_type: Union[str, None] = None) -> Union[DataExportType, None]:
    """Get the file type for exporting data to disk at the end of
    a measurement from config

    Args:
        export_type (Union[str, None], optional): Export type string format
        to convert to DataExportType. Defaults to None.

    Raises:
        ValueError: If the type is not supported, this raises an error

    Returns:
        Union[DataExportType, None]: Data export type
    """
    if isinstance(export_type, DataExportType):
        return export_type

    # If export_type is None, get value from config
    export_type = export_type or config[DATASET_CONFIG_SECTION][EXPORT_TYPE]

    if export_type:
        if hasattr(DataExportType, export_type.upper()):
            return getattr(DataExportType, export_type.upper())


def get_data_export_path() -> str:
    """Get the path to export data to at the end of a measurement from config

    Returns:
        str: Path
    """
    return normpath(expanduser(config[DATASET_CONFIG_SECTION][EXPORT_PATH]))


def set_data_export_prefix(export_prefix: str) -> None:
    """Set the data export file name prefix to export data to at the end of 
    a measurement

    Args:
        export_prefix (str): Prefix, e.g. "qcodes_"
    """
    config[DATASET_CONFIG_SECTION][EXPORT_PREFIX] = export_prefix


def get_data_export_prefix() -> str:
    """Get the data export file name prefix to export data to at the end of 
    a measurement from config

    Returns:
        str: Prefix, e.g. "qcodes_"
    """
    return config[DATASET_CONFIG_SECTION][EXPORT_PREFIX]
