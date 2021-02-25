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


def set_data_export_type(export_type: str) -> None:
    """Set data export type

    :param export_type: Export type to use
        Currently supported values: "netcdf".
    :type export_type: str
    """
    if hasattr(DataExportType, export_type.upper()):
        config[DATASET_CONFIG_SECTION][EXPORT_TYPE] = export_type.upper()


def set_data_export_path(export_path: str) -> None:
    """Set path to export data to at the end of a measurement

    :param export_path: An existing file path on disk
    :type export_path: str
    :raises ValueError: If the path does not exist, this raises an error
    """
    if not exists(export_path):
        raise ValueError(f"Cannot set export path to '{export_path}' \
        because it does not exist.")
    config[DATASET_CONFIG_SECTION][EXPORT_PATH] = export_path


def get_data_export_type(
    export_type: Union[str, None] = None) -> Union[DataExportType, None]:
    """Get the file type for exporting data to disk at the end of
    a measurement from config

    :raises ValueError: If the type is not supported, this raises an error
    :return: Data export type or None
    :rtype: Union[DataExportType, None]
    """
    # If export_type is None, get value from config
    export_type = export_type or config[DATASET_CONFIG_SECTION][EXPORT_TYPE]

    if export_type:
        if not hasattr(DataExportType, export_type):
            raise ValueError(f"Unknown export data type: {export_type}")
        export_type = getattr(DataExportType, export_type)
    else:
        return None

    return export_type


def get_data_export_path() -> str:
    """Get the path to export data to at the end of a measurement from config

    :return: Path
    :rtype: str
    """
    return normpath(expanduser(config[DATASET_CONFIG_SECTION][EXPORT_PATH]))


def set_data_export_prefix(export_prefix: str) -> None:
    """Set the data export file name prefix to export data to at the end of 
    a measurement

    :param export_prefix: Prefix, e.g. "qcodes_"
    :type export_prefix: str
    """
    config[DATASET_CONFIG_SECTION][EXPORT_PREFIX] = export_prefix


def get_data_export_prefix() -> str:
    """Get the data export file name prefix to export data to at the end of 
    a measurement from config

    :return: Prefix, e.g. "qcodes_"
    :rtype: str
    """
    return config[DATASET_CONFIG_SECTION][EXPORT_PREFIX]
