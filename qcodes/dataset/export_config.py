import enum
import logging
from os.path import exists, expanduser, normpath
from typing import List, Optional, Union

from qcodes import config

_log = logging.getLogger(__name__)


DATASET_CONFIG_SECTION = "dataset"
EXPORT_AUTOMATIC = "export_automatic"
EXPORT_TYPE = "export_type"
EXPORT_PATH = "export_path"
EXPORT_PREFIX = "export_prefix"
EXPORT_NAME_ELEMENTS = "export_name_elements"


class DataExportType(enum.Enum):
    """File extensions for supported data types to export data"""
    NETCDF = "nc"
    CSV = "csv"


def set_data_export_type(export_type: str) -> None:
    """Set data export type

    Args:
        export_type: Export type to use.
            Currently supported values: netcdf, csv.
    """
    # disable file export
    if export_type is None:
        config[DATASET_CONFIG_SECTION][EXPORT_TYPE] = None

    elif hasattr(DataExportType, export_type.upper()):
        config[DATASET_CONFIG_SECTION][EXPORT_TYPE] = export_type.upper()

    else:
        _log.warning(
            "Could not set export type to '%s' because it is not supported."
            % export_type
        )


def set_data_export_path(export_path: str) -> None:
    """Set path to export data to at the end of a measurement

    Args:
        export_path: An existing file path on disk

    Raises:
        ValueError: If the path does not exist, this raises an error
    """
    if not exists(export_path):
        raise ValueError(
            f"Cannot set export path to '{export_path}' because it does not exist."
        )
    config[DATASET_CONFIG_SECTION][EXPORT_AUTOMATIC] = export_path


def get_data_export_type(
    export_type: Optional[Union[DataExportType, str]] = None) -> Optional[DataExportType]:
    """Get the file type for exporting data to disk at the end of
    a measurement from config

    Args:
        export_type: Export type string format to convert to DataExportType.

    Returns:
        Data export type
    """
    # If export_type is None, get value from config
    export_type = export_type or config[DATASET_CONFIG_SECTION][EXPORT_TYPE]

    if isinstance(export_type, DataExportType):
        return export_type
    elif export_type:
        if hasattr(DataExportType, export_type.upper()):
            return getattr(DataExportType, export_type.upper())
    return None


def get_data_export_automatic() -> bool:
    """Should the data be exported automatically?"""
    export_automatic = config[DATASET_CONFIG_SECTION][EXPORT_AUTOMATIC]
    return export_automatic


def get_data_export_path() -> str:
    """Get the path to export data to at the end of a measurement from config

    Returns:
        Path
    """
    return normpath(expanduser(config[DATASET_CONFIG_SECTION][EXPORT_PATH]))


def set_data_export_prefix(export_prefix: str) -> None:
    """Set the data export file name prefix to export data to at the end of
    a measurement

    Args:
        export_prefix: Prefix, e.g. "qcodes_"
    """
    config[DATASET_CONFIG_SECTION][EXPORT_PREFIX] = export_prefix


def get_data_export_prefix() -> str:
    """Get the data export file name prefix to export data to at the end of
    a measurement from config

    Returns:
        Prefix, e.g. "qcodes_"
    """
    return config[DATASET_CONFIG_SECTION][EXPORT_PREFIX]


def get_data_export_name_elements() -> List[str]:
    """Get the elements to include in the export name."""
    return config[DATASET_CONFIG_SECTION][EXPORT_NAME_ELEMENTS]
