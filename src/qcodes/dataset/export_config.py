from __future__ import annotations

import enum
import logging
from pathlib import Path

import qcodes

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
        qcodes.config[DATASET_CONFIG_SECTION][EXPORT_TYPE] = None

    elif hasattr(DataExportType, export_type.upper()):
        qcodes.config[DATASET_CONFIG_SECTION][EXPORT_TYPE] = export_type.upper()

    else:
        _log.warning(
            "Could not set export type to '%s' because it is not supported.",
            export_type,
        )


def set_data_export_path(export_path: str | Path) -> None:
    """
    Set path to export data to at the end of a measurement.
    The directory is automatically created if it doesn't already
    exist.

    Args:
        export_path: An existing file path on disk

    """
    if isinstance(export_path, str):
        export_path = Path(export_path)
    export_path.mkdir(exist_ok=True)
    qcodes.config[DATASET_CONFIG_SECTION][EXPORT_AUTOMATIC] = str(export_path)


def get_data_export_type(
    export_type: DataExportType | str | None = None,
) -> DataExportType | None:
    """Get the file type for exporting data to disk at the end of
    a measurement from config

    Args:
        export_type: Export type string format to convert to DataExportType.

    Returns:
        Data export type
    """
    # If export_type is None, get value from config
    export_type = export_type or qcodes.config[DATASET_CONFIG_SECTION][EXPORT_TYPE]

    if isinstance(export_type, DataExportType):
        return export_type
    elif export_type:
        if hasattr(DataExportType, export_type.upper()):
            return getattr(DataExportType, export_type.upper())
    return None


def get_data_export_automatic() -> bool:
    """Should the data be exported automatically?"""
    export_automatic = qcodes.config[DATASET_CONFIG_SECTION][EXPORT_AUTOMATIC]
    return export_automatic


def _expand_export_path(export_path: str) -> str:
    db_location = Path(qcodes.config["core"]["db_location"]).expanduser().absolute()
    expanded_export_folder = db_location.parent / "_".join(
        (db_location.stem, db_location.suffix.replace(".", ""))
    )
    return export_path.replace("{db_location}", str(expanded_export_folder))


def get_data_export_path() -> Path:
    """Get the path to export data to at the end of a measurement from config

    Returns:
        Path
    """
    return (
        Path(_expand_export_path(qcodes.config[DATASET_CONFIG_SECTION][EXPORT_PATH]))
        .expanduser()
        .absolute()
    )


def set_data_export_prefix(export_prefix: str) -> None:
    """Set the data export file name prefix to export data to at the end of
    a measurement

    Args:
        export_prefix: Prefix, e.g. "qcodes_"
    """
    qcodes.config[DATASET_CONFIG_SECTION][EXPORT_PREFIX] = export_prefix


def get_data_export_prefix() -> str:
    """Get the data export file name prefix to export data to at the end of
    a measurement from config

    Returns:
        Prefix, e.g. "qcodes_"
    """
    return qcodes.config[DATASET_CONFIG_SECTION][EXPORT_PREFIX]


def get_data_export_name_elements() -> list[str]:
    """Get the elements to include in the export name."""
    return qcodes.config[DATASET_CONFIG_SECTION][EXPORT_NAME_ELEMENTS]
