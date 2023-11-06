"""
The extensions module contains smaller modules that extend the functionality of QCoDeS.
These modules may import from all of QCoDeS but do not themselves get imported into QCoDeS.
"""
from ._driver_test_case import DriverTestCase
from ._log_export_info import log_dataset_export_info
from .installation import register_station_schema_with_vscode

__all__ = [
    "register_station_schema_with_vscode",
    "log_dataset_export_info",
    "DriverTestCase",
]
