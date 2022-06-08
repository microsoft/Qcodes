import warnings

from qcodes.extensions.installation import register_station_schema_with_vscode

warnings.warn(
    "qcodes.utils.installation module is deprecated. "
    "Please update to import from qcodes.extensions"
)

__all__ = ["register_station_schema_with_vscode"]
