from qcodes.extensions.installation import register_station_schema_with_vscode

# todo enable warning once new api is in release
# warnings.warn(
#     "qcodes.utils.installation module is deprecated. "
#     "Please update to import from qcodes.extensions"
# )

__all__ = ["register_station_schema_with_vscode"]
