import warnings

from qcodes.extensions.installation import register_station_schema_with_vscode
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

warnings.warn(
    "The `qcodes.utils.installation` module is deprecated. "
    "Please update to import from qcodes.extensions",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
