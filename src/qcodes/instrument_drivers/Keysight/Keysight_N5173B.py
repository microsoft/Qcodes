from typing_extensions import deprecated

from qcodes.instrument_drivers.Keysight.Keysight_N5183B import KeysightN5183B
from qcodes.utils.deprecate import QCoDeSDeprecationWarning


class KeysightN5173B(KeysightN5183B):
    pass  # N5173B has the same interface as N5183B


@deprecated(
    "N5173B is deprecated. Please use qcodes.instrument_drivers.Keysight.KeysightN5173B instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class N5173B(KeysightN5173B):
    """
    Alias for backwards compatibility
    """
