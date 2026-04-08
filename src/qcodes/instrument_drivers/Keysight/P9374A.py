from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from .Keysight_P9374A import KeysightP9374A


@deprecated(
    "P9374A is deprecated. Please use qcodes.instrument_drivers.Keysight.KeysightP9374A instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class P9374A(KeysightP9374A):
    """
    Alias for backwards compatibility.
    """
