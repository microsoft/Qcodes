from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from .Keysight_N5245A import KeysightN5245A


@deprecated(
    "N5245A is deprecated. Please use qcodes.instrument_drivers.Keysight.KeysightN5245A instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class N5245A(KeysightN5245A):
    """
    Alias for backwards compatibility
    """
