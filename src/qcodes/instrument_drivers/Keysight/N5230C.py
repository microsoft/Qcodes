from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from .Keysight_N5230C import KeysightN5230C


@deprecated(
    "N5230C is deprecated. Please use qcodes.instrument_drivers.Keysight.KeysightN5230C instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class N5230C(KeysightN5230C):
    """
    Alias for backwards compatibility
    """
