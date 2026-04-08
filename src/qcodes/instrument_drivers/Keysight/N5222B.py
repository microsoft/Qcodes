from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from .Keysight_N5222B import KeysightN5222B


@deprecated(
    "N5222B is deprecated. Please use qcodes.instrument_drivers.Keysight.KeysightN5222B instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class N5222B(KeysightN5222B):
    """
    Alias for backwards compatibility

    """
