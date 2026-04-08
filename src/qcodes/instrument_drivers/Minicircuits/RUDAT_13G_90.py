from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from ._minicircuits_rudat_13g_90 import (
    MiniCircuitsRudat13G90Base,
    MiniCircuitsRudat13G90Usb,
)
from .USBHIDMixin import MiniCircuitsHIDMixin


@deprecated(
    "RUDAT_13G_90 is deprecated. Please use qcodes.instrument_drivers.Minicircuits.MiniCircuitsRudat13G90Base instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class RUDAT_13G_90(MiniCircuitsRudat13G90Base):
    """Alias for backwards compatibility."""

    pass


@deprecated(
    "RUDAT_13G_90_USB is deprecated. Please use qcodes.instrument_drivers.Minicircuits.MiniCircuitsRudat13G90Usb instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class RUDAT_13G_90_USB(MiniCircuitsRudat13G90Usb):
    """Alias for backwards compatibility."""

    pass
