from ._minicircuits_rudat_13g_90 import (
    MiniCircuitsRudat13G90Base,
    MiniCircuitsRudat13G90Usb,
)
from .USBHIDMixin import MiniCircuitsHIDMixin

RUDAT_13G_90 = MiniCircuitsRudat13G90Base
"""Alias for backwards compatibility."""

RUDAT_13G_90_USB = MiniCircuitsRudat13G90Usb
"""Alias for backwards compatibility."""
