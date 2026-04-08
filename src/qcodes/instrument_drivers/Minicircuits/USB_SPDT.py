from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from ._minicircuits_usb_spdt import (
    MiniCircuitsUsbSPDT,
    MiniCircuitsUsbSPDTSwitchChannel,
)


@deprecated(
    "SwitchChannelUSB is deprecated. Please use qcodes.instrument_drivers.Minicircuits.MiniCircuitsUsbSPDTSwitchChannel instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class SwitchChannelUSB(MiniCircuitsUsbSPDTSwitchChannel):
    """
    Alias for backwards compatibility
    """

    pass


@deprecated(
    "USB_SPDT is deprecated. Please use qcodes.instrument_drivers.Minicircuits.MiniCircuitsUsbSPDT instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class USB_SPDT(MiniCircuitsUsbSPDT):
    """
    Alias for backwards compatibility
    """

    pass
