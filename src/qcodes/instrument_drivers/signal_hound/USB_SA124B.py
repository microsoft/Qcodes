"""
Module left for backwards compatibility. Will eventually be deprecated and removed.

"""

from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from .SignalHound_USB_SA124B import (
    Constants,
    ExternalRefParameter,
    FrequencySweep,
    ScaleParameter,
    SignalHoundUSBSA124B,
    SweepTraceParameter,
    TraceParameter,
    saStatus,
)


@deprecated(
    "SignalHound_USB_SA124B is deprecated. Please use qcodes.instrument_drivers.signal_hound.SignalHoundUSBSA124B instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class SignalHound_USB_SA124B(SignalHoundUSBSA124B):
    pass
