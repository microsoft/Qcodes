"""
Module left for backwards compatibility. Will eventually be deprecated and removed.

"""

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


class SignalHound_USB_SA124B(SignalHoundUSBSA124B):
    pass
