"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

import warnings

from typing_extensions import deprecated

from qcodes.instrument_drivers.Keithley.Keithley_s46 import (
    KeithleyS46,
    KeithleyS46LockAcquisitionError,
    KeithleyS46RelayLock,
)
from qcodes.utils import QCoDeSDeprecationWarning

warnings.warn(
    "The `qcodes.instrument_drivers.tektronix.Keithley_s46` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.Keithley` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)

LockAcquisitionError = KeithleyS46LockAcquisitionError
RelayLock = KeithleyS46RelayLock


@deprecated(
    "S46 is deprecated. Please use qcodes.instrument_drivers.Keithley.KeithleyS46 instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class S46(KeithleyS46):
    """
    Alias left for backwards compatibility. Will eventually be deprecated and removed.
    """
