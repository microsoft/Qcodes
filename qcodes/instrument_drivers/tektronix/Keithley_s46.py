"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""
from qcodes.instrument_drivers.Keithley.Keithley_s46 import (
    KeithleyS46,
    KeithleyS46LockAcquisitionError,
    KeithleyS46RelayLock,
)

LockAcquisitionError = KeithleyS46LockAcquisitionError
RelayLock = KeithleyS46RelayLock


class S46(KeithleyS46):
    """
    Alias left for backwards compatibility. Will eventually be deprecated and removed.
    """
