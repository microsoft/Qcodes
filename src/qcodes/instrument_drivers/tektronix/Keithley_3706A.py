"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

from typing_extensions import deprecated

from qcodes.instrument_drivers.Keithley.Keithley_3706A import (
    Keithley3706A,
    Keithley3706AInvalidValue,
    Keithley3706AUnknownOrEmptySlot,
)
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

UnknownOrEmptySlot = Keithley3706AUnknownOrEmptySlot


InvalidValue = Keithley3706AInvalidValue


@deprecated(
    "Keithley_3706A is deprecated. Please use qcodes.instrument_drivers.Keithley.Keithley3706A instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class Keithley_3706A(Keithley3706A):
    """
    Alias left for backwards compatibility. Will eventually be deprecated and removed.
    """

    pass
