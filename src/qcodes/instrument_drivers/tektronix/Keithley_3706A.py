"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

from qcodes.instrument_drivers.Keithley.Keithley_3706A import (
    Keithley3706AInvalidValue,
    Keithley3706AUnknownOrEmptySlot,
)

UnknownOrEmptySlot = Keithley3706AUnknownOrEmptySlot


InvalidValue = Keithley3706AInvalidValue
