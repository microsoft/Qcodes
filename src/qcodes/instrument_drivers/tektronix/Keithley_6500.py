"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

from qcodes.instrument_drivers.Keithley.Keithley_6500 import (
    Keithley6500,
    Keithley6500CommandSetError,
    _parse_output_bool,
    _parse_output_string,
)

CommandSetError = Keithley6500CommandSetError


class Keithley_6500(Keithley6500):
    """
    Alias left for backwards compatibility. Will eventually be deprecated and removed.
    """

    pass
