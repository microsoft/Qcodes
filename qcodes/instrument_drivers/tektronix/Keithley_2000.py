"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""
from qcodes.instrument_drivers.Keithley.Keithley_2000 import (
    Keithley2000,
    _parse_output_bool,
    _parse_output_string,
)

parse_output_string = _parse_output_string
parse_output_bool = _parse_output_bool


class Keithley_2000(Keithley2000):
    """
    Backwards compatibility alias for Keithley 2000 driver
    """

    pass
