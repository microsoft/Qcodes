"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""
from qcodes.instrument_drivers.Keithley.Keithley_2400 import Keithley2400


class Keithley_2400(Keithley2400):
    """
    Backwards compatibility alias for Keithley 2400 driver
    """

    pass
