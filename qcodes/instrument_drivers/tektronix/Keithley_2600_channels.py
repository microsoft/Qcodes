"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""
import logging

from qcodes.instrument_drivers.Keithley._Keithley_2600 import (
    Keithley2600,
    Keithley2600Channel,
    LuaSweepParameter,
    MeasurementStatus,
    StrEnum,
    TimeAxis,
    TimeTrace,
    _MeasurementCurrentParameter,
    _MeasurementVoltageParameter,
    _ParameterWithStatus,
)

log = logging.getLogger(__name__)


KeithleyChannel = Keithley2600Channel


class Keithley_2600(Keithley2600):
    """
    Alias left for backwards compatibility. Will eventually be deprecated and removed.
    """

    pass
