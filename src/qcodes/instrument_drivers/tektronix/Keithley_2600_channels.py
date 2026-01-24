"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

import logging
import sys
from enum import Enum, StrEnum

from typing_extensions import deprecated

from qcodes.instrument_drivers.Keithley._Keithley_2600 import (
    Keithley2600,
    Keithley2600Channel,
    LuaSweepParameter,
    MeasurementStatus,
    TimeAxis,
    TimeTrace,
    _MeasurementCurrentParameter,
    _MeasurementVoltageParameter,
    _ParameterWithStatus,
)
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

log = logging.getLogger(__name__)


KeithleyChannel = Keithley2600Channel


@deprecated(
    "Keithley_2600 is deprecated. Please use qcodes.instrument_drivers.Keithley.Keithley2600 instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class Keithley_2600(Keithley2600):
    """
    Alias left for backwards compatibility. Will eventually be deprecated and removed.
    """

    pass
