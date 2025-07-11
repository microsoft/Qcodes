"""
Module left for backwards compatibility.
Do not use in new code
Will be deprecated and eventually removed.
"""

import warnings

from qcodes.parameters import Parameter
from qcodes.utils import QCoDeSDeprecationWarning

from .instrument import Instrument, InstrumentProtocol, find_or_create_instrument
from .instrument_base import InstrumentBase
from .instrument_meta import InstrumentMeta

warnings.warn(
    "The `qcodes.instrument.base` module is deprecated. "
    "Please consult the api documentation at https://microsoft.github.io/Qcodes/api/index.html for alternatives.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
