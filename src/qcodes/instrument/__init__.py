import atexit
import importlib
import warnings
from typing import TYPE_CHECKING, Any

from qcodes.utils import QCoDeSDeprecationWarning

from .channel import ChannelList, ChannelTuple, InstrumentChannel, InstrumentModule
from .instrument import Instrument, find_or_create_instrument
from .instrument_base import InstrumentBase, InstrumentBaseKWArgs
from .ip import IPInstrument
from .visa import VisaInstrument, VisaInstrumentKWArgs

# ensure that all instruments are closed when the interpreter is shut down.
# this is registered here rather than in ``qcodes.__init__`` so that importing
# the top level ``qcodes`` package does not eagerly import ``qcodes.instrument``.
atexit.register(Instrument.close_all)

__all__ = [
    "ChannelList",
    "ChannelTuple",
    "IPInstrument",
    "Instrument",
    "InstrumentBase",
    "InstrumentBaseKWArgs",
    "InstrumentChannel",
    "InstrumentModule",
    "VisaInstrument",
    "VisaInstrumentKWArgs",
    "find_or_create_instrument",
]

# The following parameter classes used to be re-exported from
# ``qcodes.instrument`` for backwards compatibility. They now live in
# ``qcodes.parameters`` and importing them from here is deprecated. They are
# provided lazily via a module level ``__getattr__`` that emits a deprecation
# warning; import them from ``qcodes.parameters`` instead.
_DEPRECATED_PARAMETER_NAMES = frozenset(
    {
        "ArrayParameter",
        "CombinedParameter",
        "DelegateParameter",
        "Function",
        "ManualParameter",
        "MultiParameter",
        "Parameter",
        "ParameterWithSetpoints",
        "ScaledParameter",
        "SweepFixedValues",
        "SweepValues",
        "combine",
    }
)


# The lazy ``__getattr__`` is intentionally hidden from static type checkers via
# ``if not TYPE_CHECKING`` so that importing these deprecated names from
# ``qcodes.instrument`` is reported as an unknown attribute; import them from
# ``qcodes.parameters`` instead.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name in _DEPRECATED_PARAMETER_NAMES:
            warnings.warn(
                f"Importing {name!r} from {__name__!r} is deprecated. "
                f"Import it from 'qcodes.parameters' instead.",
                QCoDeSDeprecationWarning,
                stacklevel=2,
            )
            return getattr(importlib.import_module("qcodes.parameters"), name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
