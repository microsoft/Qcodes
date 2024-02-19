import logging
import time
from typing import Union, Tuple
from string import ascii_letters


from qcodes import VisaInstrument
from qcodes.utils.validators import  Numbers, Enum


from __future__ import annotations

import logging
import numbers
import time
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from functools import partial
from typing import Any, Callable, ClassVar, TypeVar, cast

import numpy as np
from pyvisa import VisaIOError

from qcodes.instrument import Instrument, InstrumentChannel, VisaInstrument
from qcodes.math_utils import FieldVector
from qcodes.parameters import Parameter
from qcodes.utils import QCoDeSDeprecationWarning
from qcodes.validators import Anything, Bool, Enum, Ints, Numbers

log = logging.getLogger(__name__)

CartesianFieldLimitFunction = Callable[[float, float, float], bool]

T = TypeVar("T")



class Cryomagnetic4GException(Exception):
    pass


class Cryomagnetics4GWarning(UserWarning):
    pass



class Cryo4GSwitchHeater(InstrumentChannel):
    class _Decorators:
        @classmethod
        def check_enabled(cls, f: Callable[..., T]) -> Callable[..., T]:
            def check_enabled_decorator(
                self: Cryo4GSwitchHeater, *args: Any, **kwargs: Any
            ) -> T:
                if not self.check_enabled():
                    raise Cryomagnetic4GException("Switch not enabled")
                return f(self, *args, **kwargs)

            return check_enabled_decorator

    def __init__(self, parent: CryomagneticsModel4G) -> None:
        super().__init__(parent, "SwitchHeater")

        # Add state parameters
        self.add_parameter(
            "enabled",
            label="Switch Heater Enabled",
            get_cmd=self.check_enabled,
            set_cmd=lambda x: (self.enable() if x else self.disable()),
            vals=Bool(),
        )
        self.add_parameter(
            "state",
            label="Switch Heater On",
            get_cmd=self.check_state,
            set_cmd=lambda x: (self.on() if x else self.off()),
            vals=Bool(),
        )
        
        #Not sure about this cmd being correct
        self.add_parameter(
            "in_persistent_mode",
            label="Persistent Mode",
            get_cmd="MODE?",
            val_mapping={True: "Shim", False: "Manual"},
        )


    def disable(self) -> None:
        """Turn measurement off"""
        self.write("PSHTR 0")
        self._enabled = False

    def enable(self) -> None:
        """Turn measurement on"""
        self.write("PSHTR 1")
        self._enabled = True

    def check_enabled(self) -> bool:
        return bool(int(self.ask("PSHTR?").strip()))

    @_Decorators.check_enabled
    def on(self) -> None:
        self.write("PSHTR 1")
        while self._parent.ramping_state() == "heating switch":
            self._parent._sleep(0.5)

    @_Decorators.check_enabled
    def off(self) -> None:
        self.write("PSHTR 0")
        while self._parent.ramping_state() == "cooling switch":
            self._parent._sleep(0.5)

    @_Decorators.check_enabled
    def check_state(self) -> bool:
        return bool(int(self.ask("PSHTR?").strip()))




class CryomagneticsModel4G(VisaInstrument):


    _SHORT_UNITS: ClassVar[dict[str, str]] = {
        "seconds": "s",
        "minutes": "min",
        "tesla": "T",
        "kilogauss": "kG",
        "guass": "G",
        "amp": "A", 
    }
    _DEFAULT_CURRENT_RAMP_LIMIT = 0.06  # [A/s]
    _RETRY_WRITE_ASK = True
    _RETRY_TIME = 5

    def __init__(
        self,
        name: str,
        address: str,
        reset: bool = False,
        terminator: str = "\r\n",
        current_ramp_upper_limit: float | None = None,
        current_ramp_lower_limit: float | None = None,
    ):

        super().__init__(
            name,
            address,
            terminator=terminator,
        )

        