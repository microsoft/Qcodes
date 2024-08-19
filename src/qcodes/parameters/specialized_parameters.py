"""
Module for specialized parameters. The :mod:`qcodes.instrument.parameter`
module provides generic parameters for different generic cases. This module
provides useful/convenient specializations of such generic parameters.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from qcodes.validators import Strings, Validator

from .parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Callable

    from qcodes.instrument import InstrumentBase


class ElapsedTimeParameter(Parameter):
    """
    Parameter to measure elapsed time. Measures wall clock time since the
    last reset of the instance's clock. The clock is reset upon creation of the
    instance. The constructor passes kwargs along to the Parameter constructor.

    Args:
        name: The local name of the parameter. See the documentation of
            :class:`qcodes.parameters.Parameter` for more details.
    """

    def __init__(self, name: str, label: str = "Elapsed time", **kwargs: Any):
        hardcoded_kwargs = ["unit", "get_cmd", "set_cmd"]

        for hck in hardcoded_kwargs:
            if hck in kwargs:
                raise ValueError(f'Can not set "{hck}" for an ElapsedTimeParameter.')

        super().__init__(name=name, label=label, unit="s", set_cmd=False, **kwargs)

        self._t0: float = perf_counter()

    def get_raw(self) -> float:
        return perf_counter() - self.t0

    def reset_clock(self) -> None:
        self._t0 = perf_counter()

    @property
    def t0(self) -> float:
        return self._t0


class InstrumentRefParameter(Parameter):
    """
    An instrument reference parameter.

    This parameter is useful when one needs a reference to another instrument
    from within an instrument, e.g., when creating a meta instrument that
    sets parameters on instruments it contains.

    Args:
        name: The name of the parameter that one wants to add.

        instrument: The "parent" instrument this
            parameter is attached to, if any.

        initial_value: Starting value, may be None even if None does not
            pass the validator. None is only allowed as an initial value
            and cannot be set after initiation.

        **kwargs: Passed to InstrumentRefParameter parent class
    """

    def __init__(
        self,
        name: str,
        instrument: InstrumentBase | None = None,
        label: str | None = None,
        unit: str | None = None,
        get_cmd: str | Callable[..., Any] | Literal[False] | None = None,
        set_cmd: str | Callable[..., Any] | Literal[False] | None = None,
        initial_value: float | str | None = None,
        max_val_age: float | None = None,
        vals: Validator[Any] | None = None,
        docstring: str | None = None,
        **kwargs: Any,
    ) -> None:
        if vals is None:
            vals = Strings()
        if set_cmd is not None:
            raise RuntimeError("InstrumentRefParameter does not support set_cmd.")
        super().__init__(
            name,
            instrument=instrument,
            label=label,
            unit=unit,
            get_cmd=get_cmd,
            set_cmd=set_cmd,
            initial_value=initial_value,
            max_val_age=max_val_age,
            vals=vals,
            docstring=docstring,
            **kwargs,
        )

    # TODO(nulinspiratie) check class works now it's subclassed from Parameter
    def get_instr(self) -> InstrumentBase:
        """
        Returns the instance of the instrument with the name equal to the
        value of this parameter.
        """
        ref_instrument_name = self.get()
        # note that _instrument refers to the instrument this parameter belongs
        # to, while the ref_instrument_name is the instrument that is the value
        # of this parameter.
        if self._instrument is None:
            raise RuntimeError("InstrumentRefParameter is not bound to an instrument.")
        return self._instrument.find_instrument(ref_instrument_name)
