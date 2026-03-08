"""
Module for specialized parameters. The :mod:`qcodes.instrument.parameter`
module provides generic parameters for different generic cases. This module
provides useful/convenient specializations of such generic parameters.
"""

from __future__ import annotations

import warnings
from time import perf_counter
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from qcodes.utils import QCoDeSDeprecationWarning
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

    _DEPRECATED_POSITIONAL_ARGS: ClassVar[tuple[str, ...]] = ("label",)

    def __init__(
        self, name: str, *args: Any, label: str = "Elapsed time", **kwargs: Any
    ):
        if args:
            # TODO: After QCoDeS 0.57 remove the args argument and delete this code block.
            positional_names = __class__._DEPRECATED_POSITIONAL_ARGS
            if len(args) > len(positional_names):
                raise TypeError(
                    f"{type(self).__name__}.__init__() takes at most "
                    f"{len(positional_names) + 2} positional arguments "
                    f"({len(args) + 2} given)"
                )

            _defaults: dict[str, Any] = {"label": "Elapsed time"}
            _kwarg_vals: dict[str, Any] = {"label": label}

            for i in range(len(args)):
                arg_name = positional_names[i]
                if _kwarg_vals[arg_name] != _defaults[arg_name]:
                    raise TypeError(
                        f"{type(self).__name__}.__init__() got multiple "
                        f"values for argument '{arg_name}'"
                    )

            positional_arg_names = positional_names[: len(args)]
            names_str = ", ".join(f"'{n}'" for n in positional_arg_names)
            warnings.warn(
                f"Passing {names_str} as positional argument(s) to "
                f"{type(self).__name__} is deprecated. "
                f"Please pass them as keyword arguments.",
                QCoDeSDeprecationWarning,
                stacklevel=2,
            )

            _pos = dict(zip(positional_names, args))
            label = _pos.get("label", label)

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

    _DEPRECATED_POSITIONAL_ARGS: ClassVar[tuple[str, ...]] = (
        "instrument",
        "label",
        "unit",
        "get_cmd",
        "set_cmd",
        "initial_value",
        "max_val_age",
        "vals",
        "docstring",
    )

    def __init__(
        self,
        name: str,
        *args: Any,
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
        if args:
            # TODO: After QCoDeS 0.57 remove the args argument and delete this code block.
            positional_names = __class__._DEPRECATED_POSITIONAL_ARGS
            if len(args) > len(positional_names):
                raise TypeError(
                    f"{type(self).__name__}.__init__() takes at most "
                    f"{len(positional_names) + 2} positional arguments "
                    f"({len(args) + 2} given)"
                )

            _defaults: dict[str, Any] = {
                "instrument": None,
                "label": None,
                "unit": None,
                "get_cmd": None,
                "set_cmd": None,
                "initial_value": None,
                "max_val_age": None,
                "vals": None,
                "docstring": None,
            }

            _kwarg_vals: dict[str, Any] = {
                "instrument": instrument,
                "label": label,
                "unit": unit,
                "get_cmd": get_cmd,
                "set_cmd": set_cmd,
                "initial_value": initial_value,
                "max_val_age": max_val_age,
                "vals": vals,
                "docstring": docstring,
            }

            for i in range(len(args)):
                arg_name = positional_names[i]
                if _kwarg_vals[arg_name] is not _defaults[arg_name]:
                    raise TypeError(
                        f"{type(self).__name__}.__init__() got multiple "
                        f"values for argument '{arg_name}'"
                    )

            positional_arg_names = positional_names[: len(args)]
            names_str = ", ".join(f"'{n}'" for n in positional_arg_names)
            warnings.warn(
                f"Passing {names_str} as positional argument(s) to "
                f"{type(self).__name__} is deprecated. "
                f"Please pass them as keyword arguments.",
                QCoDeSDeprecationWarning,
                stacklevel=2,
            )

            _pos = dict(zip(positional_names, args))
            instrument = _pos.get("instrument", instrument)
            label = _pos.get("label", label)
            unit = _pos.get("unit", unit)
            get_cmd = _pos.get("get_cmd", get_cmd)
            set_cmd = _pos.get("set_cmd", set_cmd)
            initial_value = _pos.get("initial_value", initial_value)
            max_val_age = _pos.get("max_val_age", max_val_age)
            vals = _pos.get("vals", vals)
            docstring = _pos.get("docstring", docstring)

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
