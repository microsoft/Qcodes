"""
Module for specialized parameters. The :mod:`qcodes.instrument.parameter`
module provides generic parameters for different generic cases. This module
provides useful/convenient specializations of such generic parameters.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from qcodes.validators import Strings

from .parameter import Parameter, ParameterKWArgs

if TYPE_CHECKING:
    from typing import Unpack

    from qcodes.instrument import InstrumentBase


class ElapsedTimeParameter(Parameter):
    """
    Parameter to measure elapsed time. Measures wall clock time since the
    last reset of the instance's clock. The clock is reset upon creation of the
    instance. The constructor passes kwargs along to the Parameter constructor.

    Args:
        name: The local name of the parameter. See the documentation of
            :class:`qcodes.parameters.Parameter` for more details.
        **kwargs: Forwarded to the ``Parameter`` base class.
            See :class:`ParameterKWArgs` for details.
            Note that ``unit``, ``get_cmd``, and ``set_cmd`` are not allowed
            since ElapsedTimeParameter hardcodes these.
            ``label`` defaults to ``"Elapsed time"`` if not provided.

    Raises:
        ValueError: If ``unit``, ``get_cmd``, or ``set_cmd`` is provided.

    """

    def __init__(self, name: str, **kwargs: Unpack[ParameterKWArgs]):
        hardcoded_kwargs = ["unit", "get_cmd", "set_cmd"]

        for hck in hardcoded_kwargs:
            if hck in kwargs:
                raise ValueError(f'Can not set "{hck}" for an ElapsedTimeParameter.')

        kwargs.setdefault("label", "Elapsed time")
        kwargs["unit"] = "s"
        kwargs["set_cmd"] = False
        super().__init__(name=name, **kwargs)

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
        **kwargs: Forwarded to the ``Parameter`` base class.
            See :class:`ParameterKWArgs` for details.
            Note that ``set_cmd`` is not allowed since
            InstrumentRefParameter uses manual set (``set_cmd=None``).
            ``vals`` defaults to :class:`~qcodes.validators.Strings`
            if not provided.

    Raises:
        RuntimeError: If ``set_cmd`` is provided with a non-None value.

    """

    def __init__(
        self,
        name: str,
        **kwargs: Unpack[ParameterKWArgs],
    ) -> None:
        kwargs.setdefault("vals", Strings())
        if kwargs.get("set_cmd") is not None:
            raise RuntimeError("InstrumentRefParameter does not support set_cmd.")
        kwargs.setdefault("set_cmd", None)
        super().__init__(
            name,
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
