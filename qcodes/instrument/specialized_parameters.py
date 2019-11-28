"""
Module for specialized parameters. The :mod:`qcodes.instrument.parameter`
module provides generic parameters for different generic cases. This module
provides useful/convenient specializations of such generic parameters.
"""

from time import perf_counter
from typing import Any

from qcodes.instrument.parameter import Parameter


class ElapsedTimeParameter(Parameter):
    """
    Parameter to measure elapsed time. Measures wall clock time since the
    last reset of the instance's clock. The clock is reset upon creation of the
    instance. The constructor passes kwargs along to the Parameter constructor.

    Args:
        name: The local name of the parameter. See the documentation of
            :class:`qcodes.instrument.parameter.Parameter` for more details.
    """

    def __init__(self, name: str, label: str = 'Elapsed time', **kwargs: Any):

        hardcoded_kwargs = ['unit', 'get_cmd', 'set_cmd']

        for hck in hardcoded_kwargs:
            if hck in kwargs:
                raise ValueError(f'Can not set "{hck}" for an '
                                 'ElapsedTimeParameter.')

        super().__init__(name=name,
                         label=label,
                         unit='s',
                         set_cmd=False,
                         **kwargs)

        self._t0: float = perf_counter()

    def get_raw(self) -> float:
        return perf_counter() - self.t0

    def reset_clock(self) -> None:
        self._t0 = perf_counter()

    @property
    def t0(self) -> float:
        return self._t0
