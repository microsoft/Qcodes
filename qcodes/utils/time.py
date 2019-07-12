"""
This module provides helpers for time measurements
"""

from time import perf_counter

from qcodes.instrument.parameter import Parameter


class ElapsedTimeParameter(Parameter):
    """
    Parameter to measure elapsed time. Measures wall clock time since the
    last reset of the instance's clock. The clock is reset upon creation of the
    instance. The constructor passes kwargs along to the Parameter constructor.

    Args:
        name: the local name of the parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``
    """

    def __init__(self, name: str, label: str = 'Elapsed time', **kwargs):

        super().__init__(name=name,
                         label=label,
                         unit='s',
                         set_cmd=False,
                         **kwargs)

        self.t0: float = perf_counter()

    def get_raw(self) -> float:
        return perf_counter() - self.t0

    def reset_clock(self) -> None:
        self.t0 = perf_counter()
