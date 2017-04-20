from qcodes import Instrument
from qcodes.utils.validators import Enum

from .D5a_module import D5a_module

from functools import partial


class D5a(Instrument):
    """
    Qcodes driver for the D5a DAC SPI-rack module.

    functions:
    -   set_dacs_zero   set all DACs to zero voltage

    parameters:
    -   dacN:       get and set DAC voltage
    -   stepsizeN   get the minimum step size corresponding to the span
    -   spanN       get and set the DAC span: '4v uni', '4v bi', or '2.5v bi'

    where N is the DAC number from 1 up to 16
    """
    def __init__(self, name, spi_rack, module, **kwargs):
        super().__init__(name, **kwargs)

        self.d5a = D5a_module(spi_rack, module)

        self._span_set_map = {
            '4v uni': 0,
            '4v bi': 2,
            '2.5v bi': 4,
        }

        self._span_get_map = {v: k for k, v in self._span_set_map.items()}

        self.add_function('set_dacs_zero', call_cmd=self._set_dacs_zero)

        for i in range(16):
            self.add_parameter('dac{}'.format(i + 1),
                               label='DAC {}'.format(i + 1),
                               get_cmd=partial(self._get_dac, i),
                               set_cmd=partial(self.d5a.set_voltage, i),
                               unit='V',
                               delay=0.1)

            self.add_parameter('stepsize{}'.format(i + 1),
                               get_cmd=partial(self.d5a.get_stepsize, i),
                               unit='V')

            self.add_parameter('span{}'.format(i + 1),
                               get_cmd=partial(self._get_span, i),
                               set_cmd=partial(self._set_span, i),
                               vals=Enum(*self._span_set_map.keys()))

    def _set_dacs_zero(self):
        for i in range(16):
            self._set_dac(i, 0.0)

    def _get_dac(self, dac):
        return self.d5a.voltages[dac]

    def _get_span(self, dac):
        return self._span_get_map[self.d5a.span[dac]]

    def _set_span(self, dac, span_str):
        self.d5a.change_span_update(dac, self._span_set_map[span_str])
