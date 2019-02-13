from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Enum, Numbers

try:
    from spirack import D5a_module
except ImportError:
    raise ImportError(('The D5a_module class could not be found. '
                       'Try installing it using pip install spirack'))

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

    def __init__(self, name, spi_rack, module, inter_delay=0.1, dac_step=10e-3,
                 reset_voltages=False, mV=False, number_dacs=16, **kwargs):
        """ Create instrument for the D5a module.

        The D5a module works with volts as units. For backward compatibility
        there is the option to allow mV for the dacX parameters.

        The output span of the DAC module can be changed with the spanX
        command. Be carefull when executing this command with a sample
        connected as voltage jumps can occur.

        Args:
            name (str): name of the instrument.

            spi_rack (SPI_rack): instance of the SPI_rack class as defined in
                the spirack package. This class manages communication with the
                individual modules.

            module (int): module number as set on the hardware.
            inter_delay (float): time in seconds, passed to dac parameters of the object
            dac_step (float): max step size (V or mV), passed to dac parameters of the object
            reset_voltages (bool): passed to D5a_module constructor
            mV (bool): if True, then use mV as units in the dac parameters
            number_dacs (int): number of DACs available. This is 8 for the D5mux
        """
        super().__init__(name, **kwargs)

        self.d5a = D5a_module(spi_rack, module, reset_voltages=reset_voltages)
        self._mV = mV
        self._number_dacs = number_dacs

        self._span_set_map = {
            '4v uni': 0,
            '4v bi': 2,
            '2v bi': 4,
        }

        self._span_get_map = {v: k for k, v in self._span_set_map.items()}

        self.add_function('set_dacs_zero', call_cmd=self._set_dacs_zero,
                          docstring='Reset all dacs to zero voltage. No ramping is performed.')

        if self._mV:
            self._gain = 1e3
            unit = 'mV'
        else:
            self._gain = 1
            unit = 'V'

        for i in range(self._number_dacs):
            validator = self._get_validator(i)

            self.add_parameter('dac{}'.format(i + 1),
                               label='DAC {}'.format(i + 1),
                               get_cmd=partial(self._get_dac, i),
                               set_cmd=partial(self._set_dac, i),
                               unit=unit,
                               vals=validator,
                               step=dac_step,
                               inter_delay=inter_delay)

            self.add_parameter('stepsize{}'.format(i + 1),
                               get_cmd=partial(self.d5a.get_stepsize, i),
                               unit='V',
                               docstring='Returns the smallest voltage step of the DAC.')

            self.add_parameter('span{}'.format(i + 1),
                               get_cmd=partial(self._get_span, i),
                               set_cmd=partial(self._set_span, i),
                               vals=Enum(*self._span_set_map.keys()),
                               docstring='Change the output span of the DAC. This command also updates the validator.')

    def _set_dacs_zero(self):
        for i in range(self._number_dacs):
            self._set_dac(i, 0.0)

    def _set_dac(self, dac, value):
        return self.d5a.set_voltage(dac, value / self._gain)

    def _get_dac(self, dac):
        return self._gain * self.d5a.voltages[dac]

    def _get_span(self, dac):
        return self._span_get_map[self.d5a.span[dac]]

    def _set_span(self, dac, span_str):
        self.d5a.change_span_update(dac, self._span_set_map[span_str])
        self.parameters['dac{}'.format(
            dac + 1)].vals = self._get_validator(dac)

    def _get_validator(self, dac):
        span = self.d5a.span[dac]
        if span == D5a_module.range_2V_bi:
            validator = Numbers(-2 * self._gain, 2 * self._gain)
        elif span == D5a_module.range_4V_bi:
            validator = Numbers(-4 * self._gain, 4 * self._gain)
        elif span == D5a_module.range_4V_uni:
            validator = Numbers(0, 4 * self._gain)
        else:
            msg = 'The found DAC span of {} does not correspond to a known one'
            raise Exception(msg.format(span))

        return validator
