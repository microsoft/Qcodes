from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Bool, Numbers

try:
    from spirack import S5i_module
except ImportError:
    raise ImportError(('The S5i_module class could not be found. '
                       'Try installing it using pip install spirack'))


class S5i(Instrument):
    """
    Qcodes driver for the S5i RF generator SPI-rack module.

    Args:
        name (str): name of the instrument.
        spi_rack (SPI_rack): instance of the SPI_rack class as defined in
            the spirack package. This class manages communication with the
            individual modules.
        module (int): module number as set on the hardware.
    """

    def __init__(self, name, spi_rack, module, **kwargs):
        super().__init__(name, **kwargs)

        self.s5i = S5i_module(spi_rack, module, frequency=41e6, enable_output=1,
                              output_level=0)

        self.add_parameter('enabled',
                           label='RF_on/off',
                           set_cmd=self.s5i.enable_output_soft,
                           vals=Bool(),
                           docstring='Switched output on/off')

        self.add_parameter('frequency_stepsize',
                           label='Frequency stepsize',
                           get_cmd=self._get_stepsize,
                           set_cmd=self.s5i.set_stepsize,
                           units='Hz',
                           vals=Numbers(),
                           docstring='Set the optimal frequency stepzise for '
                                     'a minimal phase noise')

        self.add_parameter('frequency',
                           label='Frequency',
                           get_cmd=self._get_rf_frequency,
                           set_cmd=self.s5i.set_frequency,
                           units='Hz',
                           vals=Numbers(40e6, 4e9),
                           docstring='Set RF frequency, range 40MHz to 4GHz')

        self.add_parameter('power',
                           label='Output Power',
                           set_cmd=self.s5i.set_output_power,
                           unit='dBm',
                           vals=Numbers(-14, 20),
                           docstring='Set power in dBm, range -14 to +20 dBm')

        self.add_function('optimize_for_frequency', call_cmd=self._optimize)

    def _get_stepsize(self):
        return self.s5i.stepsize

    def _get_rf_frequency(self):
        return self.s5i.rf_frequency

    def _optimize(self):
        stepsize = self.s5i.get_optimal_stepsize(self.s5i.rf_frequency)
        self.s5i.set_stepsize(stepsize)
