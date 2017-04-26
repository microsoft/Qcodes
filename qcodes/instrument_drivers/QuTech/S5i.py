from qcodes import Instrument
from qcodes.utils.validators import Bool, Numbers

from spirack import S5i_module


class S5i(Instrument):
    """
    Qcodes driver for the S5i RF generator SPI-rack module.
    """
    def __init__(self, name, spi_rack, module, **kwargs):
        super().__init__(name, **kwargs)

        self.s5i = S5i_module(spi_rack, module)

        self.add_parameter('use_external_reference',
                           label='Use external reference',
                           get_cmd=self._use_external_reference,
                           set_cmd=self.s5i.use_external_reference,
                           vals=Bool())

        self.add_parameter('stepsize',
                           label='Stepsize',
                           get_cmd=self._get_stepsize,
                           set_cmd=self.s5i.set_stepsize,
                           units='Hz',
                           vals=Numbers())

        # TODO create 2 functions and 1 get parameter?
        self.add_parameter('frequency',
                           label='Frequency',
                           get_cmd=self._get_rf_frequency,
                           set_cmd=self.s5i.set_frequency,
                           units='Hz',
                           vals=Numbers())

        self.add_parameter('frequency_optimal',
                           label='Frequency',
                           get_cmd=self._get_rf_frequency,
                           set_cmd=self.s5i.set_frequency,
                           units='Hz',
                           vals=Numbers())

    def _use_external_reference(self):
        return self.s5i.use_external

    def _get_stepsize(self):
        return self.s5i.stepsize

    def _get_rf_frequency(self):
        return self.s5i.rfFrequency
