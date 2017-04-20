from qcodes import Instrument
from qcodes.utils.validators import Numbers

from .S5i_module import S5i_module


class S5i(Instrument):
    """
    Qcodes driver for the S5i RF generator SPI-rack module.
    """
    def __init__(self, name, spi_rack, module, **kwargs):
        super().__init__(name, **kwargs)

        self.s5i = S5i_module(spi_rack, module)

        self.add_parameter('rf_frequency',
                           label='RF Frequency',
                           get_cmd=self._get_rf_frequency,
                           set_cmd=self.s5i.set_frequency,
                           unit='MHz',
                           vals=Numbers())

    def _get_rf_frequency(self):
        return self.s5i.rfFrequency
