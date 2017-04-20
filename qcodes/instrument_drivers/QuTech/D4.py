from qcodes import Instrument
from qcodes.utils.validators import Numbers

from .D4_module import D4_module


class D4(Instrument):
    """
    Qcodes driver for the D4 ADC SPI-rack module.
    """
    def __init__(self, name, spi_rack, module, **kwargs):
        super().__init__(name, **kwargs)

        self.d4 = D4_module(spi_rack, module)

        for i in range(2):
            pass
