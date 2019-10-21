from qcodes.instrument.base import Instrument
try:
    from spirack import D4_module
except ImportError:
    raise ImportError(('The D4_module class could not be found. '
                       'Try installing it using pip install spirack'))

from functools import partial
from qcodes.utils.deprecate import deprecate_moved_to_qcd


@deprecate_moved_to_qcd(alternative='qcodes_contrib_drivers.drivers.QuTech.D4.D4')
class D4(Instrument):
    """
    Qcodes driver for the D4 ADC SPI-rack module. Requires installation
    of the 'spirack' package using pip.

    Args:
        name (str): name of the instrument.

        spi_rack (SPI_rack): instance of the SPI_rack class as defined in
            the spirack package. This class manages communication with the
            individual modules.

        module (int): module number as set on the hardware.
    """

    def __init__(self, name, spi_rack, module, **kwargs):
        super().__init__(name, **kwargs)

        self.d4 = D4_module(spi_rack, module)

        self.add_parameter('mode',
                           label='Mode',
                           get_cmd=self.get_mode)

        self.add_parameter('filter_value',
                           label='Filter value',
                           get_cmd=self.get_filter_value)

        self.add_parameter('buffers_enabled',
                           label='Buffers enabled',
                           get_cmd=self.get_buffers_enabled)

        for i in range(2):
            self.add_parameter('adc{}'.format(i + 1),
                               label='ADC {}'.format(i + 1),
                               get_cmd=partial(self.d4.singleConversion, i),
                               units='V')

    def get_mode(self):
        return self.d4.mode

    def get_filter_value(self):
        return self.d4.filter_val

    def get_buffers_enabled(self):
        return self.d4.buf_en
