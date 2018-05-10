from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Enum

try:
    from spirack import F1d_module
except ImportError:
    raise ImportError(('The F1d_module class could not be found. '
                       'Try installing it using pip install spirack'))


class F1d(Instrument):
    """
    Qcodes driver for the F1d IQ-Mixer SPI-rack module.

    Args:
        name (str): name of the instrument.

        spi_rack (SPI_rack): instance of the SPI_rack class as defined in
            the spirack package. This class manages communication with the
            individual modules.

        module (int): module number as set on the hardware.
    """

    def __init__(self, name, spi_rack, module, **kwargs):
        super().__init__(name, **kwargs)

        self.f1d = F1d_module(spi_rack, module)

        self.add_parameter('remote_settings',
                           label='Remote settings',
                           get_cmd=self.get_remote_settings)

        self.add_parameter('IQ_filter',
                           label='IQ filter',
                           set_cmd=self.f1d.set_IQ_filter,
                           units='MHz',
                           vals=Enum(1, 3, 10, 20))

        self.add_parameter('I_gain',
                           label='I gain',
                           set_cmd=self.f1d.set_I_gain,
                           vals=Enum('low', 'mid', 'high'))

        self.add_parameter('Q_gain',
                           label='Q gain',
                           set_cmd=self.f1d.set_Q_gain,
                           vals=Enum('low', 'mid', 'high'))

        self.add_parameter('RF_level',
                           label='RF level',
                           set_cmd=self.f1d.get_RF_level,
                           units='dBm')

        self.add_parameter('LO_level',
                           label='LO level',
                           set_cmd=self.f1d.get_LO_level,
                           units='dBm')

        self.add_function('enable_remote', call_cmd=self.f1d.enable_remote)
        self.add_function('clear_rf_clip', call_cmd=self.f1d.clear_rf_clip)
        self.add_function('is_rf_clipped', call_cmd=self.f1d.rf_clipped)

    def get_remote_settings(self):
        return self.f1d.remote_settings
