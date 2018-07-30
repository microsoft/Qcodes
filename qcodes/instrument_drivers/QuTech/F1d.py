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

    The `enable_remote` parameter sets the F1d module in remote mode. When settings are changed on the
    hardware front panel, then the remote mode is deactivated

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
                           unit='MHz',
                           vals=Enum(1, 3, 10),
                           docstring='Low-pass filter after demodulation')

        self.add_parameter('I_gain',
                           label='I gain',
                           set_cmd=self.f1d.set_I_gain,
                           vals=Enum('low', 'mid', 'high'),
                           docstring='Gain of amplifier of demodulatd signal')

        self.add_parameter('Q_gain',
                           label='Q gain',
                           set_cmd=self.f1d.set_Q_gain,
                           vals=Enum('low', 'mid', 'high'),
                           docstring='Gain of amplifier of demodulatd signal')

        self.add_parameter('RF_level',
                           label='RF level',
                           get_cmd=self.f1d.get_RF_level,
                           unit='dBm')

        self.add_parameter('LO_level',
                           label='LO level',
                           get_cmd=self.f1d.get_LO_level,
                           unit='dBm')

        self.add_parameter('enable_remote',
                           label='Enable remote',
                           set_cmd=self.f1d.enable_remote, docstring='getting the remote status is not possible')
        self.add_function('clear_rf_clip',
                          call_cmd=self.f1d.clear_rf_clip)
        self.add_function('is_rf_clipped',
                          call_cmd=self.f1d.rf_clipped)

    def get_remote_settings(self):
        return self.f1d.remote_settings
