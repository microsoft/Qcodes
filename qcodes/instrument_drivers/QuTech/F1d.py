from qcodes import Instrument
from qcodes.utils.validators import Enum

from .F1d_module import F1d_module


class F1d(Instrument):
    """
    Qcodes driver for the F1d IQ-Mixer SPI-rack module.
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

        self.add_function('enable_remote', self.f1d.enable_remote)
        self.add_function('clear_rf_clip', self.f1d.clear_rf_clip)
        self.add_function('is_rf_clipped', self.f1d.rf_clipped)

    def get_remote_settings(self):
        return self.f1d.remote_settings
