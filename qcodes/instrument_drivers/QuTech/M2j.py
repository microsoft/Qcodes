from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers
import numpy as np

try:
    from spirack import M2j_module, SPI_rack
except ImportError:
    raise ImportError(('The M2j_module class could not be found. '
                       'Try installing it using pip install spirack'))


class M2j(Instrument):

    def __init__(self, name: str, spi_rack: SPI_rack, module: int, **kwargs):
        """
        Qcodes driver for the M2j RF amplifier SPI-rack module.

        Args:
            name: name of the instrument.

            spi_rack: instance of the SPI_rack class as defined in
                the spirack package. This class manages communication with the
                individual modules.

            module: module number as set on the hardware.

        The gain can only be set on the device, but not read from the device.
        """
        super().__init__(name, **kwargs)

        self.m2j = M2j_module.M2j_module(spi_rack, module)

        self.add_parameter('gain',
                           label='gain',
                           set_cmd=self._set_gain,
                           unit='dB',
                           vals=Numbers(min_value=33, max_value=55),
                           docstring='Amplifier gain in dB, range 33 to 55 dB')

        self.add_parameter('RF_level',
                           label='RF level',
                           get_cmd=self._meas_rf_level,
                           unit='dBm',
                           docstring='Measured RF power after amplification '
                                     '(not calibrated)')

        self.add_function('clear_rf_clip',
                          call_cmd=self.m2j.clear_rf_clip)
        self.add_function('is_rf_clipped',
                          call_cmd=self.m2j.rf_clipped)

    def _set_gain(self, gain):
        a = 1024.45
        b = 32
        c = 4450.63

        ref_scale = int(-a * np.log(gain - b) + c)
        if ref_scale < 0:
            ref_scale = 0
        if ref_scale > 4095:
            ref_scale = 4095

        self.m2j.set_gain(ref_scale)

    def _meas_rf_level(self):
        """
        Measure the power in dBm. Calibrated using an R&S SMA100 source.
        Linear relation between set power and measured data.
        Measurement range -80 to -40 dBm.
        """
        return self.m2j.get_level()
