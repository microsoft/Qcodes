from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers
import numpy as np

try:
    from spirack import M2j_module
except ImportError:
    raise ImportError(('The M2j_module class could not be found. '
                       'Try installing it using pip install spirack'))


class M2j(Instrument):

    def __init__(self, name, spi_rack, module, **kwargs):
        """
        Qcodes driver for the M2j RF amplifier SPI-rack module.

        Args:
            name (str): name of the instrument.

            spi_rack (SPI_rack): instance of the SPI_rack class as defined in
                the spirack package. This class manages communication with the
                individual modules.

            module (int): module number as set on the hardware.

        The gain and RF_level can only be set to the device, but not read from the device.

        Parameters:
            gain: Amplifier gain in dB, range 32 to 55 dB
            RF_level: Measured RF power after amplification (not calibrated)
        """
        super().__init__(name, **kwargs)

        self.m2j = M2j_module(spi_rack, module)

        self.add_parameter('gain',
                           label='gain',
                           set_cmd=self._set_gain,
                           unit='dB',
                           vals=Numbers())

        self.add_parameter('RF_level',
                           label='RF level',
                           get_cmd=self._meas_power,
                           unit='dBm')

        self.add_function('clear_rf_clip',
                          call_cmd=self.m2j.clear_rf_clip)
        self.add_function('is_rf_clipped',
                          call_cmd=self.m2j.rf_clipped)

    def _set_gain(self, dB):
        if dB < 32:
            print(
                "notice, lower limit for the gain is 32dBm. The limit will be the set value.")
            dB = 32
        if dB > 55:
            print(
                "notice, upper limit for the gain is 55dBm. The limit will be the set value.")
        a = 1024.45
        b = 32
        c = 4450.63
        x = dB

        ref_scale = int(-a * np.log(x - b) + c)
        if ref_scale < 0:
            ref_scale = 0
        if ref_scale > 4095:
            ref_scale = 4095

        self.m2j.set_gain(ref_scale)

    def _meas_power(self):
        """ Measure the power and convert it to dBm. Calibrated using an R&S
         SMA100 source. Linear relation between set power and measured data.
         Measurement range -80 to -40 dBm.
        """
        x = self.m2j.get_level()
        a = 0
        b = 1
        dBm = a + b * x
        return dBm
