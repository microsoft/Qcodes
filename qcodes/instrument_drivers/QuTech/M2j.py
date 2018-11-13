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

        self.m2j = M2j_module(spi_rack, module)
        self._max_gain_value = 4095
        self._min_gain_value = 0
        self._gain_parameters = {'slope': -1024.45, 'offset': 4450.63,
                                 'db_offset': 32}

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
        ref_scale = int(self._gain_parameters['slope'] * np.log(
            gain - self._gain_parameters['db_offset']) + self._gain_parameters[
                            'offset'])
        if ref_scale < self._min_gain_value:
            ref_scale = self._min_gain_value
        if ref_scale > self._max_gain_value:
            ref_scale = self._max_gain_value

        self.m2j.set_gain(ref_scale)

    def _meas_rf_level(self):
        """
        Measure the power in dBm. Calibrated using an R&S SMA100 source.
        Linear relation between set power and measured data.
        Measurement range -80 to -40 dBm.
        """
        return self.m2j.get_level()
