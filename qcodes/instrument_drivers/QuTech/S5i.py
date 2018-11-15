from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Bool, Numbers

try:
    from spirack import S5i_module
except ImportError:
    raise ImportError(('The S5i_module class could not be found. '
                       'Try installing it using pip install spirack'))


class S5i(Instrument):
    """
    Qcodes driver for the S5i RF generator SPI-rack module.

    Args:
        name (str): name of the instrument.

        spi_rack (SPI_rack): instance of the SPI_rack class as defined in
            the spirack package. This class manages communication with the
            individual modules.

        module (int): module number as set on the hardware.

    Parameters:
        enabled: switched output on/off
        frequency: Frequency on Hz, range 40 MHz - 4 GHz, stepsize 1 MHz
        freq_fine: Frequency on Hz, range 40 MHz - 4 GHz, stepsize 10 kHz.
        Difference between frequency and freq_fine unclear!?
        power: power in dBm, range -20 to +15 dBm

    """

    def __init__(self, name, spi_rack, module, **kwargs):
        super().__init__(name, **kwargs)

        self.s5i = S5i_module(spi_rack, module, frequency=41e6,
                              enable_output=1, output_level=0)
        # Put low freq as it is unlikely it will be resonant with something.
        self.add_parameter('enabled',
                           label='RF_on/off',
                           set_cmd=self.s5i.enable_output_soft,
                           vals=Bool())

        self.add_parameter('frequency',
                           label='Frequency',
                           get_cmd=self._get_rf_frequency,
                           set_cmd=self.s5i.set_frequency,
                           unit='Hz',
                           vals=Numbers())

        self.add_parameter('freq_fine',
                           label='Frequency fine',
                           get_cmd=self._get_rf_frequency,
                           set_cmd=self.s5i.set_frequency_optimally,
                           unit='Hz',
                           vals=Numbers())

        self.add_parameter('power',
                           label='Power',
                           unit='dBm',
                           set_cmd=self._set_power,
                           vals=Numbers())

    def _set_power(self, dbm):
        self.s5i.set_output_power(self.convert_dbm_to_ref_scale(dbm))

    @staticmethod
    def convert_ref_scale_to_dbm(ref_scale):
        # Made using a simple fit to measured data. This is temporarily and
        # should be done in the module itselves.
        a = -30.1271
        b = 58.8666
        c = -7.29323
        d = -145.572
        x1 = 1.10351
        x2 = 0.52127

        x = ref_scale
        return a + b * x + c * (x - x1)**2 + d * (x - x2)**3

    @staticmethod
    def convert_dbm_to_ref_scale(dBm):
        if dBm < -20:
            print(
                "notice, lower limit for the power is -20dBm. The limit will be the set value.")
        if dBm > 15:
            print(
                "notice, upper limit for the power is 15dBm. The limit will be the set value.")
        a = 0.563451
        b = 0.0182626
        c = -0.000556757
        d = -8.44786e-05
        x1 = 1.29053
        x2 = -0.230088
        e = 2.82792e-07
        x3 = -2.647
        x = dBm

        ref_scale = a + b * x + c * \
            (x - x1)**2 + d * (x - x2)**3 + e * (x - x3)**5
        if ref_scale < 0:
            ref_scale = 0
        if ref_scale > 1:
            ref_scale = 1

        return ref_scale

    def _use_external_reference(self):
        return self.s5i.use_external

    def _get_stepsize(self):
        return self.s5i.stepsize

    def _get_rf_frequency(self):
        return self.s5i.rf_frequency
