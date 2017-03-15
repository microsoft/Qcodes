from qcodes.instrument_drivers.test import DriverTestCase
import signadyne_common.SD_DIG

class SD_DIG_test(DriverTestCase):
    """
    This is the qcodes driver test suite for a generic Signadyne Digitizer of the M32/33XX series.

    Status: pre-alpha

    """
    driver = SD_DIG

    @classMethod
    def setUpClass(cls):
        super().setUpClass()

    def test_device_name():
        """ Test that the device can be accessed and returns a name
        """
        name = self.instrument.SD_AIN.getProductNameBySlot(1,8)
        self.assertEqual(name, 'M3300A')

