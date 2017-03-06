from qcodes.instrument_drivers.test import DriverTestCase
from .M3201A import Signadyne_M3201A

class TestSignadyne_M3201A(DriverTestCase):
    """
    This is a test suite for testing the Signadyne M3201A AWG card driver.
    It provides test functions for each function and parameter as defined in the driver,
    as well as test functions for general things like connecting to the device.

    Status: beta

    The current test functions are not super useful yet because the driver doesn't support set-able parameters at the
    moment. Normally a more useful test function would do something like:

            self.instrument.clock_frequency(100e6)
            self.assertAlmostEqual(self.instrument.clock_frequency(), 100e6, places=4)

    Unfortunately, the Signadyne libraries don't support many variables that are both get-able and set-able.

    We can however test for ValueErrors which is a useful safety test.
    """

    driver = Signadyne_M3201A

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.instrument.off()  # Not a test but a safety measure

    def test_chassis_number(self):
        chassis_number = self.instrument.chassis_number()
        self.assertEqual(chassis_number, 1)

    def test_slot_number(self):
        slot_number = self.instrument.slot_number()
        self.assertEqual(slot_number, 8)

    def test_serial_number(self):
        serial_number = self.instrument.serial_number()
        self.assertEqual(serial_number, 1234567890)
