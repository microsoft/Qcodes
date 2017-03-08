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
        self.assertEqual(slot_number, 7)

    def test_serial_number(self):
        serial_number = self.instrument.serial_number()
        self.assertEqual(serial_number, '21L6MRU4')

    def test_chassis_and_slot(self):
        chassis_number = self.instrument.chassis_number()
        slot_number = self.instrument.slot_number()
        product_name = self.instrument.product_name()
        serial_number = self.instrument.serial_number()

        product_name_test = self.instrument.get_product_name_by_slot(chassis_number, slot_number)
        self.assertEqual(product_name_test, product_name)

        serial_number_test = self.instrument.get_serial_number_by_slot(chassis_number, slot_number)
        self.assertEqual(serial_number_test, serial_number)

    def test_open_close(self):
        chassis_number = self.instrument.chassis_number()
        slot_number = self.instrument.slot_number()
        product_name = self.instrument.product_name()
        serial_number = self.instrument.serial_number()

        self.instrument.close_soft()
        open_status = self.instrument.open()
        self.assertEqual(open_status, False)

        self.instrument.open_with_serial_number(product_name, serial_number)
        open_status = self.instrument.open()
        self.assertEqual(open_status, True)

        self.instrument.close_soft()
        self.instrument.open_with_slot(product_name, chassis_number, slot_number)
        open_status = self.instrument.open()
        self.assertEqual(open_status, True)

    def test_clock_frequency(self):
        with self.assertRaises(ValueError):
            self.instrument.clock_frequency(600e6)
        with self.assertRaises(ValueError):
            self.instrument.clock_frequency(32)

        cur_f = self.instrument.clock_frequency()
        test_f = 300e6
        self.instrument.clock_frequency(test_f)
        self.assertAlmostEqual(self.instrument.clock_frequency(), test_f, delta=1)

        test_f = 453.152e6
        self.instrument.clock_frequency(test_f)
        self.assertAlmostEqual(self.instrument.clock_frequency(), test_f, delta=1)

        # leave the setup in the initial state
        self.instrument.clock_frequency(cur_f)
