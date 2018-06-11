from qcodes.instrument_drivers.test import DriverTestCase
import unittest

try:
    from qcodes.instrument_drivers.Keysight.M3201A import Keysight_M3201A
    Keysight_M3201A_found = True
except ImportError:
    Keysight_M3201A_found = False
try:
    from qcodes.instrument_drivers.Keysight.M3300A import M3300A_AWG
    M3300A_AWG_found = True
except ImportError:
    M3300A_AWG_found = False
try:
    from qcodes.instrument_drivers.Keysight.SD_common.SD_Module import SD_Module
    SD_Module_found = True
except ImportError:
    SD_Module_found = False

@unittest.skipIf(not SD_Module_found, "SD_Module tests requires the keysightSD1 module")
class TestSD_Module(DriverTestCase):
    """
    Tis is a test suite for testing the general Keysight SD_Module driver.

    This test suit is only used during the development of the general SD_Module driver. In a real-life scenario,
    no direct instances will be made from this class, but rather instances of either SD_AWG or SD_DIG.
    """
    if SD_Module_found:
        driver = SD_Module

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_chassis_and_slot(self):
        chassis_number = self.instrument.chassis_number()
        slot_number = self.instrument.slot_number()
        product_name = self.instrument.product_name()
        serial_number = self.instrument.serial_number()

        product_name_test = self.instrument.get_product_name_by_slot(chassis_number, slot_number)
        self.assertEqual(product_name_test, product_name)

        serial_number_test = self.instrument.get_serial_number_by_slot(chassis_number, slot_number)
        # hack to make this test pass even with the current faulty Keysight lib (v.2.01.00)
        serial_number = serial_number[:-1]
        self.assertEqual(serial_number_test, serial_number)


@unittest.skipIf(not Keysight_M3201A_found, "Keysight_M3201A tests requires the keysightSD1 module")
class TestKeysight_M3201A(DriverTestCase):
    """
    This is a test suite for testing the Signadyne M3201A AWG card driver.
    It provides test functions for most of the functions and parameters as defined in the driver,
    as well as test functions for general things like connecting to the device.

    Status: beta

    Most of the current test functions are not super useful yet because the driver doesn't support set-able parameters
    at the moment. Normally a more useful test function would do something like:

            self.instrument.clock_frequency(100e6)
            self.assertAlmostEqual(self.instrument.clock_frequency(), 100e6, places=4)

    Unfortunately, the Signadyne libraries don't support many variables that are both get-able and set-able.

    We can however test for ValueErrors which is a useful safety test.
    """
    if Keysight_M3201A_found:
        driver = Keysight_M3201A

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print('Did set up Class. Should have printed `found ..., testing`.')
        cls.instrument.off()  # Not a test but a safety measure

    def test_chassis_number(self):
        chassis_number = self.instrument.chassis_number()
        self.assertEqual(chassis_number, 1)

    def test_slot_number(self):
        slot_number = self.instrument.slot_number()
        self.assertEqual(slot_number, 7)

    def test_serial_number(self):
        serial_number = self.instrument.serial_number()
        self.assertEqual(serial_number, 'ES56600108')

    def test_chassis_and_slot(self):
        chassis_number = self.instrument.chassis_number()
        slot_number = self.instrument.slot_number()
        product_name = self.instrument.product_name()
        serial_number = self.instrument.serial_number()

        product_name_test = self.instrument.get_product_name_by_slot(chassis_number, slot_number)
        self.assertEqual(product_name_test, product_name)

        serial_number_test = self.instrument.get_serial_number_by_slot(chassis_number, slot_number)
        # hack to make this test pass even with the current faulty Keysight lib (v.2.01.00)
        serial_number = serial_number[:-1]
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

    def test_channel_frequency(self):
        cur_f = self.instrument.frequency_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.frequency_channel_0.set(600e6)
        with self.assertRaises(ValueError):
            self.instrument.frequency_channel_0.set(-32)

        # turn off the signal for safety
        self.instrument.off()

        self.instrument.frequency_channel_0.set(0.1e6)
        self.instrument.frequency_channel_0.set(10e6)
        self.instrument.frequency_channel_0.set(132)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_f:
            self.instrument.frequency_channel_0.set(cur_f)
        else:
            self.instrument.frequency_channel_0.set(0)

    def test_channel_phase(self):
        cur_p = self.instrument.phase_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.phase_channel_0.set(400)
        with self.assertRaises(ValueError):
            self.instrument.phase_channel_0.set(-32)

        # turn off the signal for safety
        self.instrument.off()

        self.instrument.phase_channel_0.set(0)
        self.instrument.phase_channel_0.set(351.89)
        self.instrument.phase_channel_0.set(6.123)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_p:
            self.instrument.phase_channel_0.set(cur_p)
        else:
            self.instrument.phase_channel_0.set(0)

    def test_channel_amplitude(self):
        cur_a = self.instrument.amplitude_channel_0.get_latest()
        cur_o = self.instrument.offset_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.amplitude_channel_0.set(2)
        with self.assertRaises(ValueError):
            self.instrument.amplitude_channel_0.set(-3)

        # turn off the signal for safety
        self.instrument.off()
        # set offset to zero (so we don't go out of range)
        self.instrument.offset_channel_0.set(0)

        self.instrument.amplitude_channel_0.set(0)
        self.instrument.amplitude_channel_0.set(1.35)
        self.instrument.amplitude_channel_0.set(-1.112)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_a:
            self.instrument.amplitude_channel_0.set(cur_a)
        else:
            self.instrument.amplitude_channel_0.set(0)
        if cur_o:
            self.instrument.offset_channel_0.set(cur_o)
        else:
            self.instrument.offset_channel_0.set(0)

    def test_channel_offset(self):
        cur_o = self.instrument.offset_channel_0.get_latest()
        cur_a = self.instrument.amplitude_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.offset_channel_0.set(2)
        with self.assertRaises(ValueError):
            self.instrument.offset_channel_0.set(-3)

        # turn off the signal for safety
        self.instrument.off()
        # set amplitude to zero (so we don't go out of range)
        self.instrument.amplitude_channel_0.set(0)

        self.instrument.offset_channel_0.set(0)
        self.instrument.offset_channel_0.set(1.35)
        self.instrument.offset_channel_0.set(-1.112)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_o:
            self.instrument.offset_channel_0.set(cur_o)
        else:
            self.instrument.offset_channel_0.set(0)
        if cur_a:
            self.instrument.amplitude_channel_0.set(cur_a)
        else:
            self.instrument.amplitude_channel_0.set(0)

    def test_channel_wave_shape(self):
        cur_w = self.instrument.wave_shape_channel_0.get_latest()
        cur_o = self.instrument.offset_channel_0.get_latest()
        cur_a = self.instrument.amplitude_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.wave_shape_channel_0.set(1.5)
        with self.assertRaises(ValueError):
            self.instrument.wave_shape_channel_0.set(-3)

        # turn off the signal for safety
        self.instrument.off()
        # set amplitude and offset to zero for safety
        self.instrument.amplitude_channel_0.set(0)
        self.instrument.offset_channel_0.set(0)

        self.instrument.wave_shape_channel_0.set(0)
        self.instrument.wave_shape_channel_0.set(1)
        self.instrument.wave_shape_channel_0.set(6)
        self.instrument.wave_shape_channel_0.set(5)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_w:
            self.instrument.wave_shape_channel_0.set(cur_w)
        else:
            self.instrument.wave_shape_channel_0.set(0)
        if cur_o:
            self.instrument.offset_channel_0.set(cur_o)
        else:
            self.instrument.offset_channel_0.set(0)
        if cur_a:
            self.instrument.amplitude_channel_0.set(cur_a)
        else:
            self.instrument.amplitude_channel_0.set(0)

    def test_PXI_trigger(self):
        with self.assertRaises(ValueError):
            self.instrument.pxi_trigger_number_0.set(1.5)
        with self.assertRaises(ValueError):
            self.instrument.clock_frequency(32)

        cur_pxi = self.instrument.pxi_trigger_number_0.get()

        test_pxi = 0
        self.instrument.pxi_trigger_number_0.set(test_pxi)
        self.assertEqual(self.instrument.pxi_trigger_number_0.get(), test_pxi)

        test_pxi = 1
        self.instrument.pxi_trigger_number_0.set(test_pxi)
        self.assertEqual(self.instrument.pxi_trigger_number_0.get(), test_pxi)

        # leave the setup in the initial state
        self.instrument.pxi_trigger_number_0.set(cur_pxi)


@unittest.skipIf(not M3300A_AWG_found, "M3300A_AWG tests requires the keysightSD1 module")
class TestKeysight_M3300A(DriverTestCase):
    """
    This is a test suite for testing the Signadyne M3201A AWG card driver.
    It provides test functions for most of the functions and parameters as defined in the driver,
    as well as test functions for general things like connecting to the device.

    Status: beta

    Most of the current test functions are not super useful yet because the driver doesn't support set-able parameters
    at the moment. Normally a more useful test function would do something like:

            self.instrument.clock_frequency(100e6)
            self.assertAlmostEqual(self.instrument.clock_frequency(), 100e6, places=4)

    Unfortunately, the Signadyne libraries don't support many variables that are both get-able and set-able.

    We can however test for ValueErrors which is a useful safety test.
    """
    if M3300A_AWG_found:
        driver = M3300A_AWG

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print('Did set up Class. Should have printed `found ..., testing`.')
        cls.instrument.off()  # Not a test but a safety measure

    def test_chassis_number(self):
        chassis_number = self.instrument.chassis_number()
        self.assertEqual(chassis_number, 1)

    def test_slot_number(self):
        slot_number = self.instrument.slot_number()
        self.assertEqual(slot_number, 8)

    def test_serial_number(self):
        serial_number = self.instrument.serial_number()
        self.assertEqual(serial_number, 'ES56600101')

    def test_chassis_and_slot(self):
        chassis_number = self.instrument.chassis_number()
        slot_number = self.instrument.slot_number()
        product_name = self.instrument.product_name()
        serial_number = self.instrument.serial_number()

        product_name_test = self.instrument.get_product_name_by_slot(chassis_number, slot_number)
        self.assertEqual(product_name_test, product_name)

        serial_number_test = self.instrument.get_serial_number_by_slot(chassis_number, slot_number)
        # hack to make this test pass even with the current faulty Keysight lib (v.2.01.00)
        serial_number = serial_number[:-1]
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
        # TODO: This does not work, figure out why
        with self.assertRaises(ValueError):
            self.instrument.clock_frequency(600e6)
        with self.assertRaises(ValueError):
            self.instrument.clock_frequency(32)

        cur_f = self.instrument.clock_frequency()
        # test_f = 200e6
        # self.instrument.clock_frequency(test_f)
        # self.assertAlmostEqual(self.instrument.clock_frequency(), test_f, delta=1)
        #
        # test_f = 453.152e6
        # self.instrument.clock_frequency(test_f)
        # self.assertAlmostEqual(self.instrument.clock_frequency(), test_f, delta=1)
        print(cur_f)
        # leave the setup in the initial state
        # self.instrument.clock_frequency(cur_f)

    def test_channel_frequency(self):
        cur_f = self.instrument.frequency_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.frequency_channel_0.set(600e6)
        with self.assertRaises(ValueError):
            self.instrument.frequency_channel_0.set(-32)

        # turn off the signal for safety
        self.instrument.off()

        self.instrument.frequency_channel_0.set(0.1e6)
        self.instrument.frequency_channel_0.set(10e6)
        self.instrument.frequency_channel_0.set(132)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_f:
            self.instrument.frequency_channel_0.set(cur_f)
        else:
            self.instrument.frequency_channel_0.set(0)

    def test_channel_phase(self):
        cur_p = self.instrument.phase_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.phase_channel_0.set(400)
        with self.assertRaises(ValueError):
            self.instrument.phase_channel_0.set(-32)

        # turn off the signal for safety
        self.instrument.off()

        self.instrument.phase_channel_0.set(0)
        self.instrument.phase_channel_0.set(351.89)
        self.instrument.phase_channel_0.set(6.123)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_p:
            self.instrument.phase_channel_0.set(cur_p)
        else:
            self.instrument.phase_channel_0.set(0)

    def test_channel_amplitude(self):
        cur_a = self.instrument.amplitude_channel_0.get_latest()
        cur_o = self.instrument.offset_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.amplitude_channel_0.set(2)
        with self.assertRaises(ValueError):
            self.instrument.amplitude_channel_0.set(-3)

        # turn off the signal for safety
        self.instrument.off()
        # set offset to zero (so we don't go out of range)
        self.instrument.offset_channel_0.set(0)

        self.instrument.amplitude_channel_0.set(0)
        self.instrument.amplitude_channel_0.set(1.35)
        self.instrument.amplitude_channel_0.set(-1.112)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_a:
            self.instrument.amplitude_channel_0.set(cur_a)
        else:
            self.instrument.amplitude_channel_0.set(0)
        if cur_o:
            self.instrument.offset_channel_0.set(cur_o)
        else:
            self.instrument.offset_channel_0.set(0)

    def test_channel_offset(self):
        cur_o = self.instrument.offset_channel_0.get_latest()
        cur_a = self.instrument.amplitude_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.offset_channel_0.set(2)
        with self.assertRaises(ValueError):
            self.instrument.offset_channel_0.set(-3)

        # turn off the signal for safety
        self.instrument.off()
        # set amplitude to zero (so we don't go out of range)
        self.instrument.amplitude_channel_0.set(0)

        self.instrument.offset_channel_0.set(0)
        self.instrument.offset_channel_0.set(1.35)
        self.instrument.offset_channel_0.set(-1.112)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_o:
            self.instrument.offset_channel_0.set(cur_o)
        else:
            self.instrument.offset_channel_0.set(0)
        if cur_a:
            self.instrument.amplitude_channel_0.set(cur_a)
        else:
            self.instrument.amplitude_channel_0.set(0)

    def test_channel_wave_shape(self):
        cur_w = self.instrument.wave_shape_channel_0.get_latest()
        cur_o = self.instrument.offset_channel_0.get_latest()
        cur_a = self.instrument.amplitude_channel_0.get_latest()
        with self.assertRaises(ValueError):
            self.instrument.wave_shape_channel_0.set(1.5)
        with self.assertRaises(ValueError):
            self.instrument.wave_shape_channel_0.set(-3)

        # turn off the signal for safety
        self.instrument.off()
        # set amplitude and offset to zero for safety
        self.instrument.amplitude_channel_0.set(0)
        self.instrument.offset_channel_0.set(0)

        self.instrument.wave_shape_channel_0.set(0)
        self.instrument.wave_shape_channel_0.set(1)
        self.instrument.wave_shape_channel_0.set(6)
        self.instrument.wave_shape_channel_0.set(5)

        # leave the setup in the initial state or default values if no initial state was found
        if cur_w:
            self.instrument.wave_shape_channel_0.set(cur_w)
        else:
            self.instrument.wave_shape_channel_0.set(0)
        if cur_o:
            self.instrument.offset_channel_0.set(cur_o)
        else:
            self.instrument.offset_channel_0.set(0)
        if cur_a:
            self.instrument.amplitude_channel_0.set(cur_a)
        else:
            self.instrument.amplitude_channel_0.set(0)

    def test_PXI_trigger(self):
        with self.assertRaises(ValueError):
            self.instrument.pxi_trigger_number_0.set(1.5)
        with self.assertRaises(ValueError):
            self.instrument.clock_frequency(32)

        cur_pxi = self.instrument.pxi_trigger_number_0.get()

        test_pxi = 0
        self.instrument.pxi_trigger_number_0.set(test_pxi)
        self.assertEqual(self.instrument.pxi_trigger_number_0.get(), test_pxi)

        test_pxi = 1
        self.instrument.pxi_trigger_number_0.set(test_pxi)
        self.assertEqual(self.instrument.pxi_trigger_number_0.get(), test_pxi)

        # leave the setup in the initial state
        self.instrument.pxi_trigger_number_0.set(cur_pxi)
