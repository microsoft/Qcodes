from qcodes.instrument_drivers.test import DriverTestCase
from qcodes.instrument_drivers.weinschel.Weinschel_8320 import Weinschel_8320


class TestWeinschel_8320(DriverTestCase):
    '''
    This is a test suite for testing the weinschel/aeroflex stepped attenuator.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the com s are working.
    '''
    driver = Weinschel_8320

    def test_firmware_version(self):
        v = self.instrument.IDN.get()
        self.assertTrue(v.startswith('API Weinschel, 8320,'))

    def test_attenuation(self):
        curr_val = self.instrument.attenuation.get()

        for v in [0, 32, 60]:
            self.instrument.attenuation.set(v)
            self.assertEqual(self.instrument.attenuation.get(), v)

        for v in [-2, 3, 61]:
            with self.assertRaises(ValueError):
                self.instrument.attenuation.set(v)
        self.instrument.attenuation.set(curr_val)
