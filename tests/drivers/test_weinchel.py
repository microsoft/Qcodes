from qcodes.extensions import DriverTestCase
from qcodes.instrument_drivers.weinschel import Weinschel8320


class TestWeinschel8320(DriverTestCase):
    """
    This is a test suite for testing the weinschel/aeroflex stepped attenuator.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the com s are working.
    """

    driver = Weinschel8320

    def test_firmware_version(self) -> None:
        v = self.instrument.IDN.get()
        self.assertTrue(v.startswith("API Weinschel, 8320,"))

    def test_attenuation(self) -> None:
        curr_val = self.instrument.attenuation.get()

        for v in [0, 32, 60]:
            self.instrument.attenuation.set(v)
            self.assertEqual(self.instrument.attenuation.get(), v)

        for v in [-2, 3, 61]:
            with self.assertRaises(ValueError):
                self.instrument.attenuation.set(v)
        self.instrument.attenuation.set(curr_val)
