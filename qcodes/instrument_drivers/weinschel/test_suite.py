import unittest

instr = None


class stepped_attenuator(unittest.TestCase):
    '''
    This is a test suite for testing the weinschel/aeroflex stepped attenuator.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the com s are working.
    '''
    @classmethod
    def setUpClass(self):
        self.instr = instr

    def test_firmware_version(self):
        v = self.instr.IDN.get()
        self.assertTrue(v.startswith('API Weinschel, 8320,'))

    def test_attenuation(self):
        curr_val = self.instr.attenuation.get()

        for v in [0, 32, 60]:
            self.instr.attenuation.set(v)
            self.assertEqual(self.instr.attenuation.get(), v)

        for v in [-2, 3, 61]:
            with self.assertRaises(ValueError):
                self.instr.attenuation.set(v)
