import unittest

source = None


class mw_source(unittest.TestCase):
    '''
    This is a test suite for testing the QuTech_ControlBox Instrument.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the com s are working.
    '''
    @classmethod
    def setUpClass(self):
        self.source = source
        self.source.off()  # Not a test but a safety measure

    def test_firmware_version(self):
        v = self.source.IDN.get()
        self.assertTrue(v.startswith('Agilent Technologies, E8257D,'))

    def test_on_off(self):
        self.source.off()
        self.assertEqual(self.source.status.get(), 'Off')

        self.source.on()
        self.assertEqual(self.source.status.get(), 'On')

        self.source.status.set('off')
        self.assertEqual(self.source.status.get(), 'Off')

        self.source.status.set('On')
        self.assertEqual(self.source.status.get(), 'On')

        # Ensure test ends with source being off
        self.source.off()
        self.assertEqual(self.source.status.get(), 'Off')

        # This should raise an error because it is not a valid input
        with self.assertRaises(ValueError):
            self.source.status.set('on24')

    def test_frequency(self):
        with self.assertRaises(ValueError):
            self.source.frequency.set(32e9)
        with self.assertRaises(ValueError):
            self.source.frequency.set(32)

        cur_f = self.source.frequency.get()
        test_f = 2e9
        self.source.frequency.set(test_f)
        self.assertEqual(self.source.frequency.get(), test_f)

        test_f = 2.2435e9
        self.source.frequency.set(test_f)
        self.assertEqual(self.source.frequency.get(), test_f)

        # leave the setup in the initial state
        self.source.frequency.set(cur_f)

    def test_power(self):
        with self.assertRaises(ValueError):
            self.source.power.set(-150)
        with self.assertRaises(ValueError):
            self.source.power.set(32)

        cur_val = self.source.power.get()
        test_val = -18
        self.source.power.set(test_val)
        self.assertEqual(self.source.power.get(), test_val)

        test_val = -5
        self.source.power.set(test_val)
        self.assertEqual(self.source.power.get(), test_val)

        # leave the setup in the initial state
        self.source.power.set(cur_val)

    def test_phase(self):
        with self.assertRaises(ValueError):
            self.source.phase.set(-250)
        with self.assertRaises(ValueError):
            self.source.phase.set(181)

        cur_val = self.source.phase.get()
        test_val = 12
        self.source.phase.set(test_val)
        self.assertAlmostEqual(self.source.phase.get(), test_val, places=4)

        test_val = -80
        self.source.phase.set(test_val)
        self.assertAlmostEqual(self.source.phase.get(), test_val, places=4)

        # leave the setup in the initial state
        self.source.phase.set(cur_val)
