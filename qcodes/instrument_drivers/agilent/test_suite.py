import unittest

instr = None


class TestAgilent_E8527D(unittest.TestCase):
    '''
    This is a test suite for testing the QuTech_ControlBox Instrument.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the com s are working.
    '''
    @classmethod
    def setUpClass(self):
        if instr is None:
            raise unittest.SkipTest('no instrument found')
        self.instr = instr
        self.instr.off()  # Not a test but a safety measure

    def test_firmware_version(self):
        v = self.instr.IDN.get()
        self.assertTrue(v.startswith('Agilent Technologies, E8257D,'))

    def test_on_off(self):
        self.instr.off()
        self.assertEqual(self.instr.status.get(), 'Off')

        self.instr.on()
        self.assertEqual(self.instr.status.get(), 'On')

        self.instr.status.set('off')
        self.assertEqual(self.instr.status.get(), 'Off')

        self.instr.status.set('On')
        self.assertEqual(self.instr.status.get(), 'On')

        # Ensure test ends with instr being off
        self.instr.off()
        self.assertEqual(self.instr.status.get(), 'Off')

        # This should raise an error because it is not a valid input
        with self.assertRaises(ValueError):
            self.instr.status.set('on24')

    def test_frequency(self):
        with self.assertRaises(ValueError):
            self.instr.frequency.set(32e9)
        with self.assertRaises(ValueError):
            self.instr.frequency.set(32)

        cur_f = self.instr.frequency.get()
        test_f = 2e9
        self.instr.frequency.set(test_f)
        self.assertEqual(self.instr.frequency.get(), test_f)

        test_f = 2.2435e9
        self.instr.frequency.set(test_f)
        self.assertEqual(self.instr.frequency.get(), test_f)

        # leave the setup in the initial state
        self.instr.frequency.set(cur_f)

    def test_power(self):
        with self.assertRaises(ValueError):
            self.instr.power.set(-150)
        with self.assertRaises(ValueError):
            self.instr.power.set(32)

        cur_val = self.instr.power.get()
        test_val = -18
        self.instr.power.set(test_val)
        self.assertEqual(self.instr.power.get(), test_val)

        test_val = -5
        self.instr.power.set(test_val)
        self.assertEqual(self.instr.power.get(), test_val)

        # leave the setup in the initial state
        self.instr.power.set(cur_val)

    def test_phase(self):
        with self.assertRaises(ValueError):
            self.instr.phase.set(-250)
        with self.assertRaises(ValueError):
            self.instr.phase.set(181)

        cur_val = self.instr.phase.get()
        test_val = 12
        self.instr.phase.set(test_val)
        self.assertAlmostEqual(self.instr.phase.get(), test_val, places=4)

        test_val = -80
        self.instr.phase.set(test_val)
        self.assertAlmostEqual(self.instr.phase.get(), test_val, places=4)

        # leave the setup in the initial state
        self.instr.phase.set(cur_val)
