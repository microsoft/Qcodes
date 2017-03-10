"""
Test suite for  instument.*
"""
from unittest import TestCase

from .instrument_mocks import DummyInstrument


class TestInstrument(TestCase):

    def setUp(self):
        self.instrument = DummyInstrument(
            name='testdummy', gates=['dac1', 'dac2', 'dac3'], server_name=None)

    def tearDown(self):
        # TODO (giulioungaretti) remove ( does nothing ?)
        pass

    def test_validate_function(self):
        instrument = self.instrument
        instrument.validate_status()  # test the instrument has valid values

        instrument.dac1._save_val(1000)  # overrule the validator
        with self.assertRaises(Exception):
            instrument.validate_status()

    def test_attr_access(self):
        instrument = self.instrument

        # test the instrument works
        instrument.dac1.set(10)
        val = instrument.dac1.get()
        self.assertEqual(val, 10)

        # close the instrument
        instrument.close()

        # make sure we can still print the instrument
        instrument.__repr__()

        # make sure the gate is removed
        self.assertEqual(hasattr(instrument, 'dac1'), False)
