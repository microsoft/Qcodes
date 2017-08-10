"""
Test suite for  instument.*
"""
from unittest import TestCase
from qcodes.instrument.base import Instrument
from .instrument_mocks import DummyInstrument, MockParabola
from qcodes.instrument.parameter import ManualParameter
import gc


class TestInstrument(TestCase):

    def setUp(self):
        self.instrument = DummyInstrument(
            name='testdummy', gates=['dac1', 'dac2', 'dac3'])
        self.instrument2 = MockParabola("parabola")

    def tearDown(self):
        # force gc run
        self.instrument.close()
        self.instrument2.close()
        del self.instrument
        del self.instrument2
        gc.collect()

    def test_validate_function(self):
        instrument = self.instrument
        instrument.validate_status()  # test the instrument has valid values

        instrument.dac1._save_val(1000)  # overrule the validator
        with self.assertRaises(Exception):
            instrument.validate_status()

    def test_check_instances(self):
        with self.assertRaises(KeyError):
            DummyInstrument(name='testdummy', gates=['dac1', 'dac2', 'dac3'])

        self.assertEqual(Instrument.instances(), [])
        self.assertEqual(DummyInstrument.instances(), [self.instrument])
        self.assertEqual(self.instrument.instances(), [self.instrument])


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

    def test_repr(self):
        idn = dict(zip(('vendor', 'model', 'serial', 'firmware'),
                       [None, self.instrument.name, None, None]))
        self.assertEqual(idn, self.instrument.get_idn())

    def test_add_remove_f_p(self):
        with self.assertRaises(KeyError):
                self.instrument.add_parameter('dac1', get_cmd='foo')
        self.instrument.add_function('function', call_cmd='foo')
        with self.assertRaises(KeyError):
                self.instrument.add_function('function', call_cmd='foo')

        self.instrument.add_function('dac1', call_cmd='foo')
        # test custom __get_attr__
        self.instrument['function']
        # by desgin one gets the parameter if a function exists and has same
        # name
        dac1 = self.instrument['dac1']
        self.assertTrue(isinstance(dac1, ManualParameter))

    def test_instances(self):
        instruments = [self.instrument, self.instrument2]
        for instrument in instruments:
            for other_instrument in instruments:
                instances = instrument.instances()
                # check that each instrument is in only its own
                if other_instrument is instrument:
                    self.assertIn(instrument, instances)
                else:
                    self.assertNotIn(other_instrument, instances)

                # check that we can find each instrument from any other
                self.assertEqual(
                    instrument,
                    other_instrument.find_instrument(instrument.name))

            # check that we can find this instrument from the base class
            self.assertEqual(instrument,
                             Instrument.find_instrument(instrument.name))

    def test_snapshot_value(self):
        self.instrument.add_parameter('has_snapshot_value',
                                      parameter_class=ManualParameter,
                                      initial_value=42,
                                      snapshot_value=True)
        self.instrument.add_parameter('no_snapshot_value',
                                      parameter_class=ManualParameter,
                                      initial_value=42,
                                      snapshot_value=False)

        snapshot = self.instrument.snapshot()

        self.assertIn('value', snapshot['parameters']['has_snapshot_value'])
        self.assertEqual(42,
                         snapshot['parameters']['has_snapshot_value']['value'])
        self.assertNotIn('value', snapshot['parameters']['no_snapshot_value'])