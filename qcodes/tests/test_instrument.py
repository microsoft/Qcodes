"""
Test suite for  instument.*
"""
import weakref
from unittest import TestCase
from qcodes.instrument.base import Instrument, InstrumentBase, find_or_create_instrument
from .instrument_mocks import DummyInstrument, MockParabola
from qcodes.instrument.parameter import Parameter
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
        self.assertTrue(isinstance(dac1, Parameter))

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
                                      parameter_class=Parameter,
                                      initial_value=42,
                                      snapshot_value=True,
                                      get_cmd=None, set_cmd=None)
        self.instrument.add_parameter('no_snapshot_value',
                                      parameter_class=Parameter,
                                      initial_value=42,
                                      snapshot_value=False,
                                      get_cmd=None, set_cmd=None)

        snapshot = self.instrument.snapshot()

        self.assertIn('name', snapshot)
        self.assertEqual('testdummy', snapshot['name'])

        self.assertIn('value', snapshot['parameters']['has_snapshot_value'])
        self.assertEqual(42,
                         snapshot['parameters']['has_snapshot_value']['value'])
        self.assertNotIn('value', snapshot['parameters']['no_snapshot_value'])


class TestFindOrCreateInstrument(TestCase):
    """Tests for find_or_create_instrument function"""

    def setUp(self):
        Instrument.close_all()

    def tearDown(self):
        Instrument.close_all()

    def test_find(self):
        """Test finding an existing instrument"""
        instr = DummyInstrument(
            name='instr', gates=['dac1', 'dac2', 'dac3'])

        instr_2 = find_or_create_instrument(
            DummyInstrument, name='instr', gates=['dac1', 'dac2', 'dac3'])

        self.assertEqual(instr_2, instr)
        self.assertEqual(instr_2.name, instr.name)

        instr.close()

    def test_find_same_name_but_different_class(self):
        """Test finding an existing instrument with different class"""
        instr = DummyInstrument(
            name='instr', gates=['dac1', 'dac2', 'dac3'])

        class GammyInstrument(Instrument):
            some_other_attr = 25

        # Find an instrument with the same name but of different class
        with self.assertRaises(TypeError) as cm:
            _ = find_or_create_instrument(
                GammyInstrument, name='instr', gates=['dac1', 'dac2', 'dac3'])

        self.assertEqual("Instrument instr is <class "
                         "'qcodes.tests.instrument_mocks.DummyInstrument'> but "
                         "<class 'qcodes.tests.test_instrument"
                         ".TestFindOrCreateInstrument"
                         ".test_find_same_name_but_different_class.<locals>"
                         ".GammyInstrument'> was requested",
                         str(cm.exception))
        instr.close()

    def test_create(self):
        """Test creating an instrument that does not yet exist"""
        instr = find_or_create_instrument(
            DummyInstrument, name='instr', gates=['dac1', 'dac2', 'dac3'])

        self.assertEqual('instr', instr.name)

        instr.close()

    def test_other_exception(self):
        """Test an unexpected exception occurred during finding instrument"""
        with self.assertRaises(TypeError) as cm:
            # in order to raise an unexpected exception, and make sure it is
            # passed through the call stack, let's pass an empty dict instead
            # of a string with instrument name
            _ = find_or_create_instrument(DummyInstrument, {})
        self.assertEqual(str(cm.exception), "unhashable type: 'dict'")

    def test_recreate(self):
        """Test the case when instrument needs to be recreated"""
        instr = DummyInstrument(
            name='instr', gates=['dac1', 'dac2', 'dac3'])
        instr_ref = weakref.ref(instr)

        self.assertListEqual(
            ['instr'], list(Instrument._all_instruments.keys()))

        instr_2 = find_or_create_instrument(
            DummyInstrument, name='instr', gates=['dac1', 'dac2'],
            recreate=True
        )
        instr_2_ref = weakref.ref(instr_2)

        self.assertListEqual(
            ['instr'], list(Instrument._all_instruments.keys()))

        self.assertIn(instr_2_ref, Instrument._all_instruments.values())
        self.assertNotIn(instr_ref, Instrument._all_instruments.values())

        instr_2.close()


class TestInstrumentBase(TestCase):
    """
    This class contains tests that are relevant to the InstrumentBase class.
    """

    def test_snapshot_and_meta_attrs(self):
        """Test snapshot of InstrumentBase contains _meta_attrs attributes"""
        instr = InstrumentBase('instr')

        self.assertEqual(instr.name, 'instr')

        self.assertTrue(hasattr(instr, '_meta_attrs'))
        self.assertListEqual(instr._meta_attrs, ['name'])

        snapshot = instr.snapshot()

        self.assertIn('name', snapshot)
        self.assertEqual('instr', snapshot['name'])

        self.assertIn('__class__', snapshot)
        self.assertIn('InstrumentBase', snapshot['__class__'])
