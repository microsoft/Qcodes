"""
Test suite for instument.base.*
"""

import gc
import weakref
import io
import contextlib
from unittest import TestCase

from qcodes.instrument.base import Instrument, InstrumentBase, find_or_create_instrument
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.function import Function

from .instrument_mocks import DummyInstrument, MockParabola, MockMetaParabola



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

        instrument.dac1.cache._value = 1000  # overrule the validator
        instrument.dac1.cache._raw_value = 1000  # overrule the validator
        with self.assertRaises(Exception):
            instrument.validate_status()

    def test_check_instances(self):
        with self.assertRaises(KeyError) as cm:
            DummyInstrument(name='testdummy', gates=['dac1', 'dac2', 'dac3'])
        assert str(cm.exception) == "'Another instrument has the name: testdummy'"

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

        # make sure the name property still exists
        assert hasattr(instrument, 'name')
        assert instrument.name == 'testdummy'

        # make sure we can still print the instrument
        assert 'testdummy' in instrument.__repr__()
        assert 'testdummy' in str(instrument)

        # make sure the gate is removed
        self.assertEqual(hasattr(instrument, 'dac1'), False)

    def test_get_idn(self):
        idn = dict(zip(('vendor', 'model', 'serial', 'firmware'),
                       [None, self.instrument.name, None, None]))
        self.assertEqual(idn, self.instrument.get_idn())

    def test_repr(self):
        assert repr(self.instrument) == '<DummyInstrument: testdummy>'

    def test_add_remove_f_p(self):
        with self.assertRaises(KeyError) as cm:
            self.instrument.add_parameter('dac1', get_cmd='foo')
        assert str(cm.exception) == "'Duplicate parameter name dac1'"

        self.instrument.add_function('function', call_cmd='foo')

        with self.assertRaises(KeyError) as cm:
            self.instrument.add_function('function', call_cmd='foo')
        assert str(cm.exception) == "'Duplicate function name function'"

        self.instrument.add_function('dac1', call_cmd='foo')

        # test custom __get_attr__ for functions
        fcn = self.instrument['function']
        self.assertTrue(isinstance(fcn, Function))
        # by design, one gets the parameter if a function exists 
        # and has same name
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

    def test_is_valid(self):
        assert Instrument.is_valid(self.instrument)
        self.instrument.close()
        assert not Instrument.is_valid(self.instrument)

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

    def test_meta_instrument(self):
        mock_instrument = MockMetaParabola("mock_parabola", self.instrument2)

        # Check that the mock instrument can return values
        self.assertEqual(mock_instrument.parabola(), self.instrument2.parabola())
        mock_instrument.x(1)
        mock_instrument.y(2)
        self.assertEqual(mock_instrument.parabola(), self.instrument2.parabola())
        self.assertNotEqual(mock_instrument.parabola(), 0)

        # Add a scaling factor
        mock_instrument.gain(2)
        self.assertEqual(mock_instrument.parabola(), self.instrument2.parabola()*2)

        # Check snapshots
        snap = mock_instrument.snapshot(update=True)
        self.assertIn("parameters", snap)
        self.assertIn("gain", snap["parameters"])
        self.assertEqual(snap["parameters"]["gain"]["value"], 2)

        # Check printable snapshot
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            mock_instrument.print_readable_snapshot()
        readable_snap = f.getvalue()

        # Line length satisfied
        self.assertTrue(all(len(line) <= 80 for line in readable_snap.splitlines()))
        # Gain is included in output with correct value
        self.assertRegex(readable_snap, r"gain[ \t]+:[ \t]+2")


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


def test_instrument_metadata():
    metadatadict = {1: "data", "some": "data"}
    instrument = DummyInstrument(name='testdummy', gates=['dac1', 'dac2', 'dac3'],
                                 metadata=metadatadict)
    assert instrument.metadata == metadatadict


def test_instrumentbase_metadata():
    metadatadict = {1: "data", "some": "data"}
    instrument = InstrumentBase('instr', metadata=metadatadict)
    assert instrument.metadata == metadatadict


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
