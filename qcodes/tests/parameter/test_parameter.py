"""
Test suite for parameter
"""
from unittest import TestCase
import pytest

from qcodes.instrument.parameter import Parameter, _BaseParameter
import qcodes.utils.validators as vals
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.utils.helpers import create_on_off_val_mapping

from .conftest import (OverwriteGetParam, OverwriteSetParam,
                       GetSetRawParameter)



class TestStandardParam(TestCase):
    def set_p(self, val):
        self._p = val

    def set_p_prefixed(self, val):
        self._p = 'PVAL: {:d}'.format(val)

    def strip_prefix(self, val):
        return int(val[6:])

    def get_p(self):
        return self._p

    def parse_set_p(self, val):
        return '{:d}'.format(val)

    def test_param_cmd_with_parsing(self):
        p = Parameter('p_int', get_cmd=self.get_p, get_parser=int,
                              set_cmd=self.set_p, set_parser=self.parse_set_p)

        p(5)
        self.assertEqual(self._p, '5')
        self.assertEqual(p(), 5)

        p.cache.set(7)
        self.assertEqual(p.get_latest(), 7)
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, '5')
        self.assertEqual(p(), 5)
        self.assertEqual(p.get_latest(), 5)

    def test_settable(self):
        p = Parameter('p', set_cmd=self.set_p, get_cmd=False)

        p(10)
        self.assertEqual(self._p, 10)
        with self.assertRaises(NotImplementedError):
            p()

        self.assertTrue(hasattr(p, 'set'))
        self.assertTrue(p.settable)
        self.assertFalse(hasattr(p, 'get'))
        self.assertFalse(p.gettable)

        # For settable-only parameters, using ``cache.set`` may not make
        # sense, nevertheless, it works
        p.cache.set(7)
        self.assertEqual(p.get_latest(), 7)

    def test_gettable(self):
        p = Parameter('p', get_cmd=self.get_p)
        self._p = 21

        self.assertEqual(p(), 21)
        self.assertEqual(p.get(), 21)

        with self.assertRaises(NotImplementedError):
            p(10)

        self.assertTrue(hasattr(p, 'get'))
        self.assertTrue(p.gettable)
        self.assertFalse(hasattr(p, 'set'))
        self.assertFalse(p.settable)

        p.cache.set(7)
        self.assertEqual(p.get_latest(), 7)
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, 21)
        self.assertEqual(p(), 21)
        self.assertEqual(p.get_latest(), 21)

    def test_val_mapping_basic(self):
        p = Parameter('p', set_cmd=self.set_p, get_cmd=self.get_p,
                      val_mapping={'off': 0, 'on': 1},
                      vals=vals.Enum('off', 'on'))

        p('off')
        self.assertEqual(self._p, 0)
        self.assertEqual(p(), 'off')

        self._p = 1
        self.assertEqual(p(), 'on')

        # implicit mapping to ints
        self._p = '0'
        self.assertEqual(p(), 'off')

        # unrecognized response
        self._p = 2
        with self.assertRaises(KeyError):
            p()

        self._p = 1  # for further testing

        p.cache.set('off')
        self.assertEqual(p.get_latest(), 'off')
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, 1)
        self.assertEqual(p(), 'on')
        self.assertEqual(p.get_latest(), 'on')

    def test_val_mapping_with_parsers(self):
        # set_parser with val_mapping
        Parameter('p', set_cmd=self.set_p, get_cmd=self.get_p,
                  val_mapping={'off': 0, 'on': 1},
                  set_parser=self.parse_set_p)

        # get_parser with val_mapping
        p = Parameter('p', set_cmd=self.set_p_prefixed,
                      get_cmd=self.get_p, get_parser=self.strip_prefix,
                      val_mapping={'off': 0, 'on': 1},
                      vals=vals.Enum('off', 'on'))

        p('off')
        self.assertEqual(self._p, 'PVAL: 0')
        self.assertEqual(p(), 'off')

        self._p = 'PVAL: 1'
        self.assertEqual(p(), 'on')

        p.cache.set('off')
        self.assertEqual(p.get_latest(), 'off')
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, 'PVAL: 1')
        self.assertEqual(p(), 'on')
        self.assertEqual(p.get_latest(), 'on')

    def test_on_off_val_mapping(self):
        instrument_value_for_on = 'on_'
        instrument_value_for_off = 'off_'

        parameter_return_value_for_on = True
        parameter_return_value_for_off = False

        p = Parameter('p', set_cmd=self.set_p, get_cmd=self.get_p,
                      val_mapping=create_on_off_val_mapping(
                          on_val=instrument_value_for_on,
                          off_val=instrument_value_for_off))

        test_data = [(instrument_value_for_on,
                      parameter_return_value_for_on,
                      ('On', 'on', 'ON', 1, True)),
                     (instrument_value_for_off,
                      parameter_return_value_for_off,
                      ('Off', 'off', 'OFF', 0, False))]

        for instr_value, parameter_return_value, inputs in test_data:
            for inp in inputs:
                # Setting parameter with any of the `inputs` is allowed
                p(inp)
                # For any value from the `inputs`, what gets send to the
                # instrument is on_val/off_val which are specified in
                # `create_on_off_val_mapping`
                self.assertEqual(self._p, instr_value)
                # When getting a value of the parameter, only specific
                # values are returned instead of `inputs`
                self.assertEqual(p(), parameter_return_value)


class TestManualParameterValMapping(TestCase):
    def setUp(self):
        self.instrument = DummyInstrument('dummy_holder')

    def tearDown(self):
        self.instrument.close()
        del self.instrument


    def test_val_mapping(self):
        self.instrument.add_parameter('myparameter', set_cmd=None, get_cmd=None, val_mapping={'A': 0, 'B': 1})
        self.instrument.myparameter('A')
        assert self.instrument.myparameter() == 'A'
        assert self.instrument.myparameter() == 'A'
        assert self.instrument.myparameter.raw_value == 0


def test_parameter_with_overwritten_get_raises():
    """
    Test that creating a parameter that overwrites get and set raises runtime errors
    """

    with pytest.raises(RuntimeError) as record:
        a = OverwriteGetParam(name='foo')
    assert "Overwriting get in a subclass of _BaseParameter: foo is not allowed." == str(record.value)


def test_parameter_with_overwritten_set_raises():
    """
    Test that creating a parameter that overwrites get and set raises runtime errors
    """
    with pytest.raises(RuntimeError) as record:
        a = OverwriteSetParam(name='foo')
    assert "Overwriting set in a subclass of _BaseParameter: foo is not allowed." == str(record.value)


def test_unknown_args_to_baseparameter_warns():
    """
    Passing an unknown kwarg to _BaseParameter should trigger a warning
    """
    with pytest.warns(Warning):
        a = _BaseParameter(name='Foo',
                           instrument=None,
                           snapshotable=False)


@pytest.mark.parametrize("get_cmd, set_cmd", [(False, False), (False, None), (None, None), (None, False),
                                              (lambda: 1, lambda x: x)])
def test_gettable_settable_attributes_with_get_set_cmd(get_cmd, set_cmd):
    a = Parameter(name='foo',
                  get_cmd=get_cmd,
                  set_cmd=set_cmd)
    expected_gettable = get_cmd is not False
    expected_settable = set_cmd is not False

    assert a.gettable is expected_gettable
    assert a.settable is expected_settable


@pytest.mark.parametrize("baseclass", [_BaseParameter, Parameter])
def test_gettable_settable_attributes_with_get_set_raw(baseclass):
    """Test that parameters that have get_raw,set_raw are
    listed as gettable/settable and reverse."""

    class GetSetParam(baseclass):
        def __init__(self, *args, initial_value=None, **kwargs):
            self._value = initial_value
            super().__init__(*args, **kwargs)

        def get_raw(self):
            return self._value

        def set_raw(self, value):
            self._value = value

    a = GetSetParam('foo', instrument=None, initial_value=1)

    assert a.gettable is True
    assert a.settable is True

    b = _BaseParameter('foo', None)

    assert b.gettable is False
    assert b.settable is False


@pytest.mark.parametrize("working_get_cmd", (False, None))
@pytest.mark.parametrize("working_set_cmd", (False, None))
def test_get_raw_and_get_cmd_raises(working_get_cmd, working_set_cmd):
    with pytest.raises(TypeError, match="get_raw"):
        GetSetRawParameter(name="param1", get_cmd="GiveMeTheValue", set_cmd=working_set_cmd)
    with pytest.raises(TypeError, match="set_raw"):
        GetSetRawParameter(name="param2", set_cmd="HereIsTheValue {}", get_cmd=working_get_cmd)
    GetSetRawParameter("param3", get_cmd=working_get_cmd, set_cmd=working_set_cmd)


def test_get_on_parameter_marked_as_non_gettable_raises():
    a = Parameter("param")
    a._gettable = False
    with pytest.raises(TypeError, match="Trying to get a parameter that is not gettable."):
        a.get()


def test_set_on_parameter_marked_as_non_settable_raises():
    a = Parameter("param", set_cmd=None)
    a.set(2)
    assert a.get() == 2
    a._settable = False
    with pytest.raises(TypeError, match="Trying to set a parameter that is not settable."):
        a.set(1)
    assert a.get() == 2
