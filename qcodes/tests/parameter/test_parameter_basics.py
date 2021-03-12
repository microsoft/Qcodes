import pytest

from qcodes.instrument.parameter import Parameter, _BaseParameter
import qcodes.utils.validators as vals
from qcodes.instrument.function import Function
from .conftest import GettableParam, blank_instruments, named_instrument


def test_no_name():
    with pytest.raises(TypeError):
        Parameter()


def test_default_attributes():
    # Test the default attributes, providing only a name
    name = 'repetitions'
    p = GettableParam(name, vals=vals.Numbers())
    assert p.name == name
    assert p.label == name
    assert p.unit == ''
    assert str(p) == name

    # default validator is all numbers
    p.validate(-1000)
    with pytest.raises(TypeError):
        p.validate('not a number')

    # docstring exists, even without providing one explicitly
    assert name in p.__doc__

    # test snapshot_get by looking at _get_count
    # by default, snapshot_get is True, hence we expect ``get`` to be called
    assert p._get_count == 0
    snap = p.snapshot(update=True)
    assert p._get_count == 1
    snap_expected = {
        'name': name,
        'label': name,
        'unit': '',
        'value': 42,
        'raw_value': 42,
        'vals': repr(vals.Numbers())
    }
    for k, v in snap_expected.items():
        assert snap[k] == v
    assert snap['ts'] is not None


def test_explicit_attributes():
    # Test the explicit attributes, providing everything we can
    name = 'volt'
    label = 'Voltage'
    unit = 'V'
    docstring = 'DOCS!'
    metadata = {'gain': 100}
    p = GettableParam(name, label=label, unit=unit,
                      vals=vals.Numbers(5, 10), docstring=docstring,
                      snapshot_get=False, metadata=metadata)

    assert p.name == name
    assert p.label == label
    assert p.unit == unit
    assert str(p) == name

    with pytest.raises(ValueError):
        p.validate(-1000)
    p.validate(6)
    with pytest.raises(TypeError):
        p.validate('not a number')

    assert name in p.__doc__
    assert docstring in p.__doc__

    # test snapshot_get by looking at _get_count
    assert p._get_count == 0
    # Snapshot should not perform get since snapshot_get is False
    snap = p.snapshot(update=True)
    assert p._get_count == 0
    snap_expected = {
        'name': name,
        'label': label,
        'unit': unit,
        'vals': repr(vals.Numbers(5, 10)),
        'value': None,
        'raw_value': None,
        'ts': None,
        'metadata': metadata
    }
    for k, v in snap_expected.items():
        assert snap[k] == v

    # attributes only available in MultiParameter
    for attr in ['names', 'labels', 'setpoints', 'setpoint_names',
                 'setpoint_labels', 'full_names']:
        assert not hasattr(p, attr)


def test_has_set_get():
    # Create parameter that has no set_cmd, and get_cmd returns last value
    gettable_parameter = Parameter('one', set_cmd=False, get_cmd=None)
    assert hasattr(gettable_parameter, 'get')
    assert gettable_parameter.gettable
    assert not hasattr(gettable_parameter, 'set')
    assert not gettable_parameter.settable
    with pytest.raises(NotImplementedError):
        gettable_parameter(1)
    # Initial value is None if not explicitly set
    assert gettable_parameter() is None
    # Assert the ``cache.set`` still works for non-settable parameter
    gettable_parameter.cache.set(1)
    assert gettable_parameter() == 1

    # Create parameter that saves value during set, and has no get_cmd
    settable_parameter = Parameter('two', set_cmd=None, get_cmd=False)
    assert not hasattr(settable_parameter, 'get')
    assert not settable_parameter.gettable
    assert hasattr(settable_parameter, 'set')
    assert settable_parameter.settable
    with pytest.raises(NotImplementedError):
        settable_parameter()
    settable_parameter(42)

    settable_gettable_parameter = Parameter('three', set_cmd=None, get_cmd=None)
    assert hasattr(settable_gettable_parameter, 'set')
    assert settable_gettable_parameter.settable
    assert hasattr(settable_gettable_parameter, 'get')
    assert settable_gettable_parameter.gettable
    assert settable_gettable_parameter() is None
    settable_gettable_parameter(22)
    assert settable_gettable_parameter() == 22


def test_str_representation():
    # three cases where only name gets used for full_name
    for instrument in blank_instruments:
        p = Parameter(name='fred')
        p._instrument = instrument
        assert str(p) == 'fred'

    # and finally an instrument that really has a name
    p = Parameter(name='wilma')
    p._instrument = named_instrument
    assert str(p) == 'astro_wilma'


def test_bad_name():
    with pytest.raises(ValueError):
        Parameter('p with space')
    with pytest.raises(ValueError):
        Parameter('⛄')
    with pytest.raises(ValueError):
        Parameter('1')


def test_set_via_function():
    # not a use case we want to promote, but it's there...
    p = Parameter('test', get_cmd=None, set_cmd=None)

    def doubler(x):
        p.set(x * 2)

    f = Function('f', call_cmd=doubler, args=[vals.Numbers(-10, 10)])

    f(4)
    assert p.get() == 8
    with pytest.raises(ValueError):
        f(20)


def test_unknown_args_to_baseparameter_raises():
    """
    Passing an unknown kwarg to _BaseParameter should trigger a TypeError
    """
    with pytest.raises(TypeError):
        _ = _BaseParameter(name='Foo',
                           instrument=None,
                           snapshotable=False)
