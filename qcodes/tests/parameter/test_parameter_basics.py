import pytest

from qcodes.instrument.parameter import Parameter
import qcodes.utils.validators as vals
from .conftest import GettableParam


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
