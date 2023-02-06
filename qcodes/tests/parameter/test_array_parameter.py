from typing import Any

import pytest

from qcodes.parameters import ArrayParameter, ParamRawDataType

from .conftest import blank_instruments, named_instrument


class SimpleArrayParam(ArrayParameter):
    def __init__(self, return_val: ParamRawDataType, *args: Any, **kwargs: Any):
        self._return_val = return_val
        self._get_count = 0
        super().__init__(*args, **kwargs)

    def get_raw(self) -> ParamRawDataType:
        self._get_count += 1
        return self._return_val


class SettableArray(SimpleArrayParam):
    # this is not allowed - just created to raise an error in the test below
    def set_raw(self, value: Any) -> None:
        self.v = value


def test_default_attributes() -> None:
    name = 'array_param'
    shape = (2, 3)
    p = SimpleArrayParam([[1, 2, 3], [4, 5, 6]], name, shape)

    assert p.name == name
    assert p.shape == shape

    assert p.label == name
    assert p.unit == ''
    assert p.setpoints is None
    assert p.setpoint_names is None
    assert p.setpoint_labels is None

    assert str(p) == name

    assert p._get_count == 0
    snap = p.snapshot(update=True)
    assert p._get_count == 0
    snap_expected = {
        'name': name,
        'label': name,
        'unit': ''
    }
    for k, v in snap_expected.items():
        assert snap[k] == v
    assert 'value' not in snap
    assert 'raw_value' not in snap
    assert snap['ts'] is None

    assert p.__doc__ is not None
    assert name in p.__doc__


def test_explicit_attributes() -> None:
    name = 'tiny_array'
    shape = (2,)
    label = 'it takes two to tango'
    unit = 'steps'
    setpoints = [(0, 1)]
    setpoint_names = ['sp_index']
    setpoint_labels = ['Setpoint Label']
    docstring = 'Whats up Doc?'
    metadata = {'size': 2}
    p = SimpleArrayParam([6, 7], name, shape, label=label, unit=unit,
                         setpoints=setpoints,
                         setpoint_names=setpoint_names,
                         setpoint_labels=setpoint_labels,
                         docstring=docstring, snapshot_value=True,
                         metadata=metadata)

    assert p.name == name
    assert p.shape == shape
    assert p.label == label
    assert p.unit == unit
    assert p.setpoints == setpoints
    assert p.setpoint_names == setpoint_names
    assert p.setpoint_full_names == setpoint_names
    assert p.setpoint_labels == setpoint_labels

    assert p._get_count == 0
    snap = p.snapshot(update=True)
    assert p._get_count == 1
    snap_expected = {
        'name': name,
        'label': label,
        'unit': unit,
        'setpoint_names': setpoint_names,
        'setpoint_labels': setpoint_labels,
        'metadata': metadata,
        'value': [6, 7],
        'raw_value': [6, 7]
    }
    for k, v in snap_expected.items():
        assert snap[k] == v
    assert snap['ts'] is not None

    assert p.__doc__ is not None
    assert name in p.__doc__
    assert docstring in p.__doc__


def test_has_set_get() -> None:
    name = 'array_param'
    shape = (3,)
    with pytest.raises(AttributeError):
        ArrayParameter(name, shape)

    p = SimpleArrayParam([1, 2, 3], name, shape)

    assert hasattr(p, 'get')
    assert p.gettable
    assert not hasattr(p, 'set')
    assert not p.settable

    # Yet, it's possible to set the cached value
    p.cache.set([6, 7, 8])
    assert p.get_latest() == [6, 7, 8]
    # However, due to the implementation of this ``SimpleArrayParam``
    # test parameter it's ``get`` call will return the originally passed
    # list
    assert p.get() == [1, 2, 3]
    assert p.get_latest() == [1, 2, 3]

    with pytest.raises(AttributeError):
        SettableArray([1, 2, 3], name, shape)


def test_full_name() -> None:
    # three cases where only name gets used for full_name
    for instrument in blank_instruments:
        p = SimpleArrayParam([6, 7], 'fred', (2,),
                             setpoint_names=('barney',))
        # this is not allowed since instrument
        # here is not actually an instrument
        # but useful for testing
        p._instrument = instrument  # type: ignore[assignment]
        assert str(p) == 'fred'
        assert p.setpoint_full_names == ('barney',)

    # and then an instrument that really has a name
    p = SimpleArrayParam([6, 7], "wilma", (2,), setpoint_names=("betty",))
    p._instrument = named_instrument  # type: ignore[assignment]
    assert str(p) == "astro_wilma"
    assert p.setpoint_full_names == ("astro_betty",)

    # and with a 2d parameter to test mixed setpoint_names
    p = SimpleArrayParam(
        [[6, 7, 8], [1, 2, 3]], "wilma", (3, 2), setpoint_names=("betty", None)
    )
    p._instrument = named_instrument  # type: ignore[assignment]
    assert p.setpoint_full_names == ("astro_betty", None)


@pytest.mark.parametrize("constructor", [
    {'shape': [[3]]},  # not a depth-1 sequence
    {'shape': [3], 'setpoints': [1, 2, 3]},  # should be [[1, 2, 3]]
    {'shape': [3], 'setpoint_names': 'index'},  # should be ['index']
    {'shape': [3], 'setpoint_labels': 'the index'},  # ['the index']
    {'shape': [3], 'setpoint_names': [None, 'index2']}
])
def test_constructor_errors(constructor: dict) -> None:
    with pytest.raises(ValueError):
        SimpleArrayParam([1, 2, 3], 'p', **constructor)
