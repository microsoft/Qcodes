import pytest

from qcodes.instrument.parameter import MultiParameter
from .conftest import named_instrument, blank_instruments



class SimpleMultiParam(MultiParameter):
    def __init__(self, return_val, *args, **kwargs):
        self._return_val = return_val
        self._get_count = 0
        super().__init__(*args, **kwargs)

    def get_raw(self):
        self._get_count += 1
        return self._return_val


class SettableMulti(SimpleMultiParam):
    def set_raw(self, value):
        print("Calling set")
        self._return_val = value


def test_default_attributes():
    name = 'mixed_dimensions'
    names = ('0D', '1D', '2D')
    shapes = ((), (3,), (2, 2))
    p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                         name, names, shapes)

    assert p.name == name
    assert p.names == names
    assert p.shapes == shapes

    assert p.labels == names
    assert p.units == [''] * 3
    assert p.setpoints is None
    assert p.setpoint_names is None
    assert p.setpoint_labels is None

    assert str(p) == name

    assert p._get_count == 0
    snap = p.snapshot(update=True)
    assert p._get_count == 0
    snap_expected = {
        'name': name,
        'names': names,
        'labels': names,
        'units': [''] * 3,
        'ts': None
    }
    for k, v in snap_expected.items():
        assert snap[k] == v
    assert 'value' not in snap
    assert 'raw_value' not in snap

    assert name in p.__doc__

    # only in simple parameters
    assert not hasattr(p, 'label')
    assert not hasattr(p, 'unit')


def test_explicit_attributes():
    name = 'mixed_dimensions'
    names = ('0D', '1D', '2D')
    shapes = ((), (3,), (2, 2))
    labels = ['scalar', 'vector', 'matrix']
    units = ['V', 'A', 'W']
    setpoints = [(), ((4, 5, 6),), ((7, 8), None)]
    setpoint_names = [(), ('sp1',), ('sp2', None)]
    setpoint_labels = [(), ('setpoint1',), ('setpoint2', None)]
    docstring = 'DOCS??'
    metadata = {'sizes': [1, 3, 4]}
    p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                         name, names, shapes, labels=labels, units=units,
                         setpoints=setpoints,
                         setpoint_names=setpoint_names,
                         setpoint_labels=setpoint_labels,
                         docstring=docstring, snapshot_value=True,
                         metadata=metadata)

    assert p.name == name
    assert p.names == names
    assert p.shapes == shapes

    assert p.labels == labels
    assert p.units == units
    assert p.setpoints == setpoints
    assert p.setpoint_names == setpoint_names
    # as the parameter is not attached to an instrument the full names are
    # equivalent to the setpoint_names
    assert p.setpoint_full_names == setpoint_names
    assert p.setpoint_labels == setpoint_labels

    assert p._get_count == 0
    snap = p.snapshot(update=True)
    assert p._get_count == 1
    snap_expected = {
        'name': name,
        'names': names,
        'labels': labels,
        'units': units,
        'setpoint_names': setpoint_names,
        'setpoint_labels': setpoint_labels,
        'metadata': metadata,
        'value': [0, [1, 2, 3], [[4, 5], [6, 7]]],
        'raw_value': [0, [1, 2, 3], [[4, 5], [6, 7]]]
    }
    for k, v in snap_expected.items():
        assert snap[k] == v
    assert snap['ts'] is not None

    assert name in p.__doc__
    assert docstring in p.__doc__


def test_has_set_get():
    name = 'mixed_dimensions'
    names = ['0D', '1D', '2D']
    shapes = ((), (3,), (2, 2))
    with pytest.raises(AttributeError):
        MultiParameter(name, names, shapes)

    original_value = [0, [1, 2, 3], [[4, 5], [6, 7]]]
    p = SimpleMultiParam(original_value, name, names, shapes)

    assert hasattr(p, 'get')
    assert p.gettable
    assert not hasattr(p, 'set')
    assert not p.settable
    # Ensure that ``cache.set`` works
    new_cache = [10, [10, 20, 30], [[40, 50], [60, 70]]]
    p.cache.set(new_cache)
    assert p.get_latest() == new_cache
    # However, due to the implementation of this ``SimpleMultiParam``
    # test parameter it's ``get`` call will return the originally passed
    # list
    assert p.get() == original_value
    assert p.get_latest() == original_value

    # We allow creation of Multiparameters with set to support
    # instruments that already make use of them.
    p = SettableMulti([0, [1, 2, 3], [[4, 5], [6, 7]]], name, names, shapes)
    assert hasattr(p, 'get')
    assert p.gettable
    assert hasattr(p, 'set')
    assert p.settable
    value_to_set = [2, [1, 5, 2], [[8, 2], [4, 9]]]
    p.set(value_to_set)
    assert p.get() == value_to_set
    # Also, ``cache.set`` works as expected
    p.cache.set(new_cache)
    assert p.get_latest() == new_cache
    assert p.get() == value_to_set


def test_full_name_s():
    name = 'mixed_dimensions'
    names = ('0D', '1D', '2D')
    setpoint_names = ((),
                      ('setpoints_1D',),
                      ('setpoints_2D_1',
                       None))
    shapes = ((), (3,), (2, 2))

    # three cases where only name gets used for full_name
    for instrument in blank_instruments:
        p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                             name, names, shapes,
                             setpoint_names=setpoint_names)
        p._instrument = instrument
        assert str(p) == name
        assert p.full_names == names
        assert p.setpoint_full_names == \
                         ((), ('setpoints_1D',), ('setpoints_2D_1', None))

    # and finally an instrument that really has a name
    p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                         name, names, shapes, setpoint_names=setpoint_names)
    p._instrument = named_instrument
    assert str(p) == 'astro_mixed_dimensions'

    assert p.full_names == ('astro_0D', 'astro_1D', 'astro_2D')
    assert p.setpoint_full_names == \
                     ((), ('astro_setpoints_1D',),
                      ('astro_setpoints_2D_1', None))


@pytest.mark.parametrize("constructor", [
    {'names': 'a', 'shapes': ((3,), ())},
    {'names': ('a', 'b'), 'shapes': (3, 2)},
    {'names': ('a', 'b'), 'shapes': ((3,), ()),
     'setpoints': [(1, 2, 3), ()]},
    {'names': ('a', 'b'), 'shapes': ((3,), ()),
     'setpoint_names': (None, ('index',))},
    {'names': ('a', 'b'), 'shapes': ((3,), ()),
     'setpoint_labels': (None, None, None)}])
def test_constructor_errors(constructor):
    with pytest.raises(ValueError):
        SimpleMultiParam([1, 2, 3], 'p', **constructor)
