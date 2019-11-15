"""
Test suite for DelegateParameter
"""
import pytest

from qcodes.instrument.parameter import (
    Parameter, DelegateParameter)


def test_delegate_parameter_init():
    """
    Test that the lable and unit get used from source parameter if not
    specified otherwise.
    """
    p = Parameter('testparam', set_cmd=None, get_cmd=None,
                  label='Test Parameter', unit='V')
    d = DelegateParameter('test_delegate_parameter', p)
    assert d.label == p.label
    assert d.unit == p.unit

    d = DelegateParameter('test_delegate_parameter', p, unit='Ohm')
    assert d.label == p.label
    assert not d.unit == p.unit
    assert d.unit == 'Ohm'


def test_delegate_parameter_get_set_raises():
    """
    Test that providing a get/set_cmd kwarg raises an error.
    """
    p = Parameter('testparam', set_cmd=None, get_cmd=None)
    for kwargs in ({'set_cmd': None}, {'get_cmd': None}):
        with pytest.raises(KeyError) as e:
            DelegateParameter('test_delegate_parameter', p, **kwargs)
        assert str(e.value).startswith('\'It is not allowed to set')


def test_delegate_parameter_scaling():
    p = Parameter('testparam', set_cmd=None, get_cmd=None, offset=1, scale=2)
    d = DelegateParameter('test_delegate_parameter', p, offset=3, scale=5)

    p(1)
    assert p.raw_value == 3
    assert d() == (1-3)/5

    d(2)
    assert d.raw_value == 2*5+3
    assert d.raw_value == p()


def test_delegate_parameter_snapshot():
    p = Parameter('testparam', set_cmd=None, get_cmd=None,
                  offset=1, scale=2, initial_value=1)
    d = DelegateParameter('test_delegate_parameter', p, offset=3, scale=5,
                          initial_value=2)

    snapshot = d.snapshot()
    source_snapshot = snapshot.pop('source_parameter')
    assert source_snapshot == p.snapshot()
    assert snapshot['value'] == 2
    assert source_snapshot['value'] == 13


def test_delegate_parameter_set_cache_for_memory_source_parameter():
    initial_value = 1
    source = Parameter('p', set_cmd=None, get_cmd=None,
                       initial_value=initial_value, offset=1, scale=2)
    delegate = DelegateParameter('d', source, offset=4, scale=5)

    # Setting the cached value of the source parameter changes the
    # delegate parameter accordingly

    new_source_value = 3
    source.cache.set(new_source_value)

    assert source.raw_value == new_source_value * 2 + 1
    assert source.get_latest() == new_source_value

    assert delegate.raw_value == new_source_value

    # But then when the delegate parameter's ``get`` is called, the new
    # value of the source propagates
    gotten_delegate_value = delegate.get()

    assert gotten_delegate_value == (new_source_value - 4) / 5
    assert delegate.raw_value == new_source_value
    assert delegate.get_latest() == (new_source_value - 4) / 5

    # Setting the cached value of the delegate parameter changes the
    # the source parameter

    new_delegate_value = 2
    delegate.cache.set(new_delegate_value)

    assert delegate.raw_value == new_delegate_value * 5 + 4
    assert delegate.get_latest() == new_delegate_value

    assert source.raw_value == (new_delegate_value * 5 + 4) * 2 + 1
    assert source.get_latest() == new_delegate_value * 5 + 4


def test_delegate_parameter_set_cache_for_instrument_source_parameter():
    instrument_value = -689

    def get_instrument_value():
        nonlocal instrument_value
        return instrument_value

    def set_instrument_value(value):
        nonlocal instrument_value
        instrument_value = value

    initial_value = 1
    source = Parameter('p',
                       set_cmd=set_instrument_value,
                       get_cmd=get_instrument_value,
                       initial_value=initial_value, offset=1, scale=2)
    delegate = DelegateParameter('d', source, offset=4, scale=5)

    # Setting the cached value of the source parameter changes the
    # delegate parameter. But it has no impact on the instrument value.

    new_source_value = 3
    source.cache.set(new_source_value)

    assert source.raw_value == new_source_value * 2 + 1
    assert source.get_latest() == new_source_value

    assert instrument_value == initial_value * 2 + 1

    assert delegate.raw_value == new_source_value

    # After the delegate parameter's ``get`` is called, source
    # parameter and the instrument values are updated

    gotten_delegate_value = delegate.get()

    assert gotten_delegate_value == (initial_value - 4) / 5

    assert delegate.raw_value == initial_value
    assert delegate.get_latest() == (initial_value - 4) / 5

    assert instrument_value == initial_value * 2 + 1

    # Setting the cached value of the delegate parameter has an impact on
    # the source parameter, but not on the instrument value

    new_delegate_value = 2
    delegate.cache.set(new_delegate_value)

    assert delegate.raw_value == new_delegate_value * 5 + 4
    assert delegate.get_latest() == new_delegate_value

    assert source.raw_value == (new_delegate_value * 5 + 4) * 2 + 1
    assert source.get_latest() == new_delegate_value * 5 + 4

    assert instrument_value == initial_value * 2 + 1

