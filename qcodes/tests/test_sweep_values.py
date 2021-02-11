import pytest

from qcodes.instrument.parameter import Parameter
from qcodes.instrument.sweep_values import SweepValues

from qcodes.utils.validators import Numbers



@pytest.fixture(name='c0')
def _make_c0():
    c0 = Parameter('c0', vals=Numbers(-10, 10), get_cmd=None, set_cmd=None)
    yield c0


@pytest.fixture(name='c1')
def _make_c1():
    c1 = Parameter('c1', get_cmd=None, set_cmd=None)
    yield c1


@pytest.fixture(name='c2')
def _make_c2():
    c2 = Parameter('c2', get_cmd=lambda: 42)
    yield c2


def test_errors(c0, c1, c2):

    # only complete 3-part slices are valid
    with pytest.raises(TypeError):
        c0[1:2]  # For Int params this could be defined as step=1
    with pytest.raises(TypeError):
        c0[:2:3]
    with pytest.raises(TypeError):
        c0[1::3]
    with pytest.raises(TypeError):
        c0[:]  # For Enum params we *could* define this one too...

    # fails if the parameter has no setter
    with pytest.raises(TypeError):
        c2[0:0.1:0.01]

    # validates every step value against the parameter's Validator
    with pytest.raises(ValueError):
        c0[5:15:1]
    with pytest.raises(ValueError):
        c0[5.0:15.0:1.0]
    with pytest.raises(ValueError):
        c0[-12]
    with pytest.raises(ValueError):
        c0[-5, 12, 5]
    with pytest.raises(ValueError):
        c0[-5, 12:8:1, 5]

    # cannot combine SweepValues for different parameters
    with pytest.raises(TypeError):
        c0[0.1] + c1[0.2]

    # improper use of extend
    with pytest.raises(TypeError):
        c0[0.1].extend(5)

    # SweepValue object has no getter, even if the parameter does
    with pytest.raises(AttributeError):
        c0[0.1].get


def test_valid(c0):

    c0_sv = c0[1]
    # setter gets mapped
    assert c0_sv.set == c0.set
    # normal sequence operations access values
    assert list(c0_sv) == [1]
    assert c0_sv[0] == 1
    assert 1 in c0_sv
    assert not (2 in c0_sv)

    # in-place and copying addition
    c0_sv += c0[1.5:1.8:0.1]
    c0_sv2 = c0_sv + c0[2]
    assert list(c0_sv) == [1, 1.5, 1.6, 1.7]
    assert list(c0_sv2) == [1, 1.5, 1.6, 1.7, 2]

    # append and extend
    c0_sv3 = c0[2]
    # append only works with straight values
    c0_sv3.append(2.1)
    # extend can use another SweepValue, (even if it only has one value)
    c0_sv3.extend(c0[2.2])
    # extend can also take a sequence
    c0_sv3.extend([2.3])
    # as can addition
    c0_sv3 += [2.4]
    c0_sv4 = c0_sv3 + [2.5, 2.6]
    assert list(c0_sv3) == [2, 2.1, 2.2, 2.3, 2.4]
    assert list(c0_sv4) == [2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]

    # len
    assert len(c0_sv3) == 5

    # in-place and copying reverse
    c0_sv.reverse()
    c0_sv5 = reversed(c0_sv)
    assert list(c0_sv) == [1.7, 1.6, 1.5, 1]
    assert list(c0_sv5) == [1, 1.5, 1.6, 1.7]

    # multi-key init, where first key is itself a list
    c0_sv6 = c0[[1, 3], 4]
    # copying
    c0_sv7 = c0_sv6.copy()
    assert list(c0_sv6) == [1, 3, 4]
    assert list(c0_sv7) == [1, 3, 4]
    assert not (c0_sv6 is c0_sv7)


def test_base():
    p = Parameter('p', get_cmd=None, set_cmd=None)
    with pytest.raises(NotImplementedError):
        iter(SweepValues(p))


def test_snapshot(c0):

    assert c0[0].snapshot() == {
        'parameter': c0.snapshot(),
        'values': [{'item': 0}]
    }

    assert c0[0:5:0.3].snapshot()['values'] == [{
        'first': 0,
        'last': 4.8,
        'num': 17,
        'type': 'linear'
    }]

    sv = c0.sweep(start=2, stop=4, num=5)
    assert sv.snapshot()['values'] == [{
        'first': 2,
        'last': 4,
        'num': 5,
        'type': 'linear'
    }]

    # mixture of bare items, nested lists, and slices
    sv = c0[1, 7, 3.2, [1, 2, 3], 6:9:1, -4.5, 5.3]
    assert sv.snapshot()['values'] == [{
        'first': 1,
        'last': 5.3,
        'min': -4.5,
        'max': 8,
        'num': 11,
        'type': 'sequence'
        }]

    assert (c0[0] + c0[1]).snapshot()['values'] == [
        {'item': 0},
        {'item': 1}
        ]

    assert (c0[0:3:1] + c0[4, 6, 9]).snapshot()['values'] == [
        {'first': 0, 'last': 2, 'num': 3, 'type': 'linear'},
        {'first': 4, 'last': 9, 'min': 4, 'max': 9, 'num': 3,
         'type': 'sequence'}
        ]


def test_repr(c0):
    sv = c0[0]
    assert repr(sv) == (
        f'<qcodes.instrument.sweep_values.SweepFixedValues: c0 at {id(sv)}>'
    )
