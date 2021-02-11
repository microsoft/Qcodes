import pytest
from qcodes.utils.helpers import is_sequence


def a_func():
    raise RuntimeError('this function shouldn\'t get called')


class AClass:
    pass


yes_sequence = [
    [],
    [1, 2, 3],
    range(5),
    (),
    ('lions', 'tigers', 'bears'),

    # we do have to be careful about generators...
    # ie don't call len() or iterate twice
    (i ** 2 for i in range(5)),
]

no_sequence = [
    1,
    1.0,
    True,
    None,
    'you can iterate a string but we won\'t',
    b'nor will we iterate bytes',
    a_func,
    AClass,
    AClass(),
    # previously dicts, sets, and files all returned True, but
    # we've eliminated them now.
    {1: 2, 3: 4},
    {1, 2, 3},
]


@pytest.mark.parametrize("val", yes_sequence)
def test_yes(val):
    assert is_sequence(val)


@pytest.mark.parametrize("val", no_sequence)
def test_no(val):
    assert not is_sequence(val)


def test_open_file_is_not_sequence():
    with open(__file__) as f:
        assert not is_sequence(f)
