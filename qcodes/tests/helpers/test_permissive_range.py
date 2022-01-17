import pytest
from qcodes.utils.helpers import permissive_range

bad_args = [
    [],
    [1],
    [1, 2],
    [None, 1, .1],
    [1, None, .1],
    [1, 2, 'not too far']
]

good_args = [
    ((1, 7, 2), [1, 3, 5]),
    ((1, 7, 4), [1, 5]),
    ((1, 7, 7), [1]),
    ((1.0, 7, 2), [1.0, 3.0, 5.0]),
    ((1, 7.0, 2), [1.0, 3.0, 5.0]),
    ((1, 7, 2.0), [1.0, 3.0, 5.0]),
    ((1.0, 7.0, 2.0), [1.0, 3.0, 5.0]),
    ((1.0, 7.000000001, 2.0), [1.0, 3.0, 5.0, 7.0]),
    ((1, 7, -2), [1, 3, 5]),
    ((7, 1, 2), [7, 5, 3]),
    ((1.0, 7.0, -2.0), [1.0, 3.0, 5.0]),
    ((7.0, 1.0, 2.0), [7.0, 5.0, 3.0]),
    ((7.0, 1.0, -2.0), [7.0, 5.0, 3.0]),
    ((1.5, 1.8, 0.1), [1.5, 1.6, 1.7])
]


@pytest.mark.parametrize("args", bad_args)
def test_bad_calls(args):
    with pytest.raises(Exception):
        permissive_range(*args)


@pytest.mark.parametrize("args,result", good_args)
def test_good_calls(args, result):
    # TODO(giulioungaretti)
    # not sure what we are testing here.
    # in python 1.0 and 1 are actually the same
    # https://docs.python.org/3.5/library/functions.html#hash
    assert permissive_range(*args) == result
