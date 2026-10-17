import numpy as np
import pytest

from qcodes.parameters.permissive_range import permissive_range

bad_args = [[], [1], [1, 2], [None, 1, 0.1], [1, None, 0.1], [1, 2, "not too far"]]

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
    ((1.5, 1.8, 0.1), [1.5, 1.6, 1.7]),
]


@pytest.mark.parametrize("args", bad_args)
def test_bad_calls(args) -> None:
    with pytest.raises(TypeError):
        permissive_range(*args)


@pytest.mark.parametrize("args,result", good_args)
def test_good_calls(args, result) -> None:
    # TODO(giulioungaretti)
    # not sure what we are testing here.
    # in python 1.0 and 1 are actually the same
    # https://docs.python.org/3.5/library/functions.html#hash
    assert permissive_range(*args) == result


# numpy fixed-width integers wrap around on overflow, and an unsigned
# integer cannot represent the negative step of a descending range.
# The result must be the same as for the equivalent Python ints.
numpy_int_args = [
    ((np.uint16(10), np.uint16(5), 1), [10, 9, 8, 7, 6]),
    ((np.uint8(10), np.uint8(5), np.uint8(1)), [10, 9, 8, 7, 6]),
    ((10, 5, np.uint16(1)), [10, 9, 8, 7, 6]),
    ((np.uint16(5), np.uint16(10), np.uint16(2)), [5, 7, 9]),
    ((np.int8(100), np.int8(-100), 50), [100, 50, 0, -50]),
    ((np.int8(-100), np.int8(100), 50), [-100, -50, 0, 50]),
    (
        (np.int32(-(2**31) + 1), np.int32(2**31 - 1), 2**30),
        [-(2**31) + 1, -(2**30) + 1, 1, 2**30 + 1],
    ),
    ((np.int32(1), np.int32(7), np.int32(2)), [1, 3, 5]),
    ((np.int64(7), np.int64(1), 2), [7, 5, 3]),
]


@pytest.mark.parametrize("args,result", numpy_int_args)
def test_numpy_integer_calls(args, result) -> None:
    assert permissive_range(*args) == result
