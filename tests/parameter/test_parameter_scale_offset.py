from collections.abc import Iterable

import hypothesis.strategies as hst
import numpy as np
from hypothesis import event, given, settings

from qcodes.parameters import Parameter


def test_scale_raw_value() -> None:
    p = Parameter(name="test_scale_raw_value", set_cmd=None)
    p(42)
    assert p.raw_value == 42

    p.scale = 2
    assert p.raw_value == 42  # No set/get cmd performed
    assert p() == 21

    p(10)
    assert p.raw_value == 20
    assert p() == 10


# There are a number different scenarios for testing a parameter with scale
# and offset. Therefore a custom strategy for generating test parameters
# is implemented here. The possible cases are:
# for getting and setting a parameter: values can be
#    scalar:
#        offset and scale can be scalars
# for getting only:
#    array:
#        offset and scale can be scalars or arrays(of same legnth as values)
#        independently

# define shorthands for strategies
TestFloats = hst.floats(min_value=-1e40, max_value=1e40).filter(
    lambda x: abs(x) >= 1e-20
)
SharedSize = hst.shared(hst.integers(min_value=1, max_value=100), key="shared_size")
ValuesScalar = hst.shared(hst.booleans(), key="values_scalar")


# the following test stra
@hst.composite
def iterable_or_number(draw, values, size, values_scalar, is_values):
    if draw(values_scalar):
        # if parameter values are scalar,
        # return scalar for values and scale/offset
        return draw(values)
    elif is_values:
        # if parameter values are not scalar and parameter values are requested
        # return a list of values of the given size
        return draw(hst.lists(values, min_size=draw(size), max_size=draw(size)))
    # if parameter values are not scalar and scale/offset are requested
    # make a random choice whether to return a list of the same size as
    # the values or a simple scalar
    elif draw(hst.booleans()):
        return draw(hst.lists(values, min_size=draw(size), max_size=draw(size)))
    else:
        return draw(values)


@settings(max_examples=500)  # default:100 increased
@given(
    values=iterable_or_number(TestFloats, SharedSize, ValuesScalar, True),
    offsets=iterable_or_number(TestFloats, SharedSize, ValuesScalar, False),
    scales=iterable_or_number(TestFloats, SharedSize, ValuesScalar, False),
)
def test_scale_and_offset_raw_value_iterable(values, offsets, scales) -> None:
    p = Parameter(name="test_scale_and_offset_raw_value", set_cmd=None)

    # test that scale and offset does not change the default behaviour
    p(values)
    assert p.raw_value == values

    # test setting scale and offset does not change anything
    p.scale = scales
    p.offset = offsets
    assert p.raw_value == values

    np_values = np.array(values)
    np_offsets = np.array(offsets)
    np_scales = np.array(scales)
    np_get_values = np.array(p())
    # No set/get cmd performed
    np.testing.assert_allclose(np_get_values, (np_values - np_offsets) / np_scales)

    # test set, only for scalar values
    if not isinstance(values, Iterable):
        p(values)
        # No set/get cmd performed
        np.testing.assert_allclose(
            np.array(p.raw_value), np_values * np_scales + np_offsets
        )

        # Due to possible lack of accuracy of the floating-point operations
        # back-and-forth testing is done only for values of ``offsets`` that are
        # not too different from ``values*scales``
        tolerance = 1e7
        if (
            abs(values * scales) >= abs(offsets)
            and abs(values * scales) < tolerance * abs(offsets)
        ) or (
            abs(values * scales) < abs(offsets)
            and abs(offsets) < tolerance * abs(values * scales)
        ):
            # testing conversion back and forth
            p(values)
            np_get_values = np.array(p())
            # No set/get cmd performed
            np.testing.assert_allclose(np_get_values, np_values)

    # adding statistics
    if isinstance(offsets, Iterable):
        event("Offset is array")
    if isinstance(scales, Iterable):
        event("Scale is array")
    if isinstance(values, Iterable):
        event("Value is array")
    if isinstance(scales, Iterable) and isinstance(offsets, Iterable):
        event("Scale is array and also offset")
    if isinstance(scales, Iterable) and not isinstance(offsets, Iterable):
        event("Scale is array but not offset")


@settings(max_examples=300)
@given(
    values=iterable_or_number(TestFloats, SharedSize, ValuesScalar, True),
    offsets=iterable_or_number(TestFloats, SharedSize, ValuesScalar, False),
    scales=iterable_or_number(TestFloats, SharedSize, ValuesScalar, False),
)
def test_scale_and_offset_raw_value_iterable_for_set_cache(
    values, offsets, scales
) -> None:
    p = Parameter(name="test_scale_and_offset_raw_value", set_cmd=None)

    # test that scale and offset does not change the default behaviour
    p.cache.set(values)
    assert p.raw_value == values

    # test setting scale and offset does not change anything
    p.scale = scales
    p.offset = offsets
    assert p.raw_value == values

    np_values = np.array(values)
    np_offsets = np.array(offsets)
    np_scales = np.array(scales)
    np_get_latest_values = np.array(p.get_latest())
    # Without a call to ``get``, ``get_latest`` will just return old
    # cached values without applying the set scale and offset
    np.testing.assert_allclose(np_get_latest_values, np_values)
    np_get_values = np.array(p.get())
    # Now that ``get`` is called, the returned values are the result of
    # application of the scale and offset. Obviously, calling
    # ``get_latest`` now will also return the values with the applied
    # scale and offset
    np.testing.assert_allclose(np_get_values, (np_values - np_offsets) / np_scales)
    np_get_latest_values_after_get = np.array(p.get_latest())
    np.testing.assert_allclose(
        np_get_latest_values_after_get, (np_values - np_offsets) / np_scales
    )

    # test ``cache.set`` for scalar values
    if not isinstance(values, Iterable):
        p.cache.set(values)
        np.testing.assert_allclose(
            np.array(p.raw_value), np_values * np_scales + np_offsets
        )
        # No set/get cmd performed

        # testing conversion back and forth
        p.cache.set(values)
        np_get_latest_values = np.array(p.get_latest())
        # No set/get cmd performed
        np.testing.assert_allclose(np_get_latest_values, np_values)

    # adding statistics
    if isinstance(offsets, Iterable):
        event("Offset is array")
    if isinstance(scales, Iterable):
        event("Scale is array")
    if isinstance(values, Iterable):
        event("Value is array")
    if isinstance(scales, Iterable) and isinstance(offsets, Iterable):
        event("Scale is array and also offset")
    if isinstance(scales, Iterable) and not isinstance(offsets, Iterable):
        event("Scale is array but not offset")


def test_numpy_array_valued_parameter_preserves_type_if_scale_and_offset_are_set() -> (
    None
):
    def rands():
        return np.random.randn(5)

    param = Parameter(name="test_param", set_cmd=None, get_cmd=rands)

    param.scale = 10
    param.offset = 7

    values = param()

    assert isinstance(values, np.ndarray)


def test_setting_numpy_array_valued_param_if_scale_and_offset_are_not_none() -> None:
    param = Parameter(name="test_param", set_cmd=None, get_cmd=None)

    values = np.array([1, 2, 3, 4, 5])

    param.scale = 100
    param.offset = 10

    param(values)

    assert isinstance(param.raw_value, np.ndarray)
