from typing import Any

import pytest

from qcodes.parameters.sequence_helpers import is_sequence_of

simple_good: list[tuple[Any, ...]] = [
    # empty lists pass without even checking that we provided a
    # valid type spec
    ([], None),
    ((), None),
    ([1, 2, 3], int),
    ((1, 2, 3), int),
    ([1, 2.0], (int, float)),
    ([{}, None], (type(None), dict)),
    # omit type (or set None) and we don't test type at all
    ([1, "2", dict],),
    ([1, "2", dict], None),
]


@pytest.mark.parametrize("args", simple_good)
def test_simple_good(args) -> None:
    assert is_sequence_of(*args)


simple_bad = [(1,), (1, int), ([1, 2.0], int), ([1, 2], float), ([1, 2], (float, dict))]


@pytest.mark.parametrize("args", simple_bad)
def test_simple_bad(args) -> None:
    assert not is_sequence_of(*args)
    # second arg, if provided, must be a type or tuple of types
    # failing this doesn't return False, it raises an error


def test_examples_raises() -> None:
    with pytest.raises(TypeError):
        is_sequence_of([1], 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        is_sequence_of([1], (1, 2))  # type: ignore[arg-type]


good_depth = [
    ([1, 2], int, 1),
    ([[1, 2], [3, 4]], int, 2),
    ([[1, 2.0], []], (int, float), 2),
    ([[[1]]], int, 3),
    ([[1, 2], [3, 4]], None, 2),
]


@pytest.mark.parametrize("args", good_depth)
def test_depth_good(args) -> None:
    assert is_sequence_of(*args)


bad_depth = [([1], int, 2), ([[1]], int, 1), ([[1]], float, 2)]


@pytest.mark.parametrize("args", bad_depth)
def test_depth_bad(args) -> None:
    assert not is_sequence_of(*args)


good_shapes = [
    ([1, 2], int, (2,)),
    ([[1, 2, 3], [4, 5, 6.0]], (int, float), (2, 3)),
    ([[[1]]], int, (1, 1, 1)),
    ([[1], [2]], None, (2, 1)),
    # if you didn't have `list` as a type, the shape of this one
    # would be (2, 2) - that's tested in bad below
    ([[1, 2], [3, 4]], list, (2,)),
    (((0, 1, 2), ((0, 1), (0, 1), (0, 1))), tuple, (2,)),
    (((0, 1, 2), ((0, 1), (0, 1), (0, 1))), (tuple, int), (2, 3)),
]


@pytest.mark.parametrize("args", good_shapes)
def test_shape_good(args) -> None:
    obj, types, shape = args
    assert is_sequence_of(obj, types, shape=shape)


bad_shapes = [
    ([1], int, (2,)),
    ([[1]], int, (1,)),
    ([[1, 2], [1]], int, (2, 2)),
    ([[1]], float, (1, 1)),
    ([[1, 2], [3, 4]], int, (2,)),
    (((0, 1, 2), ((0, 1), (0, 1))), (tuple, int), (2, 3)),
]


@pytest.mark.parametrize("args", bad_shapes)
def test_shape_bad(args) -> None:
    obj, types, shape = args
    assert not is_sequence_of(obj, types, shape=shape)


def test_shape_depth() -> None:
    # there's no reason to provide both shape and depth, but
    # we allow it if they are self-consistent
    with pytest.raises(ValueError):
        is_sequence_of([], int, depth=1, shape=(2, 2))

    assert not is_sequence_of([1], int, depth=1, shape=(2,))
    assert is_sequence_of([1], int, depth=1, shape=(1,))
