from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest

from qcodes.dataset.sqlite.database import (
    _adapt_array,
    _adapt_complex,
    _convert_array,
    _convert_complex,
    _convert_numeric,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from pytest_mock import MockerFixture


def _serialize(
    array: npt.NDArray, version: tuple[int, int] | None, allow_pickle: bool = False
) -> bytes:
    out = io.BytesIO()
    if version is None:
        np.save(out, array, allow_pickle=allow_pickle)
    else:
        np.lib.format.write_array(
            out, array, version=version, allow_pickle=allow_pickle
        )
    return out.getvalue()


def _read_with_numpy(serialized: bytes) -> npt.NDArray:
    return np.lib.format.read_array(io.BytesIO(serialized), allow_pickle=False)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0), None])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("shape", [(), (0,), (1,), (7,), (3, 4), (2, 3, 4), (0, 3)])
@pytest.mark.parametrize(
    "dtype",
    [
        "f8",
        ">f8",
        "f4",
        "i1",
        ">i4",
        "u8",
        "c8",
        "c16",
        "?",
        "<U5",
        ">U3",
        "S4",
        "M8[ns]",
        [("a", "f8"), ("b", "i4")],
    ],
)
def test_convert_array_matches_numpy(
    mocker: MockerFixture,
    dtype: Any,
    shape: tuple[int, ...],
    order: Literal["C", "F"],
    version: tuple[int, int] | None,
) -> None:
    array = np.asarray(np.zeros(shape, dtype=np.dtype(dtype)), order=order)
    if array.dtype.fields is None:
        array.flat = np.arange(array.size) % 100
    serialized = _serialize(array, version)
    expected = _read_with_numpy(serialized)

    read_array_spy = mocker.spy(np.lib.format, "read_array")
    result = _convert_array(serialized)

    read_array_spy.assert_not_called()
    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert result.flags.c_contiguous == expected.flags.c_contiguous
    assert result.flags.f_contiguous == expected.flags.f_contiguous
    assert result.flags.writeable
    assert result.tobytes() == expected.tobytes()


def test_convert_array_round_trips_adapted_array() -> None:
    array = np.random.default_rng().random((3, 5))
    result = _convert_array(bytes(_adapt_array(array)))
    np.testing.assert_array_equal(result, array)
    # the returned array must not share memory with the immutable input bytes
    result[0, 0] = 1.0


def test_convert_array_non_ascii_header_falls_back_to_numpy(
    mocker: MockerFixture,
) -> None:
    serialized = _serialize(np.zeros(2, dtype=[("\u00e5", "f8")]), (3, 0))
    read_array_spy = mocker.spy(np.lib.format, "read_array")

    result = _convert_array(serialized)

    read_array_spy.assert_called_once()
    assert result.dtype == _read_with_numpy(serialized).dtype


def test_convert_array_object_array_raises() -> None:
    serialized = _serialize(np.array([1, "a"], dtype=object), (3, 0), True)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        _convert_array(serialized)


@pytest.mark.parametrize("serialized", [b"", b"not an array", b"\x93NUMP"])
def test_convert_array_invalid_input_raises(serialized: bytes) -> None:
    with pytest.raises(ValueError):
        _convert_array(serialized)


def test_convert_array_truncated_data_raises() -> None:
    serialized = bytes(_adapt_array(np.arange(10.0)))[:-8]
    with pytest.raises(ValueError, match="array data"):
        _convert_array(serialized)


@pytest.mark.parametrize(
    "value",
    [1 + 2j, np.complex64(3 - 1j), np.complex128(-0.5 + 0.25j), complex("nan+infj")],
)
def test_convert_complex(value: complex | np.complexfloating) -> None:
    serialized = bytes(_adapt_complex(value))
    expected = np.load(io.BytesIO(serialized))[0]

    result = _convert_complex(serialized)

    assert type(result) is type(expected)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"1.5", 1.5),
        (b"-2.25e-3", -2.25e-3),
        (b"3", 3),
        (b"3.0", 3),
        (b"-0.0", 0),
        (b"1e20", 10**20),
        (b"inf", math.inf),
        (b"-Infinity", -math.inf),
        (b"some text", "some text"),
    ],
)
def test_convert_numeric(value: bytes, expected: float | str) -> None:
    result = _convert_numeric(value)
    assert type(result) is type(expected)
    assert result == expected


@pytest.mark.parametrize("value", [b"nan", b"NaN", b"-nan"])
def test_convert_numeric_nan(value: bytes) -> None:
    result = _convert_numeric(value)
    assert isinstance(result, float)
    assert math.isnan(result)
