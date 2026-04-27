"""
Tests for qcodes.utils.types - numpy type tuples and aliases.
"""

import numpy as np

from qcodes.utils.types import (
    complex_types,
    concrete_complex_types,
    numpy_c_complex,
    numpy_c_floats,
    numpy_c_ints,
    numpy_complex,
    numpy_concrete_complex,
    numpy_concrete_floats,
    numpy_concrete_ints,
    numpy_floats,
    numpy_ints,
    numpy_non_concrete_ints_instantiable,
)


def test_numpy_concrete_ints_contents() -> None:
    """Test that numpy_concrete_ints contains the expected fixed-size int types."""
    expected = (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    )
    assert numpy_concrete_ints == expected


def test_numpy_concrete_ints_length() -> None:
    """Test that numpy_concrete_ints has 8 types."""
    assert len(numpy_concrete_ints) == 8


def test_numpy_c_ints_contents() -> None:
    """Test that numpy_c_ints contains the expected C-compatible int types."""
    expected = (
        np.uintp,
        np.uintc,
        np.intp,
        np.intc,
        np.short,
        np.byte,
        np.ushort,
        np.ubyte,
        np.longlong,
        np.ulonglong,
    )
    assert numpy_c_ints == expected


def test_numpy_c_ints_length() -> None:
    """Test that numpy_c_ints has 10 types."""
    assert len(numpy_c_ints) == 10


def test_numpy_non_concrete_ints_instantiable_contents() -> None:
    """Test that numpy_non_concrete_ints_instantiable contains default int types."""
    expected = (np.int_, np.uint)
    assert numpy_non_concrete_ints_instantiable == expected


def test_numpy_ints_is_combination() -> None:
    """Test that numpy_ints is the concatenation of all int sub-tuples."""
    expected = numpy_concrete_ints + numpy_c_ints + numpy_non_concrete_ints_instantiable
    assert numpy_ints == expected


def test_numpy_ints_length() -> None:
    """Test that numpy_ints has the combined length of all int sub-tuples."""
    expected_len = (
        len(numpy_concrete_ints)
        + len(numpy_c_ints)
        + len(numpy_non_concrete_ints_instantiable)
    )
    assert len(numpy_ints) == expected_len


def test_numpy_concrete_floats_contents() -> None:
    """Test that numpy_concrete_floats contains fixed-size float types."""
    expected = (np.float16, np.float32, np.float64)
    assert numpy_concrete_floats == expected


def test_numpy_c_floats_contents() -> None:
    """Test that numpy_c_floats contains C-compatible float types."""
    expected = (np.half, np.single, np.double)
    assert numpy_c_floats == expected


def test_numpy_floats_is_combination() -> None:
    """Test that numpy_floats is the concatenation of float sub-tuples."""
    assert numpy_floats == numpy_concrete_floats + numpy_c_floats


def test_numpy_floats_length() -> None:
    """Test that numpy_floats has the combined length of float sub-tuples."""
    assert len(numpy_floats) == len(numpy_concrete_floats) + len(numpy_c_floats)


def test_numpy_concrete_complex_contents() -> None:
    """Test that numpy_concrete_complex contains fixed-size complex types."""
    expected = (np.complex64, np.complex128)
    assert numpy_concrete_complex == expected


def test_numpy_c_complex_contents() -> None:
    """Test that numpy_c_complex contains C-compatible complex types."""
    expected = (np.csingle, np.cdouble)
    assert numpy_c_complex == expected


def test_numpy_complex_is_combination() -> None:
    """Test that numpy_complex is the concatenation of complex sub-tuples."""
    assert numpy_complex == numpy_concrete_complex + numpy_c_complex


def test_concrete_complex_types_includes_python_complex() -> None:
    """Test that concrete_complex_types includes numpy and Python complex."""
    assert complex in concrete_complex_types
    for t in numpy_concrete_complex:
        assert t in concrete_complex_types


def test_complex_types_includes_python_complex() -> None:
    """Test that complex_types includes numpy and Python complex."""
    assert complex in complex_types
    for t in numpy_concrete_complex:
        assert t in complex_types


def test_all_int_types_are_numpy_integer_subclass() -> None:
    """Test that all types in numpy_ints are subclasses of np.integer."""
    for t in numpy_ints:
        assert issubclass(t, np.integer), f"{t} is not a subclass of np.integer"


def test_all_float_types_are_numpy_floating_subclass() -> None:
    """Test that all types in numpy_floats are subclasses of np.floating."""
    for t in numpy_floats:
        assert issubclass(t, np.floating), f"{t} is not a subclass of np.floating"


def test_all_complex_types_are_numpy_complexfloating_subclass() -> None:
    """Test that all types in numpy_complex are subclasses of np.complexfloating."""
    for t in numpy_complex:
        assert issubclass(t, np.complexfloating), (
            f"{t} is not a subclass of np.complexfloating"
        )


def test_concrete_int_instances() -> None:
    """Test that instances of concrete int types can be created."""
    for t in numpy_concrete_ints:
        val = t(42)
        assert isinstance(val, np.integer)


def test_concrete_float_instances() -> None:
    """Test that instances of concrete float types can be created."""
    for t in numpy_concrete_floats:
        val = t(3.14)
        assert isinstance(val, np.floating)


def test_concrete_complex_instances() -> None:
    """Test that instances of concrete complex types can be created."""
    for t in numpy_concrete_complex:
        val = t(1 + 2j)
        assert isinstance(val, np.complexfloating)


def test_all_tuples_contain_types() -> None:
    """Test that every element in every tuple is a type (class)."""
    all_tuples = [
        numpy_concrete_ints,
        numpy_c_ints,
        numpy_non_concrete_ints_instantiable,
        numpy_ints,
        numpy_concrete_floats,
        numpy_c_floats,
        numpy_floats,
        numpy_concrete_complex,
        numpy_c_complex,
        numpy_complex,
    ]
    for tup in all_tuples:
        for t in tup:
            assert isinstance(t, type), f"{t} is not a type"
