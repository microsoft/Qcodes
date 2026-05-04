"""
Tests for qcodes.utils.numpy_utils - numpy array conversion utilities.
"""

import numpy as np

from qcodes.utils.numpy_utils import list_of_data_to_maybe_ragged_nd_array


def test_regular_list_converts_to_array() -> None:
    """Test that a simple list converts to a 1D numpy array."""
    data = [1, 2, 3]
    result = list_of_data_to_maybe_ragged_nd_array(data)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))
    assert result.ndim == 1


def test_nested_lists_same_length_create_2d_array() -> None:
    """Test that nested lists of equal length create a 2D array."""
    data = [[1, 2], [3, 4], [5, 6]]
    result = list_of_data_to_maybe_ragged_nd_array(data)
    expected = np.array([[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 2)


def test_ragged_nested_lists_return_object_array() -> None:
    """Test that ragged nested lists produce an object-dtype array."""
    data = [[1, 2], [3, 4, 5], [6]]
    result = list_of_data_to_maybe_ragged_nd_array(data)
    assert result.dtype == object
    assert len(result) == 3


def test_dtype_parameter_is_respected() -> None:
    """Test that the dtype parameter is used for the output array."""
    data = [1, 2, 3]
    result = list_of_data_to_maybe_ragged_nd_array(data, dtype=float)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_empty_list() -> None:
    """Test that an empty list converts to an empty array."""
    data: list = []
    result = list_of_data_to_maybe_ragged_nd_array(data)
    assert len(result) == 0


def test_single_element_list() -> None:
    """Test that a single element list converts correctly."""
    data = [42]
    result = list_of_data_to_maybe_ragged_nd_array(data)
    np.testing.assert_array_equal(result, np.array([42]))
    assert result.shape == (1,)


def test_list_of_floats() -> None:
    """Test that a list of floats converts correctly."""
    data = [1.1, 2.2, 3.3]
    result = list_of_data_to_maybe_ragged_nd_array(data)
    np.testing.assert_array_almost_equal(result, np.array([1.1, 2.2, 3.3]))


def test_list_of_strings() -> None:
    """Test that a list of strings converts to a string array."""
    data = ["a", "b", "c"]
    result = list_of_data_to_maybe_ragged_nd_array(data)
    np.testing.assert_array_equal(result, np.array(["a", "b", "c"]))


def test_3d_uniform_data() -> None:
    """Test that uniformly nested 3D data creates a 3D array."""
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = list_of_data_to_maybe_ragged_nd_array(data)
    assert result.shape == (2, 2, 2)


def test_ragged_array_preserves_inner_lists() -> None:
    """Test that ragged array elements are preserved correctly."""
    data = [[1, 2, 3], [4, 5]]
    result = list_of_data_to_maybe_ragged_nd_array(data)
    assert result.dtype == object
    assert list(result[0]) == [1, 2, 3]
    assert list(result[1]) == [4, 5]
