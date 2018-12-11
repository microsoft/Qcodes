from typing import Sequence, Tuple, Dict

import numpy as np
from numpy.testing import assert_array_equal

def verify_data_dict(data: Dict[str, Dict[str, np.ndarray]],
                     parameter_names: Sequence[str],
                     expected_names: Dict[str, Sequence[str]],
                     expected_shapes: Dict[str, Sequence[Tuple[int, ...]]],
                     expected_values: Dict[str, Sequence[np.ndarray]]) -> None:
    """
    Simple helper function to verify a dict of data

    Args:
        data: The dict data to verify the shape and content of.
        parameter_names: names of the parameters loaded as top level
            keys in the dict.
        expected_names: names of the parameters expected as keys in the second
            level.
        expected_shapes: expected shapes of the parameters loaded in the values
            of the dict.
        expected_values: expected content of the data arrays.

    """
    # check that all the expected parameters in the dict are
    # included in the list of parameters
    assert all(param in parameter_names for param in list(data.keys())) is True
    for param in parameter_names:
        innerdata = data[param]
        verify_data_dict_for_single_param(innerdata,
                                          expected_names[param],
                                          expected_shapes[param],
                                          expected_values[param])


def verify_data_dict_for_single_param(datadict: Dict[str, np.ndarray],
                                      names: Sequence[str],
                                      shapes: Sequence[Tuple[int, ...]],
                                      values):
    # check that there are no unexpected elements in the dict
    assert all(param in names for param in list(datadict.keys())) is True
    for name, shape, value in zip(names, shapes, values):
        assert datadict[name].shape == shape
        assert_array_equal(datadict[name], value)
