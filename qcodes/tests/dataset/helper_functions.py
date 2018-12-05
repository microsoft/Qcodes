from typing import Sequence, Tuple, Dict

import numpy as np


def verify_data_dict(data: Dict[str, Dict[str, np.ndarray]],
                     parameter_names: Sequence[str],
                     expected_names: Dict[str, Sequence[str]],
                     expected_shapes: Dict[str, Sequence[Tuple[int, ...]]]) -> None:
    """
    Simple helper function to verify a dict of data

    Args:
        data: The data to verify
        parameter_names: names of the parameters requested
        expected_names: names of the paramerters expected to be loaded
        expected_shapes: shapes of the paramters loaded

    Returns:

    """
    for param in parameter_names:
        innerdata = data[param]
        verify_data_dict_for_single_param(innerdata,
                                          expected_names[param],
                                          expected_shapes[param])


def verify_data_dict_for_single_param(datadict: Dict[str, np.ndarray],
                                      names: Sequence[str],
                                      shapes: Sequence[Tuple[int, ...]]):
    for name, shape in zip(names, shapes):
        assert datadict[name].shape == shape
