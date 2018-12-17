from typing import Sequence, Tuple, Dict
from operator import mul
from functools import reduce

import numpy as np
import pandas
from numpy.testing import assert_array_equal


def verify_data_dict(data: Dict[str, Dict[str, np.ndarray]],
                     dataframe: Dict[str, pandas.DataFrame],
                     parameter_names: Sequence[str],
                     expected_names: Dict[str, Sequence[str]],
                     expected_shapes: Dict[str, Sequence[Tuple[int, ...]]],
                     expected_values: Dict[str, Sequence[np.ndarray]]) -> None:
    """
    Simple helper function to verify a dict of data

    Args:
        data: The dict data to verify the shape and content of.
        dataframe: The data represented as a dict of Pandas DataFrames.
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
    assert all(param in parameter_names for
               param in list(dataframe.keys())) is True
    for param in parameter_names:
        innerdata = data[param]
        innerdataframe = dataframe[param]
        verify_data_dict_for_single_param(innerdata,
                                          innerdataframe,
                                          expected_names[param],
                                          expected_shapes[param],
                                          expected_values[param])


def verify_data_dict_for_single_param(datadict: Dict[str, np.ndarray],
                                      dataframe: pandas.DataFrame,
                                      names: Sequence[str],
                                      shapes: Sequence[Tuple[int, ...]],
                                      values):
    # check that there are no unexpected elements in the dict
    key_names = list(datadict.keys())
    assert set(key_names) == set(names)
    # check that the dataframe has the same elements as index and columns
    pandas_index_names = list(dataframe.index.names)
    pandas_column_names = list(dataframe)
    pandas_names = []
    for i in pandas_index_names:
        if i is not None:
            pandas_names.append(i)
    for i in pandas_column_names:
        if i is not None:
            pandas_names.append(i)
    assert set(pandas_names) == set(names)

    simpledf = dataframe.reset_index()

    for name, shape, value in zip(names, shapes, values):
        assert datadict[name].shape == shape
        assert len(simpledf[name]) == reduce(mul, shape)
        assert_array_equal(dataframe.reset_index()[name].values, value.ravel())
        assert_array_equal(datadict[name], value)
