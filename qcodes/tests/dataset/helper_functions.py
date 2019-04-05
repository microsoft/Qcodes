from typing import Sequence, Tuple, Dict
from operator import mul
from typing import Optional
from functools import reduce

import numpy as np
import pandas
from numpy.testing import assert_array_equal


def verify_data_dict(data: Dict[str, Dict[str, np.ndarray]],
                     dataframe: Optional[Dict[str, pandas.DataFrame]],
                     parameter_names: Sequence[str],
                     expected_names: Dict[str, Sequence[str]],
                     expected_shapes: Dict[str, Sequence[Tuple[int, ...]]],
                     expected_values: Dict[str, Sequence[np.ndarray]]) -> None:
    """
    Simple helper function to verify a dict of data. It can also optionally

    The expected names values
    and shapes should be given as a dict with keys given by the dependent
    parameters. Each value in the dicts should be the sequence of expected
    names/shapes/values for that requested parameter and its dependencies.
    The first element in the sequence must be the dependent parameter loaded.

    Args:
        data: The dict data to verify the shape and content of.
        dataframe: The data represented as a dict of Pandas DataFrames.
        parameter_names: names of the parameters requested. These are expected
            as top level keys in the dict.
        expected_names: names of the parameters expected to be loaded for a
            given parameter as a sequence indexed by the parameter.
        expected_shapes: expected shapes of the parameters loaded. The shapes
            should be stored as a tuple per parameter in a sequence containing
            all the loaded parameters for a given requested parameter.
        expected_values: expected content of the data arrays stored in a
            sequence

    """
    # check that all the expected parameters in the dict are
    # included in the list of parameters
    assert all(param in parameter_names for param in list(data.keys())) is True
    if dataframe is not None:
        assert all(param in parameter_names for
                   param in list(dataframe.keys())) is True
    for param in parameter_names:
        innerdata = data[param]
        verify_data_dict_for_single_param(innerdata,
                                          expected_names[param],
                                          expected_shapes[param],
                                          expected_values[param])
        if dataframe is not None:
            innerdataframe = dataframe[param]
            verify_dataframe_for_single_param(innerdataframe,
                                              expected_names[param],
                                              expected_shapes[param],
                                              expected_values[param])


def verify_data_dict_for_single_param(datadict: Dict[str, np.ndarray],
                                      names: Sequence[str],
                                      shapes: Sequence[Tuple[int, ...]],
                                      values):
    # check that there are no unexpected elements in the dict
    key_names = list(datadict.keys())
    assert set(key_names) == set(names)

    for name, shape, value in zip(names, shapes, values):
        if datadict[name].dtype == np.dtype('O'):
            mydata = np.concatenate(datadict[name])
        else:
            mydata = datadict[name]
        assert mydata.shape == shape
        assert_array_equal(mydata, value)


def verify_dataframe_for_single_param(dataframe: pandas.DataFrame,
                                      names: Sequence[str],
                                      shapes: Sequence[Tuple[int, ...]],
                                      values):
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

    # lets check that the index is made up
    # from all but the first column as expected
    if len(values) > 1:
        expected_index_values = values[1:]
        index_values = dataframe.index.values

        nindexes = len(expected_index_values)
        nrows = shapes[0]

        for row in range(len(nrows)):
            row_index_values = index_values[row]
            # one dimensional arrays will have single values for there indexed
            # not tuples as they don't use multiindex. Put these in tuples
            # for easy comparison
            if not isinstance(dataframe.index, pandas.MultiIndex):
                row_index_values = (row_index_values,)

            expected_values = \
                tuple(expected_index_values[indexnum].ravel()[row]
                      for indexnum in range(nindexes))
            assert row_index_values == expected_values

    simpledf = dataframe.reset_index()

    for name, shape, value in zip(names, shapes, values):
        assert len(simpledf[name]) == reduce(mul, shape)
        assert_array_equal(dataframe.reset_index()[name].values, value.ravel())
