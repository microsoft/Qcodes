from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import pandas as pd

    from qcodes.dataset.data_set_protocol import ParameterData
    from qcodes.dataset.descriptions.dependencies import InterDependencies_


def load_to_dataframe_dict(
    datadict: ParameterData, interdeps: InterDependencies_
) -> dict[str, pd.DataFrame]:
    dfs = {}
    for name, subdict in datadict.items():
        index = _generate_pandas_index(
            subdict, interdeps=interdeps, top_level_param_name=name
        )
        dfs[name] = _data_to_dataframe(subdict, index, interdeps, name)
    return dfs


def load_to_concatenated_dataframe(
    datadict: ParameterData, interdeps: InterDependencies_
) -> pd.DataFrame:
    import pandas as pd

    if not _same_setpoints(datadict):
        warnings.warn(
            "Independent parameter setpoints are not equal. "
            "Check concatenated output carefully. Please "
            "consider using `to_pandas_dataframe_dict` to export each "
            "independent parameter to its own dataframe."
        )

    dfs_dict = load_to_dataframe_dict(datadict, interdeps=interdeps)
    df = pd.concat(list(dfs_dict.values()), axis=1)

    return df


def _data_to_dataframe(
    data: Mapping[str, npt.NDArray],
    index: pd.Index | pd.MultiIndex | None,
    interdeps: InterDependencies_,
    dependent_parameter: str,
) -> pd.DataFrame:
    import pandas as pd

    if len(data) == 0:
        return pd.DataFrame()
    dependent_col_name = dependent_parameter
    dependent_data = data[dependent_col_name]
    if dependent_data.dtype == np.dtype("O"):
        # ravel will not fully unpack a numpy array of arrays
        # which are of "object" dtype. This can happen if a variable
        # length array is stored in the db. We use concatenate to
        # flatten these
        mydata = np.concatenate(dependent_data)
    else:
        mydata = dependent_data.ravel()
    df = pd.DataFrame(mydata, index=index, columns=[dependent_col_name])
    return df


def _generate_pandas_index(
    data: Mapping[str, npt.NDArray],
    interdeps: InterDependencies_,
    top_level_param_name: str,
) -> pd.Index | pd.MultiIndex | None:
    # the first element in the dict given by parameter_tree is always the dependent
    # parameter and the index is therefore formed from the rest
    import pandas as pd

    if len(data) == 0:
        return None

    _, deps, _ = interdeps.all_parameters_in_tree_by_group(
        interdeps._node_to_paramspec(top_level_param_name)
    )

    deps_data = {dep.name: data[dep.name] for dep in deps if dep.name in data}

    keys = list(data.keys())
    if len(deps_data) == 0:
        index = None
    elif len(deps_data) == 1:
        index = pd.Index(next(iter(deps_data.values())).ravel(), name=keys[1])
    else:
        index_data = []
        for key in deps_data:
            if data[key].dtype == np.dtype("O"):
                # if we have a numpy array of dtype object,
                # it could either be a variable length array
                # in which case we concatenate it, or it could
                # be a numpy array of scalar objects.
                # In the latter case concatenate will fail
                # with a value error but ravel will produce the
                # correct result
                try:
                    index_data.append(np.concatenate(data[key]))
                except ValueError:
                    index_data.append(data[key].ravel())
            else:
                index_data.append(data[key].ravel())

        index = pd.MultiIndex.from_arrays(index_data, names=list(deps_data.keys()))
    return index


def _parameter_data_identical(
    param_dict_a: Mapping[str, npt.NDArray], param_dict_b: Mapping[str, npt.NDArray]
) -> bool:
    try:
        np.testing.assert_equal(param_dict_a, param_dict_b)
    except AssertionError:
        return False

    return True


def _same_setpoints(datadict: ParameterData) -> bool:
    def _get_setpoints(dd: ParameterData) -> Iterator[dict[str, npt.NDArray]]:
        for dep_name, param_dict in dd.items():
            out = {name: vals for name, vals in param_dict.items() if name != dep_name}
            yield out

    sp_iterator = _get_setpoints(datadict)

    try:
        first = next(sp_iterator)
    except StopIteration:
        return True

    return all(_parameter_data_identical(first, rest) for rest in sp_iterator)
