from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from qcodes.dataset.sqlite.queries import (
    get_interdeps_from_result_table_name, completed, get_non_dependencies,
    get_parameter_data_for_one_paramtree)

if TYPE_CHECKING:
    import pandas as pd
    from .data_set import DataSet, ParameterData


class DataSetCache:
    """
    The DataSetCache contains a in memory representation of the
    data in this dataset as well a a method to progressively read data
    from the db as it is written. The cache is available in the same formats
    as :py:class:`.DataSet.get_parameter_data` and :py:class:`.DataSet.get_data_as_pandas_dataframe`

    """

    def __init__(self, dataset: 'DataSet'):
        self._dataset = dataset
        self._data: ParameterData = {}
        self._read_status: Dict[str, int] = {}
        self._loaded_from_completed_ds = False

    def load_data_from_db(self) -> None:
        """
        Loads data from the dataset into the cache:
        If the dataset is marked completed and data has already been loaded
        no load will be performed.
        """
        if self._loaded_from_completed_ds:
            return

        self._dataset._completed = completed(self._dataset.conn, self._dataset.run_id)
        if self._dataset.completed:
            self._loaded_from_completed_ds = True

        parameters = get_non_dependencies(self._dataset.conn, self._dataset.table_name)
        interdeps = get_interdeps_from_result_table_name(self._dataset.conn, self._dataset.table_name)
        for parameter in parameters:
            start = self._read_status.get(parameter, None)
            if start is not None:
                start += 1

            data, rows = get_parameter_data_for_one_paramtree(self._dataset.conn,
                                                              self._dataset.table_name,
                                                              interdeps=interdeps,
                                                              output_param=parameter,
                                                              start=start,
                                                              end=None)
            if self._data.get(parameter, None) is None:
                self._data[parameter] = data
                self._read_status[parameter] = rows
            elif data == {}:
                pass
            else:
                self._data[parameter] = self._merge_data_dicts_inner(self._data[parameter], data)
                self._read_status[parameter] += rows

    @staticmethod
    def _merge_data_dicts_inner(existing_data: Dict[str, np.ndarray],
                                new_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        merged_data = {}
        for (existing_name, existing_values), (new_name, new_values) in zip(existing_data.items(), new_data.items()):
            assert existing_name == new_name
            merged_data[existing_name] = np.append(existing_values, new_values, axis=0)
        return merged_data

    def data(self) -> 'ParameterData':
        """
        Loads data from the database on disk if needed and returns

        Returns:
            The cached dataset.
        """
        self.load_data_from_db()
        return self._data

    def to_pandas(self) -> Optional[Dict[str, "pd.DataFrame"]]:
        """
        Convert the cached dataset to Pandas dataframes.

        Returns:
            A dict from parameter name to Pandas Dataframes. Each dataframe
            represents one parameter tree.
        """

        self.load_data_from_db()
        if self._data is None:
            return None
        dfs = {}
        for name, subdict in self._data.items():
            index = self._dataset._generate_pandas_index(subdict)
            dfs[name] = self._dataset._data_to_dataframe(subdict, index)
        return dfs
