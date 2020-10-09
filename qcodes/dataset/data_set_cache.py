from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from qcodes.dataset.sqlite.queries import (
    completed, get_interdeps_from_result_table_name,
    get_parameter_data_for_one_paramtree,
    get_rundescriber_from_result_table_name)

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
        #: number of rows read per parameter tree (by the name of the dependent parameter)
        self._read_status: Dict[str, int] = {}
        self._loaded_from_completed_ds = False

    def load_data_from_db(self) -> None:
        """
        Loads data from the dataset into the cache.
        If new data has been added to the dataset since the last time
        this method was called, calling this method again would load
        that new portion of the data and append to the already loaded data.
        If the dataset is marked completed and data has already been loaded
        no load will be performed.
        """
        if self._loaded_from_completed_ds:
            return
        self._dataset._completed = completed(self._dataset.conn, self._dataset.run_id)
        if self._dataset.completed:
            self._loaded_from_completed_ds = True

        rundescriber = get_rundescriber_from_result_table_name(
            self._dataset.conn,
            self._dataset.table_name
        )
        interdeps = rundescriber.interdeps
        parameters = tuple(ps.name for ps in interdeps.non_dependencies)

        for parameter in parameters:
            start = self._read_status.get(parameter, 0) + 1

            data, n_rows_read = get_parameter_data_for_one_paramtree(
                self._dataset.conn,
                self._dataset.table_name,
                rundescriber=rundescriber,
                output_param=parameter,
                start=start,
                end=None)
            self._data[parameter] = self._merge_data_dicts_inner(
                self._data.get(parameter, {}),
                data
            )
            self._read_status[parameter] = self._read_status.get(parameter, 0) + n_rows_read

    @staticmethod
    def _merge_data_dicts_inner(existing_data: Dict[str, np.ndarray],
                                new_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        merged_data = {}
        parameters = set(existing_data.keys()) | set(new_data.keys())

        for parameter in parameters:
            existing_values = existing_data.get(parameter)
            new_values = new_data.get(parameter)
            if existing_values is not None and new_values is not None:
                merged_data[parameter] = np.append(existing_values, new_values, axis=0)
            elif new_values is not None:
                merged_data[parameter] = new_values
            elif existing_values is not None:
                merged_data[parameter] = existing_values
        return merged_data

    def data(self) -> 'ParameterData':
        """
        Loads data from the database on disk if needed and returns
        the cached data. The cached data is in the same format as :py:class:`.DataSet.get_parameter_data`.

        Returns:
            The cached dataset.
        """
        self.load_data_from_db()
        return self._data

    def to_pandas(self) -> Optional[Dict[str, "pd.DataFrame"]]:
        """
        Convert the cached dataset to Pandas dataframes. The returned dataframes
        are in the same format :py:class:`.DataSet.get_data_as_pandas_dataframe`.

        Returns:
            A dict from parameter name to Pandas Dataframes. Each dataframe
            represents one parameter tree.
        """

        self.load_data_from_db()
        if self._data is None:
            return None
        dfs = self._dataset._load_to_dataframes(self._data)
        return dfs
