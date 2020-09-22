from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.sqlite.queries import (
    completed, get_parameter_data_for_one_paramtree,
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
        self._write_status: Dict[str, Optional[int]] = {}
        self._loaded_from_completed_ds = False
        self.rundescriber: Optional[RunDescriber] = None

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

        if self.rundescriber is None:
            self.rundescriber = get_rundescriber_from_result_table_name(
                self._dataset.conn, self._dataset.table_name
            )

        parameters = tuple(ps.name for ps in
                           self.rundescriber.interdeps.non_dependencies)

        for parameter in parameters:
            start = self._read_status.get(parameter, 0) + 1

            data, n_rows_read = get_parameter_data_for_one_paramtree(
                self._dataset.conn,
                self._dataset.table_name,
                rundescriber=self.rundescriber,
                output_param=parameter,
                start=start,
                end=None
            )
            shapes = self.rundescriber.shapes
            if shapes is not None:
                shape = shapes.get(parameter, None)
            else:
                shape = None

            self._data[parameter], self._write_status = self._merge_data_dicts_inner(
                self._data.get(parameter, {}), data,
                shape=shape,
                write_status=self._write_status
            )
            self._read_status[parameter] = self._read_status.get(parameter, 0) + n_rows_read

    @staticmethod
    def _merge_data_dicts_inner(existing_data: Dict[str, np.ndarray],
                                new_data: Dict[str, np.ndarray],
                                shape: Optional[Tuple[int, ...]],
                                write_status: Dict[str, Optional[int]]
                                ) -> Tuple[Dict[str, np.ndarray],
                                           Dict[str, Optional[int]]]:
        merged_data = {}
        parameters = set(existing_data.keys()) | set(new_data.keys())
        new_write_status: Optional[int]

        for parameter in parameters:
            existing_values = existing_data.get(parameter)
            new_values = new_data.get(parameter)
            if existing_values is not None and new_values is not None:
                merged_data[parameter], new_write_status = DataSetCache._insert_into_data_dict(
                    existing_values,
                    new_values,
                    write_status.get(parameter)
                )
                write_status[parameter] = new_write_status
            elif new_values is not None:
                merged_data[parameter], new_write_status = DataSetCache._create_new_data_dict(
                    new_values,
                    shape
                )
                write_status[parameter] = new_write_status
            elif existing_values is not None:
                merged_data[parameter] = existing_values
        return merged_data, write_status

    @staticmethod
    def _create_new_data_dict(new_values: np.ndarray,
                              shape: Optional[Tuple[int, ...]]
                              ) -> Tuple[np.ndarray, Optional[int]]:
        if shape is None:
            return new_values, None
        else:
            data = np.zeros(shape, dtype=new_values.dtype)
            data[:] = np.nan
            data.ravel()[0:len(new_values)] = new_values
            return data, len(new_values)

    @staticmethod
    def _insert_into_data_dict(
            existing_values: np.ndarray,
            new_values: np.ndarray,
            write_status: Optional[int]) -> Tuple[np.ndarray, Optional[int]]:
        if write_status is None:
            return np.append(existing_values, new_values, axis=0), None
        else:
            existing_values.ravel()[write_status:write_status+len(new_values)] = new_values
            return existing_values, write_status+len(new_values)

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
