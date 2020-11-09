from typing import TYPE_CHECKING, Dict, Optional

from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.sqlite.queries import (
    append_shaped_parameter_data_to_existing_arrays, completed,
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
        #: number of rows written per parameter tree (by the name of the dependent parameter)
        self._write_status: Dict[str, Optional[int]] = {}
        self._loaded_from_completed_ds = False
        self._rundescriber: Optional[RunDescriber] = None

    @property
    def rundescriber(self) -> RunDescriber:
        if self._rundescriber is None:
            self._rundescriber = get_rundescriber_from_result_table_name(
                self._dataset.conn, self._dataset.table_name
            )
        return self._rundescriber

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

        (self._write_status,
         self._read_status,
         self._data) = append_shaped_parameter_data_to_existing_arrays(
            self._dataset.conn,
            self._dataset.table_name,
            self.rundescriber,
            self._write_status,
            self._read_status,
            self._data
        )

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
