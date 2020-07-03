from typing import TYPE_CHECKING, Optional, Dict

import numpy as np

from qcodes.dataset.sqlite.queries import completed, get_dataset_num_rows

if TYPE_CHECKING:
    import pandas as pd
    from .data_set import DataSet, ParameterData


class DataSetCache:

    def __init__(self, dataset: 'DataSet'):
        self._dataset = dataset
        self._data: Optional[ParameterData] = None
        self._last_read_row = 0

    def load_data_from_db(self) -> None:
        """
        (RE)Load data from db

        Returns:

        """
        num_rows = get_dataset_num_rows(self._dataset.conn, self._dataset.table_name)
        if num_rows > self._last_read_row:
            new_data_dicts = self._dataset._load_data(start=self._last_read_row + 1, end=num_rows)
            if self._data is None:
                self._data = new_data_dicts
            else:
                self._merge_data_dicts_into_data(new_data_dicts)
            self._last_read_row = num_rows
        self._dataset._completed = completed(self._dataset.conn, self._dataset.run_id)

    def _merge_data_dicts_into_data(self, new_data_dicts: 'ParameterData') -> None:
        if self._data is None:
            raise RuntimeError
        for (old_outer_name, old_outer_data), (new_outer_name, new_outer_data) in zip(self._data.items(),
                                                                                      new_data_dicts.items()):
            merged_inner_dict = {}
            for (old_name, old_value), (new_name, new_value) in zip(old_outer_data.items(), new_outer_data.items()):
                merged_inner_dict[old_name] = np.append(old_value, new_value)
            self._data[old_outer_name] = merged_inner_dict

    def data(self) -> Optional['ParameterData']:
        self.load_data_from_db()
        return self._data

    def as_pandas_dataframe(self) -> Dict[str, "pd.DataFrame"]:
        raise NotImplemented
