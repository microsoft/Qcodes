from typing import Union, Dict, Sequence
from numbers import Number
from numpy import ndarray

from .data_storage_interface import (
    DataStorageInterface, VALUES)
from .sqlite_base import (
    connect, select_one_where, insert_values, insert_many_values)
from qcodes.dataset.database import get_DB_location


class SqliteStorageInterface(DataStorageInterface):
    """
    """
    def __init__(self, guid: str, conn):
        super().__init__(guid)
        self.conn = conn
        self.table_name = select_one_where(self.conn, "runs", "result_table_name", "guid", self.guid)

    def store_results(self, results: Dict[str, VALUES]):
        self._validate_results_dict(results)
        if len(next(iter(results.values()))) == 1:
            insert_values(self.conn, self.table_name,
                            list(results.keys()),
                            [v[0] for v in results.values()])
        else:
            values_transposed = list(map(list, zip(*results.values())))
            insert_many_values(self.conn, self.table_name,
                            list(results.keys()),
                            list(values_transposed))
